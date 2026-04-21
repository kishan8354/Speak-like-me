"""
SpeechForge v3.5 — FreeVC + Neural Prosody Edition
════════════════════════════════════════════════════════════════════════════════
BASE:  v3.5 FreeVC Edition (d-freevc.pth + wavlm-base.pt)
NEW:   Six neural prosody modules grafted onto frozen XTTSv2 backbone

FROZEN (XTTSv2 backbone):
  • BERT encoder  • GPT-2  • EnCodec decoder  • BigVGAN vocoder

NEW TRAINABLE MODULES:
  1. ProsodyEncoder      — CoordConv1d + BiGRU  → [T,4] embedding
  2. ProsodyCrossAttn    — gated MultiheadAttn  → GPT-2 hidden states
  3. F0Extractor         — CREPE/WORLD/pyin  (no gradient)
  4. SpeakerNormalizer   — log-domain F0 mapping (no gradient)
  5. F0InjectionLayer    — MLP additive injection (after EnCodec)
  6. AdaINStyleLayer     — per-frame scale+shift  (before BigVGAN)

POST-PROCESSING:  Prosody → Emotion → NeuralProsody → FreeVC → Denoise → BigVGAN → PeakNorm

EVALUATION SUITE (3 modules — production-ready, dataset-level):
  Module 1: Translation Matrix  — FLEURS Hindi→English, BLEU-1/2/3/4 + Overall + chrF + COMET-22
  Module 2: Speaker Similarity  — RAVDESS + LibriTTS, Resemblyzer cosine + MCD
  Module 3: Emotion Preservation — RAVDESS ground-truth labels, accuracy + F1 + confusion
════════════════════════════════════════════════════════════════════════════════
"""
from __future__ import annotations

import csv, io, json, os, re, shutil, sys, tempfile, time, uuid, warnings
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
import torch as _torch
import torch.nn as _nn
import torch.nn.functional as _F
from pydub import AudioSegment

# ── Optional deps ─────────────────────────────────────────────────────────────
try:
    import noisereduce as nr; HAVE_NOISEREDUCE=True
except ImportError:
    HAVE_NOISEREDUCE=False

try:
    from resemblyzer import VoiceEncoder, preprocess_wav; HAVE_RESEMBLYZER=True
except ImportError:
    HAVE_RESEMBLYZER=False

try:
    import num2words as _num2words; HAVE_NUM2WORDS=True
except ImportError:
    HAVE_NUM2WORDS=False

try:
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt; import matplotlib.gridspec as gridspec
    HAVE_MPL=True
except ImportError:
    HAVE_MPL=False

try:
    from pesq import pesq as _pesq_fn; HAVE_PESQ=True
except ImportError:
    HAVE_PESQ=False

try:
    from pystoi import stoi as _stoi_fn; HAVE_STOI=True
except ImportError:
    HAVE_STOI=False

try:
    import crepe as _crepe; HAVE_CREPE=True
except ImportError:
    HAVE_CREPE=False

try:
    import pyworld as _pw; HAVE_PYWORLD=True
except ImportError:
    HAVE_PYWORLD=False

try:
    import sacrebleu as _sacrebleu; HAVE_SACREBLEU=True
except ImportError:
    HAVE_SACREBLEU=False

try:
    from comet import download_model as _comet_dl, load_from_checkpoint as _comet_load
    HAVE_COMET=True
except ImportError:
    HAVE_COMET=False

try:
    import pandas as pd; HAVE_PANDAS=True
except ImportError:
    HAVE_PANDAS=False

try:
    from neural_vocoders.freevc_wrapper import FreeVCWrapper, HAVE_FREEVC
except ImportError:
    HAVE_FREEVC=False; FreeVCWrapper=None

try:
    from neural_vocoders.bigvgan_wrapper import BigVGANVocoder, HAVE_BIGVGAN
except ImportError:
    HAVE_BIGVGAN=False; BigVGANVocoder=None

try:
    from neural_vocoders.breathing_synthesizer import BreathingSynthesizer, HAVE_BREATHING
except ImportError:
    HAVE_BREATHING=False; BreathingSynthesizer=None

from TTS.api import TTS
import importlib.util, site as _site

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
# WHISPER SAFE IMPORT
# ══════════════════════════════════════════════════════════════════════════════

def _load_openai_whisper():
    paths: List[str]=[]
    try: paths+=_site.getsitepackages()
    except: pass
    try: paths.append(_site.getusersitepackages())
    except: pass
    for sp in paths:
        init=os.path.join(sp,"whisper","__init__.py")
        if os.path.exists(init):
            d=os.path.dirname(init)
            spec=importlib.util.spec_from_file_location("openai_whisper",init,submodule_search_locations=[d])
            mod=importlib.util.module_from_spec(spec)
            sys.modules["openai_whisper"]=mod; sys.modules.setdefault("whisper",mod)
            spec.loader.exec_module(mod); return mod
    raise ImportError("openai-whisper not found.")

whisper=_load_openai_whisper()


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(page_title="SpeechForge v3.5 Neural",page_icon="🎙️",
                   layout="wide",initial_sidebar_state="expanded")

MODEL_NAME          ="tts_models/multilingual/multi-dataset/xtts_v2"
DEVICE              ="cuda" if _torch.cuda.is_available() else "cpu"
WHISPER_MODEL_PATH  =str(Path("models")/"large-v3.pt")
VOICES_DIR          =Path("voices")
OUTPUTS_DIR         =Path("outputs")
META_DIR            =OUTPUTS_DIR/"_meta"
EVAL_DIR            =Path("evaluation")
FT_DATASETS_DIR     =Path("datasets")/"xtts_finetune"
PROSODY_CKPT_DIR    =Path("checkpoints")/"prosody_modules"
XTTS_DEFAULT_SPEAKER="Claribel Dervla"
USER_DATASET_ROOT   ="/home/hp/Desktop/tts/AI-Voice-Cloner-XTTS-v2/datasets"

# Neural prosody hyper-parameters
MEL_DIM=80; PROSODY_HIDDEN=256; PROSODY_OUT_DIM=4; GPT2_DIM=1024

SUPPORTED_LANGUAGES=["en","es","fr","de","it","pt","pl","tr","ru","nl","cs","ar","zh-cn","ja","ko","hi"]
VALID_EMOTIONS={"happy","sad","angry","surprised","fearful","neutral"}
EMOTION_EMOJI={"happy":"😄","sad":"😢","angry":"😠","surprised":"😲","fearful":"😨","neutral":"😐"}
EMOTION_PROSODY_PARAMS: Dict[str,Tuple[float,float,float,float]]={
    "happy":(2.5,1.18,1.06,+2.0),"sad":(-2.0,0.82,0.92,-3.0),
    "angry":(1.0,1.35,1.10,+4.0),"fearful":(-1.0,0.90,1.08,+1.5),
    "surprised":(3.0,1.20,1.04,+2.5),"neutral":(0.0,1.00,1.00,0.0),
}
RAVDESS_EMO_MAP={"01":"neutral","02":"neutral","03":"happy","04":"sad",
                  "05":"angry","06":"fearful","07":"surprised","08":"surprised"}

# FLEURS default paths
FLEURS_AUDIO_DIR    =f"{USER_DATASET_ROOT}/fluers/audio"
FLEURS_REF_DIR      =f"{USER_DATASET_ROOT}/fluers/reference_en"
RAVDESS_ROOT        =f"{USER_DATASET_ROOT}/RAVDESS"
CREMAD_ROOT         =f"{USER_DATASET_ROOT}/CREMA-D"
LIBRI_ROOT          =f"{USER_DATASET_ROOT}/train-clean-360"
FLEURS_REF_DIR      =f"{USER_DATASET_ROOT}/fluers/reference_en"
RAVDESS_ROOT        =f"{USER_DATASET_ROOT}/RAVDESS"
LIBRI_ROOT          =f"{USER_DATASET_ROOT}/train-clean-360"

for _d in (VOICES_DIR,OUTPUTS_DIR,META_DIR,EVAL_DIR,FT_DATASETS_DIR,PROSODY_CKPT_DIR):
    _d.mkdir(parents=True,exist_ok=True)

if str(Path(__file__).parent) not in sys.path:
    sys.path.insert(0,str(Path(__file__).parent))


# ══════════════════════════════════════════════════════════════════════════════
# STYLING
# ══════════════════════════════════════════════════════════════════════════════

def apply_styling():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
    :root{--bg:#09090f;--surface:#111118;--surface2:#18181f;--border:#2a2a38;
      --accent:#7c6af7;--accent2:#40c8c0;--text:#e8e8f0;--text-dim:#7b7b9a;
      --good:#3ecf8e;--warn:#f59e0b;--bad:#ef4444;
      --raw-col:#e87c40;--pp-col:#40c8c0;--freevc-col:#f472b6;
      --prosenc:#a78bfa;--f0col:#fb923c;--adain:#38bdf8;--eval-col:#34d399;}
    html,body,.stApp{background:var(--bg)!important;color:var(--text);font-family:'DM Sans',sans-serif;}
    .stApp>header{display:none!important;}
    .block-container{padding:1.5rem 2rem 4rem!important;max-width:1440px;}
    .stTabs [data-baseweb="tab-list"]{background:var(--surface)!important;border-radius:10px;padding:4px;gap:4px;border:1px solid var(--border);}
    .stTabs [data-baseweb="tab"]{background:transparent!important;border-radius:8px!important;color:var(--text-dim)!important;font-weight:500;padding:8px 18px!important;border:none!important;}
    .stTabs [aria-selected="true"]{background:var(--accent)!important;color:white!important;}
    .stTabs [data-baseweb="tab-panel"]{padding-top:1.5rem!important;}
    textarea,input,.stTextInput>div>div>input,.stSelectbox>div>div{background:var(--surface2)!important;color:var(--text)!important;border-radius:8px!important;border:1px solid var(--border)!important;}
    .stButton>button{background:linear-gradient(135deg,var(--accent),#5a52cc)!important;color:white!important;border-radius:8px!important;border:none!important;font-weight:600!important;padding:0.55rem 1.3rem!important;}
    .stButton>button[kind="secondary"]{background:var(--surface2)!important;border:1px solid var(--border)!important;color:var(--text)!important;}
    .stDownloadButton>button{background:linear-gradient(135deg,var(--accent2),#2a8e8a)!important;color:white!important;border-radius:8px!important;font-weight:600!important;}
    .card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1.25rem 1.5rem;margin-bottom:1rem;}
    .card-accent{border-left:3px solid var(--accent);}
    .card-eval{background:var(--surface);border:1px solid rgba(52,211,153,0.35);border-radius:12px;padding:1.25rem 1.5rem;margin-bottom:1rem;}
    .banner-raw{background:rgba(232,124,64,.10);border:1px solid rgba(232,124,64,.30);border-radius:8px;padding:6px 14px;margin-bottom:.75rem;display:flex;align-items:center;gap:8px;}
    .banner-pp{background:rgba(64,200,192,.10);border:1px solid rgba(64,200,192,.30);border-radius:8px;padding:6px 14px;margin-bottom:.75rem;display:flex;align-items:center;gap:8px;}
    .banner-raw span.title{color:var(--raw-col);font-weight:700;font-size:14px;}
    .banner-pp  span.title{color:var(--pp-col);font-weight:700;font-size:14px;}
    .eval-table{width:100%;border-collapse:collapse;font-size:13px;margin-top:.5rem;}
    .eval-table th{color:var(--text-dim);font-weight:600;text-align:left;padding:7px 10px;border-bottom:2px solid var(--border);white-space:nowrap;}
    .eval-table td{padding:6px 10px;border-bottom:1px solid rgba(42,42,56,.5);}
    .eval-table tr:nth-child(even) td{background:rgba(255,255,255,.02);}
    .eval-table td.hi{color:var(--good);font-weight:700;font-family:'Space Mono',monospace;font-size:12px;}
    .eval-table td.med{color:var(--warn);font-weight:600;font-family:'Space Mono',monospace;font-size:12px;}
    .eval-table td.lo{color:var(--bad);font-family:'Space Mono',monospace;font-size:12px;}
    .eval-table td.num{font-family:'Space Mono',monospace;font-size:12px;color:var(--text);}
    .diff-table{width:100%;border-collapse:collapse;font-size:13px;margin-top:.5rem;}
    .diff-table th{color:var(--text-dim);font-weight:500;text-align:left;padding:4px 8px;border-bottom:1px solid var(--border);}
    .diff-table td{padding:4px 8px;}
    .diff-table tr:nth-child(even) td{background:rgba(255,255,255,.02);}
    .diff-val-raw{color:var(--raw-col);font-family:'Space Mono',monospace;font-size:12px;}
    .diff-val-pp{color:var(--pp-col);font-family:'Space Mono',monospace;font-size:12px;}
    .metric-row{display:flex;gap:1rem;flex-wrap:wrap;margin:.5rem 0;}
    .metric-chip{display:inline-flex;align-items:center;gap:6px;padding:5px 12px;border-radius:20px;background:var(--surface2);border:1px solid var(--border);font-size:13px;font-weight:500;}
    .metric-chip.good{border-color:var(--good);color:var(--good);}
    .metric-chip.warn{border-color:var(--warn);color:var(--warn);}
    .metric-chip.bad{border-color:var(--bad);color:var(--bad);}
    .badge{display:inline-block;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600;}
    .badge-purple{background:rgba(124,106,247,.15);color:var(--accent);border:1px solid rgba(124,106,247,.3);}
    .badge-teal{background:rgba(64,200,192,.12);color:var(--accent2);border:1px solid rgba(64,200,192,.25);}
    .badge-green{background:rgba(62,207,142,.12);color:var(--good);border:1px solid rgba(62,207,142,.25);}
    .badge-amber{background:rgba(245,158,11,.12);color:var(--warn);border:1px solid rgba(245,158,11,.3);}
    .badge-raw{background:rgba(232,124,64,.12);color:var(--raw-col);border:1px solid rgba(232,124,64,.3);}
    .badge-pp{background:rgba(64,200,192,.12);color:var(--pp-col);border:1px solid rgba(64,200,192,.3);}
    .badge-freevc{background:rgba(244,114,182,.12);color:var(--freevc-col);border:1px solid rgba(244,114,182,.3);}
    .badge-prosenc{background:rgba(167,139,250,.12);color:var(--prosenc);border:1px solid rgba(167,139,250,.3);}
    .badge-eval{background:rgba(52,211,153,.12);color:var(--eval-col);border:1px solid rgba(52,211,153,.3);}
    .mono{font-family:'Space Mono',monospace;font-size:12px;color:var(--text-dim);}
    .emo-bar-bg{background:var(--border);border-radius:4px;height:8px;margin-top:4px;}
    .emo-bar-fill{height:8px;border-radius:4px;}
    .freevc-panel{background:rgba(244,114,182,.06);border:1px solid rgba(244,114,182,.25);border-radius:10px;padding:.75rem 1rem;margin:.5rem 0;}
    .neural-card{background:var(--surface2);border:1px solid rgba(167,139,250,.3);border-radius:10px;padding:.75rem 1rem;margin:.4rem 0;}
    .stage-card{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1rem 1.25rem;margin:.5rem 0;}
    .eval-log-box{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:.75rem 1rem;font-family:'Space Mono',monospace;font-size:11px;color:var(--text-dim);max-height:220px;overflow-y:auto;}
    .improve-pill{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:700;background:rgba(62,207,142,.12);color:var(--good);border:1px solid rgba(62,207,142,.3);}
    .worsen-pill{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:700;background:rgba(239,68,68,.12);color:var(--bad);border:1px solid rgba(239,68,68,.3);}
    hr{border-color:var(--border)!important;margin:1.5rem 0;}
    .css-1d391kg,[data-testid="stSidebar"]{background:var(--surface)!important;border-right:1px solid var(--border)!important;}
    .stProgress>div>div{background:var(--accent)!important;}
    .overall-bleu-banner{background:rgba(52,211,153,.07);border:2px solid rgba(52,211,153,.45);
      border-radius:12px;padding:1rem 1.5rem;margin-bottom:1rem;text-align:center;}
    </style>""",unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

def init_state():
    defaults={
        "speaker_path":None,"ref_display_name":None,"active_voice":None,
        "multi_ref_paths":[],"pending_delete":None,
        "text_input":("Hello! This is my cloned voice. It should sound natural and expressive, "
                      "closely matching the reference speaker's tone and speaking style."),
        "stt_english_text":None,"stt_original_text":None,"stt_source_lang":None,
        "stt_detected_emotion":None,"stt_emotion_confidence":None,
        "stt_emotion_method":None,"stt_audio_path":None,"stt_prosody":None,
        "last_quality_metrics":{},"last_quality_metrics_raw":{},
        "prosody_training_stage":1,
        "eval_m1_results":None,
        "eval_m2_results":None,
        "eval_m3_results":None,
    }
    for k,v in defaults.items():
        if k not in st.session_state: st.session_state[k]=v

init_state()
apply_styling()


# ══════════════════════════════════════════════════════════════════════════════
# NEURAL PROSODY MODULES
# ══════════════════════════════════════════════════════════════════════════════

class CoordConv1d(_nn.Module):
    def __init__(self,in_ch,out_ch,kernel=3,padding="same"):
        super().__init__()
        self.conv=_nn.Conv1d(in_ch+1,out_ch,kernel,padding=padding)
    def forward(self,x):
        B,C,T=x.shape
        c=_torch.linspace(0.,1.,T,device=x.device).view(1,1,T).expand(B,1,T)
        return self.conv(_torch.cat([x,c],1))


class ProsodyEncoder(_nn.Module):
    def __init__(self,mel_dim=MEL_DIM,hidden=PROSODY_HIDDEN,out_dim=PROSODY_OUT_DIM,n_layers=2,dropout=0.10):
        super().__init__()
        self.front=_nn.Sequential(CoordConv1d(mel_dim,hidden,5),_nn.ReLU(),_nn.BatchNorm1d(hidden),
                                   CoordConv1d(hidden,hidden,3),_nn.ReLU(),_nn.BatchNorm1d(hidden))
        self.gru=_nn.GRU(hidden,hidden//2,n_layers,batch_first=True,bidirectional=True,
                          dropout=dropout if n_layers>1 else 0.)
        self.proj=_nn.Linear(hidden,out_dim); self.norm=_nn.LayerNorm(out_dim)
    def forward(self,mel):
        x=self.front(mel); x,_=self.gru(x.transpose(1,2)); return self.norm(self.proj(x))
    def encode_audio(self,audio,sr):
        try:
            if sr!=22050: audio=librosa.resample(audio,orig_sr=sr,target_sr=22050); sr=22050
            mel=librosa.feature.melspectrogram(y=audio,sr=sr,n_mels=MEL_DIM,n_fft=1024,hop_length=256)
            mel_db=librosa.power_to_db(mel,ref=np.max).astype(np.float32)
            with _torch.no_grad(): return self.forward(_torch.from_numpy(mel_db).unsqueeze(0))
        except: return None


class ProsodyCrossAttention(_nn.Module):
    def __init__(self,q_dim=GPT2_DIM,p_dim=PROSODY_OUT_DIM,heads=8,dropout=0.10):
        super().__init__()
        self.proj=_nn.Linear(p_dim,q_dim)
        self.attn=_nn.MultiheadAttention(q_dim,heads,dropout=dropout,batch_first=True)
        self.norm=_nn.LayerNorm(q_dim); self.gate=_nn.Parameter(_torch.zeros(1))
        self.drop=_nn.Dropout(dropout)
    def forward(self,hidden,pros):
        kv=self.proj(pros); out,_=self.attn(hidden,kv,kv)
        return self.norm(hidden+_torch.sigmoid(self.gate)*self.drop(out))


class F0Extractor:
    @staticmethod
    def extract(audio,sr,hop_ms=10.0):
        hop=int(sr*hop_ms/1000)
        if HAVE_CREPE:
            try:
                a16=librosa.resample(audio,orig_sr=sr,target_sr=16000) if sr!=16000 else audio
                _,f0,conf,_=_crepe.predict(a16,16000,step_size=hop_ms,viterbi=True,verbose=0)
                return f0.astype(np.float32),(conf>0.45),int(16000*hop_ms/1000)
            except: pass
        if HAVE_PYWORLD:
            try:
                f0,_=_pw.harvest(audio.astype(np.float64),sr,frame_period=hop_ms)
                f0=f0.astype(np.float32); return f0,(f0>50.),int(sr*hop_ms/1000)
            except: pass
        try:
            f0,vf,_=librosa.pyin(audio,fmin=librosa.note_to_hz("C2"),fmax=librosa.note_to_hz("C7"),
                                  sr=sr,hop_length=hop,fill_na=0.)
            f0=np.nan_to_num(f0,nan=0.).astype(np.float32)
            return f0,(vf if vf is not None else f0>0),hop
        except:
            n=max(1,len(audio)//hop); return np.zeros(n,np.float32),np.zeros(n,bool),hop


class SpeakerNormalizer:
    @staticmethod
    def normalise(f0_src,voiced,ref_f0=None,ref_voiced=None):
        vmask=(f0_src>50.)&voiced
        if not np.any(vmask): return f0_src.copy()
        ls=np.log(f0_src[vmask]+1e-6); mu_s,sig_s=float(np.mean(ls)),float(np.std(ls))+1e-6
        if ref_f0 is not None and ref_voiced is not None:
            rm=(ref_f0>50.)&ref_voiced
            if np.any(rm):
                lr=np.log(ref_f0[rm]+1e-6); mu_t,sig_t=float(np.mean(lr)),float(np.std(lr))+1e-6
            else: mu_t,sig_t=mu_s,sig_s
        else: mu_t,sig_t=0.,1.
        out=f0_src.copy(); out[vmask]=np.exp((ls-mu_s)/sig_s*sig_t+mu_t); out[~vmask]=0.
        return out.astype(np.float32)


class F0InjectionLayer(_nn.Module):
    def __init__(self,f0_dim=1,mel_dim=MEL_DIM):
        super().__init__()
        self.mlp=_nn.Sequential(_nn.Linear(f0_dim,64),_nn.GELU(),_nn.Linear(64,mel_dim))
        self.norm=_nn.LayerNorm(mel_dim)
    def forward(self,mel,f0): return self.norm(mel+self.mlp(f0))
    def apply_np(self,mel_arr,f0_arr):
        try:
            m=_torch.from_numpy(mel_arr).float().unsqueeze(0)
            f=_torch.from_numpy(f0_arr).float().unsqueeze(0).unsqueeze(-1)
            if f.shape[1]!=m.shape[1]:
                f=_F.interpolate(f.permute(0,2,1),size=m.shape[1],mode="linear",align_corners=False).permute(0,2,1)
            with _torch.no_grad(): return self.forward(m,f).squeeze(0).numpy()
        except: return mel_arr


class AdaINStyleLayer(_nn.Module):
    def __init__(self,mel_dim=MEL_DIM,p_dim=PROSODY_OUT_DIM):
        super().__init__()
        self.proj=_nn.Linear(p_dim,mel_dim*2); self.norm=_nn.InstanceNorm1d(mel_dim,affine=False)
    def forward(self,mel,pros):
        s,sh=self.proj(pros).chunk(2,dim=-1)
        return self.norm(mel.transpose(1,2)).transpose(1,2)*(1.+s)+sh
    def apply_np(self,mel_arr,pros_arr):
        try:
            m=_torch.from_numpy(mel_arr).float().unsqueeze(0)
            p=_torch.from_numpy(pros_arr).float().unsqueeze(0)
            if p.shape[1]!=m.shape[1]:
                p=_F.interpolate(p.permute(0,2,1),size=m.shape[1],mode="linear",align_corners=False).permute(0,2,1)
            with _torch.no_grad(): return self.forward(m,p).squeeze(0).numpy()
        except: return mel_arr


@dataclass
class ProsodyBundle:
    encoder: ProsodyEncoder; cross_attn: ProsodyCrossAttention
    f0_inj: F0InjectionLayer; adain: AdaINStyleLayer; stage: int=1
    def eval_mode(self):
        for m in (self.encoder,self.cross_attn,self.f0_inj,self.adain): m.eval()
        return self
    def save(self,path:Path):
        path.mkdir(parents=True,exist_ok=True)
        for mod,fn in [(self.encoder,"enc.pt"),(self.cross_attn,"ca.pt"),
                        (self.f0_inj,"f0.pt"),(self.adain,"adain.pt")]:
            _torch.save(mod.state_dict(),path/fn)
    def load(self,path:Path):
        for mod,fn in [(self.encoder,"enc.pt"),(self.cross_attn,"ca.pt"),
                        (self.f0_inj,"f0.pt"),(self.adain,"adain.pt")]:
            p=path/fn
            if p.exists(): mod.load_state_dict(_torch.load(p,map_location="cpu"))


def _make_bundle(stage=1)->ProsodyBundle:
    b=ProsodyBundle(encoder=ProsodyEncoder().to(DEVICE),cross_attn=ProsodyCrossAttention().to(DEVICE),
                     f0_inj=F0InjectionLayer().to(DEVICE),adain=AdaINStyleLayer().to(DEVICE),stage=stage)
    b.load(PROSODY_CKPT_DIR); return b


# ══════════════════════════════════════════════════════════════════════════════
# MODEL LOADERS
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=True)
def load_tts_model()->TTS: return TTS(MODEL_NAME,progress_bar=True).to(DEVICE)

@st.cache_resource(show_spinner=True)
def load_whisper_model():
    if os.path.exists(WHISPER_MODEL_PATH): return whisper.load_model(WHISPER_MODEL_PATH,device=DEVICE)
    return whisper.load_model("large-v3",device=DEVICE)

@st.cache_resource(show_spinner=False)
def load_silero_vad():
    try:
        m,u=_torch.hub.load("snakers4/silero-vad","silero_vad",force_reload=False,onnx=False,trust_repo=True)
        return m,u
    except Exception as e: return None,str(e)

@st.cache_resource(show_spinner=False)
def load_resemblyzer():
    if not HAVE_RESEMBLYZER: return None,"not installed"
    try: return VoiceEncoder(device=DEVICE),None
    except Exception as e: return None,str(e)

@st.cache_resource(show_spinner=False)
def load_freevc():
    if not HAVE_FREEVC or FreeVCWrapper is None: return None,"FreeVC not available"
    try:
        w=FreeVCWrapper(device=DEVICE); return (w,None) if w.is_available() else (None,w.get_error())
    except Exception as e: return None,str(e)

@st.cache_resource(show_spinner=False)
def load_bigvgan():
    if not HAVE_BIGVGAN or BigVGANVocoder is None: return None,"bigvgan not installed"
    try:
        v=BigVGANVocoder(device=DEVICE); return (v,None) if v.is_available() else (None,v.get_error())
    except Exception as e: return None,str(e)

@st.cache_resource(show_spinner=False)
def load_prosody_bundle(stage:int=1)->ProsodyBundle: return _make_bundle(stage)

@st.cache_resource(show_spinner=False)
def load_comet_model():
    if not HAVE_COMET: return None,"pip install unbabel-comet"
    try:
        path=_comet_dl("Unbabel/wmt22-comet-da"); model=_comet_load(path); return model,None
    except Exception as e: return None,str(e)


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _clamp(v,lo,hi): return max(lo,min(hi,v))
def _ts(): return datetime.now().strftime("%Y%m%d_%H%M%S")
def _extract_bad_kwarg(err):
    m=re.search(r"unexpected keyword argument ['\"]([^'\"]+)['\"]",err)
    return m.group(1) if m else None

def _chip(val,label,good=0.65,warn=0.4):
    cls="good" if val>=good else("warn" if val>=warn else "bad")
    return f'<span class="metric-chip {cls}">{label}</span>'

def light_denoise(audio,sr,strength=0.3):
    if not HAVE_NOISEREDUCE: return audio
    try:
        nl=min(int(0.2*sr),len(audio)//4); noise=audio[:nl] if len(audio)>nl else audio
        return nr.reduce_noise(y=audio,sr=sr,y_noise=noise,prop_decrease=min(strength,0.8),stationary=True)
    except: return audio


class TextNormalizer:
    _AB={r"\bMr\.":"Mister",r"\bMrs\.":"Missus",r"\bDr\.":"Doctor",r"\bSt\.":"Saint",
         r"\betc\.":"et cetera",r"\bvs\.":"versus",r"\be\.g\.":"for example",r"\bi\.e\.":"that is"}
    @classmethod
    def normalize(cls,text):
        for p,r in cls._AB.items(): text=re.sub(p,r,text)
        if HAVE_NUM2WORDS:
            def _rn(m):
                try: return _num2words.num2words(int(m.group(0).replace(",","")))
                except: return m.group(0)
            text=re.sub(r"\b\d{1,6}(?:,\d{3})*\b",_rn,text)
        return re.sub(r"\s{2,}"," ",text).strip()


# ══════════════════════════════════════════════════════════════════════════════
# AUDIO PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

class AudioPipeline:
    TARGET_SR=16000
    @staticmethod
    def vad_trim(audio,sr,threshold=0.35):
        m,u=load_silero_vad()
        if m is None: return AudioPipeline._evad(audio,sr)
        try:
            (gts,*_)=u; at=_torch.from_numpy(audio).float()
            if sr!=16000:
                import torchaudio; at=torchaudio.functional.resample(at,sr,16000); sv=16000
            else: sv=sr
            ts=gts(at,m,threshold=threshold,min_speech_duration_ms=250,sampling_rate=sv)
            if not ts: return audio
            f=sr/sv
            return np.concatenate([audio[int(t["start"]*f):int(t["end"]*f)] for t in ts])
        except: return AudioPipeline._evad(audio,sr)
    @staticmethod
    def _evad(audio,sr,tdb=28):
        ivs=librosa.effects.split(audio,top_db=tdb)
        ch=[audio[s:e] for s,e in ivs if (e-s)/sr>=0.1]
        return np.concatenate(ch) if ch else audio
    @staticmethod
    def denoise(audio,sr,strength=0.8):
        if not HAVE_NOISEREDUCE: return audio
        try:
            noise=audio[:int(0.5*sr)] if len(audio)>int(0.5*sr) else audio
            return nr.reduce_noise(y=audio,sr=sr,y_noise=noise,prop_decrease=_clamp(strength,0.,1.),stationary=False)
        except: return audio
    @staticmethod
    def normalize_peak(audio,target=0.97):
        pk=float(np.max(np.abs(audio))); return audio*(target/pk) if pk>1e-6 else audio
    @classmethod
    def preprocess(cls,file_bytes,do_vad=True,do_denoise=True,denoise_strength=0.8):
        audio,sr=librosa.load(io.BytesIO(file_bytes),sr=None,mono=True)
        dr=len(audio)/sr
        if do_vad: audio=cls.vad_trim(audio,sr)
        dc=len(audio)/sr
        if do_denoise: audio=cls.denoise(audio,sr,denoise_strength)
        if sr!=cls.TARGET_SR: audio=librosa.resample(audio,orig_sr=sr,target_sr=cls.TARGET_SR); sr=cls.TARGET_SR
        audio=cls.normalize_peak(audio); snr=_estimate_snr(audio,sr)
        out=VOICES_DIR/f"ref_{int(time.time())}_{uuid.uuid4().hex[:6]}.wav"
        sf.write(str(out),audio,sr,subtype="PCM_16")
        return str(out),{"duration_raw_s":round(dr,2),"duration_clean_s":round(dc,2),"snr_db":round(snr,1),"sample_rate":sr}


def _estimate_snr(audio,sr):
    try:
        ivs=librosa.effects.split(audio,top_db=30)
        if not len(ivs): return 0.
        sp=np.concatenate([audio[s:e] for s,e in ivs])
        mask=np.ones(len(audio),bool)
        for s,e in ivs: mask[s:e]=False
        noise=audio[mask]
        if len(noise)<100: return 40.
        return float(20*np.log10(np.sqrt(np.mean(sp**2))/(np.sqrt(np.mean(noise**2))+1e-9)))
    except: return 0.


# ══════════════════════════════════════════════════════════════════════════════
# PROSODY ANALYZER
# ══════════════════════════════════════════════════════════════════════════════

class ProsodyAnalyzer:
    @staticmethod
    def extract(audio_path):
        try:
            y,sr=librosa.load(audio_path,sr=None,mono=True)
            return ProsodyAnalyzer.extract_array(y,sr)
        except Exception as e: return {"ok":False,"error":str(e)}
    @staticmethod
    def extract_array(audio,sr):
        try:
            f0,voiced,_=F0Extractor.extract(audio,sr)
            f0v=f0[(f0>50.)&voiced]
            rms=librosa.feature.rms(y=audio)[0]; sc=librosa.feature.spectral_centroid(y=audio,sr=sr)[0]
            vr=len(f0v)/(len(audio)/sr*100+1e-6)
            return {"f0_mean":round(float(np.mean(f0v)) if len(f0v) else 150.,2),
                    "f0_std":round(float(np.std(f0v)) if len(f0v)>1 else 20.,2),
                    "f0_min":round(float(np.min(f0v)) if len(f0v) else 100.,2),
                    "f0_max":round(float(np.max(f0v)) if len(f0v) else 250.,2),
                    "energy_mean":round(float(np.mean(rms)),5),
                    "energy_std":round(float(np.std(rms)),5),
                    "speaking_rate_proxy":round(float(vr),3),
                    "spectral_centroid_mean":round(float(np.mean(sc)),1),
                    "duration_s":round(len(audio)/sr,2),"sample_rate":sr,
                    "f0_extractor":"crepe" if HAVE_CREPE else ("world" if HAVE_PYWORLD else "pyin"),
                    "ok":True}
        except Exception as e: return {"ok":False,"error":str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# PROSODY + EMOTION TRANSFER
# ══════════════════════════════════════════════════════════════════════════════

class ProsodyTransfer:
    @staticmethod
    def match_f0(audio,sr,ref,out,strength=1.0):
        if not ref.get("ok") or not out.get("ok"): return audio
        r,o=ref.get("f0_mean",150.),out.get("f0_mean",150.)
        if r<10 or o<10: return audio
        n=float(np.clip(12*np.log2(r/(o+1e-6)),-12,12))*strength
        if abs(n)<0.3: return audio
        try: return librosa.effects.pitch_shift(audio,sr=sr,n_steps=n,bins_per_octave=24)
        except: return audio
    @staticmethod
    def match_energy(audio,ref,out,strength=1.0):
        if not ref.get("ok") or not out.get("ok"): return audio
        rr,ro=ref.get("energy_mean",0.05),out.get("energy_mean",0.05)
        if rr<1e-6 or ro<1e-6: return audio
        gain=float(np.clip(1.+(rr/(ro+1e-6)-1.)*strength,0.3,3.))
        sc=audio*gain; pk=np.max(np.abs(sc)); return sc*(0.98/pk) if pk>0.98 else sc
    @staticmethod
    def match_rate(audio,sr,ref,out,strength=0.5):
        if not ref.get("ok") or not out.get("ok"): return audio
        rr,ro=ref.get("speaking_rate_proxy",1.),out.get("speaking_rate_proxy",1.)
        if ro<0.01: return audio
        r=float(np.clip(1.+(rr/(ro+1e-6)-1.)*strength,0.8,1.25))
        if abs(r-1.)<0.03: return audio
        try: return librosa.effects.time_stretch(audio,rate=r)
        except: return audio
    @classmethod
    def transfer(cls,wav_path,ref_prosody,f0_strength=0.75,energy_strength=0.6,rate_strength=0.3):
        if not ref_prosody or not ref_prosody.get("ok"): return wav_path
        try:
            audio,sr=librosa.load(wav_path,sr=None,mono=True)
            op=ProsodyAnalyzer.extract(wav_path)
            if f0_strength>0: audio=cls.match_f0(audio,sr,ref_prosody,op,f0_strength)
            if energy_strength>0: audio=cls.match_energy(audio,ref_prosody,op,energy_strength)
            if rate_strength>0: audio=cls.match_rate(audio,sr,ref_prosody,op,rate_strength)
            out=wav_path.replace(".wav","_pt.wav"); sf.write(out,audio,sr,subtype="PCM_16"); return out
        except: return wav_path


class EmotionPostProcessor:
    @classmethod
    def apply(cls,wav_path,emotion,intensity=0.6):
        emotion=emotion.lower().strip()
        if emotion not in EMOTION_PROSODY_PARAMS or emotion=="neutral" or intensity<0.02: return wav_path
        ps,eg,tr,td=EMOTION_PROSODY_PARAMS[emotion]
        ep=ps*intensity; ee=1.+(eg-1.)*intensity
        et=float(np.clip(1.+(tr-1.)*intensity,0.80,1.30)); ed=td*intensity
        try:
            audio,sr=librosa.load(wav_path,sr=None,mono=True)
            if abs(ep)>=0.15:
                audio=librosa.effects.pitch_shift(audio,sr=sr,n_steps=float(np.clip(ep,-6,6)),bins_per_octave=24)
            if abs(et-1.)>=0.02: audio=librosa.effects.time_stretch(audio,rate=et)
            audio=audio*float(np.clip(ee,0.4,2.5))
            pk=float(np.max(np.abs(audio)))
            if pk>0.97: audio=audio*(0.97/pk)
            if abs(ed)>=0.5:
                try:
                    D=librosa.stft(audio,n_fft=2048,hop_length=512); freqs=librosa.fft_frequencies(sr=sr,n_fft=2048)
                    mask=np.power(10.,(ed*freqs/(sr/2.+1e-9))/20.)
                    out2=librosa.istft(D*mask[:,None],hop_length=512,length=len(audio))
                    pk2=float(np.max(np.abs(out2))); audio=out2*(0.97/pk2) if pk2>0.97 else out2
                except: pass
            out=wav_path.replace(".wav","_emo.wav"); sf.write(out,audio,sr,subtype="PCM_16"); return out
        except: return wav_path


def apply_neural_prosody(wav_path,ref_path,bundle:ProsodyBundle,f0_strength=0.80,adain_strength=0.70):
    try:
        ref_audio,ref_sr=librosa.load(ref_path,sr=22050,mono=True)
        out_audio,out_sr=librosa.load(wav_path,sr=22050,mono=True)
        bundle.eval_mode()
        ref_emb=bundle.encoder.encode_audio(ref_audio,ref_sr)
        if ref_emb is None: return wav_path
        out_mel=librosa.feature.melspectrogram(y=out_audio,sr=out_sr,n_mels=MEL_DIM,n_fft=1024,hop_length=256).astype(np.float32)
        ref_f0,ref_v,_=F0Extractor.extract(ref_audio,ref_sr)
        out_f0,out_v,_=F0Extractor.extract(out_audio,out_sr)
        f0_norm=SpeakerNormalizer.normalise(out_f0,out_v,ref_f0,ref_v)
        mel_proc=out_mel.T.copy()
        if bundle.stage>=2 and f0_strength>0.:
            mf=bundle.f0_inj.apply_np(mel_proc,f0_norm)
            mel_proc=f0_strength*mf+(1.-f0_strength)*mel_proc
        if bundle.stage>=3 and adain_strength>0.:
            ref_np=ref_emb.squeeze(0).numpy()
            if ref_np.shape[0]!=mel_proc.shape[0]:
                t=_torch.from_numpy(ref_np).unsqueeze(0).permute(0,2,1)
                t=_F.interpolate(t,size=mel_proc.shape[0],mode="linear",align_corners=False).permute(0,2,1)
                ref_np=t.squeeze(0).numpy()
            ma=bundle.adain.apply_np(mel_proc,ref_np)
            mel_proc=adain_strength*ma+(1.-adain_strength)*mel_proc
        mel_power=librosa.db_to_power(np.maximum(mel_proc.T,librosa.power_to_db(np.full((MEL_DIM,1),1e-10))))
        audio_out=librosa.feature.inverse.mel_to_audio(mel_power,sr=out_sr,n_fft=1024,hop_length=256,n_iter=64)
        pk=float(np.max(np.abs(audio_out)))
        if pk>1e-6: audio_out=audio_out*(0.97/pk)
        out_p=wav_path.replace(".wav","_npros.wav")
        sf.write(out_p,audio_out.astype(np.float32),out_sr,subtype="PCM_16"); return out_p
    except: return wav_path


# ══════════════════════════════════════════════════════════════════════════════
# EMOTION DETECTION  —  SpeechEmoNet  +  rule-based fallback
# ══════════════════════════════════════════════════════════════════════════════

_EMO_NET_PATH    = ("/home/hp/Desktop/tts/AI-Voice-Cloner-XTTS-v2"
                    "/emotion_detection/saved_models/speechemo_net.pt")
_EMO_SCALER_PATH = ("/home/hp/Desktop/tts/AI-Voice-Cloner-XTTS-v2"
                    "/emotion_detection/saved_models/speechemo_scaler.pkl")
_EMO_7_CLASSES   = ["angry","disgust","fearful","happy","neutral","sad","surprised"]
_EMO_6_MAP       = {"angry":"angry","disgust":"angry","fearful":"fearful",
                    "happy":"happy","neutral":"neutral","sad":"sad","surprised":"surprised"}
_EMO_FEATURE_DIM = 248
_EMO_MAX_FRAMES  = 128


@st.cache_resource(show_spinner=False)
def _emo_load_net():
    import pickle as _pkl
    if not Path(_EMO_NET_PATH).exists():
        return None, None, None, f"Not found: {_EMO_NET_PATH}"
    try:
        ckpt = _torch.load(_EMO_NET_PATH, map_location="cpu")
        sd   = (ckpt.get("model_state_dict") or ckpt.get("state_dict") or
                (ckpt if all(isinstance(v, _torch.Tensor) for v in ckpt.values()) else None))
        if sd is None: raise ValueError("No state_dict in checkpoint")
        cfg         = ckpt.get("config", {}) or {}
        classes     = cfg.get("classes", _EMO_7_CLASSES)
        feature_dim = int(cfg.get("feature_dim", _EMO_FEATURE_DIM))
        max_frames  = int(cfg.get("max_frames",  _EMO_MAX_FRAMES))
        model       = _SpeechEmoNet(feature_dim, max_frames, len(classes))
        model.load_state_dict(sd, strict=False)
        model.eval()
        if DEVICE == "cuda": model = model.cuda()
        model._feature_dim = feature_dim
        model._max_frames  = max_frames
        model._classes     = classes
        scaler = None
        if Path(_EMO_SCALER_PATH).exists():
            with open(_EMO_SCALER_PATH, "rb") as f: scaler = _pkl.load(f)
        return model, scaler, classes, "speechemo-net"
    except Exception as e:
        return None, None, None, f"Load failed: {e}"


def _emo_extract_features(audio_path, scaler=None,
                           max_frames=_EMO_MAX_FRAMES,
                           feature_dim=_EMO_FEATURE_DIM):
    try:
        y, sr = librosa.load(audio_path, sr=22050, mono=True)
        hop, n_fft = 512, 2048
        mel   = librosa.power_to_db(
                    librosa.feature.melspectrogram(y=y,sr=sr,n_mels=128,n_fft=n_fft,hop_length=hop),
                    ref=np.max).astype(np.float32)
        mfcc  = librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40,n_fft=n_fft,hop_length=hop).astype(np.float32)
        dmfcc = librosa.feature.delta(mfcc).astype(np.float32)
        ddmfc = librosa.feature.delta(mfcc, order=2).astype(np.float32)
        feat  = np.vstack([mel, mfcc, dmfcc, ddmfc])
        if feat.shape[0] > feature_dim:  feat = feat[:feature_dim]
        elif feat.shape[0] < feature_dim: feat = np.pad(feat,((0,feature_dim-feat.shape[0]),(0,0)))
        T = feat.shape[1]
        if T >= max_frames:
            s = (T-max_frames)//2; feat = feat[:,s:s+max_frames]
        else:
            feat = np.pad(feat,((0,0),(0,max_frames-T)))
        if scaler is not None:
            try:  feat = scaler.transform(feat.T).T.astype(np.float32)
            except Exception:
                try: feat = ((feat.T-scaler.mean_)/(scaler.scale_+1e-8)).T.astype(np.float32)
                except: pass
        t = _torch.from_numpy(feat).unsqueeze(0).unsqueeze(0)
        if DEVICE == "cuda": t = t.cuda()
        return t
    except Exception:
        return None


def _emo_classify(audio_path):
    model, scaler, classes, err = _emo_load_net()
    if model is None: return "neutral", 0.0
    fd  = getattr(model, "_feature_dim", _EMO_FEATURE_DIM)
    mf  = getattr(model, "_max_frames",  _EMO_MAX_FRAMES)
    cl  = getattr(model, "_classes",     classes or _EMO_7_CLASSES)
    inp = _emo_extract_features(audio_path, scaler, mf, fd)
    if inp is None: return "neutral", 0.0
    try:
        model.eval()
        with _torch.no_grad():
            probs = _torch.softmax(model(inp), dim=-1)[0].cpu().numpy()
        conf6 = {}
        for i, lbl in enumerate(cl):
            mapped = _EMO_6_MAP.get(lbl.lower(), "neutral")
            conf6[mapped] = conf6.get(mapped, 0.) + float(probs[i])
        best = max(conf6, key=conf6.get)
        return best, round(float(conf6[best]), 4)
    except Exception:
        return "neutral", 0.0


def _emo_rule_fallback(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        e  = float(np.clip(np.mean(librosa.feature.rms(y=y))/0.15, 0, 1))
        z  = float(np.clip(np.mean(librosa.feature.zero_crossing_rate(y=y))/0.20, 0, 1))
        sc_mean = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        sf_mean = float(np.mean(librosa.feature.spectral_flatness(y=y)))
        try:
            f0, vf, _ = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=sr)
            f0v = f0[vf][~np.isnan(f0[vf])] if vf is not None and np.any(vf) else np.array([])
            pm  = float(np.clip((np.mean(f0v)-50)/350, 0, 1)) if len(f0v) else 0.3
            pv  = float(np.clip(np.std(f0v)/100,       0, 1)) if len(f0v) > 1 else 0.
        except: pm, pv = 0.3, 0.
        sc_n = float(np.clip(sc_mean/4000, 0, 1))
        scores = {
            "angry":    0.35*e +0.20*z +0.10*pv+0.15*pm+0.20*sc_n,
            "happy":    0.20*e +0.15*z +0.20*pv+0.20*pm+0.25*sc_n,
            "sad":      0.25*(1-e)+0.20*(1-z)+0.20*(1-pv)+0.15*(1-pm)+0.20*(1-sc_n),
            "fearful":  0.15*e +0.25*z +0.25*pv+0.15*(1-pm)+0.20*sc_n,
            "surprised":0.20*e +0.10*z +0.30*pv+0.25*pm+0.15*sc_n,
            "neutral":  0.15*(1-e)+0.15*(1-z)+0.15*(1-pv)+0.15*(1-pm)+0.20*(1-sc_n)+0.20*(1-sf_mean),
        }
        sv  = np.array(list(scores.values())); sv = sv / sv.sum()
        idx = int(np.argmax(sv))
        return list(scores.keys())[idx], round(float(sv[idx]), 4)
    except: return "neutral", 0.17


def detect_emotion(audio_path, mode="best"):
    emo, conf = _emo_classify(audio_path)
    if conf > 0.15:
        return {"emotion": emo, "confidence": conf, "method": "speechemo-net"}
    emo, conf = _emo_rule_fallback(audio_path)
    return {"emotion": emo, "confidence": conf, "method": "rule-based"}


# ══════════════════════════════════════════════════════════════════════════════
# SPEAKER EMBEDDINGS + AUDIO QUALITY
# ══════════════════════════════════════════════════════════════════════════════

class SpeakerEmbeddingManager:
    @staticmethod
    def extract(audio_path):
        if not HAVE_RESEMBLYZER: return None
        enc,_=load_resemblyzer()
        if enc is None: return None
        try: return enc.embed_utterance(preprocess_wav(audio_path))
        except: return None
    @staticmethod
    def cos_sim(a,b):
        if a is None or b is None: return 0.
        try: return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-9))
        except: return 0.
    @staticmethod
    def sim_score(ref_path,out_path):
        return SpeakerEmbeddingManager.cos_sim(
            SpeakerEmbeddingManager.extract(ref_path),
            SpeakerEmbeddingManager.extract(out_path))
    @staticmethod
    def average_embeddings(paths):
        if not HAVE_RESEMBLYZER: return None
        embs=[e for p in paths if (e:=SpeakerEmbeddingManager.extract(p)) is not None]
        if not embs: return None
        avg=np.mean(embs,axis=0); return avg/(np.linalg.norm(avg)+1e-9)


class AudioQuality:
    @staticmethod
    def estimate_snr(audio,sr): return _estimate_snr(audio,sr)
    @staticmethod
    def spectral_clarity(audio,sr):
        try:
            sc=librosa.feature.spectral_centroid(y=audio,sr=sr)[0]
            sf_=librosa.feature.spectral_flatness(y=audio)[0]
            return round(_clamp(float(np.mean(sc))/(sr/2)*(1-float(np.mean(sf_))),0.,1.),3)
        except: return 0.
    @classmethod
    def score(cls,audio_path,ref_path=None):
        y,sr=librosa.load(audio_path,sr=None,mono=True)
        snr=cls.estimate_snr(y,sr); clr=cls.spectral_clarity(y,sr)
        sim=SpeakerEmbeddingManager.sim_score(ref_path,audio_path) if ref_path else 0.
        ovr=_clamp((min(snr,40)/40)*0.4+clr*0.3+sim*0.3,0.,1.)
        return {"snr_db":round(snr,1),"spectral_clarity":round(clr,3),
                "speaker_similarity":round(sim,3),"overall_score":round(ovr,3)}


# ══════════════════════════════════════════════════════════════════════════════
# STT + SYNTHESIS
# ══════════════════════════════════════════════════════════════════════════════

def transcribe(audio_path,task="transcribe",lang=None):
    model=load_whisper_model()
    opts={"task":task,"fp16":DEVICE=="cuda","verbose":False}
    if lang and lang!="auto": opts["language"]=lang
    r=model.transcribe(audio_path,**opts)
    return {"text":r.get("text","").strip(),"language":r.get("language","unknown")}

def translate_to_english(audio_path,emotion_mode="ensemble"):
    t=transcribe(audio_path); orig=t["text"]; lang=t["language"]
    eng=orig if lang=="en" else transcribe(audio_path,task="translate")["text"]
    emo=detect_emotion(audio_path,emotion_mode); pros=ProsodyAnalyzer.extract(audio_path)
    return {"original_text":orig,"english_text":eng,"source_language":lang,
            "emotion":emo.get("emotion","neutral"),"emotion_confidence":emo.get("confidence"),
            "emotion_method":emo.get("method","?"),"emotion_votes":emo.get("individual_votes"),"prosody":pros}

def _safe_tts(tts,kwargs):
    cand,dropped=dict(kwargs),{}
    for _ in range(16):
        try: tts.tts_to_file(**cand); return cand,dropped
        except TypeError as e:
            bad=_extract_bad_kwarg(str(e))
            if not bad or bad not in cand: raise
            dropped[bad]=cand.pop(bad)
    tts.tts_to_file(**cand); return cand,dropped

def synthesize(tts,text,settings,language,speaker_wav=None,multi_ref_paths=None,
               builtin_speaker=None,speed=None,emotion=None,prefix="xtts"):
    if not text.strip(): raise ValueError("Empty text")
    text=TextNormalizer.normalize(text)
    out=OUTPUTS_DIR/f"{prefix}_{_ts()}.wav"
    kw={"text":text,"file_path":str(out),"language":language,"speed":speed or 1.0}
    if emotion and emotion in VALID_EMOTIONS and emotion!="neutral": kw["emotion"]=emotion
    if multi_ref_paths and len(multi_ref_paths)>1: kw["speaker_wav"]=multi_ref_paths
    elif speaker_wav: kw["speaker_wav"]=speaker_wav
    else: kw["speaker"]=builtin_speaker or XTTS_DEFAULT_SPEAKER
    applied,dropped=_safe_tts(tts,kw)
    return str(out),{"applied":{k:v for k,v in applied.items() if k not in ("text","file_path","speaker_wav","speaker")},"dropped":dropped}


def post_process(wav_path,ref_path,ref_prosody,settings,target_emotion=None):
    safe=ref_prosody if(ref_prosody and isinstance(ref_prosody,dict)) else {}
    if settings.get("prosody_transfer",True) and safe.get("ok"):
        wav_path=ProsodyTransfer.transfer(wav_path,safe,
            f0_strength=float(settings.get("prosody_f0_strength",0.65)),
            energy_strength=float(settings.get("prosody_energy_strength",0.50)),
            rate_strength=float(settings.get("prosody_rate_strength",0.30)))
    ei=float(settings.get("emotion_intensity",0.6)); emo=target_emotion or settings.get("emotion_override")
    if emo and emo in VALID_EMOTIONS and ei>0.02: wav_path=EmotionPostProcessor.apply(wav_path,emo,ei)
    if settings.get("use_neural_prosody",False) and ref_path:
        try:
            stage=int(settings.get("prosody_training_stage",1)); bundle=load_prosody_bundle(stage)
            wav_path=apply_neural_prosody(wav_path,ref_path,bundle,
                f0_strength=float(settings.get("neural_f0_strength",0.80)),
                adain_strength=float(settings.get("neural_adain_strength",0.70)))
        except: pass
    if settings.get("use_freevc",False) and ref_path:
        fv,_=load_freevc()
        if fv and fv.is_available():
            fv_out=re.sub(r"(_pt|_emo|_npros)+\.wav$","_freevc.wav",wav_path)
            try:
                rp,ok=fv.convert(source_path=wav_path,ref_path=ref_path,output_path=fv_out,
                                   strength=float(settings.get("freevc_strength",0.70)))
                if ok: wav_path=rp
            except: pass
    if settings.get("use_freevc",False):
        try:
            y,sr=librosa.load(wav_path,sr=None,mono=True); y=light_denoise(y,sr,0.30)
            dn=wav_path.replace(".wav","_dn.wav"); sf.write(dn,y,sr,subtype="PCM_16"); wav_path=dn
        except: pass
    if settings.get("use_bigvgan",False):
        voc,_=load_bigvgan()
        if voc and voc.is_available():
            bv_out=re.sub(r"(_pt|_emo|_npros|_freevc|_dn)+\.wav$","_bigvgan.wav",wav_path)
            rp,ok=voc.enhance(wav_path,output_path=bv_out)
            if ok: wav_path=rp
    try:
        y,sr=librosa.load(wav_path,sr=None,mono=True); pk=float(np.max(np.abs(y)))
        if pk>1e-6 and abs(pk-0.97)>0.01:
            fo=wav_path.replace(".wav","_final.wav"); sf.write(fo,y*(0.97/pk),sr,subtype="PCM_16"); wav_path=fo
    except: pass
    return wav_path

def wav_to_mp3(wav_path):
    mp3=re.sub(r"(_pt|_emo|_npros|_freevc|_dn|_bigvgan|_final)+","",wav_path).replace(".wav",".mp3")
    AudioSegment.from_wav(wav_path).export(mp3,format="mp3",bitrate="192k"); return mp3

def run_generation(text,settings,language="en",emotion_override=None,prefix="xtts",
                   ref_path=None,multi_refs=None,ref_prosody=None,ref_name=None):
    tts=load_tts_model()
    sp=ref_path or st.session_state.get("speaker_path")
    mr=multi_refs or(st.session_state.get("multi_ref_paths") if len(st.session_state.get("multi_ref_paths",[]))>1 else None)
    rp=ref_prosody or st.session_state.get("stt_prosody")
    ae=emotion_override or st.session_state.get("stt_detected_emotion")
    if ae not in VALID_EMOTIONS: ae=None
    raw_wav,debug=synthesize(tts=tts,text=text,settings=settings,language=language,speaker_wav=sp,multi_ref_paths=mr,
                              builtin_speaker=XTTS_DEFAULT_SPEAKER if not sp else None,emotion=ae,prefix=prefix)
    pp_wav=raw_wav.replace(".wav","_pp.wav"); shutil.copy2(raw_wav,pp_wav)
    pp_wav=post_process(wav_path=pp_wav,ref_path=sp,ref_prosody=rp,settings=settings,target_emotion=ae)
    m_raw=m_pp={}
    try:
        m_raw=AudioQuality.score(raw_wav,ref_path=sp); m_pp=AudioQuality.score(pp_wav,ref_path=sp)
        st.session_state.last_quality_metrics_raw=m_raw; st.session_state.last_quality_metrics=m_pp
    except: pass
    return raw_wav,pp_wav,{**debug,"target_emotion":ae,"ref_path":sp,"ref_prosody":rp,"metrics_raw":m_raw,"metrics_pp":m_pp}


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def plot_wav_spec(wav_path,title="",color="#7c6af7"):
    if not HAVE_MPL: return None
    try:
        y,sr=librosa.load(wav_path,sr=None,mono=True)
        t=np.linspace(0,len(y)/sr,len(y))
        fig=plt.figure(figsize=(8,3),facecolor="#0d1117")
        gs=gridspec.GridSpec(2,1,hspace=0.3)
        ax1=fig.add_subplot(gs[0]); ax1.fill_between(t,y,alpha=0.7,color=color)
        ax1.set_facecolor("#0d1117"); ax1.set_xlim(0,t[-1])
        ax1.tick_params(colors="#7b7b9a",labelsize=6); ax1.set_ylabel("Amp",color="#7b7b9a",fontsize=7)
        ax1.spines[:].set_color("#2a2a38")
        if title: ax1.set_title(title,color="#e8e8f0",fontsize=8,pad=3)
        ax2=fig.add_subplot(gs[1])
        S=librosa.feature.melspectrogram(y=y,sr=sr,n_mels=64)
        librosa.display.specshow(librosa.power_to_db(S,ref=np.max),sr=sr,x_axis="time",y_axis="mel",ax=ax2,cmap="magma")
        ax2.set_facecolor("#0d1117"); ax2.tick_params(colors="#7b7b9a",labelsize=6)
        ax2.set_ylabel("Hz",color="#7b7b9a",fontsize=7); ax2.spines[:].set_color("#2a2a38")
        plt.tight_layout(pad=0.3)
        buf=io.BytesIO(); plt.savefig(buf,format="png",dpi=110,bbox_inches="tight",facecolor="#0d1117")
        plt.close(fig); buf.seek(0); return buf
    except: return None

def show_quality_metrics(metrics):
    snr,sim=metrics.get("snr_db",0),metrics.get("speaker_similarity",0)
    clr,ovr=metrics.get("spectral_clarity",0),metrics.get("overall_score",0)
    chips=(f'{_chip(snr/40,f"🔊 SNR {snr:.1f} dB")}'
           f'{_chip(sim,f"👤 Sim {sim:.2%}")}'
           f'{_chip(clr,f"✨ Clarity {clr:.2%}")}'
           f'{_chip(ovr,f"⭐ Overall {ovr:.2%}")}')
    st.markdown(f'<div class="metric-row">{chips}</div>',unsafe_allow_html=True)

def display_audio_comparison(raw_path,pp_path,target_emotion=None,ref_path=None,ref_prosody=None,settings=None,label=""):
    settings=settings or {}; _uid=uuid.uuid4().hex[:10]
    if label: st.markdown(f"**{label}**")
    col_raw,_,col_pp=st.columns([10,1,10])
    with col_raw:
        st.markdown("""<div class="banner-raw"><span style="font-size:18px">🔴</span>
        <span class="title">RAW OUTPUT</span><span style="color:#7b7b9a;font-size:12px">straight from XTTS</span>
        </div>""",unsafe_allow_html=True)
        buf=plot_wav_spec(raw_path,"Raw XTTS","#e87c40")
        if buf: st.image(buf,use_container_width=True)
        try: show_quality_metrics(AudioQuality.score(raw_path,ref_path=ref_path))
        except: pass
        st.audio(raw_path,format="audio/wav")
        with open(raw_path,"rb") as f:
            st.download_button("⬇ RAW WAV",f.read(),file_name=f"raw_{os.path.basename(raw_path)}",mime="audio/wav",key=f"dl_raw_{_uid}")
    with col_pp:
        active=["prosody","emotion"]
        if settings.get("use_neural_prosody"): active.append("NeuralPros")
        if settings.get("use_freevc"): active.append("FreeVC")
        st.markdown(f"""<div class="banner-pp"><span style="font-size:18px">🟢</span>
        <span class="title">POST-PROCESSED</span>
        <span style="color:#7b7b9a;font-size:12px">{" · ".join(active)}</span>
        </div>""",unsafe_allow_html=True)
        buf2=plot_wav_spec(pp_path,"Post-processed","#40c8c0")
        if buf2: st.image(buf2,use_container_width=True)
        try: show_quality_metrics(AudioQuality.score(pp_path,ref_path=ref_path))
        except: pass
        st.audio(pp_path,format="audio/wav")
        with open(pp_path,"rb") as f:
            st.download_button("⬇ PP WAV",f.read(),file_name=os.path.basename(pp_path),mime="audio/wav",key=f"dl_pp_{_uid}")


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION — SHARED HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _save_eval_csv(records: List[Dict], stem: str) -> str:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_DIR / f"{stem}_{_ts()}.csv"
    if not records:
        path.write_text(""); return str(path)
    keys = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(records)
    return str(path)


def _save_eval_json(data: Any, stem: str) -> str:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    path = EVAL_DIR / f"{stem}_{_ts()}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return str(path)


def _df_to_csv(records) -> bytes:
    if not records: return b""
    try:
        buf = io.StringIO()
        keys = list(records[0].keys())
        w = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
        w.writeheader(); w.writerows(records)
        return buf.getvalue().encode("utf-8")
    except Exception: return b""


def _records_to_df(records: List[Dict]):
    if HAVE_PANDAS and records: return pd.DataFrame(records)
    return records


def _plot_bar_dark(labels, values, title, color="#7c6af7", ylabel="Score",
                   h_line=None, h_label=""):
    if not HAVE_MPL: return None
    try:
        fig, ax = plt.subplots(figsize=(max(5, len(labels)*0.9+2), 3.5), facecolor="#0d1117")
        bars = ax.bar(labels, values, color=color, alpha=0.85,
                      edgecolor="#2a2a38", linewidth=0.5, width=0.6)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max(values)*0.015,
                    f"{val:.3f}", ha="center", va="bottom",
                    color="#e8e8f0", fontsize=8, fontweight="bold")
        if h_line is not None:
            ax.axhline(h_line, color="#ef4444", lw=1.2, linestyle="--", alpha=0.7, label=h_label)
            ax.legend(fontsize=8, facecolor="#111118", labelcolor="#e8e8f0")
        ax.set_facecolor("#0d1117"); ax.set_title(title, color="#e8e8f0", fontsize=10, pad=6)
        ax.set_ylabel(ylabel, color="#7b7b9a", fontsize=9)
        ax.tick_params(colors="#7b7b9a", labelsize=8, axis="both")
        ax.spines[:].set_color("#2a2a38")
        ax.set_ylim(0, max(max(values)*1.22, (h_line or 0)*1.2) if values else 1.)
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout(pad=0.5)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig); buf.seek(0); return buf
    except: return None


def _plot_confusion(confusion, labels, title="Confusion Matrix"):
    if not HAVE_MPL: return None
    try:
        fig, ax = plt.subplots(figsize=(6.5, 5.5), facecolor="#0d1117")
        im = ax.imshow(confusion, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=40, ha="right", color="#7b7b9a", fontsize=8)
        ax.set_yticklabels(labels, color="#7b7b9a", fontsize=8)
        mx = confusion.max() if confusion.max() > 0 else 1
        for i in range(len(labels)):
            for j in range(len(labels)):
                v = confusion[i, j]
                ax.text(j, i, str(v), ha="center", va="center", fontsize=9,
                        color="#e8e8f0" if v > mx*0.5 else "#7b7b9a")
        ax.set_xlabel("Predicted", color="#7b7b9a", fontsize=9)
        ax.set_ylabel("Ground Truth", color="#7b7b9a", fontsize=9)
        ax.set_title(title, color="#e8e8f0", fontsize=10, pad=6)
        ax.set_facecolor("#0d1117"); ax.spines[:].set_color("#2a2a38")
        plt.colorbar(im, ax=ax).ax.tick_params(colors="#7b7b9a")
        plt.tight_layout(pad=0.5)
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="#0d1117")
        plt.close(fig); buf.seek(0); return buf
    except: return None


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1: TRANSLATION MATRIX — BLEU-1/2/3/4 + Overall + chrF + COMET-22
# ══════════════════════════════════════════════════════════════════════════════

class TranslationEvaluator:
    """
    Hindi→English translation quality on FLEURS.

    Metrics:
      BLEU-1   — unigram precision (most lenient)
      BLEU-2   — bigram precision
      BLEU-3   — trigram precision
      BLEU-4   — 4-gram precision (strictest n-gram)
      Overall  — geometric mean of BLEU-1/2/3/4  ← report this in paper tables
      chrF     — character n-gram F-score (robust to morphology)
      COMET-22 — neural metric (best semantic quality proxy)
    """

    @staticmethod
    def _normalise_text(text: str) -> str:
        import string
        text = text.lower().strip()
        text = text.translate(str.maketrans("", "", string.punctuation))
        text = re.sub(r"\s{2,}", " ", text)
        return text

    @staticmethod
    def scan_fleurs_dataset(audio_dir: str, ref_dir: str) -> Tuple[List[str], List[str], List[str]]:
        audio_dir_p = Path(audio_dir); ref_dir_p = Path(ref_dir)
        if not audio_dir_p.exists():
            raise FileNotFoundError(f"Audio dir not found: {audio_dir}")
        wavs = (sorted(audio_dir_p.glob("*.wav")) +
                sorted(audio_dir_p.glob("*.mp3")) +
                sorted(audio_dir_p.glob("*.flac")))
        if not wavs: raise FileNotFoundError(f"No audio files in {audio_dir}")
        single_refs: Dict[str, str] = {}
        if ref_dir_p.exists():
            for candidate in sorted(ref_dir_p.iterdir()):
                if candidate.suffix in (".tsv",".txt",".csv") and candidate.stat().st_size>100:
                    try:
                        with open(candidate, encoding="utf-8") as f: first = f.readline()
                        delim = "\t" if "\t" in first else "|"
                        with open(candidate, encoding="utf-8") as f:
                            for line in f:
                                parts = line.strip().split(delim)
                                if len(parts) >= 2:
                                    single_refs[Path(parts[0]).stem] = parts[-1].strip()
                    except: pass
                    if single_refs: break
        audio_paths, ids, refs = [], [], []
        for wav in wavs:
            stem = wav.stem
            ref_file = ref_dir_p/(stem+".txt") if ref_dir_p.exists() else None
            if ref_file and ref_file.exists(): ref_text = ref_file.read_text(encoding="utf-8").strip()
            elif stem in single_refs:           ref_text = single_refs[stem]
            else:                               ref_text = ""
            audio_paths.append(str(wav)); ids.append(stem); refs.append(ref_text)
        return audio_paths, ids, refs

    @staticmethod
    def _compute_bleu_ngrams(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
        """
        Compute corpus BLEU-1, BLEU-2, BLEU-3, BLEU-4 and chrF.
        Overall BLEU = geometric mean of all four n-gram scores.
        Falls back to manual unigram BLEU if sacrebleu is unavailable.
        """
        results = {"bleu_1": 0., "bleu_2": 0., "bleu_3": 0., "bleu_4": 0., "chrf": 0.}
        if not hypotheses: return results

        if HAVE_SACREBLEU:
            try:
                from sacrebleu.metrics import BLEU, CHRF
                # Compute each n-gram order independently
                for n, key in [(1, "bleu_1"), (2, "bleu_2"), (3, "bleu_3"), (4, "bleu_4")]:
                    try:
                        scorer = BLEU(max_ngram_order=n, effective_order=True, tokenize="13a")
                        score  = scorer.corpus_score(hypotheses, [references])
                        results[key] = round(float(score.score), 3)
                    except Exception:
                        try:
                            score = _sacrebleu.corpus_bleu(
                                hypotheses, [references],
                                smooth_method="add-k", max_ngram_order=n)
                            results[key] = round(float(score.score), 3)
                        except: pass
                # chrF
                try:
                    chrf = CHRF()
                    results["chrf"] = round(float(chrf.corpus_score(hypotheses, [references]).score), 3)
                except: pass
            except Exception:
                # Pure manual BLEU-1 fallback
                from collections import Counter
                tc, tl, tr = 0, 0, 0
                for h, r in zip(hypotheses, references):
                    hw = h.split(); rw = r.split()
                    rc = Counter(rw); hc = Counter(hw)
                    tc += sum(min(c, rc.get(w, 0)) for w, c in hc.items())
                    tl += len(hw); tr += len(rw)
                prec = tc / (tl + 1e-9)
                bp = min(1., np.exp(1 - tr / (tl + 1e-9)))
                results["bleu_1"] = round(bp * prec * 100, 3)
        else:
            from collections import Counter
            tc, tl, tr = 0, 0, 0
            for h, r in zip(hypotheses, references):
                hw = h.split(); rw = r.split()
                rc = Counter(rw); hc = Counter(hw)
                tc += sum(min(c, rc.get(w, 0)) for w, c in hc.items())
                tl += len(hw); tr += len(rw)
            prec = tc / (tl + 1e-9)
            bp = min(1., np.exp(1 - tr / (tl + 1e-9)))
            results["bleu_1"] = round(bp * prec * 100, 3)
        return results

    @staticmethod
    def _sentence_bleu_1(hyp: str, ref: str) -> float:
        if not hyp or not ref: return 0.
        if HAVE_SACREBLEU:
            try:
                from sacrebleu.metrics import BLEU
                scorer = BLEU(max_ngram_order=1, effective_order=True, tokenize="13a")
                return round(float(scorer.sentence_score(hyp, [ref]).score), 3)
            except: pass
        from collections import Counter
        hw = hyp.split(); rw = ref.split()
        rc = Counter(rw); hc = Counter(hw)
        clip = sum(min(c, rc.get(w, 0)) for w, c in hc.items())
        return round(clip / (len(hw) + 1e-9) * 100, 3)

    @classmethod
    def run(cls, audio_paths: List[str], ids: List[str], ref_texts: List[str],
            batch_size: int = 16, log_cb=None) -> Dict:
        wm = load_whisper_model()
        hypotheses: List[str] = []
        records:    List[Dict] = []
        errors: int = 0
        total = len(audio_paths)

        # ── Whisper translate ──────────────────────────────────────────────
        for i, (path, sid, ref) in enumerate(zip(audio_paths, ids, ref_texts)):
            if log_cb: log_cb(i/total, f"[{i+1}/{total}] Transcribing: {Path(path).name}")
            hyp = ""
            try:
                result = wm.transcribe(path, task="translate",
                                        fp16=(DEVICE=="cuda"), verbose=False)
                hyp = result.get("text", "").strip()
            except Exception as e:
                errors += 1
                if log_cb: log_cb(i/total, f"  ⚠ {Path(path).name}: {e}")
            hypotheses.append(hyp)
            records.append({"id": sid, "audio_path": path,
                             "hypothesis": hyp, "reference": ref,
                             "error": (hyp == "")})

        # ── Normalise for scoring ──────────────────────────────────────────
        valid_idx = [i for i, r in enumerate(records) if r["hypothesis"] and r["reference"]]
        hyps_v    = [cls._normalise_text(records[i]["hypothesis"]) for i in valid_idx]
        refs_v    = [cls._normalise_text(records[i]["reference"])  for i in valid_idx]

        if log_cb: log_cb(0.85, f"Computing BLEU-1/2/3/4 on {len(hyps_v)} valid pairs…")

        # ── Corpus BLEU-1/2/3/4 + chrF ────────────────────────────────────
        bleu_scores = cls._compute_bleu_ngrams(hyps_v, refs_v)

        # ── Overall BLEU = geometric mean of BLEU-1/2/3/4 ─────────────────
        # This is the standard single-number summary used in MT papers.
        _bvals = [bleu_scores[k] for k in ("bleu_1","bleu_2","bleu_3","bleu_4")
                  if bleu_scores[k] > 0]
        avg_bleu_overall = (
            round(float(np.exp(np.mean(np.log([v + 1e-9 for v in _bvals])))), 3)
            if _bvals else 0.
        )

        # ── Per-sample BLEU-1 back-fill ───────────────────────────────────
        for k, i in enumerate(valid_idx):
            h = records[i]["hypothesis"]; r = records[i]["reference"]
            records[i]["bleu_1"] = cls._sentence_bleu_1(
                cls._normalise_text(h), cls._normalise_text(r))
        for rec in records:
            rec.setdefault("bleu_1", 0.)
            rec.setdefault("comet",  0.)

        # ── COMET-22 ──────────────────────────────────────────────────────
        avg_comet = 0.
        if HAVE_COMET and hyps_v:
            if log_cb: log_cb(0.90, "Computing COMET-22 (may take a few minutes)…")
            try:
                comet_model, cerr = load_comet_model()
                if comet_model:
                    comet_data = [{"src": r, "mt": h, "ref": r}
                                  for h, r in zip(hyps_v, refs_v)]
                    gpus = 1 if DEVICE == "cuda" else 0
                    comet_out = comet_model.predict(comet_data, batch_size=batch_size, gpus=gpus)
                    comet_scores = comet_out.scores
                    avg_comet = round(float(np.mean(comet_scores)), 4)
                    for k, i in enumerate(valid_idx):
                        if k < len(comet_scores):
                            records[i]["comet"] = round(float(comet_scores[k]), 4)
            except Exception as e:
                if log_cb: log_cb(0.95, f"  ⚠ COMET error: {e}")

        if log_cb: log_cb(1.0, f"✅ Done — {len(hyps_v)}/{total} pairs, {errors} errors")
        return {
            "n_total":          total,
            "n_valid":          len(hyps_v),
            "n_errors":         errors,
            "avg_bleu_1":       bleu_scores["bleu_1"],
            "avg_bleu_2":       bleu_scores["bleu_2"],
            "avg_bleu_3":       bleu_scores["bleu_3"],
            "avg_bleu_4":       bleu_scores["bleu_4"],
            "avg_bleu_overall": avg_bleu_overall,
            "avg_chrf":         bleu_scores["chrf"],
            "avg_comet":        avg_comet,
            "records":          records,
        }


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2: SPEAKER SIMILARITY
# ══════════════════════════════════════════════════════════════════════════════

class SpeakerSimilarityEvaluator:
    @staticmethod
    def mcd(ref_path: str, gen_path: str, n_mfcc: int = 13) -> float:
        try:
            ry, _ = librosa.load(ref_path, sr=22050, mono=True)
            gy, _ = librosa.load(gen_path, sr=22050, mono=True)
            r_mfcc = librosa.feature.mfcc(y=ry, sr=22050, n_mfcc=n_mfcc)[1:]
            g_mfcc = librosa.feature.mfcc(y=gy, sr=22050, n_mfcc=n_mfcc)[1:]
            T = min(r_mfcc.shape[1], g_mfcc.shape[1])
            diff = r_mfcc[:, :T] - g_mfcc[:, :T]
            mcd = float(np.mean(np.sqrt(2.0*np.sum(diff**2, axis=0))))
            return round(mcd*(10.0/np.log(10.0)), 3)
        except: return -1.

    @staticmethod
    def scan_ravdess(root: str, max_speakers: int = 50) -> Dict[str, List[str]]:
        root_p = Path(root)
        if not root_p.exists(): raise FileNotFoundError(f"RAVDESS not found: {root}")
        spk_map: Dict[str, List[str]] = defaultdict(list)
        for wav in root_p.rglob("*.wav"):
            parts = wav.stem.split("-")
            if len(parts) >= 7: spk_map[f"actor_{parts[6]}"].append(str(wav))
        return dict(list(spk_map.items())[:max_speakers])

    @staticmethod
    def scan_libri(root: str, max_speakers: int = 50) -> Dict[str, List[str]]:
        root_p = Path(root)
        if not root_p.exists(): raise FileNotFoundError(f"LibriTTS not found: {root}")
        spk_map: Dict[str, List[str]] = defaultdict(list)
        for wav in root_p.rglob("*.wav"):
            if len(wav.parts) >= 3: spk_map[wav.parts[-3]].append(str(wav))
        return dict(list(spk_map.items())[:max_speakers])

    @staticmethod
    def run(spk_map: Dict[str, List[str]], tts_model,
            eval_text: str = "Hello, this is a voice cloning evaluation test.",
            settings: Optional[Dict] = None, log_cb=None) -> Dict:
        settings = settings or {}
        records: List[Dict] = []; errors = 0; total = len(spk_map)
        for idx, (spk_id, wav_list) in enumerate(spk_map.items()):
            if log_cb: log_cb(idx/total, f"[{idx+1}/{total}] {spk_id}")
            if not wav_list: continue
            ref_path = wav_list[0]; sim = 0.; mcd_val = -1.; err_msg = ""
            try:
                ref_pros = ProsodyAnalyzer.extract(ref_path)
                raw_wav, _ = synthesize(tts_model, eval_text, settings,
                                         language="en", speaker_wav=ref_path,
                                         prefix=f"spk_eval_{idx:04d}")
                pp_wav = raw_wav.replace(".wav","_pp.wav")
                shutil.copy2(raw_wav, pp_wav)
                pp_wav = post_process(pp_wav, ref_path, ref_pros, settings)
                sim     = SpeakerEmbeddingManager.sim_score(ref_path, pp_wav)
                mcd_val = SpeakerSimilarityEvaluator.mcd(ref_path, pp_wav)
            except Exception as e:
                errors += 1; err_msg = str(e)
            records.append({"speaker_id": spk_id, "ref_path": ref_path,
                             "n_clips": len(wav_list), "cosine_sim": round(sim,4),
                             "mcd_db": mcd_val, "error": err_msg})

        valid = [r for r in records if not r["error"] and r["cosine_sim"]>0]
        sims = [r["cosine_sim"] for r in valid]
        mcds = [r["mcd_db"] for r in valid if r["mcd_db"]>0]
        if log_cb: log_cb(1.0, f"✅ {len(valid)}/{total} speakers, {errors} errors")
        return {
            "n_speakers": total, "n_valid": len(valid), "n_errors": errors,
            "mean_sim": round(float(np.mean(sims)),4) if sims else 0.,
            "std_sim":  round(float(np.std(sims)), 4) if sims else 0.,
            "min_sim":  round(float(np.min(sims)), 4) if sims else 0.,
            "max_sim":  round(float(np.max(sims)), 4) if sims else 0.,
            "mean_mcd": round(float(np.mean(mcds)),3) if mcds else -1.,
            "records": records,
        }


# ══════════════════════════════════════════════════════════════════════════════
# SPEECHEMONET — SE-ResNet2D + Transformer
# ══════════════════════════════════════════════════════════════════════════════

class _SELayer2D(_nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.fc = _nn.Sequential(
            _nn.AdaptiveAvgPool2d(1), _nn.Flatten(),
            _nn.Linear(channels, max(channels//reduction, 4)), _nn.ReLU(inplace=True),
            _nn.Linear(max(channels//reduction, 4), channels), _nn.Sigmoid(),
        )
    def forward(self, x): return x * self.fc(x).view(x.size(0), x.size(1), 1, 1)


class _SEResBlock2D(_nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.main = _nn.Sequential(
            _nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            _nn.BatchNorm2d(out_ch), _nn.ReLU(inplace=True),
            _nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            _nn.BatchNorm2d(out_ch),
        )
        self.se   = _SELayer2D(out_ch)
        self.skip = (_nn.Sequential(_nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                                     _nn.BatchNorm2d(out_ch))
                     if in_ch != out_ch or stride != 1 else _nn.Identity())
        self.relu = _nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.se(self.main(x)) + self.skip(x))


class _SpeechEmoNet(_nn.Module):
    def __init__(self, feature_dim: int = 248, max_frames: int = 128,
                 n_classes: int = 7, d_model: int = 256):
        super().__init__()
        self.feature_dim = feature_dim
        self.max_frames  = max_frames
        self.n_classes   = n_classes
        self.d_model     = d_model

        self.stem = _nn.Sequential(
            _nn.Conv2d(1, 64, kernel_size=3, padding=1, bias=False),
            _nn.BatchNorm2d(64), _nn.ReLU(inplace=True),
            _nn.InstanceNorm2d(64, affine=True),
            _nn.MaxPool2d(kernel_size=(4, 2)),
        )
        self.res1 = _nn.Sequential(_SEResBlock2D(64, 128, stride=1), _nn.MaxPool2d((2,2)))
        self.res2 = _nn.Sequential(_SEResBlock2D(128, 256, stride=1), _nn.MaxPool2d((2,2)))
        self.res3 = _SEResBlock2D(256, 256, stride=1)

        with _torch.no_grad():
            dummy = _torch.zeros(1, 1, feature_dim, max_frames)
            z = self.res3(self.res2(self.res1(self.stem(dummy))))
            _, C, F_out, T_out = z.shape
        self._T_out = T_out; self._C = C; self._F_out = F_out

        self.freq_pool = _nn.AdaptiveAvgPool2d((1, None))
        self.proj      = _nn.Linear(C, d_model)
        self.pos_emb   = _nn.Parameter(_torch.randn(1, T_out+1, d_model)*0.02)
        self.cls_tok   = _nn.Parameter(_torch.zeros(1, 1, d_model))

        enc_layer = _nn.TransformerEncoderLayer(
            d_model=d_model, nhead=8, dim_feedforward=1024,
            dropout=0.15, batch_first=True, norm_first=True)
        self.transformer = _nn.TransformerEncoder(enc_layer, num_layers=3)
        self.norm = _nn.LayerNorm(d_model)
        self.head = _nn.Sequential(
            _nn.Linear(d_model*2, d_model), _nn.GELU(),
            _nn.Dropout(0.45), _nn.Linear(d_model, n_classes))
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, _nn.Linear):
                _nn.init.xavier_uniform_(m.weight)
                if m.bias is not None: _nn.init.zeros_(m.bias)
            elif isinstance(m, (_nn.Conv2d, _nn.Conv1d)):
                _nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, x):
        B = x.size(0)
        z = self.res3(self.res2(self.res1(self.stem(x))))
        z = self.freq_pool(z).squeeze(2)
        z = self.proj(z.permute(0, 2, 1))
        cls = self.cls_tok.expand(B, -1, -1)
        z   = _torch.cat([cls, z], dim=1)
        z   = z + self.pos_emb[:, :z.size(1), :]
        z   = self.transformer(z)
        z   = self.norm(z)
        seq = z[:, 1:, :]
        feat = _torch.cat([seq.mean(dim=1), seq.max(dim=1).values], dim=-1)
        return self.head(feat)


# ══════════════════════════════════════════════════════════════════════════════
# DATASET SCANNERS
# ══════════════════════════════════════════════════════════════════════════════

_RAVDESS_EMO = {"01":"neutral","02":"neutral","03":"happy","04":"sad",
                "05":"angry","06":"fearful","07":"disgust","08":"surprised"}
_CREMAD_EMO  = {"ANG":"angry","DIS":"disgust","FEA":"fearful",
                "HAP":"happy","NEU":"neutral","SAD":"sad"}


def _scan_ravdess(root: str) -> List[Tuple[str,str,str]]:
    out = []
    for wav in Path(root).rglob("*.wav"):
        parts = wav.stem.split("-")
        if len(parts) < 7: continue
        emo = _RAVDESS_EMO.get(parts[2])
        if emo is None: continue
        out.append((str(wav), emo, f"ravdess_{parts[6]}"))
    return out


def _scan_cremad(root: str) -> List[Tuple[str,str,str]]:
    out = []; seen = set()
    search_dirs = [Path(root)]
    if (Path(root)/"AudioWAV").exists(): search_dirs.append(Path(root)/"AudioWAV")
    for d in search_dirs:
        for wav in d.glob("*.wav"):
            if str(wav) in seen: continue
            seen.add(str(wav))
            parts = wav.stem.split("_")
            if len(parts) < 3: continue
            emo = _CREMAD_EMO.get(parts[2].upper())
            if emo is None: continue
            out.append((str(wav), emo, f"cremad_{parts[0]}"))
    return out


def _speaker_disjoint_split(all_samples, test_frac=0.20, seed=42):
    import random
    spk_map: dict = defaultdict(list)
    for path, label, spk in all_samples:
        spk_map[spk].append((path, label))
    speakers = sorted(spk_map.keys())
    rng = random.Random(seed); rng.shuffle(speakers)
    n_test = max(1, int(len(speakers)*test_frac))
    test_spks  = set(speakers[:n_test])
    train_spks = set(speakers[n_test:])
    train_p, train_l, test_p, test_l = [], [], [], []
    for spk in train_spks:
        for path, label in spk_map[spk]: train_p.append(path); train_l.append(label)
    for spk in test_spks:
        for path, label in spk_map[spk]: test_p.append(path);  test_l.append(label)
    return train_p, train_l, test_p, test_l


def _build_emotion_dataset(ravdess_root, cremad_root, test_frac=0.20):
    all_samples = []; stats = {}
    if ravdess_root and Path(ravdess_root).exists():
        rv = _scan_ravdess(ravdess_root); all_samples.extend(rv); stats["ravdess"] = len(rv)
    else: stats["ravdess"] = 0
    if cremad_root and Path(cremad_root).exists():
        cr = _scan_cremad(cremad_root); all_samples.extend(cr); stats["cremad"] = len(cr)
    else: stats["cremad"] = 0
    if not all_samples: raise ValueError("No audio files found — check RAVDESS and CREMA-D paths")
    train_p, train_l, test_p, test_l = _speaker_disjoint_split(all_samples, test_frac)
    classes = sorted(set(train_l)|set(test_l))
    return {"train_paths":train_p,"train_labels":train_l,"test_paths":test_p,"test_labels":test_l,
            "classes":classes,"stats":stats,"n_train":len(train_p),"n_test":len(test_p),"n_classes":len(classes)}


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

class _EmoDataset(_torch.utils.data.Dataset):
    def __init__(self, paths, labels, label_to_idx, scaler,
                 feature_dim, max_frames, augment=False, cache=None):
        self.paths=paths; self.label_idx=[label_to_idx.get(l,0) for l in labels]
        self.scaler=scaler; self.feature_dim=feature_dim; self.max_frames=max_frames
        self.augment=augment; self.cache=cache if cache is not None else {}

    def __len__(self): return len(self.paths)

    def _augment(self, feat):
        import random; feat=feat.copy(); F,T=feat.shape
        fw=random.randint(0,min(15,F-1))
        if fw>0: f0=random.randint(0,F-fw); feat[f0:f0+fw,:]=0.
        tw=random.randint(0,min(15,T-1))
        if tw>0: t0=random.randint(0,T-tw); feat[:,t0:t0+tw]=0.
        shift=random.randint(-8,8)
        if shift: feat=np.roll(feat,shift,axis=1)
        if random.random()<0.4: feat=feat+np.random.normal(0,0.015,feat.shape).astype(np.float32)
        return feat

    def __getitem__(self, idx):
        path=self.paths[idx]
        if path not in self.cache: feat=self._extract(path); self.cache[path]=feat
        else: feat=self.cache[path]
        if feat is None: feat=np.zeros((self.feature_dim,self.max_frames),dtype=np.float32)
        else: feat=feat.copy()
        if self.augment: feat=self._augment(feat)
        return _torch.from_numpy(feat).unsqueeze(0), self.label_idx[idx]

    def _extract(self, path):
        try:
            y,sr=librosa.load(path,sr=22050,mono=True)
            hop,n_fft=512,2048
            mel  =librosa.power_to_db(librosa.feature.melspectrogram(y=y,sr=sr,n_mels=128,n_fft=n_fft,hop_length=hop),ref=np.max).astype(np.float32)
            mfcc =librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40,n_fft=n_fft,hop_length=hop).astype(np.float32)
            dmfcc=librosa.feature.delta(mfcc).astype(np.float32)
            ddmfc=librosa.feature.delta(mfcc,order=2).astype(np.float32)
            feat =np.vstack([mel,mfcc,dmfcc,ddmfc])
            if feat.shape[0]>self.feature_dim: feat=feat[:self.feature_dim]
            elif feat.shape[0]<self.feature_dim: feat=np.pad(feat,((0,self.feature_dim-feat.shape[0]),(0,0)))
            T=feat.shape[1]
            if T>=self.max_frames: s=(T-self.max_frames)//2; feat=feat[:,s:s+self.max_frames]
            else: feat=np.pad(feat,((0,0),(0,self.max_frames-T)))
            if self.scaler is not None:
                try: feat=self.scaler.transform(feat.T).T.astype(np.float32)
                except:
                    try: feat=((feat.T-self.scaler.mean_)/(self.scaler.scale_+1e-8)).T.astype(np.float32)
                    except: pass
            return feat.astype(np.float32)
        except: return None


def _fit_scaler(paths, feature_dim=248, max_frames=128, n_samples=99999, log_cb=None):
    from sklearn.preprocessing import StandardScaler
    scaler=StandardScaler(); batch_size=200; total=min(n_samples,len(paths))
    for i in range(0,total,batch_size):
        batch=paths[i:i+batch_size]
        if log_cb: log_cb(i/total,f"Scaler fit: {i}/{total} files…")
        frames_batch=[]
        for path in batch:
            try:
                y,sr=librosa.load(path,sr=22050,mono=True)
                hop,n_fft=512,2048
                mel  =librosa.power_to_db(librosa.feature.melspectrogram(y=y,sr=sr,n_mels=128,n_fft=n_fft,hop_length=hop),ref=np.max).astype(np.float32)
                mfcc =librosa.feature.mfcc(y=y,sr=sr,n_mfcc=40,n_fft=n_fft,hop_length=hop).astype(np.float32)
                dmfcc=librosa.feature.delta(mfcc).astype(np.float32)
                ddmfc=librosa.feature.delta(mfcc,order=2).astype(np.float32)
                feat =np.vstack([mel,mfcc,dmfcc,ddmfc])
                if feat.shape[0]>feature_dim: feat=feat[:feature_dim]
                elif feat.shape[0]<feature_dim: feat=np.pad(feat,((0,feature_dim-feat.shape[0]),(0,0)))
                frames_batch.append(feat.T)
            except: pass
        if frames_batch: scaler.partial_fit(np.vstack(frames_batch))
    if total==0: scaler.fit(np.zeros((1,feature_dim)))
    return scaler


def _train_epoch(model, loader, optimizer, criterion, device, mixup_alpha=0.0):
    model.train(); total_loss=correct=n=0
    for xb,yb in loader:
        xb,yb=xb.to(device),yb.to(device)
        optimizer.zero_grad()
        logits=model(xb); loss=criterion(logits,yb)
        correct+=(logits.argmax(1)==yb).sum().item()
        loss.backward()
        _torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        optimizer.step()
        total_loss+=loss.item()*len(yb); n+=len(yb)
    return total_loss/n, correct/n


@_torch.no_grad()
def _eval_epoch(model, loader, criterion, device):
    model.eval(); total_loss=correct=n=0
    for xb,yb in loader:
        xb,yb=xb.to(device),yb.to(device)
        logits=model(xb); loss=criterion(logits,yb)
        total_loss+=loss.item()*len(yb); correct+=(logits.argmax(1)==yb).sum().item(); n+=len(yb)
    return total_loss/n, correct/n


def _run_emotion_training(train_paths,train_labels,test_paths,test_labels,classes,
                           feature_dim=248,max_frames=128,epochs=60,batch_size=32,lr=3e-4,
                           model_out=_EMO_NET_PATH,scaler_out=_EMO_SCALER_PATH,log_cb=None):
    import pickle as _pkl, math
    from torch.utils.data import DataLoader
    device=DEVICE; label_to_idx={c:i for i,c in enumerate(classes)}

    if log_cb: log_cb(0.,f"Fitting scaler on all {len(train_paths)} training files…")
    scaler=_fit_scaler(train_paths,feature_dim,max_frames,n_samples=len(train_paths),log_cb=log_cb)
    if log_cb: log_cb(0.10,"Scaler ready.")

    train_cache:Dict={}; test_cache:Dict={}
    train_ds=_EmoDataset(train_paths,train_labels,label_to_idx,scaler,feature_dim,max_frames,augment=True, cache=train_cache)
    test_ds =_EmoDataset(test_paths, test_labels, label_to_idx,scaler,feature_dim,max_frames,augment=False,cache=test_cache)

    from collections import Counter
    counts=Counter(train_labels); total=len(train_labels)
    weights=_torch.tensor([total/(len(classes)*counts.get(c,1)) for c in classes],dtype=_torch.float32).to(device)
    criterion=_nn.CrossEntropyLoss(weight=weights,label_smoothing=0.05)

    train_dl=DataLoader(train_ds,batch_size=batch_size,shuffle=True, num_workers=0,pin_memory=(device=="cuda"))
    test_dl =DataLoader(test_ds, batch_size=batch_size,shuffle=False,num_workers=0,pin_memory=(device=="cuda"))

    model=_SpeechEmoNet(feature_dim,max_frames,n_classes=len(classes)).to(device)
    optimizer=_torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=5e-4)
    warmup_ep=5
    def _lr_lambda(ep):
        if ep<warmup_ep: return (ep+1)/warmup_ep
        return 0.5*(1.+math.cos(math.pi*(ep-warmup_ep)/max(1,epochs-warmup_ep)))
    scheduler=_torch.optim.lr_scheduler.LambdaLR(optimizer,_lr_lambda)

    best_val_acc=0.; best_epoch=0; patience=12; no_improve=0
    history={"train_loss":[],"train_acc":[],"val_loss":[],"val_acc":[]}
    for epoch in range(1,epochs+1):
        if log_cb: log_cb(epoch/(epochs+1),f"Epoch {epoch}/{epochs}")
        t_loss,t_acc=_train_epoch(model,train_dl,optimizer,criterion,device,mixup_alpha=0.0)
        v_loss,v_acc=_eval_epoch(model,test_dl,criterion,device)
        scheduler.step()
        history["train_loss"].append(round(t_loss,4)); history["train_acc"].append(round(t_acc,4))
        history["val_loss"].append(round(v_loss,4));   history["val_acc"].append(round(v_acc,4))
        cur_lr=optimizer.param_groups[0]["lr"]
        if log_cb: log_cb(epoch/(epochs+1),f"  Epoch {epoch:3d} | train={t_acc:.3f} val={v_acc:.3f} val_loss={v_loss:.4f} lr={cur_lr:.2e}")
        if v_acc>best_val_acc:
            best_val_acc=v_acc; best_epoch=epoch; no_improve=0
            Path(model_out).parent.mkdir(parents=True,exist_ok=True)
            _torch.save({"model_state_dict":model.state_dict(),
                          "config":{"feature_dim":feature_dim,"max_frames":max_frames,
                                    "classes":classes,"n_classes":len(classes),
                                    "d_model":model.d_model,"best_epoch":best_epoch,"val_acc":best_val_acc},
                          "epoch":epoch},model_out)
        else:
            no_improve+=1
            if no_improve>=patience:
                if log_cb: log_cb(epoch/(epochs+1),f"  Early stop at epoch {epoch} (patience={patience})")
                break

    Path(scaler_out).parent.mkdir(parents=True,exist_ok=True)
    with open(scaler_out,"wb") as f: _pkl.dump(scaler,f)

    emo_set=sorted(VALID_EMOTIONS); ei={e:i for i,e in enumerate(emo_set)}
    confusion=np.zeros((len(emo_set),len(emo_set)),dtype=int)
    all_gt,all_pred=[],[]
    model.eval()
    with _torch.no_grad():
        for xb,yb in test_dl:
            xb=xb.to(device)
            pred_idx=model(xb).argmax(1).cpu().numpy()
            for pi,gi in zip(pred_idx,yb.numpy()):
                pred_lbl=_EMO_6_MAP.get(classes[pi],"neutral")
                gt_lbl  =_EMO_6_MAP.get(classes[gi],"neutral")
                all_gt.append(gt_lbl); all_pred.append(pred_lbl)
                g=ei.get(gt_lbl,-1); p=ei.get(pred_lbl,-1)
                if g>=0 and p>=0: confusion[g,p]+=1

    gen_acc=round(100.*np.mean(np.array(all_gt)==np.array(all_pred)),2) if all_gt else 0.
    if log_cb: log_cb(1.0,f"✅ Done! Best val_acc={best_val_acc:.3f} @ epoch {best_epoch} | Test acc={gen_acc:.1f}%")
    _emo_load_net.clear()
    return {"best_val_acc":round(best_val_acc,4),"best_epoch":best_epoch,"final_test_acc":gen_acc,
            "history":history,"confusion":confusion.tolist(),"emotion_labels":emo_set,"classes":classes}


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3: EMOTION EVALUATOR
# ══════════════════════════════════════════════════════════════════════════════

class EmotionEvaluator:
    @staticmethod
    def scan_ravdess_with_labels(root: str, max_per_emotion: int = 30) -> Tuple[List[str],List[str]]:
        import random
        per_emo: Dict[str,List[str]] = {}
        for path,emo,_ in _scan_ravdess(root): per_emo.setdefault(emo,[]).append(path)
        rng=random.Random(42); paths,labels=[],[]
        for emo,wav_list in per_emo.items():
            rng.shuffle(wav_list)
            for wp in wav_list[:max_per_emotion]: paths.append(wp); labels.append(emo)
        return paths,labels

    @classmethod
    def run(cls, wav_paths, gt_emotions, log_cb=None) -> Dict:
        records=[]; errors=0; total=len(wav_paths)
        emo_set=sorted(VALID_EMOTIONS); ei={e:i for i,e in enumerate(emo_set)}
        confusion=np.zeros((len(emo_set),len(emo_set)),dtype=int)
        for i,(path,gt) in enumerate(zip(wav_paths,gt_emotions)):
            if log_cb: log_cb(i/total,f"[{i+1}/{total}] {Path(path).name} (GT:{gt})")
            pred,conf="neutral",0.
            try: pred,conf=_emo_classify(path)
            except Exception: errors+=1
            g=ei.get(gt,-1); p=ei.get(pred,-1)
            if g>=0 and p>=0: confusion[g,p]+=1
            records.append({"wav_path":path,"gt_emotion":gt,"pred":pred,"conf":round(conf,4),"match":int(pred==gt)})
        valid=[r for r in records if "match" in r]
        gen_acc=round(100.*float(np.mean([r["match"] for r in valid])),2) if valid else 0.
        mean_conf=round(float(np.mean([r["conf"] for r in valid])),4) if valid else 0.
        per_emo: Dict[str,Dict]={}
        for emo in emo_set:
            recs=[r for r in valid if r["gt_emotion"]==emo]; n=len(recs)
            if n==0: per_emo[emo]={"n":0,"accuracy":0.,"precision":0.,"recall":0.,"f1":0.,"mean_conf":0.}; continue
            tp=sum(1 for r in recs if r["pred"]==emo)
            fp=sum(1 for r in valid if r["pred"]==emo and r["gt_emotion"]!=emo)
            fn=sum(1 for r in valid if r["gt_emotion"]==emo and r["pred"]!=emo)
            prec=tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9); f1=2*prec*rec/(prec+rec+1e-9)
            per_emo[emo]={"n":n,"accuracy":round(tp/n,3),"precision":round(prec,3),
                           "recall":round(rec,3),"f1":round(f1,3),
                           "mean_conf":round(float(np.mean([r["conf"] for r in recs])),4)}
        if log_cb: log_cb(1.0,f"✅ Accuracy: {gen_acc:.1f}% | {errors} errors")
        return {"n_total":total,"n_valid":len(valid),"n_errors":errors,
                "gen_accuracy":gen_acc,"mean_conf":mean_conf,
                "per_emotion":per_emo,"confusion":confusion.tolist(),
                "emotion_labels":emo_set,"records":records}


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION TAB — MODULE 1  (BLEU-1/2/3/4 + Overall + chrF + COMET)
# ══════════════════════════════════════════════════════════════════════════════

def tab_evaluation(settings: Dict):
    st.markdown("## 📊 Results & Metrics — Evaluation Suite v2")
    st.markdown(
        '<span class="badge badge-eval">Dataset-Level · Automated</span> '
        '<span class="badge badge-prosenc">BLEU-1/2/3/4 · Overall · COMET · Speaker Sim · MCD · Emotion</span>',
        unsafe_allow_html=True)

    missing=[]
    if not HAVE_SACREBLEU:   missing.append("`pip install sacrebleu`")
    if not HAVE_COMET:       missing.append("`pip install unbabel-comet`")
    if not HAVE_RESEMBLYZER: missing.append("`pip install resemblyzer`")
    if not HAVE_PANDAS:      missing.append("`pip install pandas`")
    if missing: st.warning("Optional packages: " + "  ·  ".join(missing))

    module_tabs = st.tabs([
        "📝 Module 1: Translation",
        "👤 Module 2: Speaker Similarity",
        "🎭 Module 3: Emotion",
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # MODULE 1
    # ═══════════════════════════════════════════════════════════════════════
    with module_tabs[0]:
        st.markdown("### 📝 Module 1: Translation Quality — BLEU-1 / BLEU-2 / BLEU-3 / BLEU-4 / Overall / chrF / COMET")
        st.markdown("""<div class="card-eval"><span style="font-size:13px;color:var(--text-dim)">
        Hindi audio → Whisper translate → English hypothesis vs Reference English text<br>
        <b>BLEU-1/2/3/4</b> computed independently with sacrebleu ·
        <b>Overall BLEU</b> = geometric mean of all four (report this in paper tables) ·
        <b>chrF</b> character n-gram F-score · <b>COMET-22</b> neural semantic quality
        </span></div>""",unsafe_allow_html=True)

        c1,c2=st.columns(2)
        m1_audio=c1.text_input("Hindi audio dir",  value=FLEURS_AUDIO_DIR, key="m1_audio")
        m1_ref  =c2.text_input("Reference English dir", value=FLEURS_REF_DIR, key="m1_ref")
        m1_max  =st.slider("Max samples (0=all ~418)",0,500,0,10,key="m1_max")
        m1_batch=st.slider("Batch size",1,64,16,key="m1_batch")
        m1_comet=st.toggle("Compute COMET-22 (GPU, ~5 min)",value=False,key="m1_comet",disabled=not HAVE_COMET)

        if st.button("🔍 Scan dataset",key="m1_scan"):
            try:
                ap,ids,refs=TranslationEvaluator.scan_fleurs_dataset(m1_audio,m1_ref)
                has_ref=sum(1 for r in refs if r)
                st.success(f"Found **{len(ap)}** audio files · **{has_ref}** with reference text")
                if ap: st.markdown(f"**Sample:** `{Path(ap[0]).name}` → `{refs[0][:80] or '(missing)'}`")
            except Exception as e: st.error(f"Scan error: {e}")

        if st.button("🚀 Run Translation Evaluation",type="primary",key="m1_run"):
            prog=st.progress(0.); status=st.empty(); log_bx=st.empty(); log_lines=[]
            def _log1(frac,msg):
                prog.progress(min(frac,1.)); status.markdown(f'<span class="mono">{msg}</span>',unsafe_allow_html=True)
                log_lines.append(msg)
                log_bx.markdown('<div class="eval-log-box">'+"".join(f"{l}<br>" for l in log_lines[-10:])+"</div>",unsafe_allow_html=True)
            try:
                ap,ids,refs=TranslationEvaluator.scan_fleurs_dataset(m1_audio,m1_ref)
                if m1_max>0: ap,ids,refs=ap[:m1_max],ids[:m1_max],refs[:m1_max]
                result=TranslationEvaluator.run(ap,ids,refs,batch_size=m1_batch,log_cb=_log1)
                if not m1_comet: result["avg_comet"]=0.
                st.session_state.eval_m1_results=result
                prog.progress(1.); status.success("✅ Complete!")
                _save_eval_json({k:v for k,v in result.items() if k!="records"},"m1_translation_summary")
                _save_eval_csv(result["records"],"m1_translation_per_sample")
            except Exception as e:
                st.error(f"Module 1 failed: {e}"); import traceback; st.code(traceback.format_exc())

        r1=st.session_state.get("eval_m1_results")
        if r1:
            st.markdown("---")
            st.markdown("#### Results")

            # ── Overall BLEU hero banner ──────────────────────────────────
            _ov = r1.get("avg_bleu_overall", 0)
            _ov_cls = "#3ecf8e" if _ov>15 else ("#f59e0b" if _ov>8 else "#ef4444")
            st.markdown(
                f'<div class="overall-bleu-banner">'
                f'<span class="badge badge-green" style="font-size:13px">Overall BLEU (geometric mean 1–4)</span><br>'
                f'<span style="font-size:3rem;font-weight:800;color:{_ov_cls};line-height:1.2">{_ov:.2f}</span><br>'
                f'<span style="font-size:12px;color:var(--text-dim)">'
                f'Geometric mean of BLEU-1/2/3/4 — use this as the single number in your paper tables</span>'
                f'</div>',
                unsafe_allow_html=True)

            # ── Individual metric cards (7 columns) ───────────────────────
            cols=st.columns(7)
            _card_items=[
                ("BLEU-1",  r1.get("avg_bleu_1",0),       "badge-prosenc"),
                ("BLEU-2",  r1.get("avg_bleu_2",0),       "badge-teal"),
                ("BLEU-3",  r1.get("avg_bleu_3",0),       "badge-pp"),
                ("BLEU-4",  r1.get("avg_bleu_4",0),       "badge-amber"),
                ("Overall", r1.get("avg_bleu_overall",0), "badge-green"),
                ("chrF",    r1.get("avg_chrf",0),         "badge-raw"),
                ("COMET-22",r1.get("avg_comet",0),        "badge-eval"),
            ]
            for col,(lbl,val,badge) in zip(cols,_card_items):
                col.markdown(
                    f'<div class="card" style="text-align:center;padding:.75rem .5rem">'
                    f'<span class="badge {badge}" style="font-size:11px">{lbl}</span><br>'
                    f'<span style="font-size:1.35rem;font-weight:700;color:var(--accent)">{val:.2f}</span>'
                    f'</div>',
                    unsafe_allow_html=True)

            # ── Bar chart: all n-grams + overall ─────────────────────────
            _bleu_labels=["BLEU-1","BLEU-2","BLEU-3","BLEU-4","Overall\n(geo-mean)","chrF"]
            _bleu_vals=[
                r1.get("avg_bleu_1",0), r1.get("avg_bleu_2",0),
                r1.get("avg_bleu_3",0), r1.get("avg_bleu_4",0),
                r1.get("avg_bleu_overall",0), r1.get("avg_chrf",0),
            ]
            buf=_plot_bar_dark(_bleu_labels,_bleu_vals,"Translation Quality by Metric","#7c6af7","Score")
            if buf: st.image(buf,use_container_width=True)

            # ── Summary table ─────────────────────────────────────────────
            def _cls1(v,hi,lo): return "hi" if v>=hi else ("med" if v>=lo else "lo")
            st.markdown(f"""<table class="eval-table">
            <tr><th>Metric</th><th>Value</th><th>Range</th><th>Interpretation</th></tr>
            <tr><td>BLEU-1 (unigram)</td>
                <td class="{_cls1(r1.get('avg_bleu_1',0),25,15)}">{r1.get('avg_bleu_1',0):.2f}</td>
                <td class="num">[0,100]</td><td>Word-level overlap — most lenient</td></tr>
            <tr><td>BLEU-2 (bigram)</td>
                <td class="{_cls1(r1.get('avg_bleu_2',0),15,8)}">{r1.get('avg_bleu_2',0):.2f}</td>
                <td class="num">[0,100]</td><td>Phrase-level fluency</td></tr>
            <tr><td>BLEU-3 (trigram)</td>
                <td class="{_cls1(r1.get('avg_bleu_3',0),10,5)}">{r1.get('avg_bleu_3',0):.2f}</td>
                <td class="num">[0,100]</td><td>Phrase + sentence fluency</td></tr>
            <tr><td>BLEU-4 (4-gram)</td>
                <td class="{_cls1(r1.get('avg_bleu_4',0),8,3)}">{r1.get('avg_bleu_4',0):.2f}</td>
                <td class="num">[0,100]</td><td>Long-span coherence — strictest n-gram</td></tr>
            <tr style="background:rgba(52,211,153,.07)">
                <td><b style="color:#3ecf8e">Overall BLEU ↑ (geo-mean 1–4)</b></td>
                <td class="{_cls1(r1.get('avg_bleu_overall',0),12,6)}"><b>{r1.get('avg_bleu_overall',0):.2f}</b></td>
                <td class="num">[0,100]</td>
                <td><b>Geometric mean of BLEU-1/2/3/4 — report this in your paper</b></td></tr>
            <tr><td>chrF (character n-gram)</td>
                <td class="{_cls1(r1.get('avg_chrf',0),45,30)}">{r1.get('avg_chrf',0):.2f}</td>
                <td class="num">[0,100]</td><td>Robust to morphological variation</td></tr>
            <tr><td>COMET-22 (neural)</td>
                <td class="{_cls1(r1.get('avg_comet',0),0.7,0.5)}">{r1.get('avg_comet',0):.4f}</td>
                <td class="num">[0,1]</td><td>Semantic quality — best single metric</td></tr>
            </table>""",unsafe_allow_html=True)

            # ── LaTeX export ──────────────────────────────────────────────
            with st.expander("📄 LaTeX (Translation Quality Table — paper-ready)"):
                b1 = r1.get('avg_bleu_1',0); b2 = r1.get('avg_bleu_2',0)
                b3 = r1.get('avg_bleu_3',0); b4 = r1.get('avg_bleu_4',0)
                ov = r1.get('avg_bleu_overall',0)
                cf = r1.get('avg_chrf',0);   co = r1.get('avg_comet',0)
                latex = (
                    "\\begin{table}[h]\n"
                    "  \\centering\n"
                    "  \\caption{Translation quality on FLEURS Hindi\\textrightarrow{}English}\n"
                    "  \\label{tab:translation}\n"
                    "  \\resizebox{\\columnwidth}{!}{%\n"
                    "  \\begin{tabular}{lccccccc}\n"
                    "    \\toprule\n"
                    "    \\textbf{System} & \\textbf{BLEU-1} & \\textbf{BLEU-2} & "
                    "\\textbf{BLEU-3} & \\textbf{BLEU-4} & "
                    "\\textbf{BLEU (Overall)} $\\uparrow$ & \\textbf{chrF} & \\textbf{COMET} \\\\\n"
                    "    \\midrule\n"
                    f"    SpeechForge v3.5 & {b1:.2f} & {b2:.2f} & {b3:.2f} & {b4:.2f} & "
                    f"\\textbf{{{ov:.2f}}} & {cf:.2f} & {co:.4f} \\\\\n"
                    "    \\bottomrule\n"
                    "  \\end{tabular}}\n"
                    "\\end{table}"
                )
                st.code(latex, language="latex")
                st.caption("💡 Bold Overall BLEU is already applied. Paste directly into your paper.")

            # ── Per-sample expander ───────────────────────────────────────
            with st.expander(f"Per-sample scores (first 25 of {r1['n_total']})"):
                rows="".join(
                    f"<tr><td class='num'>{rec['id']}</td>"
                    f"<td style='font-size:11px'>{rec['hypothesis'][:55]}…</td>"
                    f"<td style='font-size:11px;color:var(--text-dim)'>{rec['reference'][:55]}…</td>"
                    f"<td class=\"{_cls1(rec['bleu_1'],25,10)}\">{rec['bleu_1']:.1f}</td></tr>"
                    for rec in r1["records"][:25])
                st.markdown(f"""<table class="eval-table">
                <tr><th>ID</th><th>Hypothesis</th><th>Reference</th><th>BLEU-1</th></tr>{rows}
                </table>""",unsafe_allow_html=True)

            c1,c2=st.columns(2)
            c1.download_button("⬇ Per-sample CSV",
                _df_to_csv([{k:v for k,v in r.items() if k!="audio_path"} for r in r1["records"]]),
                file_name=f"m1_translation_{_ts()}.csv",mime="text/csv")
            c2.download_button("⬇ Summary JSON",
                json.dumps({k:v for k,v in r1.items() if k!="records"},indent=2),
                file_name=f"m1_summary_{_ts()}.json",mime="application/json")


    # ═══════════════════════════════════════════════════════════════════════
    # MODULE 2
    # ═══════════════════════════════════════════════════════════════════════
    with module_tabs[1]:
        st.markdown("### 👤 Module 2: Speaker Similarity (Cosine + MCD)")
        st.markdown("""<div class="card-eval"><span style="font-size:13px;color:var(--text-dim)">
        Per-speaker: pick reference clip → generate clone → Resemblyzer cosine + MCD
        </span></div>""",unsafe_allow_html=True)

        if not HAVE_RESEMBLYZER:
            st.error("**pip install resemblyzer** required.")
        else:
            m2_dataset=st.radio("Dataset",["RAVDESS","LibriTTS","Both"],horizontal=True,key="m2_ds")
            c1,c2=st.columns(2)
            m2_rv=c1.text_input("RAVDESS root",value=RAVDESS_ROOT,key="m2_rv")
            m2_lb=c2.text_input("LibriTTS root",value=LIBRI_ROOT,key="m2_lb")
            m2_max=st.slider("Max speakers per dataset",10,100,50,5,key="m2_max")
            m2_text=st.text_input("Evaluation text",value="Hello, this is a voice cloning evaluation sentence.",key="m2_text")

            if st.button("🚀 Run Speaker Similarity",type="primary",key="m2_run"):
                prog=st.progress(0.); status=st.empty(); log_bx=st.empty(); log_lines=[]
                def _log2(frac,msg):
                    prog.progress(min(frac,1.)); status.markdown(f'<span class="mono">{msg}</span>',unsafe_allow_html=True)
                    log_lines.append(msg)
                    log_bx.markdown('<div class="eval-log-box">'+"".join(f"{l}<br>" for l in log_lines[-8:])+"</div>",unsafe_allow_html=True)
                try:
                    tts_m=load_tts_model(); combined_map:Dict[str,List[str]]={}
                    if m2_dataset in("RAVDESS","Both"):
                        rv_map=SpeakerSimilarityEvaluator.scan_ravdess(m2_rv,m2_max)
                        combined_map.update({f"RAVDESS_{k}":v for k,v in rv_map.items()})
                    if m2_dataset in("LibriTTS","Both"):
                        lb_map=SpeakerSimilarityEvaluator.scan_libri(m2_lb,m2_max)
                        combined_map.update({f"LibriTTS_{k}":v for k,v in lb_map.items()})
                    if not combined_map: st.error("No speakers found. Check paths."); raise StopIteration
                    result=SpeakerSimilarityEvaluator.run(combined_map,tts_m,m2_text,settings,_log2)
                    st.session_state.eval_m2_results=result
                    prog.progress(1.); status.success("✅ Speaker similarity complete!")
                    _save_eval_json({k:v for k,v in result.items() if k!="records"},"m2_speaker_sim_summary")
                    _save_eval_csv(result["records"],"m2_speaker_sim_per_speaker")
                except StopIteration: pass
                except Exception as e:
                    st.error(f"Module 2 failed: {e}"); import traceback; st.code(traceback.format_exc())

            r2=st.session_state.get("eval_m2_results")
            if r2:
                st.markdown("---"); st.markdown("#### Results")
                c1,c2,c3,c4,c5=st.columns(5)
                for col,(lbl,val,badge) in zip([c1,c2,c3,c4,c5],[
                    ("Speakers", r2["n_speakers"],     "badge-teal"),
                    ("Mean Cos", r2["mean_sim"],       "badge-green"),
                    ("Std Dev",  r2["std_sim"],        "badge-pp"),
                    ("Min Cos",  r2["min_sim"],        "badge-amber"),
                    ("Mean MCD", r2.get("mean_mcd",-1),"badge-raw"),
                ]):
                    col.markdown(f'<div class="card" style="text-align:center">'
                                 f'<span class="badge {badge}">{lbl}</span><br>'
                                 f'<span style="font-size:1.5rem;font-weight:700;color:var(--accent)">{val}</span>'
                                 f'</div>',unsafe_allow_html=True)

                if HAVE_MPL:
                    sims=[r["cosine_sim"] for r in r2["records"] if not r.get("error") and r["cosine_sim"]>0]
                    if sims:
                        fig,ax=plt.subplots(figsize=(9,3.5),facecolor="#0d1117")
                        ax.hist(sims,bins=25,color="#7c6af7",alpha=0.82,edgecolor="#2a2a38")
                        ax.axvline(r2["mean_sim"],color="#40c8c0",lw=2,linestyle="--",label=f'Mean {r2["mean_sim"]:.3f}')
                        ax.axvline(r2["mean_sim"]-r2["std_sim"],color="#f59e0b",lw=1,linestyle=":",alpha=0.7)
                        ax.axvline(r2["mean_sim"]+r2["std_sim"],color="#f59e0b",lw=1,linestyle=":",alpha=0.7,label=f'±1σ {r2["std_sim"]:.3f}')
                        ax.set_facecolor("#0d1117"); ax.tick_params(colors="#7b7b9a",labelsize=8)
                        ax.set_xlabel("Cosine Similarity",color="#7b7b9a",fontsize=9); ax.set_ylabel("Count",color="#7b7b9a",fontsize=9)
                        ax.set_title("Speaker Similarity Distribution",color="#e8e8f0",fontsize=10); ax.spines[:].set_color("#2a2a38")
                        ax.legend(fontsize=8,facecolor="#111118",labelcolor="#e8e8f0")
                        plt.tight_layout(pad=0.5)
                        buf=io.BytesIO(); plt.savefig(buf,format="png",dpi=120,bbox_inches="tight",facecolor="#0d1117"); plt.close(fig); buf.seek(0)
                        st.image(buf,use_container_width=True)
                    recs_sorted=[r for r in sorted(r2["records"],key=lambda x:x["cosine_sim"],reverse=True) if not r.get("error")]
                    if recs_sorted:
                        top20=recs_sorted[:20]
                        buf=_plot_bar_dark([r["speaker_id"][-12:] for r in top20],[r["cosine_sim"] for r in top20],
                                           "Per-Speaker Cosine Similarity (top 20)","#7c6af7","Cosine Sim",
                                           h_line=r2["mean_sim"],h_label=f"Mean {r2['mean_sim']:.3f}")
                        if buf: st.image(buf,use_container_width=True)

                st.markdown(f"""<table class="eval-table">
                <tr><th>Metric</th><th>Your Result</th><th>XTTSv2 Baseline</th><th>Target</th><th>Status</th></tr>
                <tr><td>Cosine Similarity ↑</td>
                    <td class="{'hi' if r2['mean_sim']>0.65 else 'med' if r2['mean_sim']>0.5 else 'lo'}">{r2['mean_sim']:.4f} ± {r2['std_sim']:.4f}</td>
                    <td class="num">0.60–0.80</td><td class="num">≥ 0.70</td>
                    <td style="color:{'#3ecf8e' if r2['mean_sim']>0.60 else '#ef4444'}">{"✅ On target" if r2['mean_sim']>0.60 else "⚠️ Below"}</td></tr>
                <tr><td>MCD (lower=better) ↓</td>
                    <td class="{'hi' if 0<r2.get('mean_mcd',99)<8 else 'med' if r2.get('mean_mcd',99)<12 else 'lo'}">{r2.get('mean_mcd',-1):.2f} dB</td>
                    <td class="num">6–10 dB</td><td class="num">&lt; 8 dB</td>
                    <td style="color:{'#3ecf8e' if 0<r2.get('mean_mcd',99)<8 else '#f59e0b'}">{"✅" if 0<r2.get('mean_mcd',99)<8 else "—"}</td></tr>
                </table>""",unsafe_allow_html=True)

                with st.expander("📄 LaTeX (Speaker Similarity Table)"):
                    latex=(f"\\begin{{table}}[h]\n  \\centering\n"
                           f"  \\caption{{Speaker similarity (cosine $\\uparrow$, MCD $\\downarrow$)}}\n"
                           f"  \\label{{tab:spk}}\n  \\begin{{tabular}}{{lcc}}\n    \\toprule\n"
                           f"    \\textbf{{System}} & \\textbf{{Cosine}} $\\uparrow$ & \\textbf{{MCD (dB)}} $\\downarrow$ \\\\\n"
                           f"    \\midrule\n    SpeechForge v3.5 & {r2['mean_sim']:.4f} $\\pm$ {r2['std_sim']:.4f} & {r2.get('mean_mcd',-1):.2f} \\\\\n"
                           f"    \\bottomrule\n  \\end{{tabular}}\n\\end{{table}}")
                    st.code(latex,language="latex")

                c1,c2=st.columns(2)
                c1.download_button("⬇ Per-speaker CSV",_df_to_csv([{k:v for k,v in r.items() if k!="ref_path"} for r in r2["records"]]),file_name=f"m2_speaker_{_ts()}.csv",mime="text/csv")
                c2.download_button("⬇ Summary JSON",json.dumps({k:v for k,v in r2.items() if k!="records"},indent=2),file_name=f"m2_summary_{_ts()}.json",mime="application/json")


    # ═══════════════════════════════════════════════════════════════════════
    # MODULE 3
    # ═══════════════════════════════════════════════════════════════════════
    with module_tabs[2]:
        st.markdown("### 🎭 Module 3: Emotion Model — Train & Evaluate")
        st.markdown("""<div class="card-eval">
        <b style="color:var(--eval-col)">SpeechEmoNet — Multi-Scale ResNet + Transformer</b><br>
        <span style="font-size:13px;color:var(--text-dim)">
        Train on RAVDESS + CREMA-D · Speaker-disjoint 80/20 split · 248-dim features
        </span></div>""",unsafe_allow_html=True)

        etabs=st.tabs(["🏋️ Train New Model","📊 Evaluate on Test Set","🔍 Inference Test"])

        with etabs[0]:
            st.markdown("#### Dataset paths")
            tc1,tc2=st.columns(2)
            m3_rv=tc1.text_input("RAVDESS root",value=RAVDESS_ROOT,key="m3_rv_train")
            m3_cr=tc2.text_input("CREMA-D root",value=CREMAD_ROOT, key="m3_cr_train")
            st.caption("ℹ️ LibriTTS excluded — label noise in neutral class.")
            st.markdown("#### Training hyper-parameters")
            hc1,hc2,hc3,hc4=st.columns(4)
            m3_ep=hc1.number_input("Epochs",10,200,60,5,key="m3_ep")
            m3_bs=hc2.number_input("Batch size",8,128,32,8,key="m3_bs")
            m3_lr=hc3.select_slider("Learning rate",options=[1e-4,2e-4,3e-4,5e-4,1e-3],value=3e-4,key="m3_lr")
            m3_tf=hc4.slider("Test split %",10,30,20,5,key="m3_tf")/100

            if st.button("🔍 Scan datasets (preview split)",key="m3_scan"):
                with st.spinner("Scanning…"):
                    try:
                        info=_build_emotion_dataset(m3_rv,m3_cr,m3_tf)
                        st.success(f"Train: **{info['n_train']}** files | Test: **{info['n_test']}** files | Classes: {info['classes']}")
                        from collections import Counter
                        tr_cnt=Counter(info["train_labels"]); te_cnt=Counter(info["test_labels"])
                        rows="".join(f"<tr><td>{c}</td><td class='num'>{tr_cnt.get(c,0)}</td><td class='num'>{te_cnt.get(c,0)}</td></tr>" for c in info["classes"])
                        st.markdown(f"""<table class="eval-table"><tr><th>Class</th><th>Train</th><th>Test</th></tr>{rows}</table>""",unsafe_allow_html=True)
                    except Exception as e: st.error(f"Scan failed: {e}")

            st.markdown("---")
            if st.button("🚀 Start Training",type="primary",key="m3_train_btn"):
                prog=st.progress(0.); status=st.empty(); log_bx=st.empty(); log_lines=[]
                loss_chart_slot=st.empty(); train_results_slot=st.empty()
                def _log_t(frac,msg):
                    prog.progress(min(frac,1.0)); status.markdown(f'<span class="mono">{msg}</span>',unsafe_allow_html=True)
                    log_lines.append(msg)
                    log_bx.markdown('<div class="eval-log-box">'+"".join(f"{l}<br>" for l in log_lines[-12:])+"</div>",unsafe_allow_html=True)
                try:
                    _log_t(0.,"Building dataset split…")
                    info=_build_emotion_dataset(m3_rv,m3_cr,m3_tf)
                    _log_t(0.02,f"Train={info['n_train']} Test={info['n_test']} Classes={info['classes']}")
                    result=_run_emotion_training(
                        train_paths=info["train_paths"],train_labels=info["train_labels"],
                        test_paths =info["test_paths"], test_labels =info["test_labels"],
                        classes=info["classes"],feature_dim=_EMO_FEATURE_DIM,max_frames=_EMO_MAX_FRAMES,
                        epochs=int(m3_ep),batch_size=int(m3_bs),lr=float(m3_lr),
                        model_out=_EMO_NET_PATH,scaler_out=_EMO_SCALER_PATH,log_cb=_log_t)
                    prog.progress(1.0); status.success(f"✅ Best val_acc: {result['best_val_acc']:.3f} @ epoch {result['best_epoch']} | Test acc: {result['final_test_acc']:.1f}%")
                    st.session_state["m3_train_result"]=result
                    if HAVE_MPL:
                        h=result["history"]; ep=list(range(1,len(h["train_acc"])+1))
                        fig,axes=plt.subplots(1,2,figsize=(12,3.5),facecolor="#0d1117")
                        for ax,(tr,va,ttl) in zip(axes,[(h["train_loss"],h["val_loss"],"Loss"),(h["train_acc"],h["val_acc"],"Accuracy")]):
                            ax.plot(ep,tr,color="#7c6af7",lw=1.5,label="train"); ax.plot(ep,va,color="#40c8c0",lw=1.5,label="val")
                            ax.set_facecolor("#0d1117"); ax.set_title(ttl,color="#e8e8f0",fontsize=10)
                            ax.tick_params(colors="#7b7b9a",labelsize=8); ax.spines[:].set_color("#2a2a38")
                            ax.legend(fontsize=8,facecolor="#111118",labelcolor="#e8e8f0")
                        plt.tight_layout(pad=0.5)
                        buf=io.BytesIO(); plt.savefig(buf,format="png",dpi=110,bbox_inches="tight",facecolor="#0d1117"); plt.close(fig); buf.seek(0)
                        loss_chart_slot.image(buf,use_container_width=True)
                    if result.get("confusion"):
                        buf2=_plot_confusion(np.array(result["confusion"]),result["emotion_labels"],"Test Confusion Matrix")
                        if buf2: train_results_slot.image(buf2,use_container_width=True)
                    train_results_slot.download_button("⬇ Training history JSON",
                        json.dumps({k:v for k,v in result.items() if k!="confusion"},indent=2),
                        file_name=f"speechemo_training_{_ts()}.json",mime="application/json")
                except Exception as e:
                    st.error(f"Training failed: {e}"); import traceback; st.code(traceback.format_exc())

        with etabs[1]:
            st.markdown("#### Evaluate trained SpeechEmoNet on RAVDESS ground-truth")
            with st.spinner("Loading model…"):
                net_m,net_sc,net_cl,net_err=_emo_load_net()
            if net_m is not None:
                st.success(f"✅ SpeechEmoNet loaded | classes: {net_cl} | feature_dim={getattr(net_m,'_feature_dim',248)} max_frames={getattr(net_m,'_max_frames',128)}")
            else:
                st.warning(f"⚠️ Model not loaded — train first. ({net_err})")

            m3_ev_rv=st.text_input("RAVDESS root for evaluation",value=RAVDESS_ROOT,key="m3_ev_rv")
            m3_max_per=st.slider("Max clips per emotion",5,60,20,5,key="m3_ev_max")

            if st.button("🔄 Clear model cache + reload",key="m3_ev_clear"):
                _emo_load_net.clear(); st.success("Cache cleared."); st.rerun()

            if st.button("📊 Run Evaluation",type="primary",key="m3_ev_run",disabled=(net_m is None)):
                prog=st.progress(0.); status=st.empty(); log_bx=st.empty(); log_lines=[]
                def _log_e(frac,msg):
                    prog.progress(min(frac,1.0)); status.markdown(f'<span class="mono">{msg}</span>',unsafe_allow_html=True)
                    log_lines.append(msg)
                    log_bx.markdown('<div class="eval-log-box">'+"".join(f"{l}<br>" for l in log_lines[-8:])+"</div>",unsafe_allow_html=True)
                try:
                    wav_paths,gt_emos=EmotionEvaluator.scan_ravdess_with_labels(m3_ev_rv,m3_max_per)
                    _log_e(0.,f"Loaded {len(wav_paths)} clips across {len(set(gt_emos))} classes")
                    result=EmotionEvaluator.run(wav_paths,gt_emos,_log_e)
                    st.session_state["m3_eval_result"]=result
                    prog.progress(1.); status.success(f"✅ Accuracy: {result['gen_accuracy']:.1f}% | {result['n_errors']} errors")
                    _save_eval_json({k:v for k,v in result.items() if k not in("records","confusion")},"m3_emotion_eval")
                    _save_eval_csv([{k:v for k,v in r.items() if k!="wav_path"} for r in result["records"]],"m3_emotion_per_sample")
                except Exception as e:
                    st.error(f"Evaluation failed: {e}"); import traceback; st.code(traceback.format_exc())

            r3=st.session_state.get("m3_eval_result")
            if r3:
                st.markdown("---")
                c1,c2,c3=st.columns(3)
                for col,(lbl,val,badge) in zip([c1,c2,c3],[
                    ("Accuracy",f"{r3['gen_accuracy']:.1f}%","badge-green"),
                    ("Total clips",r3["n_total"],"badge-teal"),
                    ("Mean Conf",f"{r3['mean_conf']:.2%}","badge-amber"),
                ]):
                    col.markdown(f'<div class="card" style="text-align:center"><span class="badge {badge}">{lbl}</span><br>'
                                 f'<span style="font-size:1.5rem;font-weight:700;color:var(--accent)">{val}</span></div>',unsafe_allow_html=True)
                emo_set=r3["emotion_labels"]
                rows="".join(
                    f"<tr><td>{EMOTION_EMOJI.get(e,'')} {e}</td>"
                    f"<td class='num'>{r3['per_emotion'].get(e,{}).get('n',0)}</td>"
                    f"<td class=\"{'hi' if r3['per_emotion'].get(e,{}).get('accuracy',0)>0.65 else 'med' if r3['per_emotion'].get(e,{}).get('accuracy',0)>0.40 else 'lo'}\">"
                    f"{r3['per_emotion'].get(e,{}).get('accuracy',0.):.1%}</td>"
                    f"<td class='num'>{r3['per_emotion'].get(e,{}).get('f1',0.):.3f}</td>"
                    f"<td class='num'>{r3['per_emotion'].get(e,{}).get('mean_conf',0.):.3f}</td></tr>"
                    for e in emo_set if r3['per_emotion'].get(e,{}).get('n',0)>0)
                st.markdown(f"""<table class="eval-table">
                <tr><th>Emotion</th><th>N</th><th>Accuracy</th><th>F1</th><th>Conf</th></tr>{rows}
                </table>""",unsafe_allow_html=True)
                if r3.get("confusion"):
                    buf=_plot_confusion(np.array(r3["confusion"]),emo_set)
                    if buf: st.image(buf,use_container_width=True)
                with st.expander("📄 LaTeX (Emotion Results)"):
                    emos_w=[e for e in emo_set if r3['per_emotion'].get(e,{}).get('n',0)>0]
                    latex=(f"\\begin{{table}}[h]\n  \\centering\n"
                           f"  \\caption{{SpeechEmoNet emotion accuracy on RAVDESS}}\n"
                           f"  \\label{{tab:emotion}}\n  \\resizebox{{\\columnwidth}}{{!}}{{%\n"
                           f"  \\begin{{tabular}}{{l{'c'*len(emos_w)}}}\n    \\toprule\n"
                           f"    Metric & "+"& ".join(f"\\textbf{{{e}}}" for e in emos_w)+" \\\\\n"
                           f"    \\midrule\n"
                           f"    Accuracy & "+"& ".join(f"{r3['per_emotion'][e]['accuracy']:.2f}" for e in emos_w)+" \\\\\n"
                           f"    F1 & "+"& ".join(f"{r3['per_emotion'][e]['f1']:.2f}" for e in emos_w)+" \\\\\n"
                           f"    \\bottomrule\n  \\end{{tabular}}}}\n\\end{{table}}")
                    st.code(latex,language="latex")
                c1,c2=st.columns(2)
                c1.download_button("⬇ Per-sample CSV",_df_to_csv([{k:v for k,v in r.items() if k!="wav_path"} for r in r3["records"]]),file_name=f"m3_emotion_{_ts()}.csv",mime="text/csv")
                c2.download_button("⬇ Summary JSON",json.dumps({k:v for k,v in r3.items() if k!="records"},indent=2),file_name=f"m3_summary_{_ts()}.json",mime="application/json")

        with etabs[2]:
            st.markdown("#### Test inference on a single audio file")
            up=st.file_uploader("Upload audio",type=["wav","mp3","m4a","flac"],key="m3_infer_up")
            if up:
                try:
                    with tempfile.NamedTemporaryFile(suffix=".wav",delete=False) as tmp:
                        tmp.write(up.read()); tmp_path=tmp.name
                    st.audio(tmp_path)
                    with st.spinner("Running SpeechEmoNet…"):
                        emo,conf=_emo_classify(tmp_path)
                    clr="#3ecf8e" if conf>0.6 else("#f59e0b" if conf>0.3 else "#ef4444")
                    st.markdown(f"""<div class="card" style="text-align:center;padding:1.5rem">
                      <span style="font-size:3rem">{EMOTION_EMOJI.get(emo,'🎭')}</span><br>
                      <span style="font-size:1.4rem;font-weight:700;color:var(--text)">{emo.upper()}</span><br>
                      <span style="font-size:1rem;color:{clr};font-weight:600">{conf:.1%} confidence</span>
                    </div>""",unsafe_allow_html=True)
                    Path(tmp_path).unlink(missing_ok=True)
                except Exception as e: st.error(f"Inference failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

def build_sidebar():
    with st.sidebar:
        st.markdown("## 🎙️ SpeechForge v3.5")
        st.markdown('<span class="badge badge-freevc">FreeVC</span> '
                    '<span class="badge badge-prosenc">+Neural Prosody</span>',unsafe_allow_html=True)
        st.markdown('<hr style="margin:0.5rem 0">')
        language=st.selectbox("Language",SUPPORTED_LANGUAGES,0,key="sb_lang")
        st.markdown("---")
        st.markdown("**Voice Identity**")
        vs=st.slider("Stability",0.,1.,0.50,0.05,key="sb_vs")
        vm=st.slider("Similarity",0.,1.,0.75,0.05,key="sb_vm")
        st.markdown("---")
        st.markdown("**Emotion Transfer**")
        emotion_intensity=st.slider("Intensity",0.,1.,0.60,0.05)
        st.markdown("---")
        st.markdown("**Prosody Transfer**")
        prosody_transfer=st.toggle("Enable prosody transfer",value=True)
        if prosody_transfer:
            f0_str=st.slider("F0 / Pitch",0.,1.,0.65,0.05)
            ene_str=st.slider("Energy",0.,1.,0.50,0.05)
            rate_str=st.slider("Speaking rate",0.,1.,0.30,0.05)
        else: f0_str=ene_str=rate_str=0.
        st.markdown("---")
        st.markdown("**🧠 Neural Prosody**")
        f0_backend="CREPE" if HAVE_CREPE else("WORLD" if HAVE_PYWORLD else "pyin")
        st.caption(f"F0: {f0_backend} · embed dim: {PROSODY_OUT_DIM}")
        use_neural=st.toggle("Enable Neural Prosody Modules",value=False)
        if use_neural:
            np_stage=st.select_slider("Stage",options=[1,2,3],value=int(st.session_state.get("prosody_training_stage",1)))
            st.session_state.prosody_training_stage=np_stage
            neural_f0_str=st.slider("F0 injection strength",0.,1.,0.80,0.05,key="nf0")
            neural_adain_str=st.slider("AdaIN style strength",0.,1.,0.70,0.05,key="nai")
        else: np_stage=1; neural_f0_str=0.80; neural_adain_str=0.70
        st.markdown("---")
        st.markdown("**🎨 FreeVC Timbre Transfer**")
        if HAVE_FREEVC:
            fv,fve=load_freevc()
            st.caption("✅ d-freevc.pth+wavlm-base.pt" if(fv and fv.is_available()) else f"⚠️ {fve}")
        else: st.caption("⚠️ FreeVC unavailable")
        use_freevc=st.toggle("Enable FreeVC",value=False,disabled=not HAVE_FREEVC)
        freevc_strength=(st.slider("Strength",0.50,1.00,0.70,0.05,key="fvc_str") if use_freevc and HAVE_FREEVC else 0.70)
        st.markdown("---")
        st.markdown("**🔊 BigVGAN** *(optional)*")
        st.caption("✅ installed" if HAVE_BIGVGAN else "⚠️ pip install bigvgan")
        use_bigvgan=st.toggle("Enable BigVGAN",value=False,disabled=not HAVE_BIGVGAN)
        st.markdown("---")
        emo_mode=st.selectbox("Emotion detector",["ensemble","transformer","neural-ser","rule-based"],index=0)
        st.markdown("---")
        do_vad=st.toggle("Silero VAD trim",value=True); do_denoise=st.toggle("Spectral denoising",value=True)
        denoise_strength=st.slider("Denoise strength",0.2,1.,0.8,0.05) if do_denoise else 0.8
        st.markdown("---")
        av=st.session_state.get("active_voice")
        if av:
            st.markdown(f'<span class="badge badge-green">✓ {av}</span>',unsafe_allow_html=True)
            if st.button("Clear voice",use_container_width=True):
                for k in("active_voice","speaker_path","ref_display_name"): st.session_state[k]=None
                st.rerun()
        else: st.info("No active voice")
        st.markdown("---")
        st.caption(f"🖥️ {'CUDA: '+_torch.cuda.get_device_name(0) if DEVICE=='cuda' else 'CPU'}")
        flags=[]
        if HAVE_FREEVC:       flags.append("🎨 FreeVC")
        if HAVE_RESEMBLYZER:  flags.append("👤 Resemblyzer")
        if HAVE_NOISEREDUCE:  flags.append("🔇 NR")
        if HAVE_SACREBLEU:    flags.append("📐 BLEU")
        if HAVE_COMET:        flags.append("☄️ COMET")
        if HAVE_BIGVGAN:      flags.append("🔊 BigVGAN")
        if HAVE_CREPE:        flags.append("🎵 CREPE")
        if HAVE_PYWORLD:      flags.append("🌍 WORLD")
        if flags: st.caption(" · ".join(flags))
    return {"language":language,"voice_stability":vs,"voice_similarity":vm,
            "emotion_intensity":emotion_intensity,"prosody_transfer":prosody_transfer,
            "prosody_f0_strength":f0_str,"prosody_energy_strength":ene_str,"prosody_rate_strength":rate_str,
            "use_neural_prosody":use_neural,"prosody_training_stage":np_stage,
            "neural_f0_strength":neural_f0_str,"neural_adain_strength":neural_adain_str,
            "use_freevc":use_freevc,"freevc_strength":freevc_strength,"use_bigvgan":use_bigvgan,
            "emotion_detector":emo_mode,"do_vad":do_vad,"do_denoise":do_denoise,"denoise_strength":denoise_strength}


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – VOICE CLONE
# ══════════════════════════════════════════════════════════════════════════════

def tab_voice_clone(settings):
    st.markdown("## 🎙️ Voice Clone")
    st.markdown('<div class="card">',unsafe_allow_html=True)
    text=st.text_area("Text to synthesize",value=st.session_state.get("text_input",""),height=110,key="text_area")
    st.session_state.text_input=text
    st.markdown("</div>",unsafe_allow_html=True)
    mode=st.radio("Reference voice source",["Upload","Gallery","Multi-reference blend"],horizontal=True)
    if mode=="Upload":   _upload_panel(settings)
    elif mode=="Gallery": _gallery_panel()
    else:                _multi_ref_panel()
    st.markdown("---")
    has_spk=bool(st.session_state.get("speaker_path"))
    c1,c2=st.columns([2,1])
    with c1:
        if st.button("🔊 Generate Voice",type="primary",use_container_width=True,disabled=not(has_spk and text.strip())):
            with st.spinner("Synthesising…"):
                try:
                    raw,pp,dbg=run_generation(text=text,settings=settings,language=settings["language"])
                    st.success("Done!")
                    display_audio_comparison(raw,pp,target_emotion=dbg.get("target_emotion"),
                        ref_path=dbg.get("ref_path"),ref_prosody=dbg.get("ref_prosody"),settings=settings)
                except Exception as e: st.error(f"Generation failed: {e}")
    with c2:
        if st.button("🔄 Reset",use_container_width=True):
            for k in("speaker_path","ref_display_name","active_voice","multi_ref_paths"):
                st.session_state[k]=None if "paths" not in k else []
            st.rerun()
    if not has_spk: st.info("Upload a reference voice to enable generation.")


def _upload_panel(settings):
    st.markdown('<div class="card">',unsafe_allow_html=True)
    up=st.file_uploader("Reference voice (20–30 s)",type=["wav","mp3","m4a","flac","ogg","webm"])
    c1,c2,c3=st.columns(3)
    dv=c1.checkbox("VAD trim",value=settings.get("do_vad",True))
    dd=c2.checkbox("Denoise",value=settings.get("do_denoise",True))
    ds=c3.slider("Denoise %",0.2,1.,settings.get("denoise_strength",0.8),0.05)
    if up:
        with st.spinner("Preprocessing…"):
            try:
                path,info=AudioPipeline.preprocess(up.read(),do_vad=dv,do_denoise=dd,denoise_strength=ds)
                rn=os.path.basename(path)
                st.session_state.speaker_path=path; st.session_state.ref_display_name=rn
                st.session_state.active_voice=rn; st.success(f"Ready: {rn}")
                col1,col2=st.columns(2)
                with col1: st.audio(path)
                with col2:
                    snr=info.get("snr_db",0); dur=info.get("duration_clean_s",0)
                    qc="good" if snr>20 else("warn" if snr>10 else "bad")
                    st.markdown(f'<div class="metric-row"><span class="metric-chip {qc}">SNR {snr:.1f} dB</span><span class="metric-chip">⏱ {dur:.1f}s</span></div>',unsafe_allow_html=True)
                    if dur<8: st.caption("⚠️ Short – 20–30 s recommended")
                pros=ProsodyAnalyzer.extract(path); st.session_state.stt_prosody=pros
                if pros.get("ok"):
                    st.caption(f"🎵 F0: {pros['f0_mean']:.0f} Hz  σ={pros['f0_std']:.0f}  Energy: {pros['energy_mean']:.4f} [{pros.get('f0_extractor','pyin')}]")
            except Exception as e: st.error(f"Upload failed: {e}")
    st.markdown("</div>",unsafe_allow_html=True)


def _gallery_panel():
    st.markdown('<div class="card">',unsafe_allow_html=True)
    voices=[f for f in os.listdir(VOICES_DIR) if f.lower().endswith(".wav")] if VOICES_DIR.exists() else []
    if not voices: st.warning("No saved voices."); st.markdown("</div>",unsafe_allow_html=True); return
    for name in sorted(voices):
        vp=VOICES_DIR/name; is_active=name==st.session_state.get("active_voice")
        c1,c2,c3=st.columns([4,1,1])
        with c1:
            badge='<span class="badge badge-green">Active</span> ' if is_active else ""
            st.markdown(f"{badge}**{name}**",unsafe_allow_html=True); st.audio(str(vp))
        with c2:
            if not is_active:
                if st.button("Set",key=f"set_{name}",use_container_width=True):
                    st.session_state.speaker_path=str(vp); st.session_state.ref_display_name=name
                    st.session_state.active_voice=name; st.session_state.stt_prosody=ProsodyAnalyzer.extract(str(vp))
                    st.rerun()
        with c3:
            if st.session_state.get("pending_delete")==name:
                if st.button("✓",key=f"cd_{name}",use_container_width=True):
                    p=VOICES_DIR/name
                    if p.exists(): p.unlink()
                    st.session_state.pending_delete=None; st.rerun()
            else:
                if st.button("🗑",key=f"dl_{name}",use_container_width=True):
                    st.session_state.pending_delete=name; st.rerun()
        st.markdown("---")
    st.markdown("</div>",unsafe_allow_html=True)


def _multi_ref_panel():
    st.markdown('<div class="card card-accent">',unsafe_allow_html=True)
    st.markdown("Upload **2–5 clips** from the same speaker.")
    if not HAVE_RESEMBLYZER: st.warning("pip install resemblyzer")
    ups=st.file_uploader("Multiple clips",type=["wav","mp3","m4a","flac"],accept_multiple_files=True,key="multi_ref_up")
    if ups and len(ups)>=2:
        with st.spinner(f"Processing {len(ups)} clips…"):
            paths=[]
            for u in ups:
                try: p,_=AudioPipeline.preprocess(u.read()); paths.append(p)
                except: pass
            if paths:
                st.session_state.multi_ref_paths=paths; st.session_state.speaker_path=paths[0]
                st.session_state.ref_display_name=f"multi-ref ({len(paths)} clips)"
                st.session_state.active_voice=f"multi-ref ({len(paths)} clips)"
                if HAVE_RESEMBLYZER and len(paths)>1:
                    avg=SpeakerEmbeddingManager.average_embeddings(paths)
                    if avg is not None: st.success(f"✅ {len(paths)} clips averaged")
                    sims=[SpeakerEmbeddingManager.cos_sim(SpeakerEmbeddingManager.extract(paths[i]),SpeakerEmbeddingManager.extract(paths[i+1])) for i in range(len(paths)-1)]
                    avg_s=float(np.mean(sims)); cls="good" if avg_s>0.8 else("warn" if avg_s>0.6 else "bad")
                    st.markdown(f'<span class="metric-chip {cls}">👤 Intra-speaker sim: {avg_s:.2%}</span>',unsafe_allow_html=True)
    st.markdown("</div>",unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – TRANSLATION
# ══════════════════════════════════════════════════════════════════════════════

def tab_translation(settings):
    st.markdown("## 🌐 Speech → English Translation")
    src=st.radio("Source",["Active reference voice","Upload new audio"],horizontal=True,key="stt_src")
    stt_path=None
    if src=="Active reference voice":
        sp=st.session_state.get("speaker_path")
        if sp and os.path.exists(sp): stt_path=sp; st.info(f"Using: {st.session_state.get('ref_display_name')}")
        else: st.warning("No active reference voice.")
    else:
        up=st.file_uploader("Upload audio",type=["wav","mp3","m4a","flac","ogg","webm"],key="stt_up")
        if up:
            try:
                with st.spinner("Preparing…"):
                    stt_path,info=AudioPipeline.preprocess(up.read(),
                        do_vad=settings.get("do_vad",True),do_denoise=settings.get("do_denoise",True),
                        denoise_strength=settings.get("denoise_strength",0.8))
                st.success(f"Ready · {info.get('duration_clean_s',0):.1f}s · SNR {info.get('snr_db',0):.1f} dB")
                st.audio(stt_path)
            except Exception as e: st.error(f"Prep failed: {e}")
    if st.button("🔍 Translate + Detect Emotion",disabled=stt_path is None,key="stt_go"):
        with st.spinner("Whisper + emotion ensemble…"):
            try:
                t0=time.time(); r=translate_to_english(stt_path,settings.get("emotion_detector","ensemble"))
                for k,v in {"stt_english_text":r["english_text"],"stt_original_text":r["original_text"],
                            "stt_source_lang":r["source_language"],"stt_detected_emotion":r["emotion"],
                            "stt_emotion_confidence":r["emotion_confidence"],"stt_emotion_method":r["emotion_method"],
                            "stt_audio_path":stt_path,"stt_prosody":r["prosody"]}.items():
                    st.session_state[k]=v
                st.success(f"Done in {time.time()-t0:.1f}s · Lang: **{r['source_language'].upper()}** · Emotion: **{r['emotion']}** {EMOTION_EMOJI.get(r['emotion'],'')}")
            except Exception as e: st.error(f"Failed: {e}")
    if st.session_state.get("stt_english_text"):
        lang=(st.session_state.get("stt_source_lang") or "?").upper()
        emo=st.session_state.get("stt_detected_emotion") or "neutral"
        conf=st.session_state.get("stt_emotion_confidence") or 0.
        ei=float(settings.get("emotion_intensity",0.6))
        c1,c2,c3=st.columns(3)
        c1.markdown(f'<span class="badge badge-teal">{lang}→EN</span>',unsafe_allow_html=True)
        c2.markdown(f'<span class="badge badge-amber">{EMOTION_EMOJI.get(emo,"")} {emo} ({conf:.0%})</span>',unsafe_allow_html=True)
        c3.markdown(f'<span class="badge badge-purple">🎭 Intensity {ei:.0%}</span>',unsafe_allow_html=True)
        pros=st.session_state.get("stt_prosody") or {}
        if pros.get("ok"):
            st.markdown(f'<span class="mono">F0 {pros["f0_mean"]:.0f}±{pros["f0_std"]:.0f} Hz · Energy {pros["energy_mean"]:.4f} · Centroid {pros["spectral_centroid_mean"]:.0f} Hz</span>',unsafe_allow_html=True)
        orig=st.session_state.get("stt_original_text","")
        if orig and orig!=st.session_state.get("stt_english_text"):
            with st.expander("Original text"): st.text(orig)
        edited=st.text_area("English text (editable)",value=st.session_state.get("stt_english_text",""),height=100,key="stt_edit")
        has_spk=bool(st.session_state.get("speaker_path"))
        cols=st.columns(2)
        combos=[("👤🎭 Speaker+Emotion",True,True),("👤 Speaker,No Emotion",True,False),
                ("🎭 No Speaker+Emotion",False,True),("Default speaker",False,False)]
        for idx,(lbl,us,ue) in enumerate(combos):
            with cols[idx%2]:
                dis=us and not has_spk
                if st.button(lbl,key=f"sg_{idx}",use_container_width=True,disabled=dis):
                    eo=st.session_state.get("stt_detected_emotion") if ue else None
                    if eo not in VALID_EMOTIONS: eo=None
                    ref=st.session_state.get("speaker_path") if us else None
                    with st.spinner("Generating…"):
                        try:
                            raw,pp,dbg=run_generation(edited,settings,"en",eo,"stt",ref,ref_prosody=st.session_state.get("stt_prosody"))
                            display_audio_comparison(raw,pp,settings=settings)
                        except Exception as e: st.error(f"Failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════

def tab_architecture(settings):
    st.markdown("## 🏗️ Architecture: SpeechForge v3.5")
    frozen=[("BERT encoder","Text → phoneme embeddings","Unchanged"),
            ("GPT-2 (token generator)","Text emb → EnCodec tokens","Frozen; new cross-attn inserted alongside"),
            ("EnCodec decoder","EnCodec tokens → mel","Frozen; F0InjectionLayer inserted after"),
            ("BigVGAN vocoder","Mel → waveform","Frozen; AdaINStyleLayer inserted before")]
    rows="".join(f"<tr><td style='color:#a78bfa;font-weight:600'>{m}</td><td style='color:#e8e8f0'>{r}</td><td class='mono'>{n}</td></tr>" for m,r,n in frozen)
    st.markdown(f"""<div class="card"><b style="color:var(--text-dim)">Frozen backbone (XTTSv2)</b>
    <table class="diff-table"><tr><th>Module</th><th>Role</th><th>v3.5 note</th></tr>{rows}</table></div>""",unsafe_allow_html=True)

    new_mods=[
        ("ProsodyEncoder","CoordConv1d + BiGRU","Trainable","Reference mel → frame embedding [T,4]."),
        ("ProsodyCrossAttention","MultiheadAttention gated","Trainable","Injects prosody into GPT-2 hidden states."),
        ("F0Extractor","CREPE→WORLD→pyin","NOT trainable","Deterministic F0 curve extraction."),
        ("SpeakerNormalizer","Log-domain mapping","NOT trainable","Maps source F0 range → reference range."),
        ("F0InjectionLayer","MLP additive injection","Trainable","mel_out = mel_in + MLP(f0_norm). After EnCodec."),
        ("AdaINStyleLayer","Per-frame InstanceNorm","Trainable","mel_out = InstanceNorm(mel)×(1+scale)+shift."),
    ]
    rows=""
    for m,a,t,d in new_mods:
        tc="#3ecf8e" if t=="Trainable" else "#f59e0b"
        rows+=f"<tr><td style='color:#a78bfa;font-weight:600'>{m}</td><td class='mono'>{a}</td><td style='color:{tc};font-weight:600'>{t}</td><td style='font-size:12px;color:#e8e8f0'>{d}</td></tr>"
    st.markdown(f"""<div class="card" style="border-color:rgba(167,139,250,.4)">
    <b style="color:var(--prosenc)">New trainable modules (v3.5)</b>
    <table class="diff-table"><tr><th>Module</th><th>Architecture</th><th>Training</th><th>Purpose</th></tr>{rows}
    </table></div>""",unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Checkpoint management")
    c1,c2,c3=st.columns(3)
    with c1:
        if st.button("💾 Save modules",use_container_width=True):
            try:
                b=load_prosody_bundle(int(st.session_state.get("prosody_training_stage",1))); b.save(PROSODY_CKPT_DIR)
                st.success(f"Saved to {PROSODY_CKPT_DIR}")
            except Exception as e: st.error(str(e))
    with c2:
        if st.button("📂 Reload",use_container_width=True):
            load_prosody_bundle.clear(); st.success("Reloaded"); st.rerun()
    with c3:
        ckpts=list(PROSODY_CKPT_DIR.glob("*.pt")); st.caption(f"{len(ckpts)} .pt files in {PROSODY_CKPT_DIR}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.markdown("""
    <div style="display:flex;align-items:center;gap:14px;margin-bottom:.25rem">
      <span style="font-size:2.2rem">🎙️</span>
      <div>
        <h1 style="margin:0;font-size:1.75rem;font-family:'Space Mono',monospace;
                   background:linear-gradient(90deg,#7c6af7,#40c8c0,#a78bfa);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent">
          SpeechForge v3.5 — FreeVC + Neural Prosody + Evaluation Suite
        </h1>
        <p style="margin:0;color:#7b7b9a;font-size:.8rem">
          XTTSv2 · Whisper large-v3 · FreeVC · ProsodyEncoder · ProsodyCrossAttn · F0Injection · AdaIN ·
          <span style="color:#34d399">BLEU-1/2/3/4 · Overall BLEU · chrF · COMET · Speaker Sim · Emotion</span>
        </p>
      </div>
    </div>""",unsafe_allow_html=True)

    settings=build_sidebar()
    tabs=st.tabs(["🎙️ Voice Clone","🌐 Translation","🏗️ Architecture","📊 Results & Metrics"])
    with tabs[0]: tab_voice_clone(settings)
    with tabs[1]: tab_translation(settings)
    with tabs[2]: tab_architecture(settings)
    with tabs[3]: tab_evaluation(settings)


if __name__=="__main__":
    main()
