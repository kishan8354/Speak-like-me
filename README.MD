# 🎙️ Speak Like Me

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/XTTSv2-Backbone-8B5CF6?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Whisper-large--v3-10B981?style=for-the-badge"/>
</p>

<p align="center">
  <strong>Advanced speech-to-speech with voice cloning, prosody transfer, and automated research-grade evaluation.</strong><br/>
  Built on a frozen XTTSv2 backbone with four trainable prosody conditioning modules, FreeVC timbre transfer, and a three-module evaluation suite covering translation quality, speaker similarity, and emotion preservation.
</p>

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Architecture](#-architecture)
- [Neural Prosody Modules](#-neural-prosody-modules)
- [Post-Processing Pipeline](#-post-processing-pipeline)
- [Evaluation Suite](#-evaluation-suite)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Dependencies](#-dependencies)
- [Roadmap](#-roadmap)

---

## 🧠 Overview

 Speak Like Me is a research-grade, production-ready S2ST system that grafts **four trainable neural prosody modules** onto a frozen XTTSv2 backbone. The architecture preserves XTTSv2's strong voice cloning capability while adding expressive prosody control, timbre transfer via FreeVC, and a multi-stage self-supervised training curriculum.

**Key capabilities:**

- 🎯 **Zero-shot voice cloning** — single or multi-reference speaker conditioning
- 🎵 **Neural prosody transfer** — F0, energy, and speaking rate matching from a reference
- 🎭 **Emotion post-processing** — rule-parameterised shaping across 6 emotion classes
- 🎨 **FreeVC timbre transfer** — d-freevc.pth + wavlm-base.pt timbre refinement
- 📊 **Research-grade evaluation** — BLEU-1/2/3/4, Overall BLEU, chrF, COMET-22, cosine similarity, MCD, emotion accuracy
- 🌐 **Multilingual** — 16 languages via XTTSv2 + Whisper large-v3 translation

---

## 🏗️ Architecture

### Frozen Backbone (XTTSv2)

| Module | Role | v3.5 Treatment |
|---|---|---|
| **BERT Encoder** | Text → phoneme embeddings | Unchanged |
| **GPT-2** | Text embeddings → EnCodec tokens | Frozen; new cross-attention inserted alongside |
| **EnCodec Decoder** | EnCodec tokens → mel-spectrogram | Frozen; F0InjectionLayer inserted after |
| **BigVGAN Vocoder** | Mel-spectrogram → waveform | Frozen; AdaINStyleLayer inserted before |

### Input Pipeline

```
Audio Input
    │
    ├─ Silero VAD trim
    ├─ Spectral denoising (noisereduce)
    └─ TextNormalizer (abbreviations, num2words)
         │
         └─► XTTSv2 Backbone
```

---

## 🔧 Neural Prosody Modules

Four lightweight trainable modules are grafted onto the frozen backbone with no modification to backbone weights.

### 1. `ProsodyEncoder`
```
Reference Mel → CoordConv1d(5) → ReLU → BN → CoordConv1d(3) → ReLU → BN
             → BiGRU(hidden=256, layers=2) → LayerNorm → Linear → [T, 4]
```
Extracts a compact frame-level prosody embedding from the reference audio mel-spectrogram. Uses CoordConv1d to inject positional awareness without positional encodings.

### 2. `ProsodyCrossAttention`
```
GPT-2 hidden states (Q) × Prosody embeddings (K, V)
    → Gated MultiheadAttention (8 heads) → LayerNorm
    → Learnable scalar gate (sigmoid-activated)
```
Injects reference prosody into GPT-2 hidden states via gated cross-attention, enabling the token generator to condition on prosodic context.

### 3. `F0InjectionLayer`
```
F0 curve (CREPE → WORLD → pyin cascade)
    → MLP(f0_dim=1 → 64 → mel_dim=80)
    → Additive injection into mel → LayerNorm
```
Injects normalised pitch contours directly into the mel-spectrogram after EnCodec decoding. The CREPE → WORLD → pyin fallback cascade maximises F0 estimation reliability.

### 4. `AdaINStyleLayer`
```
Prosody embedding → Linear → [scale, shift] (mel_dim × 2)
Mel → InstanceNorm(per-frame) → rescale: mel × (1 + scale) + shift
```
Applies Adaptive Instance Normalisation to transfer prosodic style characteristics (rhythm, energy contour) from the reference before BigVGAN synthesis.

### Training Curriculum

| Stage | Active modules | Objective |
|---|---|---|
| **Stage 1** | ProsodyEncoder, ProsodyCrossAttention | Self-supervised prosody reconstruction |
| **Stage 2** | + F0InjectionLayer | F0 contour matching |
| **Stage 3** | + AdaINStyleLayer | Full style transfer (end-to-end) |

Training datasets:**RAVDESS**, **LibriTTS train-clean-360**  
Optimiser: AdamW + OneCycleLR · Mixed precision via AMP GradScaler

---

## 🔁 Post-Processing Pipeline

Each generated waveform passes through up to nine stages:

```
XTTSv2 raw output
    │
    ├─ 1. ProsodyTransfer      — F0 / energy / speaking-rate matching
    ├─ 2. EmotionPostProcessor — Pitch shift + time-stretch + spectral tilt
    ├─ 3. NeuralProsody        — F0Injection + AdaIN (Stage 2/3)
    ├─ 4. FreeVC               — Timbre conversion (d-freevc.pth + wavlm-base.pt)
    ├─ 5. Light Denoise        — noisereduce (prop_decrease=0.30)
    ├─ 6. BigVGAN              — Optional neural vocoder enhancement
    └─ 7. Peak Normalisation   — Target 0.97 FS
```

Emotion classes: `happy · sad · angry · fearful · surprised · neutral`

---

## 📊 Evaluation Suite

Three production-ready, dataset-level evaluation modules with LaTeX table export for direct use in research papers.

### Module 1 — Translation Quality (`TranslationEvaluator`)

**Pipeline:** Hindi audio → Whisper large-v3 translate → English hypothesis vs. reference text

| Metric | Description |
|---|---|
| BLEU-1 | Unigram precision — most lenient n-gram score |
| BLEU-2 | Bigram precision — phrase-level fluency |
| BLEU-3 | Trigram precision — phrase + sentence fluency |
| BLEU-4 | 4-gram precision — long-span coherence |
| **Overall BLEU** | **Geometric mean of BLEU-1/2/3/4 — report this in paper tables** |
| chrF | Character n-gram F-score — robust to morphological variation |
| COMET-22 | Neural semantic quality metric (`Unbabel/wmt22-comet-da`) |

Dataset: **FLEURS Hindi → English** (~418 samples)

---

### Module 2 — Speaker Similarity (`SpeakerSimilarityEvaluator`)

**Pipeline:** Reference clip → TTS clone → Resemblyzer cosine similarity + Mel-Cepstral Distortion

| Metric | Target |
|---|---|
| Cosine similarity ↑ | ≥ 0.70 |
| MCD (dB) ↓ | < 8 dB |

Datasets: **RAVDESS** + **LibriTTS train-clean-360** (up to 100 speakers each)

---

### Module 3 — Emotion Preservation (`EmotionEvaluator` + `SpeechEmoNet`)

A custom **SE-ResNet2D + Transformer** emotion classifier trained on RAVDESS + CREMA-D with speaker-disjoint 80/20 splits.

**Architecture:**
```
Input: [1, 248, 128] spectrogram tensor
    → Stem (Conv2d + InstanceNorm) → MaxPool
    → SEResBlock × 3 (Squeeze-Excitation 2D)
    → Frequency pooling → Linear projection
    → TransformerEncoder (3 layers, 8 heads, d_model=256)
    → [CLS mean + CLS max] → MLP head → 6/7 classes
```

Features: 128-dim mel + 40 MFCC + Δ MFCC + ΔΔ MFCC = **248 dimensions**  
Training: AdamW + cosine LR warmup + label smoothing + SpecAugment  
Dataset: **RAVDESS** + **CREMA-D** (speaker-disjoint split)

---

## ⚙️ Installation

### Prerequisites

- Python 3.10+
- CUDA 11.8+ (recommended; CPU fallback supported)
- 8 GB+ VRAM for full pipeline with COMET

### 1. Clone & install

```bash
git clone https://github.com/kishan8354/Speak-like-me
cd  Speak Like Me

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install TTS openai-whisper streamlit pydub librosa soundfile noisereduce resemblyzer
pip install sacrebleu unbabel-comet pandas matplotlib crepe pyworld
```

### 2. Download model weights

```bash
# FreeVC weights (place in neural_vocoders/)
# d-freevc.pth + wavlm-base.pt

# Whisper large-v3 (auto-downloads or place at):
# models/large-v3.pt

# XTTSv2 (auto-downloads via Coqui TTS on first run)
```

### 3. Prepare datasets *(for evaluation only)*

```
datasets/
├── fluers/
│   ├── audio/          # Hindi .wav files
│   └── reference_en/   # English reference .tsv or per-file .txt
├── RAVDESS/            # Actor subdirectories
├── CREMA-D/            # AudioWAV/ or flat .wav files
└── train-clean-360/    # LibriTTS speaker/chapter structure
```

### 4. Launch

```bash
streamlit run app.py
```

---

## 🖥️ Usage

### Voice Cloning

1. Upload a 20–30 second reference clip in the **Voice Clone** tab
2. The pipeline applies Silero VAD trimming, spectral denoising, and peak normalisation automatically
3. Enter text, configure prosody/emotion settings in the sidebar, and generate
4. Both raw XTTS output and fully post-processed output are presented side-by-side with waveform + spectrogram visualisations and quality metrics

### Translation + Cross-lingual TTS

1. Upload a non-English audio clip in the **Translation** tab
2. Whisper large-v3 detects language and translates to English
3. Emotion is auto-detected (SpeechEmoNet → rule-based fallback)
4. Generate with four modes: speaker + emotion / speaker only / emotion only / default speaker

### Evaluation

Run any of the three evaluation modules from the **Results & Metrics** tab. Each module produces:
- Interactive score tables with colour-coded thresholds
- Matplotlib visualisations (bar charts, confusion matrices, similarity histograms)
- One-click CSV + JSON export
- Ready-to-paste LaTeX table for research papers

---

## 📁 Project Structure

```
Speak Like Me/
├── app.py                          # Main Streamlit application
├── models/
│   └── large-v3.pt                 # Whisper weights
├── neural_vocoders/
│   ├── freevc_wrapper.py
│   ├── bigvgan_wrapper.py
│   └── breathing_synthesizer.py
├── emotion_detection/
│   └── saved_models/
│       ├── speechemo_net.pt        # Trained SpeechEmoNet checkpoint
│       └── speechemo_scaler.pkl    # StandardScaler for features
├── checkpoints/
│   └── prosody_modules/            # enc.pt, ca.pt, f0.pt, adain.pt
├── voices/                         # Processed reference voice clips
├── outputs/                        # Generated audio files
└── evaluation/                     # Saved evaluation results (CSV/JSON)
```

---

## 📦 Dependencies

| Package | Purpose |
|---|---|
| `TTS` (Coqui) | XTTSv2 backbone |
| `openai-whisper` | Transcription + translation |
| `torch` / `torchaudio` | Neural network training & inference |
| `librosa` | Audio feature extraction, F0, mel |
| `resemblyzer` | Speaker embedding (cosine similarity) |
| `noisereduce` | Spectral denoising |
| `sacrebleu` | BLEU-1/2/3/4, chrF computation |
| `unbabel-comet` | COMET-22 neural MT metric |
| `crepe` | Neural F0 estimation (primary) |
| `pyworld` | WORLD vocoder F0 (fallback) |
| `pydub` | MP3 export |
| `streamlit` | Web UI |
| `matplotlib` | Visualisations |
| `pandas` | Tabular data handling |

Optional: `pesq`, `pystoi` for additional perceptual quality metrics.

---

## 🗺️ Roadmap

- [ ] VITS2 / StyleTTS2 backbone variants
- [ ] Real-time streaming inference endpoint
- [ ] Diffusion-based prosody prior (replace BiGRU encoder)
- [ ] UTMOS / MOSNet automated MOS scoring in evaluation suite
- [ ] Multilingual emotion model (beyond RAVDESS + CREMA-D)
- [ ] Docker deployment with CUDA support

---

## 📜 License

This project is intended for research and educational use. Please refer to the individual licences of XTTSv2 (Coqui), Whisper (MIT), FreeVC, BigVGAN, and RAVDESS/CREMA-D datasets before any commercial application.

---

<p align="center">
  Built with PyTorch · Coqui TTS · OpenAI Whisper · FreeVC · Streamlit
</p>
