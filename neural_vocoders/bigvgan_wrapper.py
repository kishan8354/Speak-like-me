from __future__ import annotations
import os
from typing import Optional, Tuple
import numpy as np
try:
    import torch
    import bigvgan
    from bigvgan import BigVGAN as _BigVGANModel
    import librosa
    import soundfile as sf
    HAVE_BIGVGAN = True
except ImportError:
    HAVE_BIGVGAN = False
    _BigVGANModel = None  # type: ignore


# ── Constants matching BigVGAN-v2 24kHz config ────────────────────────────────
# RESEARCH PAPER NOTE:
#   These parameters must exactly match the pre-trained checkpoint.
#   BigVGAN-v2 (24 kHz) uses:
#       n_mels=100, n_fft=1024, hop_size=256, win_size=1024,
#       sampling_rate=24000, fmax=None (no mel-filter cutoff).
BIGVGAN_SR        = 24_000
BIGVGAN_N_MELS    = 100
BIGVGAN_N_FFT     = 1024
BIGVGAN_HOP_SIZE  = 256
BIGVGAN_WIN_SIZE  = 1024
BIGVGAN_FMIN      = 0
BIGVGAN_FMAX      = None   # BigVGAN-v2 trains without fmax clipping

# HuggingFace model ID for BigVGAN-v2 24kHz with 100 mels
_BIGVGAN_MODEL_ID = "nvidia/bigvgan_v2_24khz_100band_256x"


class BigVGANVocoder:
    """
    Lightweight wrapper around NVIDIA BigVGAN-v2.

    RESEARCH PAPER NOTE:
        The wrapper exposes a single public method, enhance(wav_path, output_path),
        which:
          1. Loads the input WAV and resamples to 24 kHz.
          2. Computes the mel spectrogram with BigVGAN's exact filterbank.
          3. Runs the BigVGAN generator forward pass (no gradient).
          4. Optionally resamples the output back to the original SR.
          5. Saves to output_path and returns (output_path, success_bool).

        Singleton pattern: the model is loaded once and kept on the selected
        device.  In Streamlit this is handled by @st.cache_resource in the
        calling code.
    """

    def __init__(self, device: str = "cpu"):
        self._device    = device
        self._model     = None
        self._error_msg = ""
        self._load()

    # ── Internal model loader ────────────────────────────────────────────────

    def _load(self) -> None:
        """Load BigVGAN-v2 from HuggingFace (cached after first download)."""
        if not HAVE_BIGVGAN:
            self._error_msg = "bigvgan package not installed (pip install bigvgan)"
            return
        try:
            # RESEARCH PAPER NOTE:
            #   use_cuda_kernel=False keeps compatibility with CPU-only machines.
            #   On CUDA the custom CUDA kernel is not required for correctness,
            #   only for ~10% speed gain; we skip it for portability.
            self._model = _BigVGANModel.from_pretrained(
                _BIGVGAN_MODEL_ID,
                use_cuda_kernel=False,
            )
            self._model.remove_weight_norm()
            self._model = self._model.eval().to(self._device)
        except Exception as exc:
            self._error_msg = str(exc)
            self._model     = None

    # ── Public helpers ────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """Return True iff the model loaded successfully."""
        return self._model is not None

    def get_error(self) -> str:
        return self._error_msg

    # ── Mel extraction ────────────────────────────────────────────────────────

    @staticmethod
    def _wav_to_mel(wav: np.ndarray, sr: int) -> "torch.Tensor":
        """
        Compute mel spectrogram matching BigVGAN-v2's training filterbank.

        RESEARCH PAPER NOTE:
            The mel filterbank parameters MUST match the model's h.json config
            exactly, otherwise the vocoder produces garbled output.  We
            replicate the exact librosa call BigVGAN uses in its inference
            script rather than using torch.stft, because librosa's implementation
            handles centre-padding and window normalisation identically to the
            training pipeline.
        """
        # Resample to BigVGAN SR if needed
        if sr != BIGVGAN_SR:
            wav = librosa.resample(wav, orig_sr=sr, target_sr=BIGVGAN_SR)

        # Peak-normalise to ±1 to match training distribution
        peak = float(np.max(np.abs(wav)))
        if peak > 1e-6:
            wav = wav / peak

        mel = librosa.feature.melspectrogram(
            y=wav,
            sr=BIGVGAN_SR,
            n_fft=BIGVGAN_N_FFT,
            hop_length=BIGVGAN_HOP_SIZE,
            win_length=BIGVGAN_WIN_SIZE,
            n_mels=BIGVGAN_N_MELS,
            fmin=BIGVGAN_FMIN,
            fmax=BIGVGAN_FMAX,
            center=True,
            window="hann",
            pad_mode="reflect",
        )
        # Convert to log-mel (BigVGAN uses log10, clip floor at 1e-5)
        log_mel = np.log10(np.maximum(mel, 1e-5))
        # Shape: (1, n_mels, T) as expected by BigVGAN generator
        return torch.FloatTensor(log_mel).unsqueeze(0)

    # ── Main enhancement method ───────────────────────────────────────────────

    def enhance(
        self,
        wav_path: str,
        output_path: Optional[str] = None,
    ) -> Tuple[str, bool]:
        """
        Re-synthesise wav_path through BigVGAN and write to output_path.

        RESEARCH PAPER NOTE:
            This method is called at Stage 5 of post_process() in app.py.
            The input wav_path already contains all prosodic and emotion
            corrections; BigVGAN's job is purely artefact removal.

        Parameters
        ----------
        wav_path    : path to the input (corrected) WAV file.
        output_path : destination path.  Defaults to wav_path with
                      _bigvgan suffix.

        Returns
        -------
        (output_path, success): output_path is the written file,
                                success is False on any error.
        """
        if not self.is_available():
            return wav_path, False

        if output_path is None:
            output_path = wav_path.replace(".wav", "_bigvgan.wav")

        try:
            import soundfile as sf  # local import keeps top-level optional

            # ── Load original audio ────────────────────────────────────────
            wav_orig, sr_orig = librosa.load(wav_path, sr=None, mono=True)

            # ── Build mel tensor ───────────────────────────────────────────
            mel_tensor = self._wav_to_mel(wav_orig, sr_orig)
            mel_tensor = mel_tensor.to(self._device)

            # ── Forward pass (no grad) ─────────────────────────────────────
            # RESEARCH PAPER NOTE:
            #   torch.no_grad() is mandatory; omitting it would accumulate
            #   computation graph nodes and OOM on long clips.
            with torch.no_grad():
                wav_out = self._model(mel_tensor)        # (1, 1, T_out)
                wav_out = wav_out.squeeze().cpu().numpy()  # (T_out,)

            # ── Resample back to original SR if needed ─────────────────────
            # RESEARCH PAPER NOTE:
            #   BigVGAN always outputs at 24 kHz.  We resample back to the
            #   pipeline SR so downstream quality metrics and audio players
            #   are SR-agnostic.
            if sr_orig != BIGVGAN_SR:
                wav_out = librosa.resample(wav_out, orig_sr=BIGVGAN_SR, target_sr=sr_orig)
                sr_write = sr_orig
            else:
                sr_write = BIGVGAN_SR

            # ── Peak normalise output ──────────────────────────────────────
            pk = float(np.max(np.abs(wav_out)))
            if pk > 1e-6:
                wav_out = wav_out * (0.97 / pk)

            sf.write(output_path, wav_out, sr_write, subtype="PCM_16")
            return output_path, True

        except Exception as exc:
            # RESEARCH PAPER NOTE:
            #   Graceful fallback: return the original path and False so
            #   post_process() can skip BigVGAN without crashing.
            self._error_msg = str(exc)
            return wav_path, False