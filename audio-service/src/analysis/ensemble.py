"""BS-Roformer + MDX-Net specialist separators via `audio-separator`.

Two public functions: `separate_roformer_vocals` and `separate_roformer_bass`.
Both lazy-load their separator on first call. Memory-conscious sequential
use is the caller's responsibility; call `free_model()` between models.

`_Separator` is a module attribute initialized to `None` so tests can
monkeypatch it without importing `audio_separator`.
"""
from __future__ import annotations

import gc
import os
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf

VOCALS_MODEL_DEFAULT = os.environ.get(
    "DESPIECE_ROFORMER_VOCALS_MODEL",
    "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
)
BASS_MODEL_DEFAULT = os.environ.get(
    "DESPIECE_ROFORMER_BASS_MODEL",
    "MDX23C-InstVoc_HQ.ckpt",
)

_Separator = None  # type: ignore[assignment]
_vocals_sep: Optional[object] = None
_bass_sep: Optional[object] = None


def _load_separator_class() -> object:
    global _Separator
    if _Separator is None:
        from audio_separator.separator import Separator  # type: ignore

        _Separator = Separator
    return _Separator


def _ensure(model_filename: str, slot: str) -> object:
    global _vocals_sep, _bass_sep
    existing = _vocals_sep if slot == "vocals" else _bass_sep
    if existing is not None:
        return existing
    cls = _load_separator_class()
    sep = cls(log_level=30)
    sep.load_model(model_filename)
    if slot == "vocals":
        _vocals_sep = sep
    else:
        _bass_sep = sep
    return sep


def _run_separator(sep, audio: np.ndarray, sr: int, want_keyword: str) -> np.ndarray:
    """Write audio to a temp WAV, run the separator, locate the desired stem,
    return mono float32 at the original sr."""
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "in.wav")
        sf.write(in_path, audio, sr)
        out_paths = sep.separate(in_path)
        chosen = None
        for p in out_paths:
            base = os.path.basename(p).lower()
            if want_keyword.lower() in base:
                chosen = p
                break
        if chosen is None:
            chosen = out_paths[0]
        data, out_sr = sf.read(chosen, dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        if out_sr != sr:
            import librosa

            data = librosa.resample(
                data.astype(np.float32), orig_sr=out_sr, target_sr=sr
            )
        return data.astype(np.float32)


def separate_roformer_vocals(audio: np.ndarray, sr: int) -> np.ndarray:
    """Run the vocals specialist. Returns mono float32 at the original sr."""
    sep = _ensure(VOCALS_MODEL_DEFAULT, "vocals")
    return _run_separator(sep, audio.astype(np.float32), sr, want_keyword="vocals")


def separate_roformer_bass(audio: np.ndarray, sr: int) -> np.ndarray:
    """Run the bass specialist. Returns mono float32 at the original sr."""
    sep = _ensure(BASS_MODEL_DEFAULT, "bass")
    return _run_separator(sep, audio.astype(np.float32), sr, want_keyword="bass")


def free_model() -> None:
    """Release both cached separators + force gc + empty torch cache."""
    global _vocals_sep, _bass_sep
    _vocals_sep = None
    _bass_sep = None
    gc.collect()
    try:
        import torch  # type: ignore

        torch.cuda.empty_cache()
    except Exception:
        pass
