"""Wrapper around PANNs CNN14 audio tagger.

Lazy-loads the model on first call. Caches under module global `_model`.
Caller should invoke `free_model()` when finished to release ~500 MiB.

`_AudioTagging` is a module attribute initialized to `None` and assigned
lazily on first use. Tests substitute it with a fake via monkeypatch
without ever importing `panns_inference`.
"""
from __future__ import annotations

import gc
from typing import Optional

import numpy as np
import librosa

PANNS_SR = 32000  # CNN14 was trained at 32 kHz.

_AudioTagging = None  # type: ignore[assignment]
_model: Optional[object] = None
_labels: Optional[np.ndarray] = None


def _load_audio_tagging_class() -> object:
    global _AudioTagging
    if _AudioTagging is None:
        from panns_inference import AudioTagging  # type: ignore

        _AudioTagging = AudioTagging
    return _AudioTagging


def _ensure_model() -> object:
    global _model, _labels
    if _model is None:
        cls = _load_audio_tagging_class()
        _model = cls(checkpoint_path=None, device="cpu")
        _labels = np.asarray(getattr(_model, "labels"), dtype=object)
    return _model


def tag_clip(audio: np.ndarray, sr: int, top_k: int = 5) -> list[tuple[str, float]]:
    """Tag a mono audio clip.

    Resamples to 32 kHz internally if `sr != PANNS_SR`. Returns the top-k
    (label, probability) pairs sorted by probability descending.
    Empty input -> empty list.
    """
    if audio.size == 0:
        return []

    if sr != PANNS_SR:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=PANNS_SR)

    model = _ensure_model()
    audio_batch = np.asarray(audio, dtype=np.float32)[None, :]
    clipwise, _ = model.inference(audio_batch)

    probs = np.asarray(clipwise[0], dtype=np.float32)
    labels = _labels
    assert labels is not None
    k = min(top_k, probs.shape[0], labels.shape[0])
    top_idx = np.argsort(-probs)[:k]
    return [(str(labels[i]), float(probs[i])) for i in top_idx]


def free_model() -> None:
    """Release the cached model + force garbage collection."""
    global _model, _labels
    _model = None
    _labels = None
    gc.collect()
