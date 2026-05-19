import os
import numpy as np
import pytest
import soundfile as sf

from src.analysis import ensemble


@pytest.fixture(autouse=True)
def _reset_models():
    ensemble.free_model()
    yield
    ensemble.free_model()


class _FakeSeparator:
    """Mock that mimics audio-separator's `Separator.separate(path)` API."""

    def __init__(self, *args, **kwargs):
        self.loaded = None

    def load_model(self, model_filename: str):
        self.loaded = model_filename

    def separate(self, audio_path: str):
        sr = 44100
        out = np.zeros(sr * 1, dtype=np.float32)
        base, _ = os.path.splitext(audio_path)
        if "vocals" in (self.loaded or "").lower() or "roformer" in (self.loaded or "").lower():
            keyword = "Vocals"
        else:
            keyword = "Bass"
        out_path = f"{base}_({keyword})_dummy.wav"
        sf.write(out_path, out, sr)
        return [out_path]


def test_separate_roformer_vocals_returns_mono_float32(monkeypatch):
    monkeypatch.setattr(ensemble, "_Separator", _FakeSeparator)
    sr = 44100
    audio = np.random.RandomState(0).randn(sr * 2).astype(np.float32) * 0.1

    out = ensemble.separate_roformer_vocals(audio, sr=sr)
    assert out.dtype == np.float32
    assert out.ndim == 1
    assert out.shape[0] > 0


def test_separate_roformer_bass_returns_mono_float32(monkeypatch):
    monkeypatch.setattr(ensemble, "_Separator", _FakeSeparator)
    sr = 44100
    audio = np.zeros(sr, dtype=np.float32)

    out = ensemble.separate_roformer_bass(audio, sr=sr)
    assert out.dtype == np.float32
    assert out.ndim == 1


def test_free_model_resets_state(monkeypatch):
    monkeypatch.setattr(ensemble, "_Separator", _FakeSeparator)
    ensemble.separate_roformer_vocals(np.zeros(44100, dtype=np.float32), sr=44100)
    assert ensemble._vocals_sep is not None
    ensemble.free_model()
    assert ensemble._vocals_sep is None
    assert ensemble._bass_sep is None
