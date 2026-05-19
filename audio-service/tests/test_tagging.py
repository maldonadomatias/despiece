import numpy as np
import pytest

from src.analysis import tagging


@pytest.fixture(autouse=True)
def _reset_model():
    tagging.free_model()
    yield
    tagging.free_model()


class _FakeAT:
    """Mimics panns_inference.AudioTagging."""

    def __init__(self, *args, **kwargs):
        self.labels = np.array(["A", "B", "C", "D"], dtype=object)

    def inference(self, audio):
        # clipwise probs shape (1, 4) for the 4 fake labels
        return np.array([[0.1, 0.7, 0.05, 0.15]], dtype=np.float32), None


def test_tag_clip_returns_top_k_sorted(monkeypatch):
    monkeypatch.setattr(tagging, "_AudioTagging", _FakeAT)

    audio = np.zeros(32000, dtype=np.float32)
    out = tagging.tag_clip(audio, sr=32000, top_k=3)
    assert len(out) == 3
    assert out[0] == ("B", pytest.approx(0.7))
    assert out[1] == ("D", pytest.approx(0.15))
    assert out[2] == ("A", pytest.approx(0.1))


def test_resamples_when_sr_mismatch(monkeypatch):
    captured = {}

    class _CapturingFake(_FakeAT):
        def inference(self, audio):
            captured["len"] = audio.shape[-1]
            return np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32), None

    monkeypatch.setattr(tagging, "_AudioTagging", _CapturingFake)
    audio = np.zeros(44100, dtype=np.float32)  # 1 sec at 44.1 kHz
    tagging.tag_clip(audio, sr=44100, top_k=1)
    assert 31950 <= captured["len"] <= 32050


def test_free_model_resets_state(monkeypatch):
    monkeypatch.setattr(tagging, "_AudioTagging", _FakeAT)

    tagging.tag_clip(np.zeros(32000, dtype=np.float32), sr=32000, top_k=1)
    assert tagging._model is not None
    tagging.free_model()
    assert tagging._model is None


def test_empty_audio_returns_empty_list(monkeypatch):
    monkeypatch.setattr(tagging, "_AudioTagging", _FakeAT)
    out = tagging.tag_clip(np.array([], dtype=np.float32), sr=32000, top_k=5)
    assert out == []
