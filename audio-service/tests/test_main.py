import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from unittest.mock import patch

from src.analysis.demucs_runner import STEMS


@pytest.fixture
def client():
    from src.main import app
    return TestClient(app)


@pytest.fixture
def dummy_wav(tmp_path, sample_audio):
    y, sr = sample_audio
    path = str(tmp_path / "test.wav")
    sf.write(path, y, sr)
    return path


def test_health(client):
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "ok"


def test_analyze_returns_stems(client, dummy_wav, sample_audio):
    y, sr = sample_audio

    def fake_download(key, dest):
        import shutil
        shutil.copy(dummy_wav, dest)

    def fake_separate(path, out_dir):
        return {stem: (y * (0.5 ** i), sr) for i, stem in enumerate(STEMS)}

    with patch("src.main.download_to_path", side_effect=fake_download), \
         patch("src.main.separate_stems", side_effect=fake_separate), \
         patch("src.main.detect_bpm_and_beats", return_value=(120.0, [0.5, 1.0, 1.5])), \
         patch("src.main.encode_mp3"), \
         patch("src.main.upload_from_path"), \
         patch("src.main.detect_regions", return_value=[]):
        res = client.post("/analyze", json={"storage_key": "songs/test.wav"})

    assert res.status_code == 200
    body = res.json()
    assert set(body["stems"].keys()) == set(STEMS)
    assert body["duration_sec"] > 0
    assert body["bpm"] > 0
    assert body["key"] != "unknown"
    assert all("audio_key" in v for v in body["stems"].values())
    assert all("regions" in v for v in body["stems"].values())
    assert isinstance(body["beat_grid"], list)
    assert isinstance(body["bar_grid"], list)
    assert isinstance(body["sections"], list)
    assert len(body["sections"]) > 0
