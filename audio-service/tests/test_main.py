import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient
from unittest.mock import patch


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
        return {
            "drums": (y, sr),
            "bass": (y * 0.5, sr),
            "other": (y * 0.3, sr),
            "vocals": (y * 0.2, sr),
        }

    with patch("src.main.download_to_path", side_effect=fake_download), \
         patch("src.main.separate_stems", side_effect=fake_separate), \
         patch("src.main.detect_bpm_and_beats", return_value=(120.0, [0.5, 1.0, 1.5])):
        res = client.post("/analyze", json={"storage_key": "songs/test.wav"})

    assert res.status_code == 200
    body = res.json()
    assert set(body["stems"].keys()) == {"drums", "bass", "other", "vocals"}
    assert body["duration_sec"] > 0
    assert body["bpm"] > 0
    assert body["key"] != "unknown"
    assert all("envelope" in v for v in body["stems"].values())
    assert isinstance(body["beat_grid"], list)
    assert isinstance(body["sections"], list)
    assert len(body["sections"]) > 0
