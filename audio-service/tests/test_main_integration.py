import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.main as main_mod


@pytest.fixture
def client(monkeypatch):
    sr = 22050
    duration_sec = 2.0
    n = int(sr * duration_sec)
    mono = np.random.RandomState(0).randn(n).astype(np.float32) * 0.05

    stems_data = {
        "drums":  (mono.copy(), sr),
        "bass":   (mono.copy(), sr),
        "other":  (mono.copy(), sr),
        "vocals": (mono.copy(), sr),
        "guitar": (mono.copy(), sr),
        "piano":  (mono.copy(), sr),
    }

    def fake_download(storage_key, dest_path):
        import soundfile as sf
        sf.write(dest_path, mono, sr, format="WAV")

    def fake_upload(local_path, key):
        pass

    monkeypatch.setattr(main_mod, "download_to_path", fake_download)
    monkeypatch.setattr(main_mod, "upload_from_path", fake_upload)
    monkeypatch.setattr(main_mod, "separate_stems", lambda input_path, tmpdir: stems_data)

    monkeypatch.setattr(main_mod, "separate_roformer_vocals", lambda a, sr: a.copy())
    monkeypatch.setattr(main_mod, "separate_roformer_bass", lambda a, sr: a.copy())
    monkeypatch.setattr(main_mod, "ensemble_free_model", lambda: None)
    monkeypatch.setattr(main_mod, "tag_clip", lambda a, sr, top_k=5: [("Pad", 0.8)])
    monkeypatch.setattr(main_mod, "tagging_free_model", lambda: None)
    monkeypatch.setattr(
        main_mod,
        "encode_mp3",
        lambda audio, sr, path, bitrate_kbps=128: open(path, "wb").close(),
    )

    monkeypatch.setattr(
        main_mod,
        "detect_regions",
        lambda audio, sr, beat_grid, beats_per_bar=4: [
            {"start_sec": 0.0, "end_sec": 1.0, "envelope": [[0.0, 0.5]]}
        ],
    )

    # Map "Pad" to our taxonomy via the existing sub_label module.
    # No monkeypatch needed — collapse_to_sub_label already maps "Pad" → "pad".

    return TestClient(main_mod.app)


def test_analyze_returns_version_2_and_sub_label(client):
    resp = client.post(
        "/analyze",
        json={"song_id": "abc-123", "storage_key": "songs/abc-123.wav"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["analysis_version"] == 2
    assert set(body["stems"].keys()) == {"drums", "bass", "other", "vocals", "guitar", "piano"}

    other_regions = body["stems"]["other"]["regions"]
    assert len(other_regions) == 1
    assert other_regions[0]["sub_label"] == "pad"
    assert 0.79 <= other_regions[0]["sub_label_confidence"] <= 0.81

    assert "sub_label" not in body["stems"]["vocals"]["regions"][0]
    assert "sub_label_confidence" not in body["stems"]["bass"]["regions"][0]
