import os
import numpy as np
from src.analysis.mp3_encoder import encode_mp3


def test_encode_mp3_writes_valid_mp3(tmp_path):
    sr = 22050
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    y = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    out_path = tmp_path / "test.mp3"
    encode_mp3(y, sr, str(out_path), bitrate_kbps=128)
    assert out_path.exists()
    # MP3 files start with ID3 tag or 0xFF 0xFB/0xFA/0xF3/0xF2 sync word
    header = out_path.read_bytes()[:3]
    assert header.startswith(b"ID3") or header[0] == 0xFF
    assert out_path.stat().st_size > 1000  # sanity: not empty
