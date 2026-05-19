from __future__ import annotations

import numpy as np
import pytest

from src.analysis.drum_hits import detect_drum_hits


def _make_kick(sr: int, t0_sec: float, dur_sec: float = 0.2) -> np.ndarray:
    """60 Hz sine with exponential decay starting at t0_sec, length dur_sec."""
    n = int(sr * dur_sec)
    t = np.arange(n) / sr
    env = np.exp(-t / 0.05).astype(np.float32)
    sig = np.sin(2 * np.pi * 60 * t).astype(np.float32) * env
    return sig


def _make_silence(sr: int, dur_sec: float) -> np.ndarray:
    return np.zeros(int(sr * dur_sec), dtype=np.float32)


def _two_kicks(sr: int) -> np.ndarray:
    """Two kicks 500ms apart on a 1-sec leading silence."""
    return np.concatenate(
        [_make_silence(sr, 1.0), _make_kick(sr, 0.0), _make_silence(sr, 0.3),
         _make_kick(sr, 0.0), _make_silence(sr, 0.5)]
    ).astype(np.float32)


def test_empty_audio_returns_five_empty_arrays():
    out = detect_drum_hits(np.zeros(0, dtype=np.float32), sr=22050)
    assert set(out.keys()) == {"kick", "snare", "hihat", "cymbal", "unknown"}
    for v in out.values():
        assert v == []


def test_no_onsets_returns_five_empty_arrays():
    sr = 22050
    out = detect_drum_hits(np.zeros(sr * 2, dtype=np.float32), sr=sr)
    for v in out.values():
        assert v == []


def test_two_kicks_produce_two_total_hits():
    sr = 22050
    audio = _two_kicks(sr)
    out = detect_drum_hits(audio, sr=sr)
    total = sum(len(v) for v in out.values())
    assert total == 2


def test_each_hit_has_three_fields():
    sr = 22050
    out = detect_drum_hits(_two_kicks(sr), sr=sr)
    for cls, hits in out.items():
        for h in hits:
            assert set(h.keys()) == {"t_sec", "velocity", "confidence"}
            assert isinstance(h["t_sec"], float)
            assert isinstance(h["velocity"], float)
            assert isinstance(h["confidence"], float)
            assert 0.0 <= h["velocity"] <= 1.0
            assert 0.0 <= h["confidence"] <= 1.0


def test_velocity_normalized_to_one():
    sr = 22050
    soft = _make_kick(sr, 0.0) * 0.3
    loud = _make_kick(sr, 0.0) * 1.0
    audio = np.concatenate(
        [_make_silence(sr, 0.5), soft, _make_silence(sr, 0.5), loud, _make_silence(sr, 0.5)]
    ).astype(np.float32)
    out = detect_drum_hits(audio, sr=sr)
    all_velocities = [h["velocity"] for hits in out.values() for h in hits]
    assert len(all_velocities) == 2
    assert max(all_velocities) == pytest.approx(1.0)
    assert min(all_velocities) < 0.5
