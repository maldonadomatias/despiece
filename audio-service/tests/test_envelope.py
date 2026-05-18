import numpy as np
import pytest
from src.analysis.envelope import compute_envelope


def test_returns_list_of_pairs(sample_audio):
    y, sr = sample_audio
    result = compute_envelope(y, sr, target_fps=25.0)
    assert isinstance(result, list)
    assert len(result) > 0
    assert all(len(p) == 2 for p in result)


def test_energy_normalized_0_to_1(sample_audio):
    y, sr = sample_audio
    result = compute_envelope(y, sr, target_fps=25.0)
    energies = [p[1] for p in result]
    assert max(energies) <= 1.0
    assert min(energies) >= 0.0


def test_silence_has_zero_energy():
    sr = 22050
    silence = np.zeros(sr * 3, dtype=np.float32)
    result = compute_envelope(silence, sr)
    assert all(p[1] == 0.0 for p in result)


def test_time_increases_monotonically(sample_audio):
    y, sr = sample_audio
    result = compute_envelope(y, sr)
    times = [p[0] for p in result]
    assert times == sorted(times)
