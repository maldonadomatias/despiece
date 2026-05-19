from __future__ import annotations

import numpy as np
import pytest

from src.analysis.chord_track import detect_chords, _idx_to_label, _build_templates


def test_idx_to_label_majors():
    assert _idx_to_label(0) == "C"
    assert _idx_to_label(7) == "G"
    assert _idx_to_label(11) == "B"


def test_idx_to_label_minors():
    assert _idx_to_label(12) == "Cm"
    assert _idx_to_label(19) == "Gm"
    assert _idx_to_label(23) == "Bm"


def test_idx_to_label_no_chord():
    assert _idx_to_label(24) == "N"


def test_build_templates_shape_and_norm():
    t = _build_templates()
    assert t.shape == (24, 12)
    norms = np.linalg.norm(t, axis=1)
    np.testing.assert_allclose(norms, np.ones(24), atol=1e-6)


def test_detect_chords_empty_audio_returns_empty():
    out = detect_chords(np.zeros(0, dtype=np.float32), sr=22050, bar_grid=[0.0, 2.0, 4.0])
    assert out == []


def test_detect_chords_short_bar_grid_returns_empty():
    sr = 22050
    audio = np.zeros(sr, dtype=np.float32)
    assert detect_chords(audio, sr=sr, bar_grid=[]) == []
    assert detect_chords(audio, sr=sr, bar_grid=[0.0]) == []


def test_detect_chords_returns_list_of_dicts():
    """Non-empty audio + valid bar_grid -> list of dicts shaped correctly.
    Task 2 stub returns []; this test will be tightened in Task 3."""
    sr = 22050
    audio = np.zeros(sr * 4, dtype=np.float32)
    out = detect_chords(audio, sr=sr, bar_grid=[0.0, 2.0, 4.0])
    assert isinstance(out, list)


def _make_triad(sr: int, dur_sec: float, freqs: list[float]) -> np.ndarray:
    """Sum of equal-amplitude sines at the given frequencies."""
    n = int(sr * dur_sec)
    t = np.arange(n) / sr
    sig = np.zeros(n, dtype=np.float32)
    for f in freqs:
        sig += np.sin(2 * np.pi * f * t).astype(np.float32)
    sig /= max(1.0, len(freqs))
    return sig.astype(np.float32)


def test_synthetic_c_major_is_classified_as_c():
    sr = 22050
    audio = _make_triad(sr, dur_sec=4.0, freqs=[261.63, 329.63, 392.00])
    out = detect_chords(audio, sr=sr, bar_grid=[0.0, 2.0, 4.0])
    labels = [c["label"] for c in out]
    assert "C" in labels
    assert all(c["label"] == "C" for c in out)


def test_synthetic_a_minor_is_classified_as_am():
    sr = 22050
    # Add A1 + A2 (low octaves) to reinforce root
    audio = _make_triad(sr, dur_sec=4.0, freqs=[55.00, 110.00, 220.00, 261.63, 329.63])
    out = detect_chords(audio, sr=sr, bar_grid=[0.0, 2.0, 4.0])
    labels = [c["label"] for c in out]
    assert "Am" in labels
    assert all(c["label"] == "Am" for c in out)


def test_chord_change_at_bar_boundary_produces_two_entries():
    sr = 22050
    c_maj = _make_triad(sr, dur_sec=4.0, freqs=[261.63, 329.63, 392.00])
    g_maj = _make_triad(sr, dur_sec=4.0, freqs=[392.00, 493.88, 587.33])
    audio = np.concatenate([c_maj, g_maj]).astype(np.float32)
    out = detect_chords(audio, sr=sr, bar_grid=[0.0, 2.0, 4.0, 6.0, 8.0])
    assert len(out) == 2
    assert out[0]["label"] == "C"
    assert out[1]["label"] == "G"
    assert out[0]["start_sec"] == 0.0
    assert out[1]["end_sec"] == 8.0


def test_consecutive_identical_bars_merge_into_one_entry():
    sr = 22050
    audio = _make_triad(sr, dur_sec=4.0, freqs=[261.63, 329.63, 392.00])
    out = detect_chords(audio, sr=sr, bar_grid=[0.0, 1.0, 2.0, 3.0, 4.0])
    assert len(out) == 1
    assert out[0]["label"] == "C"
    assert out[0]["start_sec"] == 0.0
    assert out[0]["end_sec"] == 4.0


def test_no_chord_threshold_raised_forces_unknown(monkeypatch):
    monkeypatch.setenv("DESPIECE_CHORD_NO_CHORD_THRESHOLD", "1.5")
    import importlib, src.analysis.chord_track as ct
    importlib.reload(ct)
    sr = 22050
    n = sr * 4
    t = np.arange(n) / sr
    audio_sig = (
        np.sin(2 * np.pi * 261.63 * t).astype(np.float32) +
        np.sin(2 * np.pi * 329.63 * t).astype(np.float32) +
        np.sin(2 * np.pi * 392.00 * t).astype(np.float32)
    ) / 3.0
    out = ct.detect_chords(audio_sig, sr=sr, bar_grid=[0.0, 2.0, 4.0])
    assert all(c["label"] == "N" for c in out)
    monkeypatch.setenv("DESPIECE_CHORD_NO_CHORD_THRESHOLD", "0.3")
    importlib.reload(ct)
