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
