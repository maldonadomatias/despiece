import numpy as np
from src.analysis.regions import _find_runs


def test_find_runs_single_active_span():
    active = np.array([False, True, True, True, False, False])
    times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    runs = _find_runs(active, times)
    assert runs == [(0.1, 0.3)]


def test_find_runs_two_spans():
    active = np.array([True, True, False, False, True, True])
    times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    runs = _find_runs(active, times)
    assert runs == [(0.0, 0.1), (0.4, 0.5)]


def test_find_runs_active_to_end():
    active = np.array([False, False, True, True])
    times = np.array([0.0, 0.1, 0.2, 0.3])
    runs = _find_runs(active, times)
    assert runs == [(0.2, 0.3)]


def test_find_runs_all_inactive():
    active = np.array([False, False, False])
    times = np.array([0.0, 0.1, 0.2])
    assert _find_runs(active, times) == []
