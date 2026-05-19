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


from src.analysis.regions import _merge_close


def test_merge_close_within_gap():
    runs = [(0.0, 1.0), (1.3, 2.0)]
    merged = _merge_close(runs, gap_sec=0.5)
    assert merged == [(0.0, 2.0)]


def test_merge_close_outside_gap_kept_separate():
    runs = [(0.0, 1.0), (2.0, 3.0)]
    merged = _merge_close(runs, gap_sec=0.5)
    assert merged == [(0.0, 1.0), (2.0, 3.0)]


def test_merge_close_empty():
    assert _merge_close([], gap_sec=0.5) == []


def test_merge_close_chain():
    runs = [(0.0, 1.0), (1.2, 2.0), (2.1, 3.0)]
    merged = _merge_close(runs, gap_sec=0.5)
    assert merged == [(0.0, 3.0)]
