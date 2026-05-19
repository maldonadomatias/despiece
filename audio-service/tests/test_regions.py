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


from src.analysis.regions import _build_bar_grid, _snap


def test_build_bar_grid_120_bpm_4_4():
    # 120 BPM 4/4: beats at 0.5s spacing, downbeats every 4th beat (2.0s spacing)
    beats = [i * 0.5 for i in range(20)]
    grid = _build_bar_grid(beats, beats_per_bar=4, audio_duration=10.0)
    # downbeats: 0.0, 2.0, 4.0, 6.0, 8.0 — plus audio_duration tail (10.0)
    assert grid[0] == 0.0
    assert grid[1] == 2.0
    assert grid[2] == 4.0
    assert grid[3] == 6.0
    assert grid[-1] == 10.0


def test_build_bar_grid_few_beats():
    grid = _build_bar_grid([0.0], beats_per_bar=4, audio_duration=10.0)
    assert grid == [0.0, 10.0]


def test_snap_picks_nearest():
    grid = [0.0, 2.0, 4.0, 6.0, 8.0]
    assert _snap(0.3, grid) == 0.0
    assert _snap(4.7, grid) == 4.0
    assert _snap(5.5, grid) == 6.0
    assert _snap(8.0, grid) == 8.0


def test_snap_empty_grid_returns_input():
    assert _snap(2.5, []) == 2.5
