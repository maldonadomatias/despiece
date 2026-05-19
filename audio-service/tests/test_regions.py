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


from src.analysis.regions import detect_regions


def _make_audio(sr, segments):
    """segments = [(start_sec, end_sec, amp)]; rest is silence."""
    total_sec = max(seg[1] for seg in segments)
    y = np.zeros(int(sr * total_sec), dtype=np.float32)
    for s, e, amp in segments:
        i0 = int(s * sr)
        i1 = int(e * sr)
        t = np.arange(i1 - i0) / sr
        y[i0:i1] = (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return y


def test_detect_regions_two_active_intervals():
    sr = 22050
    # 5s tone (1-6s), silence (6-8s), 8s tone (8-16s) -> total 16s
    y = _make_audio(sr, [(1.0, 6.0, 0.5), (8.0, 16.0, 0.5)])
    # 120 BPM 4/4: beats at 0.5s spacing, downbeats every 2s
    beats = [i * 0.5 for i in range(int(16 / 0.5))]
    regions = detect_regions(y, sr, beats, beats_per_bar=4)

    assert len(regions) == 2
    assert abs(regions[0]["start_sec"] - 0.0) < 0.5 or abs(regions[0]["start_sec"] - 2.0) < 0.5
    assert regions[0]["end_sec"] > regions[0]["start_sec"]
    assert regions[1]["start_sec"] > regions[0]["end_sec"]


def test_detect_regions_silence_returns_empty():
    sr = 22050
    y = np.zeros(sr * 4, dtype=np.float32)
    beats = [i * 0.5 for i in range(8)]
    assert detect_regions(y, sr, beats, beats_per_bar=4) == []


def test_detect_regions_short_active_filtered_out():
    sr = 22050
    # 0.2s active burst — should be dropped (< 0.5s min run)
    y = _make_audio(sr, [(1.0, 1.2, 0.5)])
    beats = [i * 0.5 for i in range(8)]
    # Pad to 4s
    pad = np.zeros(sr * 4 - len(y), dtype=np.float32)
    y = np.concatenate([y, pad])
    assert detect_regions(y, sr, beats, beats_per_bar=4) == []


def test_detect_regions_envelope_relative_to_region_start():
    sr = 22050
    y = _make_audio(sr, [(1.0, 6.0, 0.5)])
    beats = [i * 0.5 for i in range(12)]
    regions = detect_regions(y, sr, beats, beats_per_bar=4)
    assert len(regions) >= 1
    env = regions[0]["envelope"]
    assert env[0][0] == 0.0  # relative time starts at 0
    assert all(0.0 <= v <= 1.0 for _, v in env)
