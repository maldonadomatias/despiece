import numpy as np


def _find_runs(active, times):
    runs = []
    in_run = False
    start_idx = 0
    for i, a in enumerate(active):
        if a and not in_run:
            in_run = True
            start_idx = i
        elif not a and in_run:
            in_run = False
            runs.append((float(times[start_idx]), float(times[i - 1])))
    if in_run:
        runs.append((float(times[start_idx]), float(times[-1])))
    return runs


def _merge_close(runs, gap_sec):
    if not runs:
        return []
    merged = [runs[0]]
    for s, e in runs[1:]:
        ps, pe = merged[-1]
        if s - pe <= gap_sec:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged


def _build_bar_grid(beat_times, beats_per_bar, audio_duration):
    if len(beat_times) < 2:
        return [0.0, float(audio_duration)]
    bars = [0.0]
    bars.extend([float(beat_times[i]) for i in range(0, len(beat_times), beats_per_bar)])
    if bars[-1] < audio_duration:
        bars.append(float(audio_duration))
    return bars


def _snap(t, bar_grid):
    if not bar_grid:
        return float(t)
    return float(min(bar_grid, key=lambda b: abs(b - t)))
