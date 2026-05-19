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
