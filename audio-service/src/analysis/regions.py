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
