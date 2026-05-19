import numpy as np
import librosa
from scipy.ndimage import median_filter


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
    bars = [float(beat_times[i]) for i in range(0, len(beat_times), beats_per_bar)]
    if bars[0] > 0.0:
        bars.insert(0, 0.0)
    if bars[-1] < audio_duration:
        bars.append(float(audio_duration))
    return bars


def _snap(t, bar_grid):
    if not bar_grid:
        return float(t)
    return float(min(bar_grid, key=lambda b: abs(b - t)))


def detect_regions(audio, sr, beat_times, beats_per_bar=4):
    """
    Detect active intervals in a stem, snap edges to nearest bar, attach mini-envelopes.

    Returns: list of {"start_sec", "end_sec", "envelope": [[t_rel, e_0_1], ...]}.
    """
    if audio.size == 0:
        return []

    target_fps = 50.0
    hop_length = max(1, int(sr / target_fps))
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    window_frames = max(3, int(target_fps * 0.3))
    if window_frames % 2 == 0:
        window_frames += 1
    smoothed = median_filter(rms, size=window_frames)

    peak = float(smoothed.max()) if smoothed.size else 0.0
    if peak <= 0:
        return []
    noise_floor = float(np.percentile(smoothed, 10))
    threshold = max(noise_floor * 2.0, 0.05 * peak)
    active = smoothed > threshold

    runs = _find_runs(active, times)
    runs = _merge_close(runs, gap_sec=0.5)
    runs = [r for r in runs if (r[1] - r[0]) >= 0.5]

    audio_duration = float(times[-1]) if times.size else 0.0
    bar_grid = _build_bar_grid(beat_times, beats_per_bar, audio_duration)

    out = []
    for s, e in runs:
        s_snap = _snap(s, bar_grid)
        e_snap = _snap(e, bar_grid)
        if e_snap <= s_snap:
            continue
        env = _region_envelope(smoothed, times, s_snap, e_snap, peak)
        out.append({
            "start_sec": round(s_snap, 3),
            "end_sec": round(e_snap, 3),
            "envelope": env,
        })
    return out


def _region_envelope(smoothed, times, start_sec, end_sec, peak):
    mask = (times >= start_sec) & (times <= end_sec)
    sub_rms = smoothed[mask]
    sub_times = times[mask]
    if len(sub_rms) == 0 or peak <= 0:
        return []
    norm = np.clip(sub_rms / peak, 0.0, 1.0)
    return [
        [round(float(t - start_sec), 4), round(float(v), 4)]
        for t, v in zip(sub_times, norm)
    ]
