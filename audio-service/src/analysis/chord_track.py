"""Chord track detection on a harmonic mix.

Algorithm:
    1. CQT chroma per frame.
    2. Cosine similarity against 24 chord templates (12 major + 12 minor).
    3. Augment with a no-chord score (max template score < threshold).
    4. Viterbi decode with self-transition bias.
    5. Per-bar mode aggregation.
    6. Merge consecutive identical bars.

Public API:
    detect_chords(audio, sr, bar_grid) -> list[dict]
"""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import librosa

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
HOP_LENGTH = 2048


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


NO_CHORD_THRESHOLD = _env_float("DESPIECE_CHORD_NO_CHORD_THRESHOLD", 0.3)
SELF_TRANSITION = _env_float("DESPIECE_CHORD_SELF_TRANSITION", 0.9)


def _build_templates() -> np.ndarray:
    """Return a (24, 12) array of unit-norm chord templates."""
    major = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
    minor = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
    rows = [np.roll(major, i) for i in range(12)] + [np.roll(minor, i) for i in range(12)]
    t = np.stack(rows).astype(np.float32)
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return t


def _idx_to_label(i: int) -> str:
    if i == 24:
        return "N"
    if i < 12:
        return NOTE_NAMES[i]
    return NOTE_NAMES[i - 12] + "m"


def _viterbi_path(audio: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Run the full per-frame pipeline. Returns (path_indices, frame_times)."""
    chroma = librosa.feature.chroma_cqt(
        y=audio, sr=sr, hop_length=HOP_LENGTH,
        bins_per_octave=36, n_octaves=6, fmin=librosa.note_to_hz("C2"),
    )
    n_frames = chroma.shape[1]
    chroma_norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)

    templates = _build_templates()
    scores = templates @ chroma_norm  # (24, n_frames)
    # no-chord score: the threshold itself, broadcast across all frames.
    # When threshold <= typical cosine scores (~0.8-1.0), chord templates win.
    # When threshold > 1.0 (impossible for cosine similarity to reach), the
    # no-chord score exceeds all template scores, forcing "N" every frame.
    no_chord = np.full((1, n_frames), NO_CHORD_THRESHOLD, dtype=np.float32)
    scores = np.vstack([scores, no_chord])  # (25, n_frames)
    # Normalize columns to [0, 1] so librosa.sequence.viterbi accepts them.
    col_max = scores.max(axis=0, keepdims=True)
    col_max = np.where(col_max == 0, 1.0, col_max)
    scores = scores / col_max

    n_states = 25
    trans = np.full((n_states, n_states), (1.0 - SELF_TRANSITION) / (n_states - 1), dtype=np.float64)
    np.fill_diagonal(trans, SELF_TRANSITION)

    path = librosa.sequence.viterbi(scores.astype(np.float64), trans)
    frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=HOP_LENGTH)
    return path, frame_times


def detect_chords(
    audio: np.ndarray,
    sr: int,
    bar_grid: Sequence[float],
) -> list[dict]:
    """Detect per-bar chords on a harmonic-mix mono signal.

    Returns a list of `{start_sec, end_sec, label}` dicts where consecutive
    bars with the same label have been merged into a single entry.
    """
    if audio.size == 0:
        return []
    if len(bar_grid) < 2:
        return []

    path, frame_times = _viterbi_path(audio, sr)

    out: list[dict] = []
    for bar_idx in range(len(bar_grid) - 1):
        bar_start = float(bar_grid[bar_idx])
        bar_end = float(bar_grid[bar_idx + 1])
        in_bar = (frame_times >= bar_start) & (frame_times < bar_end)
        if not in_bar.any():
            label_idx = 24  # no-chord
        else:
            label_idx = int(np.bincount(path[in_bar], minlength=25).argmax())
        label = _idx_to_label(label_idx)

        start_rounded = round(bar_start, 3)
        end_rounded = round(bar_end, 3)
        if out and out[-1]["label"] == label:
            out[-1]["end_sec"] = end_rounded
        else:
            out.append({
                "start_sec": start_rounded,
                "end_sec": end_rounded,
                "label": label,
            })
    return out
