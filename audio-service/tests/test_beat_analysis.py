import numpy as np
import pytest
from src.analysis.beat_analysis import detect_bpm_and_beats, detect_key


def test_bpm_returns_positive_float():
    # Use a rhythmic signal (kick-drum-like impulses) so librosa detects beats.
    sr = 22050
    duration = 5.0
    y = np.zeros(int(sr * duration), dtype=np.float32)
    # Place impulses at 120 BPM (every 0.5 s)
    for beat_sample in range(0, len(y), sr // 2):
        y[beat_sample] = 1.0
    bpm, beats = detect_bpm_and_beats(y, sr)
    assert isinstance(bpm, float)
    assert bpm > 0
    assert isinstance(beats, list)


def test_beats_are_sorted(sample_audio):
    y, sr = sample_audio
    _, beats = detect_bpm_and_beats(y, sr)
    assert beats == sorted(beats)


def test_key_returns_valid_string(sample_audio):
    y, sr = sample_audio
    key = detect_key(y, sr)
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    assert any(key.startswith(n) for n in notes)
    assert key.endswith('major') or key.endswith('minor')
