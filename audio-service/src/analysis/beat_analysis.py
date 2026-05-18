import numpy as np
import librosa

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def detect_bpm_and_beats(audio: np.ndarray, sr: int) -> tuple[float, list[float]]:
    """Return (bpm, beat_times_in_seconds)."""
    tempo, beat_frames = librosa.beat.beat_track(y=audio, sr=sr)
    bpm = float(np.atleast_1d(tempo)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr).tolist()
    return round(bpm, 2), [round(t, 4) for t in beat_times]


def detect_key(audio: np.ndarray, sr: int) -> str:
    """Estimate musical key using Krumhansl-Schmuckler profiles on chroma_cqt."""
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    chroma_mean = chroma.mean(axis=1)  # shape (12,)

    best_score = -np.inf
    best_key = 'C major'

    for i in range(12):
        major_score = float(np.corrcoef(chroma_mean, np.roll(_MAJOR, i))[0, 1])
        minor_score = float(np.corrcoef(chroma_mean, np.roll(_MINOR, i))[0, 1])

        if major_score > best_score:
            best_score = major_score
            best_key = f'{NOTE_NAMES[i]} major'
        if minor_score > best_score:
            best_score = minor_score
            best_key = f'{NOTE_NAMES[i]} minor'

    return best_key
