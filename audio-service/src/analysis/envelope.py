import numpy as np
import librosa


def compute_envelope(
    audio: np.ndarray, sr: int, target_fps: float = 25.0
) -> list[list[float]]:
    """Return [[time_sec, energy_0_1], ...] at target_fps frames per second."""
    hop_length = max(1, int(sr / target_fps))
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]

    max_rms = float(rms.max())
    if max_rms > 0:
        rms = rms / max_rms

    times = librosa.frames_to_time(
        np.arange(len(rms)), sr=sr, hop_length=hop_length
    )
    return [[round(float(t), 4), round(float(e), 4)] for t, e in zip(times, rms)]
