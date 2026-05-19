"""Spectral subtraction helper. Removes a reference's magnitude spectrum from
a target's magnitude spectrum, preserving the target's phase."""

import numpy as np
import librosa


def _length_match(reference: np.ndarray, length: int) -> np.ndarray:
    if reference.shape[0] == length:
        return reference
    if reference.shape[0] > length:
        return reference[:length]
    out = np.zeros(length, dtype=reference.dtype)
    out[: reference.shape[0]] = reference
    return out


def spectral_subtract(
    target: np.ndarray,
    reference: np.ndarray,
    sr: int,
    alpha: float = 0.5,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """target' = ISTFT( max(|STFT(target)| - alpha * |STFT(reference)|, 0)
                        * exp(j * phase(STFT(target))) ).

    Phase from target is preserved. Magnitudes clamped >= 0.
    Reference is trimmed or zero-padded to the target's length.
    Output dtype is float32.
    """
    target = np.asarray(target, dtype=np.float32)
    reference = np.asarray(reference, dtype=np.float32)
    reference = _length_match(reference, target.shape[0])

    s_target = librosa.stft(target, n_fft=n_fft, hop_length=hop_length)
    s_ref = librosa.stft(reference, n_fft=n_fft, hop_length=hop_length)

    mag_t = np.abs(s_target)
    phase_t = np.angle(s_target)
    mag_r = np.abs(s_ref)

    mag_out = np.maximum(mag_t - alpha * mag_r, 0.0)
    s_out = mag_out * np.exp(1j * phase_t)

    out = librosa.istft(s_out, hop_length=hop_length, length=target.shape[0])
    return out.astype(np.float32)
