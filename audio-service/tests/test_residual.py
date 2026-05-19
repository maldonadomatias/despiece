import numpy as np
import librosa

from src.analysis.residual import spectral_subtract


def _band_energy(audio: np.ndarray, sr: int, fmin: float, fmax: float) -> float:
    """Energy inside [fmin, fmax] using a wide-window STFT."""
    n_fft = 2048
    spec = np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=512))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return float(np.sum(spec[mask, :] ** 2))


def _make_bass(sr: int, dur: float) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return 0.5 * np.sin(2 * np.pi * 80 * t).astype(np.float32)


def _make_guitar(sr: int, dur: float) -> np.ndarray:
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)
    return 0.4 * np.sin(2 * np.pi * 1200 * t).astype(np.float32)


def test_subtract_reduces_low_band():
    sr = 22050
    bass = _make_bass(sr, 2.0)
    guitar_clean = _make_guitar(sr, 2.0)
    target = (guitar_clean + bass).astype(np.float32)

    out = spectral_subtract(target, bass, sr=sr, alpha=0.9)

    low_before = _band_energy(target, sr, 50, 150)
    low_after = _band_energy(out, sr, 50, 150)
    assert low_after < 0.4 * low_before  # >60% reduction


def test_subtract_preserves_high_band():
    sr = 22050
    bass = _make_bass(sr, 2.0)
    guitar_clean = _make_guitar(sr, 2.0)
    target = (guitar_clean + bass).astype(np.float32)

    out = spectral_subtract(target, bass, sr=sr, alpha=0.9)

    high_before = _band_energy(target, sr, 1100, 1300)
    high_after = _band_energy(out, sr, 1100, 1300)
    assert 0.9 * high_before <= high_after <= 1.1 * high_before


def test_length_match_pads_or_trims_reference():
    sr = 22050
    target = _make_guitar(sr, 1.0)
    shorter_bass = _make_bass(sr, 0.5)
    out = spectral_subtract(target, shorter_bass, sr=sr, alpha=0.5)
    assert out.shape == target.shape


def test_output_dtype_float32():
    sr = 22050
    target = _make_guitar(sr, 0.5)
    bass = _make_bass(sr, 0.5)
    out = spectral_subtract(target, bass, sr=sr, alpha=0.5)
    assert out.dtype == np.float32
