"""Sub-classify drum hits inside an isolated drums stem.

Pure DSP — onset detection + per-onset spectral features + rule cascade.
No model load. No global state.

Public API:
    detect_drum_hits(audio, sr) -> dict[str, list[dict]]
        Returns five class arrays: kick, snare, hihat, cymbal, unknown.
        Each hit dict has t_sec, velocity, confidence.
"""
from __future__ import annotations

import os
from typing import TypedDict

import numpy as np
import librosa


class _OnsetFeatures(TypedDict):
    t_sec: float
    low_energy: float
    mid_energy: float
    hi_energy: float
    centroid: float
    zcr: float
    decay_ms: float
    peak_rms: float


SHORT_WINDOW_PRE_SEC = 0.010
SHORT_WINDOW_POST_SEC = 0.050
DECAY_WINDOW_SEC = 0.400


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


KICK_LOW_THRESHOLD = _env_float("DESPIECE_DRUMS_KICK_LOW_THRESHOLD", 0.5)
SNARE_MID_THRESHOLD = _env_float("DESPIECE_DRUMS_SNARE_MID_THRESHOLD", 0.35)
HIHAT_HI_THRESHOLD = _env_float("DESPIECE_DRUMS_HIHAT_HI_THRESHOLD", 0.4)
CYMBAL_DECAY_MS = _env_float("DESPIECE_DRUMS_CYMBAL_DECAY_MS", 80.0)
CONFIDENCE_MIN = _env_float("DESPIECE_DRUMS_CONFIDENCE_MIN", 0.4)


def _empty_result() -> dict[str, list[dict]]:
    return {"kick": [], "snare": [], "hihat": [], "cymbal": [], "unknown": []}


def _extract_features(
    audio: np.ndarray,
    sr: int,
    onset_times_sec: np.ndarray,
) -> list[_OnsetFeatures]:
    out: list[_OnsetFeatures] = []
    n_total = audio.shape[0]
    onset_samps = (onset_times_sec * sr).astype(int)

    for i, t in enumerate(onset_times_sec):
        s_start = max(0, int((t - SHORT_WINDOW_PRE_SEC) * sr))
        s_end = min(n_total, int((t + SHORT_WINDOW_POST_SEC) * sr))
        ws = audio[s_start:s_end]
        if ws.size < 64:
            continue

        d_start = max(0, int(t * sr))
        d_end_target = min(n_total, int((t + DECAY_WINDOW_SEC) * sr))
        if i + 1 < len(onset_samps):
            d_end_target = min(d_end_target, onset_samps[i + 1])
        wd = audio[d_start:d_end_target]
        if wd.size < 32:
            continue

        n_fft = 512
        hop_length = 128

        S = np.abs(librosa.stft(ws, n_fft=n_fft, hop_length=hop_length)) + 1e-9
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        total = float(np.sum(S ** 2))
        low_mask = (freqs >= 30) & (freqs <= 150)
        mid_mask = (freqs > 150) & (freqs <= 1000)
        hi_mask = (freqs >= 6000) & (freqs <= 12000)
        low_energy = float(np.sum(S[low_mask, :] ** 2) / total)
        mid_energy = float(np.sum(S[mid_mask, :] ** 2) / total)
        hi_energy = float(np.sum(S[hi_mask, :] ** 2) / total)

        centroid_arr = librosa.feature.spectral_centroid(
            y=ws, sr=sr, n_fft=n_fft, hop_length=hop_length
        )
        zcr_arr = librosa.feature.zero_crossing_rate(y=ws, hop_length=hop_length)
        centroid = float(np.mean(centroid_arr)) if centroid_arr.size else 0.0
        zcr = float(np.mean(zcr_arr)) if zcr_arr.size else 0.0

        rms_short = librosa.feature.rms(y=ws, hop_length=hop_length)[0]
        peak_rms = float(np.max(rms_short)) if rms_short.size else 0.0

        rms_decay = librosa.feature.rms(y=wd, hop_length=hop_length)[0]
        if rms_decay.size == 0:
            decay_ms = DECAY_WINDOW_SEC * 1000.0
        else:
            peak_idx = int(np.argmax(rms_decay))
            peak_val = float(rms_decay[peak_idx])
            target = peak_val * 0.501187  # -6 dB ≈ 0.501
            decay_ms = float(rms_decay.size) * (hop_length / sr) * 1000.0
            for j in range(peak_idx + 1, rms_decay.size):
                if rms_decay[j] <= target:
                    decay_ms = float(j - peak_idx) * (hop_length / sr) * 1000.0
                    break

        out.append(_OnsetFeatures(
            t_sec=float(t),
            low_energy=low_energy,
            mid_energy=mid_energy,
            hi_energy=hi_energy,
            centroid=centroid,
            zcr=zcr,
            decay_ms=decay_ms,
            peak_rms=peak_rms,
        ))
    return out


def _classify(f: _OnsetFeatures) -> tuple[str, float]:
    """Rule-based 5-class cascade. Returns (label, confidence in [0, 1])."""
    low_e = f["low_energy"]
    mid_e = f["mid_energy"]
    hi_e = f["hi_energy"]
    centroid = f["centroid"]
    zcr = f["zcr"]
    decay_ms = f["decay_ms"]

    if low_e > KICK_LOW_THRESHOLD:
        return ("kick", float(min(1.0, low_e)))

    if mid_e > SNARE_MID_THRESHOLD and centroid < 4000:
        return ("snare", float(min(1.0, mid_e)))

    if hi_e > HIHAT_HI_THRESHOLD and zcr > 0.15 and decay_ms < CYMBAL_DECAY_MS:
        return ("hihat", float(min(1.0, hi_e * zcr / 0.15)))

    if hi_e > 0.3 and decay_ms >= CYMBAL_DECAY_MS:
        return ("cymbal", float(min(1.0, hi_e * (decay_ms / 200.0))))

    unknown_conf = float(1.0 - max(low_e, mid_e, hi_e))
    unknown_conf = float(min(1.0, max(0.0, unknown_conf)))
    return ("unknown", unknown_conf)


def _apply_confidence_floor(label: str, confidence: float) -> tuple[str, float]:
    if confidence < CONFIDENCE_MIN:
        return ("unknown", confidence)
    return (label, confidence)


def detect_drum_hits(audio: np.ndarray, sr: int) -> dict[str, list[dict]]:
    """Detect + classify drum hits in a mono drums stem.

    Returns five class arrays. Each hit is `{t_sec, velocity, confidence}`,
    all numeric fields rounded to 4 decimal places.
    """
    if audio.size == 0:
        return _empty_result()

    onset_times = librosa.onset.onset_detect(
        y=audio, sr=sr, units="time", backtrack=True, hop_length=512
    )
    onset_times = np.asarray(onset_times, dtype=np.float64)
    if onset_times.size == 0:
        return _empty_result()

    features = _extract_features(audio, sr, onset_times)
    if not features:
        return _empty_result()

    max_peak = max((f["peak_rms"] for f in features), default=0.0) or 1.0
    classified: list[tuple[str, float, float, float]] = []
    for f in features:
        label, conf = _classify(f)
        label, conf = _apply_confidence_floor(label, conf)
        velocity = min(1.0, f["peak_rms"] / max_peak)
        classified.append((label, f["t_sec"], velocity, conf))

    out = _empty_result()
    for label, t_sec, vel, conf in classified:
        out[label].append({
            "t_sec": round(t_sec, 4),
            "velocity": round(vel, 4),
            "confidence": round(conf, 4),
        })
    return out
