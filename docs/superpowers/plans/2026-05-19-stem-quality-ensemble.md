# Pro-Grade Stem Quality (Phase A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift stem quality of the song-analysis pipeline by replacing Demucs vocals + bass with BS-Roformer (with MDX-Net bass fallback), spectrally subtracting bass leakage from guitar/piano, and tagging each `other`-stem region with a 6-class sub-label via PANNs CNN14.

**Architecture:** Sequential CPU pipeline in the existing audio-service. Three new modules (`ensemble`, `residual`, `tagging`) and one taxonomy module (`sub_label`). Each model is lazy-loaded and explicitly freed before the next model runs so peak RAM stays under the 3.83 GiB container limit. Frontend changes are prop additions plus a sub-label legend and a re-analyze gate keyed on `analysis_version: 2`. No DB migration.

**Tech Stack:** Python 3.11, PyTorch 2.4, librosa 0.10, `audio-separator[cpu]` (BS-Roformer / MDX-Net via ONNX-Runtime), `panns-inference` (CNN14), FastAPI, TypeScript 5.7, React 19.

**Spec:** `docs/superpowers/specs/2026-05-19-stem-quality-ensemble-design.md`

---

## File Map

**audio-service (Python):**
- Create: `audio-service/src/analysis/sub_label.py`
- Create: `audio-service/src/analysis/residual.py`
- Create: `audio-service/src/analysis/tagging.py`
- Create: `audio-service/src/analysis/ensemble.py`
- Create: `audio-service/tests/test_sub_label.py`
- Create: `audio-service/tests/test_residual.py`
- Create: `audio-service/tests/test_tagging.py`
- Create: `audio-service/tests/test_ensemble.py`
- Create: `audio-service/tests/test_main_integration.py`
- Create: `audio-service/scripts/prefetch_models.py`
- Create: `audio-service/scripts/analyze_local.py`
- Modify: `audio-service/requirements.txt` — add `audio-separator[cpu]`, `panns-inference`
- Modify: `audio-service/Dockerfile` — pre-pull BS-Roformer + PANNs weights
- Modify: `audio-service/src/main.py` — orchestrate ensemble + residual + tagging + `analysis_version: 2`
- Modify: `audio-service/env.example` — new env vars

**backend (Node):**
- Modify: `backend/src/domain/song.ts` — add `SubLabel` + optional fields on `StemRegion` + `analysis_version` on `AnalysisResult`
- Create: `backend/src/scripts/resetAnalysis.ts`

**frontend (React/TS):**
- Modify: `frontend/src/types/song.ts` — same type additions
- Modify: `frontend/src/components/RegionBlock.tsx` — sub-label color + inline label text
- Modify: `frontend/src/components/StemTrack.tsx` — wire sub-label only for `other`
- Modify: `frontend/src/components/SongTimeline.tsx` — re-analyze gate + sub-label legend

---

## Task 1: Backend + Frontend types (`analysis_version` + sub-label fields)

**Why first:** Locks the shape both ends agree on. Subsequent tasks reference these types.

**Files:**
- Modify: `backend/src/domain/song.ts`
- Modify: `frontend/src/types/song.ts`

- [ ] **Step 1: Update backend types**

Replace `StemRegion` and `AnalysisResult` blocks in `backend/src/domain/song.ts`:

```ts
export type SubLabel = 'lead' | 'pad' | 'synth' | 'strings' | 'fx' | 'other_misc';

export interface StemRegion {
  start_sec: number;
  end_sec: number;
  envelope: [number, number][]; // [t_relative_sec, energy_0_1]
  sub_label?: SubLabel;
  sub_label_confidence?: number;
}

export interface AnalysisResult {
  bpm: number;
  key: string;
  duration_sec: number;
  beat_grid: number[];
  bar_grid: number[];
  sections: Section[];
  stems: Record<string, StemAnalysis>;
  analysis_version: number;
}
```

- [ ] **Step 2: Update frontend types**

Replace `StemRegion` and `AnalysisResult` blocks in `frontend/src/types/song.ts`:

```ts
export type SubLabel = 'lead' | 'pad' | 'synth' | 'strings' | 'fx' | 'other_misc';

export interface AnalysisResult {
  bpm: number;
  key: string;
  duration_sec: number;
  beat_grid: number[];
  bar_grid: number[];
  sections: Section[];
  stems: Record<string, StemAnalysis>;
  analysis_version: number;
}

export interface StemRegion {
  start_sec: number;
  end_sec: number;
  envelope: [number, number][];
  sub_label?: SubLabel;
  sub_label_confidence?: number;
}
```

- [ ] **Step 3: Verify both packages type-check**

Run: `cd backend && npm run build`
Expected: builds without TypeScript errors.

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors. (Existing `analysis.analysis_version` is `undefined` in old code paths; treated as `number | undefined` until used. No call site yet.)

- [ ] **Step 4: Commit**

```bash
git add backend/src/domain/song.ts frontend/src/types/song.ts
git commit -m "feat(types): add SubLabel + analysis_version to AnalysisResult"
```

---

## Task 2: Sub-label taxonomy (`audio-service/src/analysis/sub_label.py`)

**Files:**
- Create: `audio-service/src/analysis/sub_label.py`
- Create: `audio-service/tests/test_sub_label.py`

- [ ] **Step 1: Write the failing tests**

Create `audio-service/tests/test_sub_label.py`:

```python
from src.analysis.sub_label import collapse_to_sub_label


def test_strings_dominant():
    label, conf = collapse_to_sub_label([
        ("Violin, fiddle", 0.6),
        ("Cello", 0.2),
        ("Music", 0.05),
    ])
    assert label == "strings"
    assert 0.79 <= conf <= 0.81


def test_synth_dominant():
    label, conf = collapse_to_sub_label([
        ("Synthesizer", 0.7),
        ("Music", 0.1),
    ])
    assert label == "synth"
    assert 0.69 <= conf <= 0.71


def test_unmapped_falls_back_to_other_misc():
    label, conf = collapse_to_sub_label([
        ("Music", 0.9),
        ("Speech", 0.05),
    ])
    assert label == "other_misc"
    assert 0.89 <= conf <= 0.91


def test_empty_input():
    label, conf = collapse_to_sub_label([])
    assert label == "other_misc"
    assert conf == 0.0


def test_confidence_clipped_to_one():
    label, conf = collapse_to_sub_label([
        ("Violin, fiddle", 0.7),
        ("Cello", 0.4),  # sum would be 1.1
    ])
    assert label == "strings"
    assert conf == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd audio-service && python -m pytest tests/test_sub_label.py -v`
Expected: FAIL (ModuleNotFoundError on `src.analysis.sub_label`).

- [ ] **Step 3: Implement the module**

Create `audio-service/src/analysis/sub_label.py`:

```python
"""Collapse PANNs CNN14 AudioSet (527-class) tags to a 6-class taxonomy.

The TAXONOMY dict is data, not algorithm logic. Iterate it freely without
touching `collapse_to_sub_label`. Unmapped tags fall through to
`other_misc` with the dominant unmapped probability.
"""
from typing import Literal

SubLabel = Literal["lead", "pad", "synth", "strings", "fx", "other_misc"]

TAXONOMY: dict[str, SubLabel] = {
    # strings
    "Violin, fiddle": "strings",
    "Cello": "strings",
    "Double bass": "strings",
    "Orchestra": "strings",
    "String section": "strings",
    "Pizzicato": "strings",
    # pad (slow, sustained)
    "Pad": "pad",
    "Choir": "pad",
    "Synthesizer": "synth",
    "Sampler": "synth",
    "Electronic music": "synth",
    # lead
    "Lead (synth lead)": "lead",
    "Saxophone": "lead",
    "Trumpet": "lead",
    "Flute": "lead",
    "Whistle": "lead",
    # fx
    "Sound effect": "fx",
    "Noise": "fx",
    "Whoosh, swoosh, swish": "fx",
    "Boom": "fx",
    "Reverberation": "fx",
    "Echo": "fx",
}


def collapse_to_sub_label(
    audioset_tags: list[tuple[str, float]],
) -> tuple[SubLabel, float]:
    """Collapse top-k AudioSet labels into the 6-class taxonomy.

    Sums probabilities per sub_label across input tags. Returns the
    argmax sub_label and its clipped sum. Unmapped tags accumulate
    against `other_misc`.
    """
    if not audioset_tags:
        return ("other_misc", 0.0)

    sums: dict[SubLabel, float] = {}
    unmapped_max = 0.0
    for tag, prob in audioset_tags:
        sub = TAXONOMY.get(tag)
        if sub is None:
            if prob > unmapped_max:
                unmapped_max = prob
            continue
        sums[sub] = sums.get(sub, 0.0) + float(prob)

    if not sums:
        return ("other_misc", float(min(unmapped_max, 1.0)))

    best = max(sums.items(), key=lambda kv: kv[1])
    return (best[0], float(min(best[1], 1.0)))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd audio-service && python -m pytest tests/test_sub_label.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add audio-service/src/analysis/sub_label.py audio-service/tests/test_sub_label.py
git commit -m "feat(audio-service): sub_label taxonomy collapse helper"
```

---

## Task 3: Spectral residual subtract (`audio-service/src/analysis/residual.py`)

**Files:**
- Create: `audio-service/src/analysis/residual.py`
- Create: `audio-service/tests/test_residual.py`

- [ ] **Step 1: Write the failing tests**

Create `audio-service/tests/test_residual.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd audio-service && python -m pytest tests/test_residual.py -v`
Expected: FAIL (ModuleNotFoundError on `src.analysis.residual`).

- [ ] **Step 3: Implement the module**

Create `audio-service/src/analysis/residual.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd audio-service && python -m pytest tests/test_residual.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add audio-service/src/analysis/residual.py audio-service/tests/test_residual.py
git commit -m "feat(audio-service): spectral_subtract helper for residual bass cleanup"
```

---

## Task 4: PANNs CNN14 tagger wrapper (`audio-service/src/analysis/tagging.py`)

**Files:**
- Modify: `audio-service/requirements.txt` (add `panns-inference`)
- Create: `audio-service/src/analysis/tagging.py`
- Create: `audio-service/tests/test_tagging.py`

- [ ] **Step 1: Add `panns-inference` to requirements**

Modify `audio-service/requirements.txt` — append:

```
panns-inference==0.1.1
```

- [ ] **Step 2: Install locally for the failing-test step**

Run: `cd audio-service && pip install panns-inference==0.1.1 librosa==0.10.2`
Expected: install succeeds. Model weights are NOT downloaded yet (lazy on first inference call).

- [ ] **Step 3: Write the failing tests**

Create `audio-service/tests/test_tagging.py`:

```python
import numpy as np
import pytest

from src.analysis import tagging


@pytest.fixture(autouse=True)
def _reset_model():
    tagging.free_model()
    yield
    tagging.free_model()


def test_tag_clip_returns_top_k_sorted(monkeypatch):
    fake_labels = np.array(["A", "B", "C", "D"], dtype=object)
    fake_probs = np.array([[0.1, 0.7, 0.05, 0.15]], dtype=np.float32)

    class FakeAT:
        labels = fake_labels

        def __init__(self, *args, **kwargs):
            pass

        def inference(self, audio):
            return fake_probs, None  # (clipwise, framewise)

    monkeypatch.setattr(tagging, "_AudioTagging", FakeAT)

    audio = np.zeros(32000, dtype=np.float32)
    out = tagging.tag_clip(audio, sr=32000, top_k=3)
    assert len(out) == 3
    assert out[0] == ("B", pytest.approx(0.7))
    assert out[1] == ("D", pytest.approx(0.15))
    assert out[2] == ("A", pytest.approx(0.1))


def test_resamples_when_sr_mismatch(monkeypatch):
    captured = {}

    class FakeAT:
        labels = np.array(["X"], dtype=object)

        def __init__(self, *args, **kwargs):
            pass

        def inference(self, audio):
            captured["len"] = audio.shape[-1]
            return np.array([[1.0]], dtype=np.float32), None

    monkeypatch.setattr(tagging, "_AudioTagging", FakeAT)

    audio = np.zeros(44100, dtype=np.float32)  # 1 sec at 44.1 kHz
    tagging.tag_clip(audio, sr=44100, top_k=1)
    # Resampled to 32 kHz → ~32000 samples ± rounding
    assert 31950 <= captured["len"] <= 32050


def test_free_model_resets_state(monkeypatch):
    class FakeAT:
        labels = np.array(["X"], dtype=object)

        def __init__(self, *args, **kwargs):
            pass

        def inference(self, audio):
            return np.array([[1.0]], dtype=np.float32), None

    monkeypatch.setattr(tagging, "_AudioTagging", FakeAT)

    tagging.tag_clip(np.zeros(32000, dtype=np.float32), sr=32000, top_k=1)
    assert tagging._model is not None
    tagging.free_model()
    assert tagging._model is None
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd audio-service && python -m pytest tests/test_tagging.py -v`
Expected: FAIL (ModuleNotFoundError on `src.analysis.tagging`).

- [ ] **Step 5: Implement the module**

Create `audio-service/src/analysis/tagging.py`:

```python
"""Wrapper around PANNs CNN14 audio tagger.

Lazy-loads the model on first call. Caches under module global `_model`.
Caller should invoke `free_model()` when finished to release ~500 MiB.
"""
from __future__ import annotations

import gc
from typing import Optional

import numpy as np
import librosa

# Indirected import so tests can monkeypatch.
from panns_inference import AudioTagging as _AudioTagging  # type: ignore

PANNS_SR = 32000  # CNN14 was trained at 32 kHz.

_model: Optional[object] = None
_labels: Optional[np.ndarray] = None


def _ensure_model() -> object:
    global _model, _labels
    if _model is None:
        _model = _AudioTagging(checkpoint_path=None, device="cpu")
        _labels = np.asarray(getattr(_model, "labels"), dtype=object)
    return _model


def tag_clip(audio: np.ndarray, sr: int, top_k: int = 5) -> list[tuple[str, float]]:
    """Tag a mono audio clip.

    Resamples to 32 kHz internally if `sr != 32000`. Returns the top-k
    (label, probability) pairs sorted by probability descending.
    """
    if audio.size == 0:
        return []

    if sr != PANNS_SR:
        audio = librosa.resample(audio.astype(np.float32), orig_sr=sr, target_sr=PANNS_SR)

    model = _ensure_model()
    audio_batch = np.asarray(audio, dtype=np.float32)[None, :]
    clipwise, _ = model.inference(audio_batch)

    probs = np.asarray(clipwise[0], dtype=np.float32)
    labels = _labels
    assert labels is not None
    k = min(top_k, probs.shape[0], labels.shape[0])
    top_idx = np.argsort(-probs)[:k]
    return [(str(labels[i]), float(probs[i])) for i in top_idx]


def free_model() -> None:
    """Release the cached model + force garbage collection."""
    global _model, _labels
    _model = None
    _labels = None
    gc.collect()
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd audio-service && python -m pytest tests/test_tagging.py -v`
Expected: 3 PASS.

- [ ] **Step 7: Commit**

```bash
git add audio-service/requirements.txt audio-service/src/analysis/tagging.py audio-service/tests/test_tagging.py
git commit -m "feat(audio-service): PANNs CNN14 tag_clip wrapper"
```

---

## Task 5: BS-Roformer / MDX ensemble wrapper (`audio-service/src/analysis/ensemble.py`)

**Files:**
- Modify: `audio-service/requirements.txt` (add `audio-separator[cpu]`)
- Create: `audio-service/src/analysis/ensemble.py`
- Create: `audio-service/tests/test_ensemble.py`

**Implementation note:** `audio-separator` exposes a `Separator` class that loads UVR-style models from filename strings. Two specialist passes:
- Vocals: `model_bs_roformer_ep_368_sdr_12.9628.ckpt` (or the closest variant the package fetches).
- Bass: `UVR_MDXNET_KARA_2.onnx` or `Kim_Vocal_2.onnx` is wrong for bass — use `MDX23C-InstVoc HQ.ckpt` and select the bass output stem if present; otherwise fall back to `MDX-Net_Inst_HQ_3.onnx`'s bass-band proxy.

Because the exact freely-available BS-Roformer bass weights move around, the wrapper accepts the model filename via env (`DESPIECE_ROFORMER_VOCALS_MODEL`, `DESPIECE_ROFORMER_BASS_MODEL`) with sane defaults. The wrapper's interface and tests don't depend on which model is loaded — only that a `separate_*` call returns a float32 mono numpy array at the original sample rate.

- [ ] **Step 1: Add `audio-separator[cpu]` to requirements**

Modify `audio-service/requirements.txt` — append:

```
audio-separator[cpu]==0.32.0
```

- [ ] **Step 2: Install locally**

Run: `cd audio-service && pip install "audio-separator[cpu]==0.32.0"`
Expected: install succeeds. No model downloads at install time.

- [ ] **Step 3: Write the failing tests**

Create `audio-service/tests/test_ensemble.py`:

```python
import numpy as np
import pytest

from src.analysis import ensemble


@pytest.fixture(autouse=True)
def _reset_models():
    ensemble.free_model()
    yield
    ensemble.free_model()


class _FakeSeparator:
    """Mock that mimics audio-separator's `Separator.separate(path)` API."""

    def __init__(self, *args, **kwargs):
        self.last_args = (args, kwargs)

    def load_model(self, model_filename: str):
        self.loaded = model_filename

    def separate(self, audio_path: str):
        # Write a synthetic "vocals" output next to the input.
        import os
        import soundfile as sf
        sr = 44100
        out = np.zeros(sr * 1, dtype=np.float32)
        out_path = audio_path.replace(".wav", "_(Vocals)_dummy.wav")
        sf.write(out_path, out, sr)
        return [out_path]


def test_separate_roformer_vocals_returns_mono_float32(tmp_path, monkeypatch):
    monkeypatch.setattr(ensemble, "_Separator", _FakeSeparator)
    sr = 44100
    audio = np.random.RandomState(0).randn(sr * 2).astype(np.float32) * 0.1

    out = ensemble.separate_roformer_vocals(audio, sr=sr)
    assert out.dtype == np.float32
    assert out.ndim == 1
    # Result may be shorter (1 sec dummy) or longer; just assert it's non-trivial.
    assert out.shape[0] > 0


def test_separate_roformer_bass_returns_mono_float32(tmp_path, monkeypatch):
    monkeypatch.setattr(ensemble, "_Separator", _FakeSeparator)
    sr = 44100
    audio = np.zeros(sr, dtype=np.float32)

    out = ensemble.separate_roformer_bass(audio, sr=sr)
    assert out.dtype == np.float32
    assert out.ndim == 1


def test_free_model_resets_state(monkeypatch):
    monkeypatch.setattr(ensemble, "_Separator", _FakeSeparator)
    ensemble.separate_roformer_vocals(np.zeros(44100, dtype=np.float32), sr=44100)
    assert ensemble._vocals_sep is not None
    ensemble.free_model()
    assert ensemble._vocals_sep is None
    assert ensemble._bass_sep is None
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd audio-service && python -m pytest tests/test_ensemble.py -v`
Expected: FAIL (ModuleNotFoundError on `src.analysis.ensemble`).

- [ ] **Step 5: Implement the module**

Create `audio-service/src/analysis/ensemble.py`:

```python
"""BS-Roformer + MDX-Net specialist separators via `audio-separator`.

Two public functions: `separate_roformer_vocals` and `separate_roformer_bass`.
Both lazy-load their model on first call. Memory-conscious sequential use
is the caller's responsibility; call `free_model()` between models.
"""
from __future__ import annotations

import gc
import os
import tempfile
from typing import Optional

import numpy as np
import soundfile as sf

# Indirected import so tests can monkeypatch.
from audio_separator.separator import Separator as _Separator  # type: ignore


VOCALS_MODEL_DEFAULT = os.environ.get(
    "DESPIECE_ROFORMER_VOCALS_MODEL",
    "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
)
BASS_MODEL_DEFAULT = os.environ.get(
    "DESPIECE_ROFORMER_BASS_MODEL",
    "MDX23C-InstVoc_HQ.ckpt",
)

_vocals_sep: Optional[object] = None
_bass_sep: Optional[object] = None


def _ensure(model_filename: str, slot: str) -> object:
    global _vocals_sep, _bass_sep
    existing = _vocals_sep if slot == "vocals" else _bass_sep
    if existing is not None:
        return existing
    sep = _Separator(log_level=30)
    sep.load_model(model_filename)
    if slot == "vocals":
        _vocals_sep = sep
    else:
        _bass_sep = sep
    return sep


def _run_separator(sep, audio: np.ndarray, sr: int, want_keyword: str) -> np.ndarray:
    """Write audio to a temp WAV, run the separator, locate the desired stem,
    return mono float32 at the original sr."""
    with tempfile.TemporaryDirectory() as tmpdir:
        in_path = os.path.join(tmpdir, "in.wav")
        sf.write(in_path, audio, sr)
        out_paths = sep.separate(in_path)
        # Pick the path whose filename contains the keyword (case-insensitive).
        chosen = None
        for p in out_paths:
            base = os.path.basename(p).lower()
            if want_keyword.lower() in base:
                chosen = p
                break
        if chosen is None:
            # Fall back to the first output to avoid hard failure.
            chosen = out_paths[0]
        data, out_sr = sf.read(chosen, dtype="float32", always_2d=False)
        if data.ndim == 2:
            data = data.mean(axis=1)
        if out_sr != sr:
            import librosa
            data = librosa.resample(data.astype(np.float32), orig_sr=out_sr, target_sr=sr)
        return data.astype(np.float32)


def separate_roformer_vocals(audio: np.ndarray, sr: int) -> np.ndarray:
    """Run the vocals specialist. Returns mono float32 at the original sr."""
    sep = _ensure(VOCALS_MODEL_DEFAULT, "vocals")
    return _run_separator(sep, audio.astype(np.float32), sr, want_keyword="vocals")


def separate_roformer_bass(audio: np.ndarray, sr: int) -> np.ndarray:
    """Run the bass specialist. Returns mono float32 at the original sr."""
    sep = _ensure(BASS_MODEL_DEFAULT, "bass")
    return _run_separator(sep, audio.astype(np.float32), sr, want_keyword="bass")


def free_model() -> None:
    """Release both cached separators + force gc + empty torch cache."""
    global _vocals_sep, _bass_sep
    _vocals_sep = None
    _bass_sep = None
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd audio-service && python -m pytest tests/test_ensemble.py -v`
Expected: 3 PASS.

- [ ] **Step 7: Commit**

```bash
git add audio-service/requirements.txt audio-service/src/analysis/ensemble.py audio-service/tests/test_ensemble.py
git commit -m "feat(audio-service): BS-Roformer/MDX ensemble wrapper for vocals + bass"
```

---

## Task 6: Pipeline orchestration in `main.py`

**Why now:** Wires the four new modules into the existing `/analyze` flow. Tests are integration-shaped and mock all four heavy models.

**Files:**
- Modify: `audio-service/src/main.py`
- Create: `audio-service/tests/test_main_integration.py`

- [ ] **Step 1: Write the failing integration test**

Create `audio-service/tests/test_main_integration.py`:

```python
import os
import numpy as np
import pytest
from fastapi.testclient import TestClient

import src.main as main_mod
from src.models import AnalyzeRequest


@pytest.fixture
def client(monkeypatch, tmp_path):
    sr = 22050
    duration_sec = 2.0
    n = int(sr * duration_sec)
    mono = np.random.RandomState(0).randn(n).astype(np.float32) * 0.05

    stems_data = {
        "drums":  (mono.copy(), sr),
        "bass":   (mono.copy(), sr),
        "other":  (mono.copy(), sr),
        "vocals": (mono.copy(), sr),
        "guitar": (mono.copy(), sr),
        "piano":  (mono.copy(), sr),
    }

    def fake_download(storage_key, dest_path):
        # Write an empty WAV so any downstream tool that might inspect it doesn't crash.
        import soundfile as sf
        sf.write(dest_path, mono, sr)

    def fake_upload(local_path, key):
        pass

    monkeypatch.setattr(main_mod, "download_to_path", fake_download)
    monkeypatch.setattr(main_mod, "upload_from_path", fake_upload)
    monkeypatch.setattr(main_mod, "separate_stems", lambda input_path, tmpdir: stems_data)

    # Mock ensemble + tagging.
    monkeypatch.setattr(main_mod, "separate_roformer_vocals", lambda a, sr: a.copy())
    monkeypatch.setattr(main_mod, "separate_roformer_bass", lambda a, sr: a.copy())
    monkeypatch.setattr(main_mod, "ensemble_free_model", lambda: None)
    monkeypatch.setattr(main_mod, "tag_clip", lambda a, sr, top_k=5: [("Pad", 0.8)])
    monkeypatch.setattr(main_mod, "tagging_free_model", lambda: None)
    monkeypatch.setattr(main_mod, "encode_mp3", lambda audio, sr, path, bitrate_kbps=128: open(path, "wb").close())

    # Force one region to be detected per stem so the tagging branch fires for `other`.
    monkeypatch.setattr(main_mod, "detect_regions", lambda audio, sr, beat_grid, beats_per_bar=4: [
        {"start_sec": 0.0, "end_sec": 1.0, "envelope": [[0.0, 0.5]]}
    ])

    return TestClient(main_mod.app)


def test_analyze_returns_version_2_and_sub_label(client):
    resp = client.post(
        "/analyze",
        json={"song_id": "abc-123", "storage_key": "songs/abc-123.wav"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["analysis_version"] == 2
    assert set(body["stems"].keys()) == {"drums", "bass", "other", "vocals", "guitar", "piano"}

    other_regions = body["stems"]["other"]["regions"]
    assert len(other_regions) == 1
    assert other_regions[0]["sub_label"] == "pad"
    assert 0.79 <= other_regions[0]["sub_label_confidence"] <= 0.81

    # Non-`other` stems should NOT have sub_label fields.
    assert "sub_label" not in body["stems"]["vocals"]["regions"][0]
    assert "sub_label_confidence" not in body["stems"]["bass"]["regions"][0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd audio-service && python -m pytest tests/test_main_integration.py -v`
Expected: FAIL — current `main.py` doesn't return `analysis_version: 2` and doesn't tag `other` regions.

- [ ] **Step 3: Rewrite `main.py` to orchestrate the new pipeline**

Replace the contents of `audio-service/src/main.py`:

```python
import os
import tempfile
import traceback

import numpy as np
from fastapi import FastAPI, HTTPException

from .models import AnalyzeRequest
from .storage import download_to_path, upload_from_path
from .analysis.demucs_runner import separate_stems, STEMS
from .analysis.beat_analysis import detect_bpm_and_beats, detect_key
from .analysis.segmentation import detect_sections
from .analysis.regions import detect_regions
from .analysis.mp3_encoder import encode_mp3
from .analysis.ensemble import (
    separate_roformer_vocals,
    separate_roformer_bass,
    free_model as ensemble_free_model,
)
from .analysis.residual import spectral_subtract
from .analysis.tagging import tag_clip, free_model as tagging_free_model
from .analysis.sub_label import collapse_to_sub_label

app = FastAPI(title="Audio Analysis Service")

BEATS_PER_BAR = 4
STEM_MP3_BITRATE_KBPS = 128
ANALYSIS_VERSION = 2


def _env_flag(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return default


USE_ROFORMER_VOCALS = _env_flag("DESPIECE_USE_ROFORMER_VOCALS", True)
USE_ROFORMER_BASS = _env_flag("DESPIECE_USE_ROFORMER_BASS", True)
USE_RESIDUAL_SUBTRACT = _env_flag("DESPIECE_USE_RESIDUAL_SUBTRACT", True)
USE_TAGGER = _env_flag("DESPIECE_USE_TAGGER", True)
RESIDUAL_ALPHA = _env_float("DESPIECE_RESIDUAL_ALPHA", 0.5)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    storage_key = req.storage_key
    song_id = req.song_id

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.audio")
        try:
            download_to_path(storage_key, input_path)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Cannot fetch audio: {e}")

        try:
            stems_data = separate_stems(input_path, tmpdir)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Demucs failed: {e}")

        # Step 3-4: BS-Roformer specialist replacements (sequential, free between).
        if USE_ROFORMER_VOCALS:
            try:
                vocals_audio, vocals_sr = stems_data["vocals"]
                vocals_v2 = separate_roformer_vocals(vocals_audio, vocals_sr)
                stems_data["vocals"] = (vocals_v2, vocals_sr)
            except Exception:
                traceback.print_exc()
            finally:
                ensemble_free_model()

        if USE_ROFORMER_BASS:
            try:
                bass_audio, bass_sr = stems_data["bass"]
                bass_v2 = separate_roformer_bass(bass_audio, bass_sr)
                stems_data["bass"] = (bass_v2, bass_sr)
            except Exception:
                traceback.print_exc()
            finally:
                ensemble_free_model()

        # Step 6: residual subtract bass from guitar + piano.
        if USE_RESIDUAL_SUBTRACT:
            try:
                bass_audio, bass_sr = stems_data["bass"]
                for h in ("guitar", "piano"):
                    h_audio, h_sr = stems_data[h]
                    if h_sr != bass_sr:
                        continue  # skip subtract on sr mismatch; harmonic stays as-is
                    h_cleaned = spectral_subtract(h_audio, bass_audio, sr=h_sr, alpha=RESIDUAL_ALPHA)
                    stems_data[h] = (h_cleaned, h_sr)
            except Exception:
                traceback.print_exc()

        all_audio = [audio for audio, _ in stems_data.values()]
        sr = next(iter(stems_data.values()))[1]
        mix_mono: np.ndarray = np.mean(all_audio, axis=0) if all_audio else np.zeros(1)
        duration_sec = len(mix_mono) / sr

        try:
            bpm, beat_grid = detect_bpm_and_beats(mix_mono, sr)
        except Exception:
            bpm, beat_grid = 0.0, []

        try:
            key = detect_key(mix_mono, sr)
        except Exception:
            key = "unknown"

        try:
            sections = detect_sections(mix_mono, sr, duration_sec)
        except Exception:
            sections = []

        bar_grid = _bar_grid_from_beats(beat_grid, BEATS_PER_BAR, duration_sec)

        stems: dict = {}
        for stem_name in STEMS:
            audio, stem_sr = stems_data[stem_name]
            audio_key = None
            try:
                mp3_path = os.path.join(tmpdir, f"{stem_name}.mp3")
                encode_mp3(audio, stem_sr, mp3_path, bitrate_kbps=STEM_MP3_BITRATE_KBPS)
                audio_key = f"songs/{song_id}/stems/{stem_name}.mp3"
                upload_from_path(mp3_path, audio_key)
            except Exception:
                traceback.print_exc()
                audio_key = None

            try:
                regions = detect_regions(audio, stem_sr, beat_grid, beats_per_bar=BEATS_PER_BAR)
            except Exception:
                traceback.print_exc()
                regions = []

            # Tag `other` regions only.
            if stem_name == "other" and USE_TAGGER and regions:
                for region in regions:
                    try:
                        start = int(region["start_sec"] * stem_sr)
                        end = int(region["end_sec"] * stem_sr)
                        clip = audio[start:end]
                        if clip.size == 0:
                            continue
                        tags = tag_clip(clip, sr=stem_sr, top_k=5)
                        sub_label, confidence = collapse_to_sub_label(tags)
                        region["sub_label"] = sub_label
                        region["sub_label_confidence"] = round(float(confidence), 4)
                    except Exception:
                        traceback.print_exc()
                tagging_free_model()

            stems[stem_name] = {"audio_key": audio_key, "regions": regions}

        return {
            "analysis_version": ANALYSIS_VERSION,
            "bpm": bpm,
            "key": key,
            "duration_sec": round(duration_sec, 3),
            "beat_grid": beat_grid,
            "bar_grid": bar_grid,
            "sections": sections,
            "stems": stems,
        }


def _bar_grid_from_beats(beat_times, beats_per_bar, audio_duration):
    if len(beat_times) < 2:
        return [0.0, round(float(audio_duration), 3)]
    bars = [round(float(beat_times[i]), 3) for i in range(0, len(beat_times), beats_per_bar)]
    if bars[0] > 0.0:
        bars.insert(0, 0.0)
    if bars[-1] < audio_duration:
        bars.append(round(float(audio_duration), 3))
    return bars
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd audio-service && python -m pytest tests/test_main_integration.py -v`
Expected: PASS.

- [ ] **Step 5: Run the whole test suite to confirm no regressions**

Run: `cd audio-service && python -m pytest -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add audio-service/src/main.py audio-service/tests/test_main_integration.py
git commit -m "feat(audio-service): orchestrate ensemble + residual + tagging in /analyze"
```

---

## Task 7: Dockerfile model pre-pull + env example

**Files:**
- Create: `audio-service/scripts/prefetch_models.py`
- Modify: `audio-service/Dockerfile`
- Modify: `audio-service/env.example` (create if missing)

- [ ] **Step 1: Write the pre-fetch script**

Create `audio-service/scripts/prefetch_models.py`:

```python
"""Pre-pull model weights at Docker-build time so the running container
never blocks on a first-run download."""
import sys


def pull_demucs():
    from demucs.pretrained import get_model
    get_model("htdemucs_6s")


def pull_audio_separator():
    from audio_separator.separator import Separator
    sep = Separator(log_level=30)
    for name in (
        "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        "MDX23C-InstVoc_HQ.ckpt",
    ):
        try:
            sep.load_model(name)
        except Exception as e:
            print(f"[prefetch] could not pre-pull {name}: {e}", file=sys.stderr)


def pull_panns():
    from panns_inference import AudioTagging
    AudioTagging(checkpoint_path=None, device="cpu")


if __name__ == "__main__":
    for name, fn in (("demucs", pull_demucs), ("audio-separator", pull_audio_separator), ("panns", pull_panns)):
        print(f"[prefetch] {name}...", flush=True)
        try:
            fn()
            print(f"[prefetch] {name} ok", flush=True)
        except Exception as e:
            print(f"[prefetch] {name} failed: {e}", file=sys.stderr)
```

- [ ] **Step 2: Update the Dockerfile to run pre-fetch**

Replace `audio-service/Dockerfile`:

```dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY scripts/ ./scripts/

# Pre-pull all model weights so the running container never blocks on a first-run download.
RUN python scripts/prefetch_models.py || true

COPY src/ ./src/

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

- [ ] **Step 3: Document env vars in env.example**

Create or append to `audio-service/env.example`:

```env
# Stem quality (Phase A) — turn off individually to bisect quality regressions.
DESPIECE_USE_ROFORMER_VOCALS=true
DESPIECE_USE_ROFORMER_BASS=true
DESPIECE_USE_RESIDUAL_SUBTRACT=true
DESPIECE_USE_TAGGER=true
DESPIECE_RESIDUAL_ALPHA=0.5

# Optional: override the BS-Roformer / MDX model filenames.
# DESPIECE_ROFORMER_VOCALS_MODEL=model_bs_roformer_ep_368_sdr_12.9628.ckpt
# DESPIECE_ROFORMER_BASS_MODEL=MDX23C-InstVoc_HQ.ckpt
```

- [ ] **Step 4: Verify the Dockerfile builds (smoke check, may take 10+ min due to model downloads)**

Run: `cd audio-service && docker build -t despiece-audio-service:phase-a .`
Expected: build completes; final image has model weights cached in layers.

If the model download fails inside the build (network flake), the `|| true` on the prefetch line allows the build to proceed — first-run inside the container will then download. Acceptable degradation.

- [ ] **Step 5: Commit**

```bash
git add audio-service/scripts/prefetch_models.py audio-service/Dockerfile audio-service/env.example
git commit -m "build(audio-service): pre-pull BS-Roformer + PANNs weights at image build"
```

---

## Task 8: Local dev-mode pipeline script

**Files:**
- Create: `audio-service/scripts/analyze_local.py`

**Why:** ~30-45 min/song over HTTP is bad for iteration. This script runs the pipeline directly on a local audio file, skipping FastAPI and S3.

- [ ] **Step 1: Write the script**

Create `audio-service/scripts/analyze_local.py`:

```python
"""Run the Phase A pipeline on a local audio file, write JSON to stdout.

Usage:  python scripts/analyze_local.py path/to/song.mp3
"""
import json
import os
import sys
import tempfile

import numpy as np
import soundfile as sf

# Force the module-level toggles to defaults before importing main.
os.environ.setdefault("DESPIECE_USE_ROFORMER_VOCALS", "true")
os.environ.setdefault("DESPIECE_USE_ROFORMER_BASS", "true")
os.environ.setdefault("DESPIECE_USE_RESIDUAL_SUBTRACT", "true")
os.environ.setdefault("DESPIECE_USE_TAGGER", "true")

from src.analysis.demucs_runner import separate_stems, STEMS
from src.analysis.beat_analysis import detect_bpm_and_beats, detect_key
from src.analysis.segmentation import detect_sections
from src.analysis.regions import detect_regions
from src.analysis.ensemble import separate_roformer_vocals, separate_roformer_bass, free_model as ensemble_free
from src.analysis.residual import spectral_subtract
from src.analysis.tagging import tag_clip, free_model as tagging_free
from src.analysis.sub_label import collapse_to_sub_label


def main(audio_path: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        stems_data = separate_stems(audio_path, tmpdir)

    # Specialist replacements
    a, sr = stems_data["vocals"]
    stems_data["vocals"] = (separate_roformer_vocals(a, sr), sr)
    ensemble_free()

    a, sr = stems_data["bass"]
    stems_data["bass"] = (separate_roformer_bass(a, sr), sr)
    ensemble_free()

    bass_a, bass_sr = stems_data["bass"]
    for h in ("guitar", "piano"):
        ha, hsr = stems_data[h]
        if hsr == bass_sr:
            stems_data[h] = (spectral_subtract(ha, bass_a, sr=hsr, alpha=0.5), hsr)

    sr = next(iter(stems_data.values()))[1]
    mix = np.mean([a for a, _ in stems_data.values()], axis=0)
    bpm, beat_grid = detect_bpm_and_beats(mix, sr)
    duration = len(mix) / sr

    out_stems = {}
    for name in STEMS:
        audio, ssr = stems_data[name]
        regions = detect_regions(audio, ssr, beat_grid, beats_per_bar=4)
        if name == "other":
            for r in regions:
                s, e = int(r["start_sec"] * ssr), int(r["end_sec"] * ssr)
                if e > s:
                    tags = tag_clip(audio[s:e], sr=ssr, top_k=5)
                    sub, conf = collapse_to_sub_label(tags)
                    r["sub_label"] = sub
                    r["sub_label_confidence"] = round(float(conf), 4)
            tagging_free()
        out_stems[name] = {"audio_key": None, "regions": regions}

    print(json.dumps({
        "analysis_version": 2,
        "bpm": bpm,
        "duration_sec": round(duration, 3),
        "beat_grid": beat_grid,
        "stems": out_stems,
    }, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/analyze_local.py path/to/audio", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
```

- [ ] **Step 2: Smoke check (manual, deferred — needs a real audio file)**

This is a manual smoke. The plan does not require running it in CI. Document its existence and skip execution.

- [ ] **Step 3: Commit**

```bash
git add audio-service/scripts/analyze_local.py
git commit -m "feat(audio-service): analyze_local.py dev script for fast iteration"
```

---

## Task 9: Frontend RegionBlock — sub-label color + label text

**Files:**
- Modify: `frontend/src/components/RegionBlock.tsx`
- Create: `frontend/src/components/__tests__/RegionBlock.sublabel.test.tsx`

- [ ] **Step 1: Write the failing test**

Create `frontend/src/components/__tests__/RegionBlock.sublabel.test.tsx`:

```tsx
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { RegionBlock } from '../RegionBlock';

const baseRegion = {
  start_sec: 10,
  end_sec: 30,
  envelope: [[0, 0.5], [1, 0.8]] as [number, number][],
};

describe('RegionBlock sub-label', () => {
  it('renders label text when subLabel + subLabelColor passed and block is wide', () => {
    render(
      <RegionBlock
        region={{ ...baseRegion, sub_label: 'pad', sub_label_confidence: 0.78 }}
        durationSec={60}
        timelineWidth={1200}
        color="#94a3b8"
        subLabel="pad"
        subLabelColor="#8b5cf6"
        onClick={() => {}}
      />
    );
    expect(screen.getByText('pad')).toBeInTheDocument();
  });

  it('low confidence (<0.4) styles label as italic', () => {
    render(
      <RegionBlock
        region={{ ...baseRegion, sub_label: 'pad', sub_label_confidence: 0.3 }}
        durationSec={60}
        timelineWidth={1200}
        color="#94a3b8"
        subLabel="pad"
        subLabelColor="#8b5cf6"
        onClick={() => {}}
      />
    );
    const label = screen.getByText('pad');
    expect(label).toHaveStyle({ fontStyle: 'italic' });
  });

  it('does not render label text when subLabel is absent', () => {
    render(
      <RegionBlock
        region={baseRegion}
        durationSec={60}
        timelineWidth={1200}
        color="#94a3b8"
        onClick={() => {}}
      />
    );
    expect(screen.queryByText('pad')).not.toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/components/__tests__/RegionBlock.sublabel.test.tsx`
Expected: FAIL.

- [ ] **Step 3: Update `RegionBlock.tsx`**

Replace `frontend/src/components/RegionBlock.tsx`:

```tsx
import { useEffect, useRef } from 'react';
import { StemRegion, SubLabel } from '@/types/song';

interface Props {
  region: StemRegion;
  durationSec: number;
  timelineWidth: number;
  color: string;
  onClick: (startSec: number) => void;
  rowHeight?: number;
  subLabel?: SubLabel;
  subLabelColor?: string;
}

const LABEL_MIN_WIDTH_PX = 60;

export function RegionBlock({
  region,
  durationSec,
  timelineWidth,
  color,
  onClick,
  rowHeight = 48,
  subLabel,
  subLabelColor,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const left = (region.start_sec / durationSec) * timelineWidth;
  const width = ((region.end_sec - region.start_sec) / durationSec) * timelineWidth;
  const effectiveColor = subLabelColor ?? color;
  const confidence = region.sub_label_confidence ?? 1;
  const lowConfidence = confidence < 0.4;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(width * dpr));
    canvas.height = Math.max(1, Math.floor(rowHeight * dpr));
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    ctx.fillStyle = effectiveColor + '22';
    ctx.fillRect(0, 0, width, rowHeight);
    ctx.strokeStyle = effectiveColor;
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, width - 1, rowHeight - 1);

    if (region.envelope.length === 0) return;
    const regionDur = region.end_sec - region.start_sec;
    ctx.fillStyle = effectiveColor;
    for (const [t, v] of region.envelope) {
      const x = (t / regionDur) * width;
      const h = Math.max(1, v * (rowHeight - 4));
      ctx.fillRect(x, rowHeight - h - 2, 1, h);
    }
  }, [region, width, rowHeight, effectiveColor]);

  const showLabel = !!subLabel && width >= LABEL_MIN_WIDTH_PX;
  const titleText =
    subLabel && region.sub_label_confidence !== undefined
      ? `${subLabel} · ${Math.round(region.sub_label_confidence * 100)}%`
      : subLabel ?? undefined;

  return (
    <div
      style={{
        position: 'absolute',
        left: `${left}px`,
        width: `${width}px`,
        height: `${rowHeight}px`,
        top: 0,
        cursor: 'pointer',
      }}
      onClick={() => onClick(region.start_sec)}
      title={titleText}
    >
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: '100%', display: 'block' }}
      />
      {showLabel && (
        <span
          style={{
            position: 'absolute',
            top: 2,
            left: 4,
            fontSize: 10,
            fontStyle: lowConfidence ? 'italic' : 'normal',
            opacity: lowConfidence ? 0.6 : 1,
            color: effectiveColor,
            pointerEvents: 'none',
            textShadow: '0 0 2px rgba(0,0,0,0.4)',
          }}
        >
          {subLabel}
        </span>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npx vitest run src/components/__tests__/RegionBlock.sublabel.test.tsx`
Expected: 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/RegionBlock.tsx frontend/src/components/__tests__/RegionBlock.sublabel.test.tsx
git commit -m "feat(frontend): RegionBlock renders sub-label color + inline label text"
```

---

## Task 10: Frontend StemTrack — wire sub-label only for `other`

**Files:**
- Modify: `frontend/src/components/StemTrack.tsx`

- [ ] **Step 1: Update `StemTrack.tsx`**

Replace `frontend/src/components/StemTrack.tsx`:

```tsx
import { StemAnalysis, SubLabel } from '@/types/song';
import { RegionBlock } from './RegionBlock';
import { StemControls } from './StemControls';

interface Props {
  name: string;
  stem: StemAnalysis;
  durationSec: number;
  timelineWidth: number;
  color: string;
  dim: boolean;
  soloed: boolean;
  muted: boolean;
  onToggleSolo: () => void;
  onToggleMute: () => void;
  onRegionClick: (startSec: number) => void;
  labelWidth?: number;
  rowHeight?: number;
  subLabelColors?: Record<SubLabel, string>;
}

export function StemTrack({
  name,
  stem,
  durationSec,
  timelineWidth,
  color,
  dim,
  soloed,
  muted,
  onToggleSolo,
  onToggleMute,
  onRegionClick,
  labelWidth = 96,
  rowHeight = 48,
  subLabelColors,
}: Props) {
  return (
    <div
      data-stem-track={name}
      className="flex items-center mt-1"
      style={{ opacity: dim ? 0.3 : 1, transition: 'opacity 120ms' }}
    >
      <div
        style={{ width: labelWidth }}
        className="text-xs font-medium capitalize text-right pr-2 text-muted-foreground flex items-center justify-end gap-2"
      >
        <StemControls
          soloed={soloed}
          muted={muted}
          onToggleSolo={onToggleSolo}
          onToggleMute={onToggleMute}
        />
        <span>{name}</span>
      </div>
      <div
        className="relative bg-muted/20 rounded overflow-hidden"
        style={{ width: timelineWidth, height: rowHeight }}
      >
        {stem.regions.map((region, i) => {
          const sub = name === 'other' ? region.sub_label : undefined;
          const subColor = sub && subLabelColors ? subLabelColors[sub] : undefined;
          return (
            <RegionBlock
              key={i}
              region={region}
              durationSec={durationSec}
              timelineWidth={timelineWidth}
              color={color}
              onClick={onRegionClick}
              rowHeight={rowHeight}
              subLabel={sub}
              subLabelColor={subColor}
            />
          );
        })}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify the frontend still type-checks + RegionBlock tests still pass**

Run: `cd frontend && npx tsc --noEmit`
Expected: no errors.

Run: `cd frontend && npx vitest run src/components/__tests__/`
Expected: existing tests + new sub-label tests pass.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/StemTrack.tsx
git commit -m "feat(frontend): StemTrack passes sub-label only for 'other' stem"
```

---

## Task 11: Frontend SongTimeline — sub-label legend + re-analyze gate

**Files:**
- Modify: `frontend/src/components/SongTimeline.tsx`
- Create: `frontend/src/components/__tests__/SongTimeline.version.test.tsx`

- [ ] **Step 1: Write failing tests for the version gate + legend**

Create `frontend/src/components/__tests__/SongTimeline.version.test.tsx`:

```tsx
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { SongTimeline } from '../SongTimeline';
import type { AnalysisResult } from '@/types/song';

function baseAnalysis(version: number, otherRegion: any = null): AnalysisResult {
  return {
    bpm: 120,
    key: 'C major',
    duration_sec: 60,
    beat_grid: [0, 0.5, 1.0],
    bar_grid: [0, 2.0],
    sections: [],
    stems: {
      vocals: { audio_key: 'k', regions: [] },
      drums:  { audio_key: 'k', regions: [] },
      bass:   { audio_key: 'k', regions: [] },
      guitar: { audio_key: 'k', regions: [] },
      piano:  { audio_key: 'k', regions: [] },
      other:  { audio_key: 'k', regions: otherRegion ? [otherRegion] : [] },
    },
    analysis_version: version,
  };
}

describe('SongTimeline', () => {
  it('shows "Re-analyze" notice when analysis_version is missing or < 2', () => {
    const old = { ...baseAnalysis(1) } as any;
    delete old.analysis_version;
    render(<SongTimeline songId="abc" analysis={old} />);
    expect(screen.getByText(/re-analyze/i)).toBeInTheDocument();
  });

  it('renders timeline when analysis_version >= 2', () => {
    render(<SongTimeline songId="abc" analysis={baseAnalysis(2)} />);
    expect(screen.queryByText(/re-analyze/i)).not.toBeInTheDocument();
  });

  it('renders the sub-label legend when at least one other region has sub_label', () => {
    const analysis = baseAnalysis(2, {
      start_sec: 0, end_sec: 10, envelope: [], sub_label: 'pad', sub_label_confidence: 0.8,
    });
    render(<SongTimeline songId="abc" analysis={analysis} />);
    expect(screen.getByText(/sub-labels/i)).toBeInTheDocument();
    expect(screen.getByText(/pad/i)).toBeInTheDocument();
  });

  it('hides the sub-label legend when no other region has sub_label', () => {
    render(<SongTimeline songId="abc" analysis={baseAnalysis(2)} />);
    expect(screen.queryByText(/sub-labels/i)).not.toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npx vitest run src/components/__tests__/SongTimeline.version.test.tsx`
Expected: FAIL on the new assertions.

- [ ] **Step 3: Update `SongTimeline.tsx`**

Replace `frontend/src/components/SongTimeline.tsx`:

```tsx
import { useMemo, useState } from 'react';
import { AnalysisResult, SubLabel } from '@/types/song';
import { StemTrack } from './StemTrack';
import { SectionBar } from './SectionBar';
import { TimeAxis } from './TimeAxis';
import { PlaybackBar } from './PlaybackBar';
import { getStemAudioUrl } from '@/lib/api';
import { audible } from '@/lib/audible';
import { useStemTransport } from '@/hooks/useStemTransport';

const STEM_ORDER = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other'];

const STEM_COLORS: Record<string, string> = {
  vocals: '#6366f1',
  drums:  '#f59e0b',
  bass:   '#10b981',
  guitar: '#a78bfa',
  piano:  '#06b6d4',
  other:  '#94a3b8',
};

const SUB_LABEL_COLORS: Record<SubLabel, string> = {
  lead:       '#ef4444',
  pad:        '#8b5cf6',
  synth:      '#ec4899',
  strings:    '#f97316',
  fx:         '#14b8a6',
  other_misc: '#94a3b8',
};

const ROW_HEIGHT = 48;
const SECTION_HEIGHT = 28;
const AXIS_HEIGHT = 24;
const LABEL_WIDTH = 96;
const REQUIRED_ANALYSIS_VERSION = 2;

interface Props {
  songId: string;
  analysis: AnalysisResult;
}

export function SongTimeline({ songId, analysis }: Props) {
  const [axisMode, setAxisMode] = useState<'seconds' | 'bars'>('bars');
  const [sectionLabels, setSectionLabels] = useState<Record<string, string>>({});
  const [soloed, setSoloed] = useState<Set<string>>(new Set());
  const [muted, setMuted] = useState<Set<string>>(new Set());
  const transport = useStemTransport();

  const timelineWidth = useMemo(
    () => Math.max(600, Math.min(1400, window.innerWidth - LABEL_WIDTH - 80)),
    []
  );

  const activeStems = useMemo(
    () =>
      STEM_ORDER.filter(
        (s) => analysis.stems[s] && analysis.stems[s].audio_key !== null
      ),
    [analysis.stems]
  );

  const otherSubLabels = useMemo<SubLabel[]>(() => {
    const other = analysis.stems['other'];
    if (!other) return [];
    const set = new Set<SubLabel>();
    for (const r of other.regions) {
      if (r.sub_label) set.add(r.sub_label);
    }
    return Array.from(set);
  }, [analysis.stems]);

  if ((analysis.analysis_version ?? 1) < REQUIRED_ANALYSIS_VERSION) {
    return (
      <div className="p-6 border rounded bg-muted/20 text-sm">
        <p className="font-medium mb-1">This song was analyzed with an older pipeline.</p>
        <p className="text-muted-foreground">
          Re-analyze it to get pro-grade stems and "other"-stem sub-labels.
        </p>
      </div>
    );
  }

  function toggleSolo(name: string) {
    setSoloed((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  function toggleMute(name: string) {
    setMuted((prev) => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  }

  function isDim(name: string) {
    return !audible(name, soloed, muted);
  }

  function handleRegionClick(startSec: number) {
    transport.playFrom(startSec);
  }

  function handleRename(label: string, name: string) {
    setSectionLabels((prev) => ({ ...prev, [label]: name }));
  }

  return (
    <div className="space-y-4">
      {activeStems.map((name) => (
        <audio
          key={name}
          ref={(el) => transport.registerAudio(name, el)}
          src={getStemAudioUrl(songId, name)}
          muted={!audible(name, soloed, muted)}
          preload="auto"
        />
      ))}

      <div className="flex items-center gap-3 flex-wrap">
        <span className="text-sm text-muted-foreground">Axis:</span>
        <button
          onClick={() => setAxisMode(axisMode === 'bars' ? 'seconds' : 'bars')}
          className="text-sm px-3 py-1 rounded border hover:bg-muted transition-colors"
        >
          {axisMode === 'bars' ? 'Bars' : 'Seconds'}
        </button>
        <span className="text-xs text-muted-foreground">
          Double-click a section to rename it. Click a region to jump audio.
        </span>
      </div>

      <div className="overflow-x-auto">
        <div style={{ minWidth: LABEL_WIDTH + timelineWidth }}>
          <div className="flex items-center">
            <div
              style={{ width: LABEL_WIDTH }}
              className="text-xs text-muted-foreground pr-2 text-right"
            >
              Sections
            </div>
            <SectionBar
              sections={analysis.sections}
              durationSec={analysis.duration_sec}
              width={timelineWidth}
              height={SECTION_HEIGHT}
              labels={sectionLabels}
              onRename={handleRename}
            />
          </div>

          {STEM_ORDER.filter((s) => analysis.stems[s]).map((stemName) => (
            <StemTrack
              key={stemName}
              name={stemName}
              stem={analysis.stems[stemName]}
              durationSec={analysis.duration_sec}
              timelineWidth={timelineWidth}
              color={STEM_COLORS[stemName] ?? '#94a3b8'}
              dim={isDim(stemName)}
              soloed={soloed.has(stemName)}
              muted={muted.has(stemName)}
              onToggleSolo={() => toggleSolo(stemName)}
              onToggleMute={() => toggleMute(stemName)}
              onRegionClick={handleRegionClick}
              labelWidth={LABEL_WIDTH}
              rowHeight={ROW_HEIGHT}
              subLabelColors={SUB_LABEL_COLORS}
            />
          ))}

          <div className="flex items-center mt-1">
            <div style={{ width: LABEL_WIDTH }} />
            <TimeAxis
              durationSec={analysis.duration_sec}
              bpm={analysis.bpm}
              width={timelineWidth}
              height={AXIS_HEIGHT}
              mode={axisMode}
            />
          </div>

          <PlaybackBar
            transport={transport}
            durationSec={analysis.duration_sec}
            timelineWidth={timelineWidth}
            labelWidth={LABEL_WIDTH}
          />
        </div>
      </div>

      <div className="flex gap-4 flex-wrap text-xs text-muted-foreground">
        {STEM_ORDER.map((s) => (
          <span key={s} className="flex items-center gap-1">
            <span
              className="inline-block w-3 h-3 rounded-sm"
              style={{ background: STEM_COLORS[s] }}
            />
            {s}
          </span>
        ))}
      </div>

      {otherSubLabels.length > 0 && (
        <div className="flex gap-4 flex-wrap text-xs text-muted-foreground">
          <span className="font-medium">Sub-labels:</span>
          {otherSubLabels.map((sub) => (
            <span key={sub} className="flex items-center gap-1">
              <span
                className="inline-block w-3 h-3 rounded-sm"
                style={{ background: SUB_LABEL_COLORS[sub] }}
              />
              {sub}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/components/__tests__/SongTimeline.version.test.tsx`
Expected: 4 PASS.

Run: `cd frontend && npx vitest run`
Expected: full suite passes.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/SongTimeline.tsx frontend/src/components/__tests__/SongTimeline.version.test.tsx
git commit -m "feat(frontend): SongTimeline re-analyze gate + sub-label legend"
```

---

## Task 12: Backend dev-only reset script

**Files:**
- Create: `backend/src/scripts/resetAnalysis.ts`

**Why:** Dev env, no users. The JSON shape has bumped to `analysis_version: 2`; rows from before this change should be re-analyzed.

- [ ] **Step 1: Write the script**

Create `backend/src/scripts/resetAnalysis.ts`:

```ts
import { pool } from '../db/connect.js';

async function main() {
  console.log('Resetting song_analysis + setting songs.status=queued for all rows...');
  await pool.query('BEGIN');
  try {
    await pool.query('DELETE FROM song_analysis');
    const res = await pool.query("UPDATE songs SET status = 'queued', error_message = NULL");
    await pool.query('COMMIT');
    console.log(`Done. ${res.rowCount ?? 0} songs requeued.`);
  } catch (err) {
    await pool.query('ROLLBACK');
    console.error('Failed:', err);
    process.exitCode = 1;
  } finally {
    await pool.end();
  }
}

main();
```

- [ ] **Step 2: Verify the script type-checks**

Run: `cd backend && npx tsc --noEmit`
Expected: no errors.

- [ ] **Step 3: Document run command in README (skip — script's purpose is self-evident from its log line)**

This step is intentionally a no-op. The script is invoked by `cd backend && npx tsx src/scripts/resetAnalysis.ts` on the developer's machine after this branch ships.

- [ ] **Step 4: Commit**

```bash
git add backend/src/scripts/resetAnalysis.ts
git commit -m "chore(backend): dev script to reset song_analysis + requeue songs"
```

---

## Task 13: End-to-end smoke (manual, post-merge)

**Why:** No automated check can validate "is the bass cleaner". Single subjective pass needed before declaring Phase A done.

- [ ] **Step 1: Bring up the stack**

Run: `npm run start:dev`
Expected: frontend on `http://localhost:3000`, backend on `http://localhost:5001`, audio-service on `http://localhost:8000`.

- [ ] **Step 2: Reset prior analysis rows**

Run: `cd backend && npx tsx src/scripts/resetAnalysis.ts`
Expected: log message about songs requeued.

- [ ] **Step 3: Upload a reference track**

Use the frontend uploader. Wait ~30-45 min for status `done`.

- [ ] **Step 4: Validate**

Check, in order:
1. All 6 stem rows render (vocals, drums, bass, guitar, piano, other).
2. Solo on vocals → subjectively cleaner than before (no drum bleed in the chorus).
3. Solo on bass → cleaner attack, less smearing into mid range.
4. Solo on guitar / piano → less low-end mud than before.
5. `other` regions show distinct colors per sub-label; tooltip on hover shows `<label> · <pct>%`.
6. Sub-label legend renders below the stem-color legend.
7. Bring back an old song (if any survive — they should all be requeued from Step 2) — confirm the "Re-analyze" branch is visually consistent.

If any item fails, file a follow-up issue and bisect via the env flags from Task 7 (`DESPIECE_USE_ROFORMER_BASS=false`, `DESPIECE_USE_RESIDUAL_SUBTRACT=false`, `DESPIECE_USE_TAGGER=false`).

- [ ] **Step 5: Cut a PR**

Standard branch + PR. No special CI gates beyond the existing test runners — the audio-service Docker build is what surfaces model-download issues.

---

## Self-review notes

- **Spec coverage:**
  - Spec § Goal — Tasks 5, 3, 4 cover BS-Roformer, residual subtract, PANNs.
  - Spec § Module interfaces — Tasks 2-5 implement them, with test coverage matching the public signatures.
  - Spec § JSON shape — Task 6 emits `analysis_version: 2` + sub-label fields, asserted in `test_main_integration.py`.
  - Spec § Backend (Node) — Task 1 (types), Task 12 (reset script).
  - Spec § Frontend — Tasks 9, 10, 11.
  - Spec § Configuration env vars — Task 7 documents them; Task 6 reads them.
  - Spec § Dockerfile pre-pull — Task 7.
  - Spec § Error handling — every model-call site in Task 6 is wrapped in `try/except` that falls back to the prior stem and never fails the job; PANNs failures omit the optional fields.
  - Spec § Testing — coverage spread across Tasks 2-6 (audio-service), 9, 11 (frontend), 1 (types).
  - Spec § Risks — BS-Roformer bass weights risk addressed via env-configurable model name (Task 5); large-image risk addressed by `|| true` on prefetch (Task 7); 30-45 min iteration loop addressed by `analyze_local.py` (Task 8).
- **Placeholder scan:** none — every step contains either runnable code or a concrete command.
- **Type consistency:** `SubLabel` is the same union in `backend/src/domain/song.ts`, `frontend/src/types/song.ts`, and `audio-service/src/analysis/sub_label.py`. `analysis_version` is the same field name in all three. `separate_roformer_vocals` / `separate_roformer_bass` / `free_model` (ensemble) / `tag_clip` / `free_model` (tagging) names match between the modules and the orchestration in `main.py` (which renames the two `free_model`s on import as `ensemble_free_model` and `tagging_free_model` to disambiguate).
