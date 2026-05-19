# Chord Track (Phase B1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect a song's chord progression (24 major + minor classes plus a no-chord class), render it as a per-bar chord row between the section bar and the stem rows, and bump `analysis_version` to 4.

**Architecture:** New `audio-service/src/analysis/chord_track.py` implements CQT-chroma → 24-template matching → Viterbi decoding → per-bar mode aggregation → merge consecutive identical bars. `main.py` builds a harmonic mix from `bass + guitar + piano + other` stems and calls `detect_chords`. Frontend adds a `ChordBar` component, palette, legend, and updates the soft-notice version gate.

**Tech Stack:** Python 3.11, numpy, librosa 0.10 (`chroma_cqt`, `sequence.viterbi`); TypeScript 5.7, React 19; pytest, Jest, Vitest.

**Spec:** `docs/superpowers/specs/2026-05-19-chord-track-design.md`

**Predecessor branches:** Phase A + Phase B2 are both merged into `main`. This plan builds on that foundation.

---

## File Map

**audio-service (Python):**
- Create: `audio-service/src/analysis/chord_track.py`
- Create: `audio-service/tests/test_chord_track.py`
- Modify: `audio-service/src/main.py` — read `DESPIECE_USE_CHORDS`, build harmonic mix, call `detect_chords`, attach to `chords`, compute response version.
- Modify: `audio-service/tests/test_main_integration.py` — assert v4 + `chords` field, plus a USE_CHORDS=false v3 path.
- Modify: `audio-service/env.example` — document the three new env vars.

**backend (Node):**
- Modify: `backend/src/domain/song.ts` — widen `AnalysisVersion` to `1 | 2 | 3 | 4`, add `ChordLabel`, `Chord`, and `chords?: Chord[]` on `AnalysisResult`.

**frontend (React/TS):**
- Modify: `frontend/src/types/song.ts` — mirror the backend additions.
- Create: `frontend/src/components/ChordBar.tsx`
- Create: `frontend/src/components/__tests__/ChordBar.test.tsx`
- Modify: `frontend/src/components/SongTimeline.tsx` — palette + `<ChordBar>` placement + updated soft notice + chord legend + bump `PREFERRED_ANALYSIS_VERSION` to 4.
- Create: `frontend/src/components/__tests__/SongTimeline.v4.test.tsx`

---

## Task 1: Types (backend + frontend)

**Why first:** Locks the wire format both sides agree on. All later code references these types.

**Files:**
- Modify: `backend/src/domain/song.ts`
- Modify: `frontend/src/types/song.ts`

- [ ] **Step 1: Update backend types**

In `backend/src/domain/song.ts`:

1. Widen `AnalysisVersion` from `1 | 2 | 3` to `1 | 2 | 3 | 4`.
2. Add the new types directly below `AnalysisVersion`:
```ts
export type ChordLabel =
  | 'C' | 'C#' | 'D' | 'D#' | 'E' | 'F' | 'F#' | 'G' | 'G#' | 'A' | 'A#' | 'B'
  | 'Cm' | 'C#m' | 'Dm' | 'D#m' | 'Em' | 'Fm' | 'F#m' | 'Gm' | 'G#m' | 'Am' | 'A#m' | 'Bm'
  | 'N';

export interface Chord {
  start_sec: number;
  end_sec: number;
  label: ChordLabel;
}
```
3. Add `chords?: Chord[]` to `AnalysisResult`:
```ts
export interface AnalysisResult {
  // existing fields…
  analysis_version: AnalysisVersion;
  chords?: Chord[];
}
```

- [ ] **Step 2: Update frontend types**

Apply the identical three changes to `frontend/src/types/song.ts`.

- [ ] **Step 3: Verify**

```bash
cd /Users/matiasagustinmaldonado/Coding/despiece/.claude/worktrees/phase-b1-chord-track/backend && npm run build
cd /Users/matiasagustinmaldonado/Coding/despiece/.claude/worktrees/phase-b1-chord-track/frontend && npx tsc --noEmit
cd /Users/matiasagustinmaldonado/Coding/despiece/.claude/worktrees/phase-b1-chord-track/backend && npm test
cd /Users/matiasagustinmaldonado/Coding/despiece/.claude/worktrees/phase-b1-chord-track/frontend && npx vitest run
```

Expected: backend build clean, frontend tsc clean, jest pass (16/16), vitest pass (36/36).

- [ ] **Step 4: Commit**

```bash
git add backend/src/domain/song.ts frontend/src/types/song.ts
git commit -m "feat(types): add Chord + ChordLabel + widen AnalysisVersion to 1|2|3|4"
```

---

## Task 2: `chord_track.py` — chroma + templates + Viterbi (no per-bar yet)

**Files:**
- Create: `audio-service/src/analysis/chord_track.py`
- Create: `audio-service/tests/test_chord_track.py`

This task builds the module: chroma extraction, template construction, per-frame scoring, no-chord scoring, Viterbi decoding, and the index-to-label helper. The per-bar aggregation + merge logic lives in a separate function but is wired only as a stub returning `[]`. Task 3 fills it in.

- [ ] **Step 1: Write failing tests for the public API shape**

Create `audio-service/tests/test_chord_track.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from src.analysis.chord_track import detect_chords, _idx_to_label, _build_templates


def test_idx_to_label_majors():
    assert _idx_to_label(0) == "C"
    assert _idx_to_label(7) == "G"
    assert _idx_to_label(11) == "B"


def test_idx_to_label_minors():
    assert _idx_to_label(12) == "Cm"
    assert _idx_to_label(19) == "Gm"
    assert _idx_to_label(23) == "Bm"


def test_idx_to_label_no_chord():
    assert _idx_to_label(24) == "N"


def test_build_templates_shape_and_norm():
    t = _build_templates()
    assert t.shape == (24, 12)
    # Each row is unit-norm
    norms = np.linalg.norm(t, axis=1)
    np.testing.assert_allclose(norms, np.ones(24), atol=1e-6)


def test_detect_chords_empty_audio_returns_empty():
    out = detect_chords(np.zeros(0, dtype=np.float32), sr=22050, bar_grid=[0.0, 2.0, 4.0])
    assert out == []


def test_detect_chords_short_bar_grid_returns_empty():
    sr = 22050
    audio = np.zeros(sr, dtype=np.float32)
    assert detect_chords(audio, sr=sr, bar_grid=[]) == []
    assert detect_chords(audio, sr=sr, bar_grid=[0.0]) == []


def test_detect_chords_returns_list_of_dicts():
    """Non-empty audio + valid bar_grid → list of dicts shaped correctly.

    Task 2 stub returns []; this test will be tightened in Task 3.
    Until Task 3, we only assert the return type is a list."""
    sr = 22050
    audio = np.zeros(sr * 4, dtype=np.float32)
    out = detect_chords(audio, sr=sr, bar_grid=[0.0, 2.0, 4.0])
    assert isinstance(out, list)
```

- [ ] **Step 2: Run, expect ModuleNotFoundError**

```bash
cd audio-service && python3 -m pytest tests/test_chord_track.py -v
```

- [ ] **Step 3: Create `audio-service/src/analysis/chord_track.py`**

```python
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
    """Run the full per-frame pipeline. Returns (path_indices, frame_times).

    path[i] is the Viterbi-decoded chord index in [0, 24] for frame i.
    frame_times[i] is the time in seconds of frame i's center.
    """
    chroma = librosa.feature.chroma_cqt(
        y=audio, sr=sr, hop_length=HOP_LENGTH,
        bins_per_octave=36, n_octaves=6, fmin=librosa.note_to_hz("C2"),
    )
    n_frames = chroma.shape[1]
    chroma_norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)

    templates = _build_templates()
    scores = templates @ chroma_norm  # (24, n_frames)
    no_chord = np.maximum(0.0, NO_CHORD_THRESHOLD - scores.max(axis=0))[None, :]
    scores = np.vstack([scores, no_chord])  # (25, n_frames)

    n_states = 25
    trans = np.full((n_states, n_states), (1.0 - SELF_TRANSITION) / (n_states - 1), dtype=np.float64)
    np.fill_diagonal(trans, SELF_TRANSITION)

    # librosa.sequence.viterbi expects nonnegative probabilities and
    # normalises rows of the transition matrix internally.
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
    # Stubbed in Task 2 — Task 3 replaces with real aggregation + merge.
    _viterbi_path(audio, sr)
    return []
```

- [ ] **Step 4: Run, expect all 7 PASS**

```bash
cd audio-service && python3 -m pytest tests/test_chord_track.py -v
```

- [ ] **Step 5: Commit**

```bash
git add audio-service/src/analysis/chord_track.py audio-service/tests/test_chord_track.py
git commit -m "feat(audio-service): chord_track chroma + templates + Viterbi (stubbed merge)"
```

---

## Task 3: Per-bar aggregation + merge consecutive identical bars

**Files:**
- Modify: `audio-service/src/analysis/chord_track.py`
- Modify: `audio-service/tests/test_chord_track.py`

- [ ] **Step 1: Append the class-assignment tests**

At the bottom of `audio-service/tests/test_chord_track.py`, append:

```python
def _make_triad(sr: int, dur_sec: float, freqs: list[float]) -> np.ndarray:
    """Sum of equal-amplitude sines at the given frequencies."""
    n = int(sr * dur_sec)
    t = np.arange(n) / sr
    sig = np.zeros(n, dtype=np.float32)
    for f in freqs:
        sig += np.sin(2 * np.pi * f * t).astype(np.float32)
    sig /= max(1.0, len(freqs))
    return sig.astype(np.float32)


def test_synthetic_c_major_is_classified_as_c():
    sr = 22050
    audio = _make_triad(sr, dur_sec=4.0, freqs=[261.63, 329.63, 392.00])
    out = detect_chords(audio, sr=sr, bar_grid=[0.0, 2.0, 4.0])
    labels = [c["label"] for c in out]
    assert "C" in labels
    # Merge should collapse adjacent identical bars
    assert all(c["label"] == "C" for c in out)


def test_synthetic_a_minor_is_classified_as_am():
    sr = 22050
    audio = _make_triad(sr, dur_sec=4.0, freqs=[220.00, 261.63, 329.63])
    out = detect_chords(audio, sr=sr, bar_grid=[0.0, 2.0, 4.0])
    labels = [c["label"] for c in out]
    assert "Am" in labels
    assert all(c["label"] == "Am" for c in out)


def test_chord_change_at_bar_boundary_produces_two_entries():
    sr = 22050
    c_maj = _make_triad(sr, dur_sec=4.0, freqs=[261.63, 329.63, 392.00])
    g_maj = _make_triad(sr, dur_sec=4.0, freqs=[392.00, 493.88, 587.33])
    audio = np.concatenate([c_maj, g_maj]).astype(np.float32)
    out = detect_chords(audio, sr=sr, bar_grid=[0.0, 2.0, 4.0, 6.0, 8.0])
    assert len(out) == 2
    assert out[0]["label"] == "C"
    assert out[1]["label"] == "G"
    assert out[0]["start_sec"] == 0.0
    assert out[1]["end_sec"] == 8.0


def test_consecutive_identical_bars_merge_into_one_entry():
    """4 short 1-second bars of C major → single merged entry spanning 0–4 s."""
    sr = 22050
    audio = _make_triad(sr, dur_sec=4.0, freqs=[261.63, 329.63, 392.00])
    out = detect_chords(audio, sr=sr, bar_grid=[0.0, 1.0, 2.0, 3.0, 4.0])
    assert len(out) == 1
    assert out[0]["label"] == "C"
    assert out[0]["start_sec"] == 0.0
    assert out[0]["end_sec"] == 4.0


def test_no_chord_threshold_raised_forces_unknown(monkeypatch):
    monkeypatch.setenv("DESPIECE_CHORD_NO_CHORD_THRESHOLD", "1.5")
    import importlib, src.analysis.chord_track as ct
    importlib.reload(ct)
    sr = 22050
    audio = ct._make_triad if False else None  # ignore — kept for type hint clarity
    audio_sig = (
        np.sin(2 * np.pi * 261.63 * np.arange(sr * 4) / sr).astype(np.float32) +
        np.sin(2 * np.pi * 329.63 * np.arange(sr * 4) / sr).astype(np.float32) +
        np.sin(2 * np.pi * 392.00 * np.arange(sr * 4) / sr).astype(np.float32)
    ) / 3.0
    out = ct.detect_chords(audio_sig, sr=sr, bar_grid=[0.0, 2.0, 4.0])
    assert all(c["label"] == "N" for c in out)
    # Restore default for subsequent tests
    monkeypatch.setenv("DESPIECE_CHORD_NO_CHORD_THRESHOLD", "0.3")
    importlib.reload(ct)
```

- [ ] **Step 2: Run, expect class-assignment failures**

```bash
cd audio-service && python3 -m pytest tests/test_chord_track.py -v
```

The 5 new tests fail because `detect_chords` currently returns `[]`.

- [ ] **Step 3: Replace the stub body in `detect_chords`**

In `audio-service/src/analysis/chord_track.py`, replace the body of `detect_chords` after the `if len(bar_grid) < 2:` guard:

```python
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
```

- [ ] **Step 4: Run all tests, expect 12 PASS**

```bash
cd audio-service && python3 -m pytest tests/test_chord_track.py -v
```

If `test_synthetic_a_minor_is_classified_as_am` fails because the chroma scoring lands on `C` (the relative major) instead of `Am` for a bare A-minor triad — the right fix is to extend the synthetic fixture so it reinforces the A root (e.g. add the A one octave lower: `freqs=[110.00, 220.00, 261.63, 329.63]`). Don't relax the assertion. Document any fixture tuning in the report.

- [ ] **Step 5: Full audio-service suite still green**

```bash
cd audio-service && python3 -m pytest tests/ -v
```

Expected: all Phase A + B2 + new B1 tests pass. The pre-existing ffmpeg + MinIO failures are unrelated.

- [ ] **Step 6: Commit**

```bash
git add audio-service/src/analysis/chord_track.py audio-service/tests/test_chord_track.py
git commit -m "feat(audio-service): per-bar aggregation + merge identical bars"
```

---

## Task 4: Wire chord detection into `main.py`

**Files:**
- Modify: `audio-service/src/main.py`
- Modify: `audio-service/tests/test_main_integration.py`

- [ ] **Step 1: Append failing integration tests**

Open `audio-service/tests/test_main_integration.py` and append at the end:

```python
def test_analyze_returns_version_4_when_chords_enabled(monkeypatch):
    import src.main as main_mod
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
        import soundfile as sf
        sf.write(dest_path, mono, sr, format="WAV")

    monkeypatch.setattr(main_mod, "download_to_path", fake_download)
    monkeypatch.setattr(main_mod, "upload_from_path", lambda local_path, key: None)
    monkeypatch.setattr(main_mod, "separate_stems", lambda input_path, tmpdir: stems_data)
    monkeypatch.setattr(main_mod, "separate_roformer_vocals", lambda a, sr: a.copy())
    monkeypatch.setattr(main_mod, "separate_roformer_bass", lambda a, sr: a.copy())
    monkeypatch.setattr(main_mod, "ensemble_free_model", lambda: None)
    monkeypatch.setattr(main_mod, "tag_clip", lambda a, sr, top_k=5: [("Pad", 0.8)])
    monkeypatch.setattr(main_mod, "tagging_free_model", lambda: None)
    monkeypatch.setattr(
        main_mod, "encode_mp3",
        lambda audio, sr, path, bitrate_kbps=128: open(path, "wb").close(),
    )
    monkeypatch.setattr(
        main_mod, "detect_regions",
        lambda audio, sr, beat_grid, beats_per_bar=4: [
            {"start_sec": 0.0, "end_sec": 1.0, "envelope": [[0.0, 0.5]]}
        ],
    )
    monkeypatch.setattr(
        main_mod, "detect_drum_hits",
        lambda audio, sr: {
            "kick": [], "snare": [], "hihat": [], "cymbal": [], "unknown": [],
        },
    )
    monkeypatch.setattr(
        main_mod, "detect_chords",
        lambda audio, sr, bar_grid: [
            {"start_sec": 0.0, "end_sec": 8.0, "label": "Am"}
        ],
    )

    from fastapi.testclient import TestClient
    client = TestClient(main_mod.app)
    resp = client.post(
        "/analyze", json={"song_id": "abc-123", "storage_key": "songs/abc-123.wav"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["analysis_version"] == 4
    assert body["chords"][0]["label"] == "Am"


def test_analyze_returns_version_3_when_chords_disabled(monkeypatch):
    monkeypatch.setenv("DESPIECE_USE_CHORDS", "false")
    import importlib, src.main as main_mod
    importlib.reload(main_mod)

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
        import soundfile as sf
        sf.write(dest_path, mono, sr, format="WAV")

    monkeypatch.setattr(main_mod, "download_to_path", fake_download)
    monkeypatch.setattr(main_mod, "upload_from_path", lambda local_path, key: None)
    monkeypatch.setattr(main_mod, "separate_stems", lambda input_path, tmpdir: stems_data)
    monkeypatch.setattr(main_mod, "separate_roformer_vocals", lambda a, sr: a.copy())
    monkeypatch.setattr(main_mod, "separate_roformer_bass", lambda a, sr: a.copy())
    monkeypatch.setattr(main_mod, "ensemble_free_model", lambda: None)
    monkeypatch.setattr(main_mod, "tag_clip", lambda a, sr, top_k=5: [("Pad", 0.8)])
    monkeypatch.setattr(main_mod, "tagging_free_model", lambda: None)
    monkeypatch.setattr(
        main_mod, "encode_mp3",
        lambda audio, sr, path, bitrate_kbps=128: open(path, "wb").close(),
    )
    monkeypatch.setattr(
        main_mod, "detect_regions",
        lambda audio, sr, beat_grid, beats_per_bar=4: [
            {"start_sec": 0.0, "end_sec": 1.0, "envelope": [[0.0, 0.5]]}
        ],
    )
    monkeypatch.setattr(
        main_mod, "detect_drum_hits",
        lambda audio, sr: {
            "kick": [], "snare": [], "hihat": [], "cymbal": [], "unknown": [],
        },
    )

    from fastapi.testclient import TestClient
    client = TestClient(main_mod.app)
    resp = client.post(
        "/analyze", json={"song_id": "abc-123", "storage_key": "songs/abc-123.wav"}
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["analysis_version"] == 3
    assert "chords" not in body

    monkeypatch.setenv("DESPIECE_USE_CHORDS", "true")
    importlib.reload(main_mod)
```

Also update the existing `test_analyze_returns_version_3_and_sub_label` to expect v4 by default (since drum hits + chord track are both on). Rename it to `test_analyze_returns_version_4_and_sub_label` and change `assert body["analysis_version"] == 3` to `assert body["analysis_version"] == 4`. Add a `detect_chords` monkeypatch returning one chord entry so the response carries v4.

- [ ] **Step 2: Run, expect failures**

```bash
cd audio-service && python3 -m pytest tests/test_main_integration.py -v
```

- [ ] **Step 3: Modify `audio-service/src/main.py`**

Apply five changes:

**A. Import `detect_chords`** near the existing `.analysis.*` imports:

```python
from .analysis.chord_track import detect_chords
```

**B. Add a third version constant.** Replace the existing two constants with:

```python
ANALYSIS_VERSION_BASE = 2
ANALYSIS_VERSION_WITH_HITS = 3
ANALYSIS_VERSION_WITH_CHORDS = 4
```

**C. Add the env flag** after the existing `USE_DRUM_HITS` line:

```python
USE_CHORDS = _env_flag("DESPIECE_USE_CHORDS", True)
```

**D. Run chord detection.** After the drum-hits block (the `if USE_DRUM_HITS and ...` block ending with the `try / except / pass` for `stems["drums"]["hits"]`), and before the `version = ...` computation, insert:

```python
chords: list[dict] = []
if USE_CHORDS and bar_grid and len(bar_grid) >= 2:
    try:
        present = [s for s in ("bass", "guitar", "piano", "other") if s in stems_data]
        if present:
            ref_sr = stems_data[present[0]][1]
            min_len = min(stems_data[s][0].shape[0] for s in present)
            harmonic_mix = np.zeros(min_len, dtype=np.float32)
            for s in present:
                harmonic_mix += stems_data[s][0][:min_len].astype(np.float32)
            chords = detect_chords(harmonic_mix, ref_sr, bar_grid)
    except Exception:
        traceback.print_exc()
        chords = []
```

**E. Update the version selection** to prefer v4 when chords were attached, fall back to v3 if drum hits were attached, otherwise v2:

```python
        if chords:
            version = ANALYSIS_VERSION_WITH_CHORDS
        elif "hits" in stems.get("drums", {}):
            version = ANALYSIS_VERSION_WITH_HITS
        else:
            version = ANALYSIS_VERSION_BASE

        response: dict = {
            "analysis_version": version,
            "bpm": bpm,
            "key": key,
            "duration_sec": round(duration_sec, 3),
            "beat_grid": beat_grid,
            "bar_grid": bar_grid,
            "sections": sections,
            "stems": stems,
        }
        if chords:
            response["chords"] = chords
        return response
```

This replaces the existing `return {...}` block at the end of `analyze`.

- [ ] **Step 4: Run, expect all 4 integration tests PASS**

```bash
cd audio-service && python3 -m pytest tests/test_main_integration.py -v
```

- [ ] **Step 5: Full audio-service suite**

```bash
cd audio-service && python3 -m pytest tests/ -v
```

Expected: all Phase A + B2 + B1 tests pass. Pre-existing ffmpeg + MinIO failures unrelated.

- [ ] **Step 6: Commit**

```bash
git add audio-service/src/main.py audio-service/tests/test_main_integration.py
git commit -m "feat(audio-service): wire detect_chords into /analyze, bump version to 4"
```

---

## Task 5: `env.example` documentation

**Files:**
- Modify: `audio-service/env.example`

- [ ] **Step 1: Append the three new env vars**

Append to `audio-service/env.example`:

```env

# Chord track (Phase B1) — toggle + tune.
DESPIECE_USE_CHORDS=true
DESPIECE_CHORD_NO_CHORD_THRESHOLD=0.3
DESPIECE_CHORD_SELF_TRANSITION=0.9
```

- [ ] **Step 2: Commit**

```bash
git add audio-service/env.example
git commit -m "docs(audio-service): document Phase B1 chord-track env vars"
```

---

## Task 6: `ChordBar` frontend component

**Files:**
- Create: `frontend/src/components/ChordBar.tsx`
- Create: `frontend/src/components/__tests__/ChordBar.test.tsx`

- [ ] **Step 1: Write failing tests**

Create `frontend/src/components/__tests__/ChordBar.test.tsx`:

```tsx
import { render, screen } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { ChordBar } from '../ChordBar';
import type { Chord } from '@/types/song';

const chords: Chord[] = [
  { start_sec: 0,  end_sec: 8,  label: 'Am' },
  { start_sec: 8,  end_sec: 12, label: 'F'  },
  { start_sec: 12, end_sec: 16, label: 'C'  },
  { start_sec: 16, end_sec: 24, label: 'N'  },
];

const palette: Record<string, string> = {
  C: '#ef4444', 'C#': '#f97316', D: '#f59e0b', 'D#': '#eab308',
  E: '#84cc16', F: '#22c55e', 'F#': '#14b8a6', G: '#06b6d4',
  'G#': '#3b82f6', A: '#8b5cf6', 'A#': '#a855f7', B: '#ec4899',
};
const noChordColor = '#475569';

describe('ChordBar', () => {
  it('renders one block per chord', () => {
    render(
      <ChordBar
        chords={chords}
        durationSec={24}
        width={1200}
        rootColors={palette}
        noChordColor={noChordColor}
        onChordClick={() => {}}
      />
    );
    const blocks = screen.getAllByTestId('chord-block');
    expect(blocks).toHaveLength(4);
  });

  it('positions blocks by start_sec / durationSec', () => {
    render(
      <ChordBar
        chords={chords}
        durationSec={24}
        width={1200}
        rootColors={palette}
        noChordColor={noChordColor}
        onChordClick={() => {}}
      />
    );
    const blocks = screen.getAllByTestId('chord-block');
    expect(blocks[0]).toHaveStyle({ left: '0px',   width: '400px' });
    expect(blocks[1]).toHaveStyle({ left: '400px', width: '200px' });
    expect(blocks[2]).toHaveStyle({ left: '600px', width: '200px' });
    expect(blocks[3]).toHaveStyle({ left: '800px', width: '400px' });
  });

  it('major chord uses root color at full opacity (no overlay)', () => {
    render(
      <ChordBar
        chords={[{ start_sec: 0, end_sec: 4, label: 'C' }]}
        durationSec={4}
        width={400}
        rootColors={palette}
        noChordColor={noChordColor}
        onChordClick={() => {}}
      />
    );
    const block = screen.getByTestId('chord-block');
    expect(block).toHaveStyle({ background: 'rgb(239, 68, 68)' });
    expect(screen.queryByTestId('chord-block-overlay')).not.toBeInTheDocument();
  });

  it('minor chord uses root color + 30% black overlay', () => {
    render(
      <ChordBar
        chords={[{ start_sec: 0, end_sec: 4, label: 'Am' }]}
        durationSec={4}
        width={400}
        rootColors={palette}
        noChordColor={noChordColor}
        onChordClick={() => {}}
      />
    );
    const block = screen.getByTestId('chord-block');
    expect(block).toHaveStyle({ background: 'rgb(139, 92, 246)' });
    const overlay = screen.getByTestId('chord-block-overlay');
    expect(overlay).toBeInTheDocument();
  });

  it('"N" chord uses noChordColor and renders no label text', () => {
    render(
      <ChordBar
        chords={[{ start_sec: 0, end_sec: 4, label: 'N' }]}
        durationSec={4}
        width={400}
        rootColors={palette}
        noChordColor={noChordColor}
        onChordClick={() => {}}
      />
    );
    const block = screen.getByTestId('chord-block');
    expect(block).toHaveStyle({ background: 'rgb(71, 85, 105)' });
    expect(screen.queryByText('N')).not.toBeInTheDocument();
  });

  it('click on a block fires onChordClick with start_sec', () => {
    const spy = vi.fn();
    render(
      <ChordBar
        chords={chords}
        durationSec={24}
        width={1200}
        rootColors={palette}
        noChordColor={noChordColor}
        onChordClick={spy}
      />
    );
    const blocks = screen.getAllByTestId('chord-block');
    blocks[1].click();
    expect(spy).toHaveBeenCalledWith(8);
  });
});
```

- [ ] **Step 2: Run, expect failure**

```bash
cd /Users/matiasagustinmaldonado/Coding/despiece/.claude/worktrees/phase-b1-chord-track/frontend && npx vitest run src/components/__tests__/ChordBar.test.tsx
```

- [ ] **Step 3: Create `frontend/src/components/ChordBar.tsx`**

```tsx
import { Chord, ChordLabel } from '@/types/song';

interface Props {
  chords: Chord[];
  durationSec: number;
  width: number;
  height?: number;
  rootColors: Record<string, string>;
  noChordColor: string;
  onChordClick?: (startSec: number) => void;
}

const NO_CHORD: ChordLabel = 'N';

function getRoot(label: ChordLabel): string {
  if (label === NO_CHORD) return NO_CHORD;
  return label.endsWith('m') ? label.slice(0, -1) : label;
}

function isMinor(label: ChordLabel): boolean {
  return label !== NO_CHORD && label.endsWith('m');
}

export function ChordBar({
  chords,
  durationSec,
  width,
  height = 28,
  rootColors,
  noChordColor,
  onChordClick,
}: Props) {
  return (
    <div
      className="relative bg-muted/20 rounded overflow-hidden"
      style={{ width, height }}
    >
      {chords.map((chord, i) => {
        const left = (chord.start_sec / durationSec) * width;
        const blockWidth = ((chord.end_sec - chord.start_sec) / durationSec) * width;
        const root = getRoot(chord.label);
        const isNoChord = chord.label === NO_CHORD;
        const background = isNoChord ? noChordColor : rootColors[root] ?? noChordColor;
        const minor = isMinor(chord.label);
        const startWhole = Math.round(chord.start_sec);
        const endWhole = Math.round(chord.end_sec);
        return (
          <div
            key={i}
            data-testid="chord-block"
            onClick={() => onChordClick?.(chord.start_sec)}
            title={`${chord.label} · ${startWhole}s – ${endWhole}s`}
            style={{
              position: 'absolute',
              left: `${left}px`,
              top: 0,
              width: `${blockWidth}px`,
              height: '100%',
              background,
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              fontSize: 11,
              fontWeight: 600,
              color: '#fff',
              textShadow: '0 0 2px rgba(0,0,0,0.6)',
              overflow: 'hidden',
              whiteSpace: 'nowrap',
            }}
          >
            {minor && (
              <div
                data-testid="chord-block-overlay"
                style={{
                  position: 'absolute',
                  inset: 0,
                  background: 'rgba(0, 0, 0, 0.3)',
                  pointerEvents: 'none',
                }}
              />
            )}
            {!isNoChord && (
              <span
                style={{
                  position: 'relative',
                  zIndex: 1,
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  maxWidth: '100%',
                  padding: '0 4px',
                }}
              >
                {chord.label}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
```

- [ ] **Step 4: Run, expect 6 PASS**

```bash
cd /Users/matiasagustinmaldonado/Coding/despiece/.claude/worktrees/phase-b1-chord-track/frontend && npx vitest run src/components/__tests__/ChordBar.test.tsx
```

- [ ] **Step 5: Full frontend suite + type-check**

```bash
cd /Users/matiasagustinmaldonado/Coding/despiece/.claude/worktrees/phase-b1-chord-track/frontend && npx tsc --noEmit
cd /Users/matiasagustinmaldonado/Coding/despiece/.claude/worktrees/phase-b1-chord-track/frontend && npx vitest run
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src/components/ChordBar.tsx frontend/src/components/__tests__/ChordBar.test.tsx
git commit -m "feat(frontend): ChordBar component renders per-bar chord blocks"
```

---

## Task 7: `SongTimeline` integration — chord row + legend + v4 notice

**Files:**
- Modify: `frontend/src/components/SongTimeline.tsx`
- Create: `frontend/src/components/__tests__/SongTimeline.v4.test.tsx`

- [ ] **Step 1: Write failing tests**

Create `frontend/src/components/__tests__/SongTimeline.v4.test.tsx`:

```tsx
import { render, screen } from '@testing-library/react';
import { describe, it, expect } from 'vitest';
import { SongTimeline } from '../SongTimeline';
import type { AnalysisResult, Chord } from '@/types/song';

function baseAnalysis(version: 1 | 2 | 3 | 4, chords?: Chord[]): AnalysisResult {
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
      other:  { audio_key: 'k', regions: [] },
    },
    analysis_version: version,
    chords,
  };
}

describe('SongTimeline v4', () => {
  it('v3 with no chords → soft notice mentions both drum sub-rows and chord track', () => {
    render(<SongTimeline songId="abc" analysis={baseAnalysis(3)} />);
    expect(
      screen.getByText(/newer analysis features available/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText(/drum sub-rows.*chord track/i)
    ).toBeInTheDocument();
  });

  it('v4 with chords renders chord row between sections and stems', () => {
    const chords: Chord[] = [
      { start_sec: 0, end_sec: 30, label: 'Am' },
      { start_sec: 30, end_sec: 60, label: 'F' },
    ];
    render(<SongTimeline songId="abc" analysis={baseAnalysis(4, chords)} />);
    const blocks = screen.getAllByTestId('chord-block');
    expect(blocks).toHaveLength(2);
    expect(screen.queryByText(/newer analysis features available/i)).not.toBeInTheDocument();
  });

  it('v4 with no chords → no chord row, no chord legend', () => {
    render(<SongTimeline songId="abc" analysis={baseAnalysis(4)} />);
    expect(screen.queryByTestId('chord-block')).not.toBeInTheDocument();
    expect(screen.queryByText(/chord roots:/i)).not.toBeInTheDocument();
  });

  it('chord legend renders only when chords array non-empty', () => {
    const chords: Chord[] = [{ start_sec: 0, end_sec: 60, label: 'C' }];
    render(<SongTimeline songId="abc" analysis={baseAnalysis(4, chords)} />);
    expect(screen.getByText(/chord roots:/i)).toBeInTheDocument();
  });
});
```

- [ ] **Step 2: Run, expect failures**

```bash
cd /Users/matiasagustinmaldonado/Coding/despiece/.claude/worktrees/phase-b1-chord-track/frontend && npx vitest run src/components/__tests__/SongTimeline.v4.test.tsx
```

- [ ] **Step 3: Modify `frontend/src/components/SongTimeline.tsx`**

Apply six changes.

**A. Import additions.** Add `Chord` and `ChordLabel` to the existing import:

```ts
import { AnalysisResult, Chord, ChordLabel, DrumHitClass, DrumHits, SubLabel } from '@/types/song';
```

Import the new component:

```ts
import { ChordBar } from './ChordBar';
```

**B. Add the chord palette + height constant** below the existing `DRUM_HIT_COLORS`:

```ts
const CHORD_ROOT_COLORS: Record<string, string> = {
  C:    '#ef4444',
  'C#': '#f97316',
  D:    '#f59e0b',
  'D#': '#eab308',
  E:    '#84cc16',
  F:    '#22c55e',
  'F#': '#14b8a6',
  G:    '#06b6d4',
  'G#': '#3b82f6',
  A:    '#8b5cf6',
  'A#': '#a855f7',
  B:    '#ec4899',
};
const NO_CHORD_COLOR = '#475569';
const CHORD_HEIGHT = 28;
```

**C. Bump version constant** — change `PREFERRED_ANALYSIS_VERSION` from 3 to 4:

```ts
const PREFERRED_ANALYSIS_VERSION = 4;
```

**D. Update the soft-notice banner text** (the JSX block currently saying "Drum sub-rows are available — re-analyze to see kick / snare / hi-hat / cymbal per hit."). Replace its inner text with:

```tsx
Newer analysis features available — re-analyze for drum sub-rows + chord track.
```

**E. Render the chord row** between the existing `<SectionBar>` block and the `STEM_ORDER.filter(...).map(...)` block. Right after the section bar's parent `<div className="flex items-center">` closes, insert:

```tsx
{analysis.chords && analysis.chords.length > 0 && (
  <div className="flex items-center mt-1">
    <div
      style={{ width: LABEL_WIDTH }}
      className="text-xs text-muted-foreground pr-2 text-right"
    >
      Chords
    </div>
    <ChordBar
      chords={analysis.chords}
      durationSec={analysis.duration_sec}
      width={timelineWidth}
      height={CHORD_HEIGHT}
      rootColors={CHORD_ROOT_COLORS}
      noChordColor={NO_CHORD_COLOR}
      onChordClick={handleRegionClick}
    />
  </div>
)}
```

**F. Render the chord legend** at the very bottom, after the existing drum-hit legend:

```tsx
{analysis.chords && analysis.chords.length > 0 && (
  <div className="flex gap-3 flex-wrap text-xs text-muted-foreground">
    <span className="font-medium">Chord roots:</span>
    {(['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'] as const).map((root) => (
      <span key={root} className="flex items-center gap-1">
        <span
          className="inline-block w-3 h-3 rounded-sm"
          style={{ background: CHORD_ROOT_COLORS[root] }}
        />
        {root}
      </span>
    ))}
  </div>
)}
```

- [ ] **Step 4: Run new tests, expect PASS**

```bash
cd /Users/matiasagustinmaldonado/Coding/despiece/.claude/worktrees/phase-b1-chord-track/frontend && npx vitest run src/components/__tests__/SongTimeline.v4.test.tsx
```

- [ ] **Step 5: Adjust the prior Phase B2 SongTimeline test** if the soft-notice text matcher fails because the text changed.

In `frontend/src/components/__tests__/SongTimeline.version.test.tsx` (Phase A test from earlier), if any test references `/drum sub-rows are available/i`, update the matcher to `/newer analysis features available/i` to match the new banner copy. Run the full vitest suite after this change.

```bash
cd /Users/matiasagustinmaldonado/Coding/despiece/.claude/worktrees/phase-b1-chord-track/frontend && npx vitest run
```

- [ ] **Step 6: Type-check**

```bash
cd /Users/matiasagustinmaldonado/Coding/despiece/.claude/worktrees/phase-b1-chord-track/frontend && npx tsc --noEmit
```

Expected: zero errors.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/SongTimeline.tsx \
        frontend/src/components/__tests__/SongTimeline.v4.test.tsx \
        frontend/src/components/__tests__/SongTimeline.version.test.tsx
git commit -m "feat(frontend): SongTimeline chord row + v4 notice + chord legend"
```

---

## Task 8: Final cross-task sweep

No new files. Verify nothing drifted.

- [ ] **Step 1: Branch-wide test sweep**

```bash
cd /Users/matiasagustinmaldonado/Coding/despiece/.claude/worktrees/phase-b1-chord-track/backend && npm test
cd /Users/matiasagustinmaldonado/Coding/despiece/.claude/worktrees/phase-b1-chord-track/frontend && npx vitest run
cd /Users/matiasagustinmaldonado/Coding/despiece/.claude/worktrees/phase-b1-chord-track/audio-service && python3 -m pytest tests/ -v
```

Expected:
- backend: 16/16
- frontend: prior 36 + ChordBar 6 + SongTimeline.v4 4 = 46 passing
- audio-service: prior tests + chord_track 12 + 2 new integration tests + 1 renamed = all green except pre-existing ffmpeg + MinIO failures

- [ ] **Step 2: Type-check both TS packages**

```bash
cd /Users/matiasagustinmaldonado/Coding/despiece/.claude/worktrees/phase-b1-chord-track/backend && npm run build
cd /Users/matiasagustinmaldonado/Coding/despiece/.claude/worktrees/phase-b1-chord-track/frontend && npx tsc --noEmit
```

- [ ] **Step 3: Sanity greps**

```bash
grep -rn 'ANALYSIS_VERSION_WITH_CHORDS\|USE_CHORDS' /Users/matiasagustinmaldonado/Coding/despiece/.claude/worktrees/phase-b1-chord-track/audio-service/src
grep -rn 'CHORD_ROOT_COLORS\|PREFERRED_ANALYSIS_VERSION' /Users/matiasagustinmaldonado/Coding/despiece/.claude/worktrees/phase-b1-chord-track/frontend/src
```

Expected:
- `ANALYSIS_VERSION_WITH_CHORDS` appears in `main.py`.
- `USE_CHORDS` appears in `main.py`.
- `CHORD_ROOT_COLORS` appears in `SongTimeline.tsx`.
- `PREFERRED_ANALYSIS_VERSION` appears once in `SongTimeline.tsx` with value `4`.

No `PREFERRED_ANALYSIS_VERSION = 3` should remain.

- [ ] **Step 4: No commit needed if sweep is clean**

If anything fails, fix it before declaring the branch ready.

---

## Self-review notes

- **Spec coverage:**
  - Spec § Algorithm — Task 2 (chroma + templates + Viterbi) + Task 3 (per-bar + merge).
  - Spec § Env vars (`DESPIECE_USE_CHORDS`, `DESPIECE_CHORD_NO_CHORD_THRESHOLD`, `DESPIECE_CHORD_SELF_TRANSITION`) — Task 2 reads the two thresholds; Task 4 reads `USE_CHORDS`; Task 5 documents all three in env.example.
  - Spec § JSON shape — Task 4 emits `chords` field; integration tests assert it.
  - Spec § Backend (Node) types — Task 1.
  - Spec § Frontend types — Task 1.
  - Spec § `ChordBar` component — Task 6 (props, palette, minor overlay, no-chord rendering, click).
  - Spec § `SongTimeline` integration — Task 7 (palette, version constant bump, v4 notice text, chord-row placement, chord legend).
  - Spec § Layout impact (chord row 28 px above stems) — Task 7's `CHORD_HEIGHT = 28`.
  - Spec § Error handling — Task 2 guards empty audio + short bar_grid; Task 4 wraps `detect_chords` in try/except and falls back to v3/v2 if `chords` empty.
  - Spec § Testing — coverage spread across Tasks 2, 3, 4, 6, 7.
  - Spec § `librosa.sequence.viterbi` usage — Task 2 calls it with `(scores: float64, trans: float64)` consistent with librosa 0.10 API.

- **Placeholder scan:** none. Every step contains either runnable code or a concrete command. The "synthetic A minor" test note explains how to retune the fixture if the cosine match prefers C — the engineer has a concrete fallback rather than a TODO.

- **Type consistency:**
  - `Chord` and `ChordLabel` declared in Task 1 and used unchanged in Tasks 4, 6, 7.
  - `detect_chords` signature `(audio: np.ndarray, sr: int, bar_grid: Sequence[float]) -> list[dict]` is consistent across Task 2 (stub), Task 3 (implementation), Task 4's monkeypatch signatures.
  - `_idx_to_label` and `_build_templates` are private helpers declared in Task 2 and consumed only internally.
  - `CHORD_ROOT_COLORS` defined once (Task 7, `SongTimeline.tsx`) and passed via prop into `ChordBar` (Task 6). The legend renders 12 root letters in a fixed order matching the palette keys.
  - `ANALYSIS_VERSION_BASE / WITH_HITS / WITH_CHORDS` constants live in `main.py`; `PREFERRED_ANALYSIS_VERSION = 4` lives in `SongTimeline.tsx`. These are parallel constants in independent systems — by design.

- **Spec § Test 7 (`test_no_chord_threshold_raised_forces_unknown`):** the test reloads `chord_track` after setting `DESPIECE_CHORD_NO_CHORD_THRESHOLD=1.5` so the module-level constant picks up the new value. The reload-then-reset pattern mirrors the Phase B2 confidence-floor test (`test_low_confidence_goes_to_unknown`).
