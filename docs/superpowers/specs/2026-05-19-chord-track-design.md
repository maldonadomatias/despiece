# Chord Track (Phase B1) — Design Spec

**Status:** Draft for review
**Phase:** B1 of Phase B (B1 chord track, B2 drum hits — done, B3 bassline notes, B4 sections v2)
**Date:** 2026-05-19
**Predecessor:** Phase A (clean harmonic stems) + Phase B2 (drum hits) — both merged into main.

## Goal

Detect a song's chord progression and render it as a per-bar chord row between the section bar and the stem rows. 24-class vocabulary (12 major + 12 minor) plus an `"N"` no-chord class. Pure DSP — CQT chroma + 24 templates + Viterbi smoothing. No new ML deps.

## Scope (B1 only)

**In scope:**

- New `audio-service/src/analysis/chord_track.py` — `detect_chords(audio, sr, bar_grid) -> list[dict]`.
- Input audio = **harmonic mix** = sum of `bass + guitar + piano + other` stems (drums + vocals excluded). Built in `main.py` after stems_data is finalized.
- Per-frame CQT chroma → score against 24 chord templates → Viterbi decode with self-transition bias → per-bar dominant label → merge consecutive identical bars.
- Bump `analysis_version` from 3 to 4. Frontend v < 4 shows updated soft notice (mentions both drum sub-rows + chord track).
- New top-level field `chords: Chord[]` in the response JSON.
- New `ChordBar` frontend component. `SongTimeline` renders it between `SectionBar` and the first stem row.
- 12-color root palette + minor-variant overlay. Legend at the bottom when chords render.
- Env flags: `DESPIECE_USE_CHORDS` (master), `DESPIECE_CHORD_NO_CHORD_THRESHOLD`, `DESPIECE_CHORD_SELF_TRANSITION`.

**Out of scope (later phases):**

- 7ths, sus, dim, slash chords, extensions.
- Per-beat or per-half-bar granularity (Phase B1.5 if needed).
- Roman-numeral notation; chord-chart export.
- Key-aware HMM transitions.
- Chord-row playback (solo / loop / etc.).
- B3 bassline notes, B4 sections v2.

## Architecture

```
[audio-service POST /analyze, after Phase A + B2 pipeline]
   ...stems_data populated; per-stem regions built...
   ↓
If USE_CHORDS:
   harmonic_audio = stems_data["bass"][0]
                  + stems_data["guitar"][0]
                  + stems_data["piano"][0]
                  + stems_data["other"][0]
   sr = stems_data["bass"][1]                 # all stems share sr post-Demucs
   chords = detect_chords(harmonic_audio, sr, bar_grid)
   ↓
version = 4 if chords attached else 3
   ↓
Response JSON with new "chords" field
```

`detect_chords` is a pure-numpy + librosa function. No model load. No global state.

### Module layout (audio-service)

```
audio-service/src/analysis/
├── …existing Phase A modules + drum_hits.py from B2…
└── chord_track.py        (NEW — chroma + templates + Viterbi)
```

`main.py` gets a small wedge between region detection and the version computation, gated by `USE_CHORDS`.

## Algorithm

`detect_chords(audio: np.ndarray, sr: int, bar_grid: list[float]) -> list[dict]`

### Step 1 — CQT chroma

```python
chroma = librosa.feature.chroma_cqt(
    y=audio, sr=sr,
    hop_length=2048,
    bins_per_octave=36,
    n_octaves=6,
    fmin=librosa.note_to_hz("C2"),
)
# shape (12, n_frames)
```

Hop 2048 @ 44.1 kHz ≈ 46 ms/frame, ≈ 22 frames/sec.

### Step 2 — 24 chord templates

```python
MAJOR = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
MINOR = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)

# templates[0..11]  = major triads, root C..B
# templates[12..23] = minor triads, root C..B
templates = np.stack(
    [np.roll(MAJOR, i) for i in range(12)] +
    [np.roll(MINOR, i) for i in range(12)]
).astype(np.float32)
templates /= np.linalg.norm(templates, axis=1, keepdims=True)
```

### Step 3 — Per-frame chord scores

```python
chroma_norm = chroma / (np.linalg.norm(chroma, axis=0, keepdims=True) + 1e-9)
scores = templates @ chroma_norm                          # (24, n_frames) cosine similarity
no_chord = np.maximum(0.0, NO_CHORD_THRESHOLD - scores.max(axis=0))[None, :]
scores = np.vstack([scores, no_chord])                    # (25, n_frames)
```

`NO_CHORD_THRESHOLD` defaults to 0.3 (env: `DESPIECE_CHORD_NO_CHORD_THRESHOLD`).

### Step 4 — Viterbi decoding

Transition matrix `T (25, 25)`:

- Self-transition: `T[i, i] = SELF_TRANSITION` (default 0.9, env-tunable).
- All other transitions: `T[i, j] = (1 - SELF_TRANSITION) / 24`.

```python
path = librosa.sequence.viterbi(scores, T)   # (n_frames,)  values 0..24
```

`librosa.sequence.viterbi` accepts probability matrices and returns the most likely state sequence.

### Step 5 — Per-bar aggregation

For each adjacent pair `(bar_grid[i], bar_grid[i+1])`:

```python
frame_times = librosa.frames_to_time(np.arange(n_frames), sr=sr, hop_length=2048)
in_bar = (frame_times >= bar_start) & (frame_times < bar_end)
if not in_bar.any():
    label_idx = 24                    # no-chord
else:
    label_idx = int(np.bincount(path[in_bar], minlength=25).argmax())
```

The dominant Viterbi class within the bar wins (mode aggregation).

### Step 6 — Index → label

```python
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

def _idx_to_label(i: int) -> str:
    if i == 24:
        return "N"
    if i < 12:
        return NOTE_NAMES[i]
    return NOTE_NAMES[i - 12] + "m"
```

### Step 7 — Merge consecutive identical bars

```python
out: list[dict] = []
for bar_idx in range(len(bar_grid) - 1):
    label = _idx_to_label(per_bar_idx[bar_idx])
    start = float(bar_grid[bar_idx])
    end = float(bar_grid[bar_idx + 1])
    if out and out[-1]["label"] == label:
        out[-1]["end_sec"] = round(end, 3)
    else:
        out.append({
            "start_sec": round(start, 3),
            "end_sec": round(end, 3),
            "label": label,
        })
return out
```

### Step 8 — Edge cases

- `audio.size == 0` → return `[]`.
- `len(bar_grid) < 2` → return `[]`.
- All frames classified as `"N"` → emit one merged `{0, duration, "N"}` entry.

### Env vars

| Var | Default | Effect |
|---|---|---|
| `DESPIECE_USE_CHORDS` | `true` | Master switch. False → step skipped, `chords` absent, version stays at 3. |
| `DESPIECE_CHORD_NO_CHORD_THRESHOLD` | `0.3` | Below this max template score per frame → `"N"`. Higher = stricter (more `"N"`). |
| `DESPIECE_CHORD_SELF_TRANSITION` | `0.9` | Viterbi self-transition probability. Higher = smoother / fewer chord changes. |

### Runtime + memory budget (5-min song)

| Stage | Time |
|---|---|
| Build harmonic mix (sum 4 stems) | < 100 ms |
| `chroma_cqt` (hop 2048) | 3-5 s |
| Template scoring + no-chord scoring | < 1 s |
| Viterbi (25 states, ~6600 frames) | < 1 s |
| Per-bar aggregation + merge | < 500 ms |
| **Total B1 overhead** | **< 10 s** |

Memory: negligible. All numpy on in-memory stems.

## JSON shape

```jsonc
{
  "analysis_version": 4,           // bumped from 3
  "bpm": 120.2,
  "key": "A minor",
  "duration_sec": 484.123,
  "beat_grid": [...],
  "bar_grid": [...],
  "sections": [...],
  "chords": [                      // NEW top-level field
    { "start_sec": 0.0,  "end_sec": 8.0,  "label": "Am" },
    { "start_sec": 8.0,  "end_sec": 12.0, "label": "F"  },
    { "start_sec": 12.0, "end_sec": 16.0, "label": "C"  },
    { "start_sec": 16.0, "end_sec": 24.0, "label": "G"  }
  ],
  "stems": { /* unchanged */ }
}
```

**Field constraints:**

- `chords` is top-level (parallel to `sections`), not nested under any stem.
- `label` ∈ exactly 25 values: 12 major roots, 12 minor roots, `"N"`.
- Consecutive bars with the same label are merged into a single entry.
- Coverage is contiguous from `bar_grid[0]` to the last bar boundary (no gaps).
- `chords` field is optional from the v2 reader's perspective. v3 readers also tolerate it.

## Backend (Node) changes

`backend/src/domain/song.ts`:

```ts
export type AnalysisVersion = 1 | 2 | 3 | 4;

export type ChordLabel =
  | 'C' | 'C#' | 'D' | 'D#' | 'E' | 'F' | 'F#' | 'G' | 'G#' | 'A' | 'A#' | 'B'
  | 'Cm' | 'C#m' | 'Dm' | 'D#m' | 'Em' | 'Fm' | 'F#m' | 'Gm' | 'G#m' | 'Am' | 'A#m' | 'Bm'
  | 'N';

export interface Chord {
  start_sec: number;
  end_sec: number;
  label: ChordLabel;
}

export interface AnalysisResult {
  // existing fields…
  analysis_version: AnalysisVersion;
  chords?: Chord[];
}
```

`src/workers/jobWorker.ts`, `src/routes/songs.ts`, `src/services/songService.ts`: no changes.

No DB migration.

## Frontend changes

### Types

`frontend/src/types/song.ts` mirrors the backend additions exactly: widens `AnalysisVersion` to `1 | 2 | 3 | 4`, adds `ChordLabel` union, `Chord` interface, optional `chords?: Chord[]` on `AnalysisResult`.

### New component

`frontend/src/components/ChordBar.tsx`:

```ts
interface Props {
  chords: Chord[];
  durationSec: number;
  width: number;       // timelineWidth
  height?: number;     // default 28
  onChordClick?: (startSec: number) => void;
}
```

- Renders one horizontal row containing absolutely-positioned chord blocks.
- For each chord:
  - `left: (chord.start_sec / durationSec) * width`
  - `width: ((chord.end_sec - chord.start_sec) / durationSec) * width`
  - Fill: chord-root color from `CHORD_ROOT_COLORS`; minor chords overlay 30% black for darker variant.
  - `"N"` chord → slate fill, no text label.
  - Label text centered, truncated with ellipsis if narrower than text.
  - `title` tooltip: `"<label> · <start>s – <end>s"` (start/end rounded to whole seconds).
  - Click → `onChordClick?.(chord.start_sec)`.

### Modified `SongTimeline.tsx`

1. Add the chord palette:

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
```

2. Widen `PREFERRED_ANALYSIS_VERSION` from 3 to 4.

3. Update the soft-notice banner text (only when `version < 4`):
   > Newer analysis features available — re-analyze for drum sub-rows + chord track.

4. Insert `<ChordBar>` between the `<SectionBar>` row and the first `<StemTrack>`:

```tsx
<div className="flex items-center mt-1">
  <div style={{ width: LABEL_WIDTH }} className="text-xs text-muted-foreground pr-2 text-right">
    Chords
  </div>
  {analysis.chords && analysis.chords.length > 0 && (
    <ChordBar
      chords={analysis.chords}
      durationSec={analysis.duration_sec}
      width={timelineWidth}
      height={CHORD_HEIGHT}
      onChordClick={handleRegionClick}
    />
  )}
</div>
```

5. Add a chord-row legend at the bottom (after the drum-hit legend), only when `analysis.chords && analysis.chords.length > 0`:

```tsx
<div className="flex gap-3 flex-wrap text-xs text-muted-foreground">
  <span className="font-medium">Chord roots:</span>
  {['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'].map((root) => (
    <span key={root} className="flex items-center gap-1">
      <span className="inline-block w-3 h-3 rounded-sm"
            style={{ background: CHORD_ROOT_COLORS[root] }} />
      {root}
    </span>
  ))}
</div>
```

### Layout impact

```
Sections row (28px)
↓
Chords row   (28px)   ← NEW
↓
vocals row   (48px)
drums row    (48px)
drums.kick   (24px)
…
```

Total chrome above stems grows from 28 px to 56 px. Acceptable.

## Configuration summary

| Var | Default | Read by |
|---|---|---|
| `DESPIECE_USE_CHORDS` | `true` | `main.py` |
| `DESPIECE_CHORD_NO_CHORD_THRESHOLD` | `0.3` | `chord_track.py` |
| `DESPIECE_CHORD_SELF_TRANSITION` | `0.9` | `chord_track.py` |

All documented in `audio-service/env.example`.

## Error handling

| Failure mode | Behavior |
|---|---|
| `detect_chords` raises | Log via `traceback.print_exc()`. `chords` field omitted. Version falls back to 3. Job completes. |
| `bar_grid` has < 2 entries | Return `[]`. `chords: []` emitted. Frontend renders no chord row. |
| Harmonic stems missing | Build mix from whatever stems exist. If still empty array, return `[]`. |
| `chroma_cqt` raises on edge audio | Caught, returns `[]`. |
| `DESPIECE_USE_CHORDS=false` | Step skipped. `chords` absent. Version stays at 3. |

## Testing

### audio-service (pytest)

`audio-service/tests/test_chord_track.py`:

1. **Synthetic C major** — sum of sines at C4 (261.63 Hz), E4 (329.63 Hz), G4 (392.00 Hz) for 4 s, bar_grid `[0.0, 2.0, 4.0]` → `chords` contains 1 merged `"C"` entry spanning 0–4 s (or 2 unmerged entries both labeled `"C"`; merge logic must collapse to 1).
2. **Synthetic A minor** — sines at A3 (220 Hz), C4 (261.63 Hz), E4 (329.63 Hz), 4 s → `"Am"`.
3. **Silence** — zero audio, bar_grid `[0, 2, 4]` → either `[]` or `[{0, 4, "N"}]`.
4. **Chord change at the bar boundary** — 4 s of C major triad concatenated with 4 s of G major triad, bar_grid `[0, 2, 4, 6, 8]` → 2 merged entries `{0, 4, "C"}` + `{4, 8, "G"}`.
5. **Merge consecutive identical bars** — 4 s of C across 4 one-second bars (`[0, 1, 2, 3, 4]`) → single entry `{0, 4, "C"}`.
6. **Viterbi smooths flicker** — synthetic mix where frame-level argmax flips C/F every 200 ms but C is dominant overall → Viterbi locks the bar to `"C"`.
7. **No-chord threshold escalated** — `monkeypatch.setenv("DESPIECE_CHORD_NO_CHORD_THRESHOLD", "1.5")`, reload `chord_track`, run on a strong C major triad → all bars labeled `"N"`.

### audio-service main_integration (extend existing)

8. Mock `detect_chords` returns `[{"start_sec": 0.0, "end_sec": 1.0, "label": "Am"}]`. Assert `body["analysis_version"] == 4` and `body["chords"][0]["label"] == "Am"`.
9. With `DESPIECE_USE_CHORDS=false`, response has no `chords` key and `analysis_version` stays at 3 (with drum-hits still enabled; if drum-hits is also disabled, version is 2).

### frontend (vitest)

`frontend/src/components/__tests__/ChordBar.test.tsx`:

10. Renders one block per chord at correct `left` / `width` positions for a known `durationSec`.
11. Major chord block uses the root color at full opacity.
12. Minor chord block uses the root color with a 30% black overlay (assert via inline style or computed style).
13. `"N"` chord uses the slate `NO_CHORD_COLOR` and renders no text label.
14. Click on a block fires `onChordClick(chord.start_sec)`.

`frontend/src/components/__tests__/SongTimeline.v4.test.tsx`:

15. `analysis_version === 3` with no `chords` → soft-notice banner reads "Newer analysis features available — re-analyze for drum sub-rows + chord track."; chord row not rendered.
16. `analysis_version === 4` with `chords` non-empty → chord row renders between `SectionBar` and the first `StemTrack`.
17. `analysis_version === 4` with `chords === []` → no chord row, no chord legend.
18. Chord legend renders only when at least one chord exists.

### Manual smoke (post-merge)

1. Reset analysis (`npx tsx backend/src/scripts/resetAnalysis.ts`) and re-analyze a pop/rock reference track with a known progression (e.g. C-Am-F-G or Am-F-C-G).
2. Chord row renders blocks with the expected labels in the right order.
3. Chord boundaries roughly align with section boundaries on visual inspection.
4. Click a chord block → transport seeks to that bar.
5. Old v3 song shows the updated soft notice; timeline still functional.

## Risks + mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| 24-class vocabulary misses 7ths / sus / dim / slash chords common in jazz, funk, neo-soul → fallback to closest triad | Medium | Acknowledged. Phase B1.5 widens vocab if user feedback warrants. |
| Drumless harmonic mix still contains percussive bleed (snare into piano, kick into bass) → chroma noise | Low | Phase A's residual subtract already removes most bass leakage. Acceptable for v4. |
| Viterbi self-transition over-smooths fast chord changes (1 chord per beat in fast pop) | Low | Default 0.9 allows ~10 % of frames to switch per second. Tunable via `DESPIECE_CHORD_SELF_TRANSITION`. |
| Songs without a usable `bar_grid` (free-tempo, no detected beat) → empty `chords` | Low | Acceptable. v3 fallback. |
| Chord row visually collides with section bar at narrow window widths | Low | Both rows are 28 px; labels truncate with ellipsis. |
| `librosa.sequence.viterbi` API drift between versions | Low | Pin librosa 0.10.2 (already in `requirements.txt`). |

## Follow-up phases (not in scope)

- **Phase B1.5** — widen vocabulary to 7ths / sus / slash; add key-aware HMM transitions; expose confidence per chord.
- **Phase B3** — bassline note track via `crepe` / `basic-pitch` on the cleaned bass stem.
- **Phase B4** — sections v2 using stem-activity patterns + repetition matrices.
- **Phase B5** — vocal melody contour + lyric alignment.
