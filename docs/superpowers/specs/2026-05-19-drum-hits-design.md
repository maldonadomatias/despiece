# Drum-Hit Classification (Phase B2) — Design Spec

**Status:** Draft for review
**Phase:** B2 of Phase B (B1 chord track, B2 drum hits, B3 bassline notes, B4 sections v2 are independent subprojects)
**Date:** 2026-05-19
**Predecessor:** `2026-05-19-stem-quality-ensemble-design.md` (Phase A — required for clean drum stems)

## Goal

Sub-classify hits inside the Demucs `drums` stem into **kick / snare / hi-hat / cymbal / unknown** via pure DSP (no new ML deps). Render each class as its own indented sub-row beneath the parent drums row on the timeline. Per-hit velocity + per-hit classification confidence emitted so the UI can convey both dynamics and uncertainty.

## Scope (B2 only)

**In scope:**

- New `audio-service/src/analysis/drum_hits.py` — onset detection on the drums stem + per-onset spectral feature vector + rule-based decision tree.
- New top-level field `stems.drums.hits` in the analysis JSON: `{kick, snare, hihat, cymbal, unknown}`, each an array of `{t_sec, velocity, confidence}`.
- Bump `analysis_version` from 2 to 3. The v2 reader still works on v3 JSON (drum hits field is optional from the v2 perspective).
- Pipeline integration: in `audio-service/src/main.py`, after region detection on the drums stem, run `detect_drum_hits` and attach the result. Gated by env flag `DESPIECE_USE_DRUM_HITS=true`.
- Frontend: new `DrumHitRow` component. `StemTrack` extended to render 4 (or 5 if `unknown` non-empty) sub-rows beneath the drums row, each showing per-hit ticks.
- `SongTimeline` gains a soft-notice banner for `analysis_version === 2` ("re-analyze to enable drum sub-rows"). v1 still triggers the hard re-analyze gate from Phase A.
- New legend entry below the existing stem-color and sub-label legends.

**Out of scope (future Phase B subprojects):**

- B1 chord track per bar.
- B3 bassline note pitch tracking.
- B4 section detection v2.
- B5 melody contour + lyric alignment.

Also explicitly out of B2:

- Per-sub-class drum **audio** synthesis (filtering drum stem by frequency band into 4 separate stems). Deferred to a hypothetical B2.5.
- Trained sklearn classifier. The rule-based layer is intentionally a drop-in replacement target for B2.5.
- Solo / mute per sub-class. Parent drums solo/mute is unchanged.
- Toms as a first-class sub-class. Tom hits land in `snare` (low-mid energy) or `unknown` depending on feature signature.

## Architecture

```
[audio-service POST /analyze]
   ...existing Phase A pipeline (Demucs → BS-Roformer ensemble → residual → tagging) ...
   ↓
After per-stem region detection:
   ↓
If USE_DRUM_HITS and drums stem present:
   drums_audio, drums_sr = stems_data["drums"]
   hits = detect_drum_hits(drums_audio, drums_sr)
   stems["drums"]["hits"] = hits
   ↓
analysis_version = 3
   ↓
Response JSON
```

`detect_drum_hits` is a pure-numpy + librosa function. No model load. No global state. No external service. Sequential within the existing `/analyze` call.

### Module placement

```
audio-service/src/analysis/
├── …existing Phase A modules…
└── drum_hits.py        (NEW — onset + features + rule cascade)
```

`main.py` gets two new lines (env flag read + call). No other modules touched.

## Algorithm

`detect_drum_hits(audio: np.ndarray, sr: int) -> dict[str, list[dict]]`

### Step 1 — Onset detection

```python
onset_times = librosa.onset.onset_detect(
    y=audio, sr=sr, units='time', backtrack=True, hop_length=512
)
```

Returns onset timestamps in seconds. `backtrack=True` snaps each onset to the nearest local minimum of the onset envelope, which improves transient alignment for percussive sounds.

If `len(onset_times) == 0`, return five empty arrays.

### Step 2 — Per-onset windows

Two windows per onset are needed: a short one for transient-driven features (energy ratios, centroid, ZCR) and a longer one for the decay measurement.

**Short window** (transient features):
- `start_samp = max(0, int((t - 0.010) * sr))`
- `end_samp = min(len(audio), int((t + 0.050) * sr))`
- `window_short = audio[start_samp:end_samp]`

60 ms wide. Captures the transient + early decay portion.

**Long window** (decay only):
- `decay_start = max(0, int(t * sr))`
- `decay_end = min(len(audio), int((t + 0.400) * sr))`
- `window_decay = audio[decay_start:decay_end]`

400 ms wide. Long enough to measure cymbal decay (typically 200–1000 ms). If the next onset arrives sooner than 400 ms after `t`, the long window is trimmed to `min(decay_end, next_onset_samp)` so we don't read another hit's energy into this hit's decay.

### Step 3 — Feature vector

For each window:

| Feature | Computation | Class signal |
|---|---|---|
| `low_energy` | sum(STFT magnitude where 30 Hz ≤ f ≤ 150 Hz) / total energy | high for kick |
| `mid_energy` | sum(STFT magnitude where 150 Hz < f ≤ 1000 Hz) / total energy | high for snare |
| `hi_energy` | sum(STFT magnitude where 6000 Hz ≤ f ≤ 12000 Hz) / total energy | high for HH and cymbal |
| `centroid` | mean of `librosa.feature.spectral_centroid` over window | low for kick, mid for snare, high for HH/cymbal |
| `zcr` | mean of `librosa.feature.zero_crossing_rate` over window | high for HH, lower for cymbal |
| `decay_ms` | computed on `window_decay`: time in ms for RMS envelope to drop 6 dB from peak. If no 6 dB drop within the window, treat decay as the window length (~400 ms). | short for HH (~30–60 ms), long for cymbal (~150 ms+) |
| `peak_rms` | max(`librosa.feature.rms(window_short)`); used for velocity | per-onset loudness |

The energy ratios, centroid, ZCR, and peak_rms all read `window_short`. Only `decay_ms` reads `window_decay`. Energy ratios use a single `librosa.stft(window_short, n_fft=512, hop_length=128)`. Centroid and ZCR use librosa's frame-level helpers and average over the resulting frames.

### Step 4 — Velocity normalization

After processing all onsets, normalize:

```python
max_peak = max(o["peak_rms"] for o in onsets) or 1.0
for o in onsets:
    o["velocity"] = min(1.0, o["peak_rms"] / max_peak)
```

Velocity is **per-song** — values are relative to the loudest hit in *this* song. This produces consistent dynamics on the UI regardless of overall mix loudness.

### Step 5 — Decision cascade (rule-based classifier)

For each onset's feature vector:

```python
if low_energy > KICK_LOW_THRESHOLD:                   # default 0.5
    label = "kick"
    confidence = low_energy
elif mid_energy > SNARE_MID_THRESHOLD and centroid < 4000:   # default 0.35, 4000 Hz
    label = "snare"
    confidence = mid_energy
elif hi_energy > HIHAT_HI_THRESHOLD and zcr > 0.15 and decay_ms < CYMBAL_DECAY_MS:
    label = "hihat"                                   # default hi=0.4, decay<80ms
    confidence = min(1.0, hi_energy * zcr / 0.15)
elif hi_energy > 0.3 and decay_ms >= CYMBAL_DECAY_MS:
    label = "cymbal"                                  # default decay≥80ms
    confidence = min(1.0, hi_energy * (decay_ms / 200))
else:
    label = "unknown"
    confidence = 1.0 - max(low_energy, mid_energy, hi_energy)
```

Then a confidence floor:

```python
if confidence < CONFIDENCE_MIN:                       # default 0.4
    label = "unknown"
```

Onsets in `unknown` keep their `confidence` value (post-floor, so always < CONFIDENCE_MIN).

### Step 6 — Output shape

```python
return {
    "kick":    [{"t_sec": <float>, "velocity": <0-1>, "confidence": <0-1>}, ...],
    "snare":   [...],
    "hihat":   [...],
    "cymbal":  [...],
    "unknown": [...],
}
```

All numeric fields rounded to 4 decimal places before JSON emission.

### Step 7 — Tunable env vars

| Var | Default | Effect |
|---|---|---|
| `DESPIECE_USE_DRUM_HITS` | `true` | Master switch. False → drum hit detection skipped entirely; `analysis_version` stays at 2. |
| `DESPIECE_DRUMS_KICK_LOW_THRESHOLD` | `0.5` | Min `low_energy` for kick. |
| `DESPIECE_DRUMS_SNARE_MID_THRESHOLD` | `0.35` | Min `mid_energy` for snare. |
| `DESPIECE_DRUMS_HIHAT_HI_THRESHOLD` | `0.4` | Min `hi_energy` for hi-hat. |
| `DESPIECE_DRUMS_CYMBAL_DECAY_MS` | `80` | Decay threshold (ms) separating hi-hat from cymbal. |
| `DESPIECE_DRUMS_CONFIDENCE_MIN` | `0.4` | Below this, the hit is reclassified as `unknown`. |

These exist to let us bisect quality on a per-genre basis without recompiling. They are not exposed in the UI.

### Runtime + memory cost (5-minute song)

| Stage | Time | Memory |
|---|---|---|
| `onset_detect` | ~1 s | negligible |
| Per-onset feature extraction (~500 onsets typical for pop/rock) | ~3-5 s | negligible |
| Decision cascade + velocity normalization | < 100 ms | negligible |
| **Total Phase B2 overhead** | **< 10 s** | **< 100 MiB** |

No new model load. Runs entirely on the in-memory drums stem that already exists from Demucs.

## JSON shape

```jsonc
{
  "analysis_version": 3,             // was 2 in Phase A
  "bpm": 120.2,
  "key": "A minor",
  "duration_sec": 484.123,
  "beat_grid": [...],
  "bar_grid": [...],
  "sections": [...],
  "stems": {
    "drums": {
      "audio_key": "songs/<id>/stems/drums.mp3",
      "regions": [
        { "start_sec": 0, "end_sec": 16, "envelope": [[0, 0.5], ...] }
      ],
      "hits": {
        "kick":    [
          { "t_sec": 0.45, "velocity": 0.82, "confidence": 0.71 },
          { "t_sec": 0.90, "velocity": 0.78, "confidence": 0.69 }
        ],
        "snare":   [
          { "t_sec": 0.50, "velocity": 0.71, "confidence": 0.62 }
        ],
        "hihat":   [
          { "t_sec": 0.225, "velocity": 0.40, "confidence": 0.55 }
        ],
        "cymbal":  [
          { "t_sec": 12.0, "velocity": 0.95, "confidence": 0.66 }
        ],
        "unknown": []
      }
    },
    "vocals": { /* unchanged from v2 */ },
    "bass":   { /* unchanged */ },
    "guitar": { /* unchanged */ },
    "piano":  { /* unchanged */ },
    "other":  { /* unchanged — still carries sub_label on regions */ }
  }
}
```

**Constraints:**

- `hits` appears **only** on the `drums` stem. Other stems omit the field.
- All five class arrays are present even if empty (so the frontend can deterministically decide whether to render the `unknown` row).
- A v2-shape parser ignores the `hits` field and the `analysis_version` value; nothing crashes.

## Backend (Node) changes

`backend/src/domain/song.ts`:

```ts
export type AnalysisVersion = 1 | 2 | 3;   // widened from `1 | 2`

export type DrumHitClass = 'kick' | 'snare' | 'hihat' | 'cymbal' | 'unknown';

export interface DrumHit {
  t_sec: number;
  velocity: number;    // 0-1
  confidence: number;  // 0-1
}

export interface DrumHits {
  kick: DrumHit[];
  snare: DrumHit[];
  hihat: DrumHit[];
  cymbal: DrumHit[];
  unknown: DrumHit[];
}

export interface StemAnalysis {
  audio_key: string | null;
  regions: StemRegion[];
  hits?: DrumHits;     // only present on drums stem
}
```

`src/workers/jobWorker.ts`: no logic change.
`src/routes/songs.ts`: no route changes.
`src/services/songService.ts`: no change.

No DB migration. The new field lives inside `song_analysis.result_json`.

## Frontend changes

### Types

`frontend/src/types/song.ts` mirrors the backend additions exactly (`AnalysisVersion`, `DrumHitClass`, `DrumHit`, `DrumHits`, optional `hits` on `StemAnalysis`).

### New component

`frontend/src/components/DrumHitRow.tsx`:

```ts
interface Props {
  hits: DrumHit[];
  durationSec: number;
  timelineWidth: number;
  color: string;
  rowHeight?: number;      // default 24
  isUnknown?: boolean;     // dashed border on ticks
  onHitClick: (tSec: number) => void;
}
```

- Renders a row at the given `rowHeight` (24px) and `timelineWidth`.
- For each `hit`, renders a vertical tick at `left: (hit.t_sec / durationSec) * timelineWidth`.
- Tick is 2px wide × full row height.
- Tick opacity = `0.4 + 0.6 * hit.velocity` (low-vel = faint, high-vel = bold).
- Tick fill = `color`.
- `isUnknown` → tick gets a 1px dashed outline + slight italic hint via `border-style: dashed`.
- `title` tooltip: `"<class> · vel <pct>% · conf <pct>%"`.
- Click → `onHitClick(hit.t_sec)`.

### Modified `StemTrack.tsx`

When `name === 'drums'` and `stem.hits` is set:

1. Render the regular drums row exactly as today (single row with region blocks).
2. Below it, render up to 5 `DrumHitRow` instances, each indented 16px from the left of the timeline:
   - `kick` → red
   - `snare` → yellow
   - `hihat` → cyan
   - `cymbal` → lime
   - `unknown` → slate (only if `hits.unknown.length > 0`)
3. Each `DrumHitRow` gets a small label in the label column (`kick`, `snare`, etc.), 16px right-indent for visual hierarchy.
4. Parent drums row solo/mute unchanged. Sub-rows have no controls.

Other stems: behavior unchanged. The `hits` prop is undefined for them.

### Modified `SongTimeline.tsx`

```ts
const SUPPORTED_ANALYSIS_VERSION = 2;     // soft floor — render UI
const PREFERRED_ANALYSIS_VERSION = 3;     // soft notice if below

const version = analysis.analysis_version ?? 1;

if (version < SUPPORTED_ANALYSIS_VERSION) {
  return <ReanalyzeRequired />;       // existing Phase A hard gate
}

const showV3Notice = version < PREFERRED_ANALYSIS_VERSION;
```

When `showV3Notice` is true, render a small banner above the timeline:

> Drum sub-rows are available — re-analyze to see kick / snare / hi-hat / cymbal per hit.

Add the drum-hit palette constant:

```ts
const DRUM_HIT_COLORS: Record<DrumHitClass, string> = {
  kick:    '#dc2626',   // red
  snare:   '#facc15',   // yellow
  hihat:   '#22d3ee',   // cyan
  cymbal:  '#a3e635',   // lime
  unknown: '#64748b',   // slate
};
```

Pass `drumHitColors={DRUM_HIT_COLORS}` to `StemTrack`. (StemTrack only uses it when `name === 'drums'`.)

### Legend update

A third legend row below the existing stem-color and sub-label legends, **only when `analysis.stems.drums.hits` exists** AND at least one class has hits:

> Drum hits: ● kick ● snare ● hihat ● cymbal *( ● unknown if any )*

## Configuration summary

| Var | Default | Source |
|---|---|---|
| `DESPIECE_USE_DRUM_HITS` | `true` | `main.py` |
| `DESPIECE_DRUMS_KICK_LOW_THRESHOLD` | `0.5` | `drum_hits.py` |
| `DESPIECE_DRUMS_SNARE_MID_THRESHOLD` | `0.35` | `drum_hits.py` |
| `DESPIECE_DRUMS_HIHAT_HI_THRESHOLD` | `0.4` | `drum_hits.py` |
| `DESPIECE_DRUMS_CYMBAL_DECAY_MS` | `80` | `drum_hits.py` |
| `DESPIECE_DRUMS_CONFIDENCE_MIN` | `0.4` | `drum_hits.py` |

All read at module-import time and documented in `audio-service/env.example`.

## Error handling

| Failure mode | Behavior |
|---|---|
| `detect_drum_hits` raises | Log via `traceback.print_exc()`. `stems["drums"]["hits"]` omitted. Job completes. Frontend renders parent drums row only — no sub-rows. |
| Onset detection returns 0 onsets | All five class arrays empty. `hits` field still present. Frontend renders parent row + no sub-rows (every class has length 0). |
| Per-onset feature extraction raises on one onset | That onset skipped, others continue. Logged. |
| Drums stem missing from `stems_data` | Drum-hit step skipped silently. `analysis_version` stays at 2 in this edge case. (Should not happen — htdemucs_6s always emits drums.) |
| `DESPIECE_USE_DRUM_HITS=false` | Step skipped. `analysis_version` stays at 2. v2 JSON emitted. |
| Frontend receives `hits` field with malformed structure | TypeScript types enforce the shape at compile time; at runtime the optional-chain operators in `DrumHitRow` short-circuit on missing class arrays. |

## Testing

### audio-service (pytest)

`tests/test_drum_hits.py`:

1. **Synthetic kick** — 60 Hz sine windowed by `exp(-t/0.05)` for 200 ms. `low_energy > 0.5`. Expect classified as `kick`.
2. **Synthetic snare** — band-passed white noise (200–2000 Hz) with sharp envelope (5 ms attack, 80 ms decay). `mid_energy > 0.35`, `centroid < 4000`. Expect `snare`.
3. **Synthetic hi-hat** — band-passed white noise (8–12 kHz) with `exp(-t/0.02)` (50 ms decay). High `hi_energy`, high `zcr`, low `decay_ms`. Expect `hihat`.
4. **Synthetic cymbal** — same band but `exp(-t/0.15)` (300 ms decay). High `hi_energy`, lower `zcr`, high `decay_ms`. Expect `cymbal`.
5. **Empty audio** → 5 empty arrays, no crash.
6. **Confidence floor** — synthetic ambiguous hit (energy spread roughly evenly across bands AND short decay so it doesn't fall through to cymbal) → confidence < 0.4 → `unknown`.
7. **Velocity ordering** — two synthetic kicks at different amplitudes → louder one has `velocity == 1.0`, softer one < 1.0.
8. **Multi-class mix** — concatenate the four synthetic hits with 100 ms gaps → each lands in the right class array (4 hits total, 1 per class).

`tests/test_main_integration.py` (extend the existing Phase A test):

9. Mock `detect_drum_hits` to return `{kick: [{t_sec: 0.1, velocity: 0.8, confidence: 0.7}], snare: [], hihat: [], cymbal: [], unknown: []}`. Assert response JSON has `analysis_version: 3` and `stems.drums.hits.kick[0].t_sec == 0.1`.
10. Assert `stems.vocals` and all non-drums stems have no `hits` key.
11. With `DESPIECE_USE_DRUM_HITS=false`, response has `analysis_version: 2` and no `hits` field.

### backend (jest)

`backend/src/domain/__tests__/song.test.ts`:

12. A v2 JSON sample (no `hits`) parses to `StemAnalysis` cleanly — optional field absent.
13. A v3 JSON sample with `hits` parses; `DrumHits.kick[0]` has exactly the 3 expected numeric fields.

### frontend (vitest)

`frontend/src/components/__tests__/DrumHitRow.test.tsx`:

14. Renders N tick `<div>`s for N hits at correct `left` positions.
15. Tick opacity = `0.4 + 0.6 * velocity` (assert via inline style).
16. Click on a tick fires `onHitClick(hit.t_sec)`.
17. `isUnknown` prop adds dashed border to ticks.

`frontend/src/components/__tests__/StemTrack.drums.test.tsx`:

18. `name === 'drums'` with `hits` containing kick/snare/hihat/cymbal but no unknown → 4 sub-rows render.
19. With `hits.unknown.length > 0` → 5 sub-rows render.
20. Other stem names with no `hits` → no sub-rows (just the parent row).

`frontend/src/components/__tests__/SongTimeline.v3.test.tsx`:

21. `analysis_version === 2` → v3 notice banner renders, timeline still renders.
22. `analysis_version === 3` → no v3 notice.
23. `analysis_version === 1` (or missing) → hard re-analyze gate from Phase A (existing test still passes).
24. Drum-hit legend appears only when `analysis.stems.drums.hits` exists AND at least one class array is non-empty.

### Manual smoke (post-merge)

1. `npx tsx backend/src/scripts/resetAnalysis.ts` then re-analyze a pop/rock reference track.
2. Drums row has 4 colored sub-rows (kick/snare/hihat/cymbal) below it.
3. Kicks align audibly with felt downbeats.
4. Snares align with backbeats (2 and 4 in 4/4).
5. Hi-hats appear on 8th-notes (or 16th-notes) where played.
6. Cymbal hits land at crash points.
7. `unknown` row absent or sparse (< 15% of total hits) for typical pop production.
8. An old v2 row shows the soft notice banner; timeline functional.
9. With `DESPIECE_USE_DRUM_HITS=false`, the v3 banner appears (because the song row reports v2) and no sub-rows render.

## Risks + mitigations

| Risk | Severity | Mitigation |
|---|---|---|
| Genre mismatch (metal, EDM, jazz) yields high `unknown` rate or wrong classification. | Medium | Tunable env thresholds. Phase B2.5 can swap the rule layer for a small sklearn classifier without changing the API surface or test fixtures. The `unknown` row makes uncertainty visible rather than hiding it as miscategorization. |
| `librosa.onset.onset_detect` misses ghost notes or rolls. | Low | Conservative onset behavior is fine for visual purposes. If a track shows obvious missing hits, env vars on librosa (`pre_max`, `post_max`, `delta`) can be exposed later. |
| Cymbal vs open-hi-hat confusion (long-decay hi-hat). | Medium | The decay-ms feature is the distinguisher. Borderline cases land in `unknown` rather than wrong row. Tunable via `DESPIECE_DRUMS_CYMBAL_DECAY_MS`. |
| Velocity normalization unstable across songs (one loud kit, one quiet kit). | Low | Velocity is **per-song** normalized (max = 1.0). UI shows relative dynamics, not absolute SPL. |
| 5 sub-rows make the drums area visually dominate the timeline. | Low | Sub-row height = 24px (vs 48px for stems). Total drums area ≈ 168px when all 5 render. Acceptable; a collapsible "expand drum details" toggle can ship in Phase B2.5 if user feedback warrants. |
| `analysis_version: 3` confuses Phase A code paths that only check `< 2`. | Low | All Phase A code uses `< 2` checks (correct). The `< 3` notice is a new branch added in this phase only. |

## Follow-up phases (sketched, not in scope)

- **B2.5** — Swap the rule cascade for a small sklearn classifier (RandomForest, ~50 KB pickled) trained on MDB-Drums or ENST-Drums. Same input features. Same output shape. Drop-in replacement.
- **B2.5** — Per-sub-class audio synthesis: filter the drum stem by frequency band into 4 short stems (`drums_kick.mp3`, `drums_snare.mp3`, …) so per-sub-class solo/mute becomes real audio.
- **B2.5** — Collapsible drum sub-rows (UX setting).
- **B1, B3, B4, B5** — separate Phase B subprojects (see Phase A spec's follow-up phases section).
