# Pro-Grade Stem Quality (Phase A) — Design Spec

**Status:** Draft for review
**Phase:** A of 4 (A: stem quality, B: deeper analysis, C: DAW UX, D: visual polish)
**Date:** 2026-05-19
**Predecessor:** `2026-05-18-logic-pro-mirror-design.md` (P1 foundation: htdemucs_6s + regions + per-stem playback)

## Goal

Lift song-analysis stem quality from "decent but not professional" to a pro-grade baseline by:

1. Replacing the weakest stems (vocals, bass) from `htdemucs_6s` with **BS-Roformer** specialist outputs.
2. Spectrally subtracting residual bass leakage from the guitar and piano stems.
3. Giving the `other` stem semantic structure: each detected region inside `other` is tagged with a dominant sub-label (`lead | pad | synth | strings | fx | other_misc`) via PANNs CNN14 audio tagging.

Phase A is purely an **audio-service + frontend rendering** change. No DB migration, no infra changes beyond a Dockerfile model pre-pull. Existing API contracts unchanged.

## Scope (Phase A only)

**In scope:**
- Ensemble separation in audio-service: BS-Roformer (vocals model + bass model) replaces those two stems from the `htdemucs_6s` output. Drums, guitar, piano, other remain from `htdemucs_6s`.
- Spectral residual subtract: `guitar' = ISTFT(STFT(guitar) − α · STFT(bass'))`; same for piano. Constant `α = 0.5`. Per-stem disable via env.
- PANNs CNN14 audio tagger run on each detected region of the `other` stem → top-1 sub-label after collapsing 527 AudioSet labels to a 6-class taxonomy.
- Frontend: render `other` region blocks colored by sub-label, with inline label text + confidence tooltip, plus a legend.
- `analysis_version: 2` sentinel for forward back-compat detection. Old rows (no field or version < 2) show "Re-analyze" button.

**Out of scope (later phases):**
- **Phase B:** chord track, drum-hit sub-classification (kick/snare/HH), bassline notes, melody contour, lyric alignment, better section detection.
- **Phase C:** real waveform rendering (replacing envelopes), zoom + horizontal scroll, region loop, per-stem volume/pan/EQ, keyboard shortcuts.
- **Phase D:** typography, color system, animation polish.

Also explicitly out of Phase A:
- Drum stem cleanup (current Demucs drums are acceptable for now).
- Sub-stem **audio** for `other` — labels only.
- Adaptive α tuning per song.
- GPU support (sequential CPU sequence stays under the 3.83 GiB container limit).

## Architecture

```
[Frontend uploads song]
   ↓
[Backend creates song row + job]
   ↓
[Worker → audio-service POST /analyze]
   ↓
[audio-service]:
   1. Download mix → tmpdir
   2. Demucs htdemucs_6s → 6 raw stems  (CPU, ~12 min)
   3. BS-Roformer (vocals weights)  → vocals_v2     (CPU, ~3-5 min)
   4. BS-Roformer (bass weights)    → bass_v2       (CPU, ~3-5 min)
   5. Replace: stems["vocals"] = vocals_v2
              stems["bass"]   = bass_v2
   6. Residual subtract:
         guitar' = ISTFT( STFT(guitar) − α·STFT(bass_v2) )
         piano'  = ISTFT( STFT(piano)  − α·STFT(bass_v2) )
   7. Region detection (existing, unchanged) on the final 6 stems
   8. PANNs CNN14 on each "other" region's audio slice →
         top-1 sub_label + confidence (after taxonomy collapse)
   9. MP3 encode + S3 upload each final stem  (overwrites prior keys)
  10. Build analysis JSON (analysis_version=2)
   ↓
[Backend writes song_analysis row + updates song status]
   ↓
[Frontend renders timeline; "other" region blocks colored by sub_label]
```

No database schema change. All new fields live inside `song_analysis.result_json`. The S3 key conventions are unchanged: `songs/<song_id>/stems/<stem>.mp3` for stems, `songs/<uuid>.<ext>` for the mix.

### Module layout (audio-service)

```
audio-service/src/analysis/
├── demucs_runner.py    (existing — untouched)
├── beat_analysis.py    (existing — untouched)
├── segmentation.py     (existing — untouched)
├── regions.py          (existing — untouched)
├── envelope.py         (existing — untouched)
├── mp3_encoder.py      (existing — untouched)
├── ensemble.py         (NEW — BS-Roformer wrapper, lazy-load, sequential)
├── residual.py         (NEW — STFT spectral subtract)
├── tagging.py          (NEW — PANNs CNN14 wrapper, lazy-load)
└── sub_label.py        (NEW — AudioSet-527 → 6-class taxonomy)
```

`main.py` orchestrates the pipeline. Each new module has one responsibility and a small, mockable interface (see "Module interfaces" below).

### Memory budget (sequential)

| Step | Peak RAM | Notes |
|------|----------|-------|
| Demucs htdemucs_6s | ~2.0 GiB | Free model + tensors before step 3. |
| BS-Roformer vocals | ~1.5 GiB | Free before step 4. |
| BS-Roformer bass | ~1.5 GiB | Free before step 7. |
| Region detect + STFT subtract | ~0.5 GiB | librosa + numpy. |
| PANNs CNN14 | ~0.5 GiB | Per-region clip tagging — short audio. |

Container limit: 3.83 GiB. Sequential loading + `gc.collect()` + `torch.cuda.empty_cache()` between steps keeps headroom.

### Runtime budget

5-minute song, CPU-only:

| Step | Time |
|------|------|
| Demucs htdemucs_6s | ~12 min |
| BS-Roformer × 2 (vocals + bass) | ~6-10 min |
| Residual subtract (STFT × 2) | ~5 s |
| Region detect (existing) | ~5 s |
| PANNs tagging (per region, typically 5-15 clips of 10-30 s each) | ~10-30 s |
| MP3 encode × 6 + S3 upload | ~40 s |
| **Total** | **~30-45 min/song** |

~2-3× current. Acceptable per user budget choice.

## Module interfaces

### `analysis/ensemble.py`

```python
def separate_roformer_vocals(audio: np.ndarray, sr: int) -> np.ndarray:
    """Run BS-Roformer vocals model. Returns mono float32 at original sr.

    Lazy-loads model on first call. Caches loaded model in module global.
    Caller is responsible for freeing memory between bass and vocals calls
    if memory pressure is high (call `free_model()`).
    """

def separate_roformer_bass(audio: np.ndarray, sr: int) -> np.ndarray:
    """Run BS-Roformer bass model. Returns mono float32 at original sr."""

def free_model() -> None:
    """Drop cached model from memory + gc + torch.cuda.empty_cache()."""
```

Implementation: thin wrapper around the `audio-separator` package (which wraps ONNX-Runtime inference for many UVR-style models). If `audio-separator` proves unsuitable (large transitive deps, version conflicts), fall back to a direct `onnxruntime` loader against the BS-Roformer ONNX weights.

Model identifiers (initial choice, may need refinement during impl):
- Vocals: `model_bs_roformer_ep_368_sdr_12.9628.ckpt` or its ONNX equivalent
- Bass: `bs_roformer_ep_937_sdr_10.5309.ckpt` or instrumental → bass-band extraction (TBD during impl spike — fall back to MDX-Net bass model if BS-Roformer bass weights are not freely available)

If only a vocals BS-Roformer is freely available and bass-specific weights aren't, fall back to MDX-Net (`mdxnet_bass`) for the bass replacement. Implementation spike will lock this; the architecture is identical either way (two specialist passes).

### `analysis/residual.py`

```python
def spectral_subtract(
    target: np.ndarray,
    reference: np.ndarray,
    sr: int,
    alpha: float = 0.5,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """target' = ISTFT( STFT(target) − α · STFT(reference) ).

    - Magnitudes only (phase preserved from target).
    - Clamp post-subtract magnitudes to >= 0.
    - Length-match: trim/pad reference to len(target) before STFT.
    - Returns float32, same length as target.
    """
```

Pure numpy + librosa. No model. Deterministic.

### `analysis/tagging.py`

```python
def tag_clip(audio: np.ndarray, sr: int, top_k: int = 5) -> list[tuple[str, float]]:
    """Run PANNs CNN14 on the given clip.

    - Resamples to 32 kHz internally (CNN14's training sr) if sr != 32000.
    - Returns list of (audioset_label, probability), sorted desc, length top_k.
    - Lazy-loads model on first call.
    """

def free_model() -> None:
    """Drop cached model from memory."""
```

Backed by the `panns-inference` pip package (lightweight wrapper around the CNN14 PyTorch checkpoint).

### `analysis/sub_label.py`

```python
SubLabel = Literal['lead', 'pad', 'synth', 'strings', 'fx', 'other_misc']

def collapse_to_sub_label(
    audioset_tags: list[tuple[str, float]],
) -> tuple[SubLabel, float]:
    """Collapse top-k AudioSet labels into the 6-class taxonomy.

    Algorithm:
      1. For each (audioset_label, prob) in input:
         look up its sub_label via TAXONOMY dict.
      2. Sum probabilities per sub_label.
      3. Return argmax sub_label + its summed probability (clipped to [0, 1]).
      4. If no input label maps (all are "Music", "Speech", or unknown),
         return ('other_misc', max_unmapped_prob).
    """

TAXONOMY: dict[str, SubLabel] = {
    # strings family
    'Violin, fiddle': 'strings',
    'Cello': 'strings',
    'Orchestra': 'strings',
    'String section': 'strings',
    # pad family
    'Synthesizer': 'synth',
    'Sampler': 'synth',
    # ... (full table during implementation)
}
```

The taxonomy is **data**, not algorithm logic. Iterating on it is a one-file change (and overridable via `DESPIECE_TAXONOMY_PATH` env pointing to a JSON file in dev).

## JSON shape

```jsonc
{
  "analysis_version": 2,
  "bpm": 120.2,
  "key": "A minor",
  "duration_sec": 484.123,
  "beat_grid": [...],
  "bar_grid": [...],
  "sections": [...],
  "stems": {
    "vocals": { "audio_key": "songs/<id>/stems/vocals.mp3", "regions": [...] },
    "drums":  { "audio_key": "...", "regions": [...] },
    "bass":   { "audio_key": "...", "regions": [...] },
    "guitar": { "audio_key": "...", "regions": [...] },
    "piano":  { "audio_key": "...", "regions": [...] },
    "other":  {
      "audio_key": "...",
      "regions": [
        {
          "start_sec": 12.0,
          "end_sec": 28.5,
          "envelope": [[0.0, 0.3], [0.04, 0.5], ...],
          "sub_label": "pad",
          "sub_label_confidence": 0.78
        }
      ]
    }
  }
}
```

**New fields:**
- `analysis_version` (top-level): integer, currently `2`.
- `stems.other.regions[].sub_label`: one of the six taxonomy classes.
- `stems.other.regions[].sub_label_confidence`: 0-1.

**Removed:** none. P1 shape preserved.

**Constraints:**
- `sub_label` / `sub_label_confidence` only appear on `other` regions. Other stems omit the fields entirely (cleaner type discrimination over `null`).
- If PANNs fails for a specific region, the fields are also omitted (no `null`).

## Backend (Node) changes

- `backend/src/domain/song.ts`:
  ```ts
  export type SubLabel = 'lead' | 'pad' | 'synth' | 'strings' | 'fx' | 'other_misc';

  export interface StemRegion {
    start_sec: number;
    end_sec: number;
    envelope: [number, number][];
    sub_label?: SubLabel;
    sub_label_confidence?: number;
  }

  export interface AnalysisResult {
    // existing fields...
    analysis_version: number;
  }
  ```
- `backend/src/workers/jobWorker.ts`: no logic change. The worker is shape-agnostic.
- `backend/src/routes/songs.ts`: no route additions. Existing presigned stem URL endpoint works as-is (vocals/bass keys point at the new BS-Roformer outputs).
- `backend/src/services/songService.ts`: no change.

### Dev-only reset script

`backend/src/scripts/resetAnalysis.ts`:

```ts
// One-shot: DELETE FROM song_analysis; UPDATE songs SET status='pending';
// Used in dev only after switching to analysis_version=2.
// Run via:  cd backend && npx tsx src/scripts/resetAnalysis.ts
```

Not wired into migrations. Not run by CI. Manual trigger only.

## Frontend changes

### Types

`frontend/src/types/song.ts`:

```ts
export type SubLabel = 'lead' | 'pad' | 'synth' | 'strings' | 'fx' | 'other_misc';

export interface StemRegion {
  start_sec: number;
  end_sec: number;
  envelope: [number, number][];
  sub_label?: SubLabel;
  sub_label_confidence?: number;
}

export interface AnalysisResult {
  // existing fields...
  analysis_version: number;
}
```

### Components

| File | Change |
|------|--------|
| `frontend/src/components/RegionBlock.tsx` | Accept `subLabel?: SubLabel`, `subLabelColor?: string`. When set: fill the block with the sub-label color, render label text inside (top-left) if block width > 60 px, italicize at confidence < 0.4. |
| `frontend/src/components/StemTrack.tsx` | For `name === 'other'` only, pass `region.sub_label` and the corresponding color from `SUB_LABEL_COLORS` lookup. Other stems unaffected. |
| `frontend/src/components/SongTimeline.tsx` | (1) Detect `analysis.analysis_version !== 2` → render "Re-analyze" button instead of timeline. (2) Add `SUB_LABEL_COLORS` constant. (3) Render sub-label legend below the existing stem-color legend, only when at least one `other` region has a `sub_label`. |

### Sub-label color palette

```ts
const SUB_LABEL_COLORS: Record<SubLabel, string> = {
  lead:       '#ef4444',
  pad:        '#8b5cf6',
  synth:      '#ec4899',
  strings:    '#f97316',
  fx:         '#14b8a6',
  other_misc: '#94a3b8',
};
```

### Interactions

- Hover on an `other` region with a sub-label → native `title` tooltip: `"pad · 78%"`.
- Click behavior on regions: unchanged (seeks via `useStemTransport.playFrom`).
- Solo/mute on `other` stem: unchanged — operates on the whole `other` audio, not sub-labels. Sub-labels are visual only in Phase A.

### No new components

All changes are prop additions and conditional rendering inside existing components. Phase A adds zero new React component files.

## Configuration

New env vars (all default-on, default-conservative):

| Var | Default | Effect |
|-----|---------|--------|
| `DESPIECE_USE_ROFORMER_VOCALS` | `true` | Replace Demucs vocals with BS-Roformer vocals. |
| `DESPIECE_USE_ROFORMER_BASS` | `true` | Replace Demucs bass with BS-Roformer/MDX bass. |
| `DESPIECE_USE_RESIDUAL_SUBTRACT` | `true` | Apply STFT subtract to guitar + piano. |
| `DESPIECE_RESIDUAL_ALPHA` | `0.5` | Strength of subtract (0 = none, 1 = full). |
| `DESPIECE_USE_TAGGER` | `true` | Run PANNs on `other` regions. |
| `DESPIECE_TAXONOMY_PATH` | unset | Optional path to override taxonomy JSON. |

These let us bisect quality regressions without code changes and are documented in `audio-service/env.example`.

## Dockerfile / requirements

- `audio-service/requirements.txt` adds:
  - `audio-separator[cpu]` (or `onnxruntime` + manual model loader if dep conflict — decision during impl)
  - `panns-inference`
- `audio-service/Dockerfile`:
  - Add a `RUN` step that runs a small `prefetch_models.py` script during the image build. The script downloads:
    - BS-Roformer vocals weights
    - BS-Roformer (or MDX) bass weights
    - PANNs CNN14 weights
  - Models cached under `/root/.cache/audio-separator` and `/root/panns_data` inside the image. First-run latency in the running container is eliminated.

## Error handling

| Failure mode | Behavior |
|--------------|----------|
| BS-Roformer fails for vocals or bass | Log; fall back to Demucs's original stem for that one. Job completes. |
| PANNs fails or model missing on a region | Omit `sub_label` + `sub_label_confidence` from that region. Job completes. |
| Residual subtract fails (NaN, length mismatch) | Keep original Demucs guitar/piano. Log. |
| Model weight download fails during build | Build fails. Caught in CI / local build, not at request time. |
| Memory blow at runtime | Container restarts (existing `restart: unless-stopped`); job re-queued; up to existing `MAX_ATTEMPTS=3`. |
| All three new steps disabled via env | Pipeline degrades to current htdemucs_6s-only behavior. Useful for bisecting. |

## Backward compatibility

- Old `song_analysis` rows have no `analysis_version` field (or value `1`). Frontend detects this and shows a "Re-analyze" button (already exists from P1). No DB migration written; dev resets via `resetAnalysis.ts` script.
- New shape is a superset of P1 shape (only adds optional fields and the top-level version sentinel). A frontend or backend running off P1 types still parses without crashing (TypeScript widens, JSON ignores unknowns).

## Testing

### audio-service (pytest)

- `tests/test_ensemble.py`
  - Synthetic mix (vocal-like sine + drum-like noise) → BS-Roformer vocals output has length matching input and sample rate matching input.
  - `free_model()` after call → next call still works (cold-load path covered).
- `tests/test_residual.py`
  - Synthetic `guitar = pluck + α·bass` → `spectral_subtract(guitar, bass, α=0.5)` reduces low-band energy (< 200 Hz) by > 50%, leaves high-band (> 1 kHz) within ±10%.
  - Length mismatch (reference shorter than target) → handled via pad, no crash.
- `tests/test_tagging.py`
  - Synthetic pad-like sound (slow attack, sustained low-mid harmonics) → `tag_clip` returns top labels that collapse to `pad` or `synth`.
- `tests/test_sub_label.py`
  - `collapse_to_sub_label([('Violin, fiddle', 0.6), ('Cello', 0.2)])` → `('strings', 0.8)`.
  - `collapse_to_sub_label([('Music', 0.9)])` → `('other_misc', 0.9)` (Music is unmapped sentinel).
- `tests/test_main_integration.py`
  - Mock `ensemble`, `tagging`, `demucs_runner`, `regions`. Assert the response JSON contains `analysis_version: 2`, `stems.other.regions[*].sub_label`, and stems for all 6 names.

### backend (jest)

- `domain/song.test.ts`: type-level parse of a v2 JSON sample + a v1 sample (no `analysis_version`). v1 still parses (optional fields).
- No new route or service tests (no new routes or services).

### frontend (vitest)

- `RegionBlock.test.tsx`: passing `subLabel="pad"` + a color overrides the stem color in the rendered SVG/canvas fill.
- `RegionBlock.test.tsx`: low confidence (< 0.4) renders label text with italic + reduced opacity styling.
- `SongTimeline.test.tsx`: `analysis_version: 1` → "Re-analyze" button visible, stem rows hidden.
- `SongTimeline.test.tsx`: sub-label legend appears only when at least one `other` region has a `sub_label`.

### Manual smoke (post-merge)

1. Run `resetAnalysis.ts`; upload a reference track. Wait for `done` (~30-45 min).
2. Solo vocals → subjectively cleaner than the prior build (no drum bleed in the chorus).
3. Solo bass → cleaner attack envelope, less smearing into mid range.
4. Solo guitar / piano → less low-end mud than before.
5. `other` regions show distinct colors per sub-label; hover gives tooltip; legend renders below.
6. An old song row (pre-version-2) shows "Re-analyze".

## Risks + mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| BS-Roformer bass weights aren't freely available — only vocals | Medium | Spike during impl. Fall back to MDX-Net bass (proven, widely available via UVR / `audio-separator`). |
| Roformer model weights are large (~1.5 GB) → slow Docker builds, big image | High | Pre-pull during image build into a named cache. Optional: mount as Docker volume in compose to share across rebuilds. |
| `audio-separator` package has heavy transitive deps that conflict with current torch / librosa pins | Medium | Fall back to a direct `onnxruntime` loader. Both options keep the `ensemble.py` public interface identical. |
| Residual subtract degrades harmonic stems on songs without bass leakage | Medium | Ship α=0.5 (conservative). Per-stem disable env var. Phase B may add adaptive α (per-song bass energy gate). |
| PANNs taxonomy mismatch — pop synths get tagged as generic "Music" | High | Taxonomy is data, not code. Iterate the dict; allow env-mounted JSON override. Many regions will land in `other_misc` initially — that is acceptable so long as `pad`, `strings`, `lead` work for clear cases. |
| 30-45 min/song slows iteration | Medium | Dev script `audio-service/scripts/analyze_local.py <file>` runs the pipeline outside FastAPI for fast tuning. Optional Phase A.5: add `quality=fast\|hq` query param on `/analyze`. |
| Memory blow from running multiple models | Low | Sequential load + explicit `free_model()` + `gc.collect()` between steps. Existing container restart path catches the edge case. |

## Follow-up phases (not in scope)

- **Phase B — Deeper analysis:** chord track per bar (via `chord-extractor` or trained CRNN on the harmonic ensemble), drum-hit classification inside the drums stem (kick/snare/HH/cymbal via onset + spectral features), bass note track (via `crepe`/`basic-pitch`), section detection v2 using stem-activity patterns.
- **Phase C — DAW UX:** swap envelope rendering for real waveform peaks (decode MP3 in the browser via WebAudio + offline analyser), horizontal zoom + scroll, region loop, per-stem volume/pan/EQ, keyboard shortcuts (space, j/k/l, m, s).
- **Phase D — Visual polish:** typography system, animation passes, full responsive layout, dark-mode parity.
