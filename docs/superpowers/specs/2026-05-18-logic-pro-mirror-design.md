# Logic-Pro-Mirror Analysis View — Design Spec

**Status:** Draft for review
**Phase:** P1 of 3 (Foundation)
**Date:** 2026-05-18

## Goal

Replace the current per-stem energy-ribbon visualization with a Logic-Pro-like view: discrete region blocks per stem, bar-snapped to a beat grid, with mini-waveforms inside each block, full-mix playback, and per-stem solo/mute controls. Used by the user to inspect song structure of reference tracks and inform their own productions.

## Scope (P1 only)

Includes:
- Switch source separation model from `htdemucs` (4 stems) to `htdemucs_6s` (6 stems: vocals, drums, bass, guitar, piano, other)
- New per-stem region-detection pipeline (active intervals, bar-snapped)
- Per-region mini-envelope for inside-block rendering
- Per-stem MP3 encoded and stored to S3 under `songs/<song_id>/stems/<stem>.mp3`
- Frontend rewrite of `SongTimeline` to render region blocks (canvas-based mini-waves) instead of continuous ribbons
- Full-mix audio playback with playhead cursor
- Click region → seek mix audio to that timestamp
- Per-stem solo/mute toggle (visual dimming in P1; true mix solo deferred to P2)

Out of scope (future phases):
- **P2:** drum stem sub-classification (kick/snare/hi-hat/cymbals), true stem-mix solo via Web Audio API, click-region-to-solo-stem playback
- **P3:** classification of segments inside "other" into lead/pad/fx/strings via YAMNet, section-detection improvements using stem-activity patterns

## Architecture

```
[Frontend uploads song]
   ↓
[Backend creates song row + job, returns]
   ↓
[Worker → audio-service POST /analyze]
   ↓
[audio-service]:
   1. Demucs htdemucs_6s → 6 stem WAVs (in-memory)
   2. For each stem: encode MP3 → upload to S3 as `songs/<song_id>/stems/<stem>.mp3`
   3. For each stem: detect active regions, bar-snap to beat grid
   4. For each region: compute mini-envelope (RMS @ ~50fps)
   5. Existing: BPM, key, sections (kept, untouched in P1)
   6. Return analysis JSON
   ↓
[Backend writes song_analysis row + updates song]
   ↓
[Frontend renders timeline: region blocks per stem,
 mini-wave inside, full-mix <audio> playback]
```

No database schema changes. All new data lives inside `song_analysis.result_json` plus new S3 object keys under each song's prefix.

## Backend pipeline (audio-service)

### New module: `audio-service/src/analysis/regions.py`

```python
def detect_regions(audio, sr, beat_times, beats_per_bar=4) -> list[dict]:
    """
    Detect active intervals in a stem, snap edges to nearest bar.

    Steps:
    1. RMS @ 50fps → smooth with median filter (~0.3s window)
    2. Threshold: active = rms > max(noise_floor, 0.05 * peak_rms)
    3. Find contiguous runs of `active=True`
    4. Merge runs separated by < 0.5 sec gap
    5. Drop runs shorter than 0.5 sec
    6. Build bar grid from beat_times (beats_per_bar beats/bar)
    7. Snap each run start/end to nearest bar boundary
    8. Compute per-region mini-envelope (RMS @ 50fps, normalized 0-1)

    Returns: [{"start_sec", "end_sec", "envelope": [[t_relative, e_0_1], ...]}]
    """
```

### Changes to existing files

- **`demucs_runner.py`**: model name `"htdemucs"` → `"htdemucs_6s"`. `STEMS` list grows to `["drums", "bass", "other", "vocals", "guitar", "piano"]` (htdemucs_6s order).
- **`main.py` `/analyze`**:
  - After Demucs separation, encode each stem to MP3 128kbps (via `ffmpeg` subprocess; ffmpeg already in image) → upload to S3 via `storage.upload_from_path`
  - Compute `beat_times` once on the mix; pass into `detect_regions` for every stem so all rows share one bar grid
  - Replace per-stem global `compute_envelope` call with `detect_regions(stem, sr, beat_times)`; per-region envelopes are computed inside `detect_regions`
- **`storage.py`**: add `upload_from_path(local_path: str, key: str) -> None`. Symmetric to existing `download_to_path`.
- **`requirements.txt`**: no new Python deps (ffmpeg via subprocess; already installed in Dockerfile)

### Response shape (new fields in bold)

```json
{
  "bpm": 120.2,
  "key": "A minor",
  "duration_sec": 484.123,
  "beat_grid": [0.5, 1.0, ...],
  "bar_grid": [0.5, 2.5, 4.5, ...],
  "sections": [{"label": "A", "start_sec": 0, "end_sec": 120}, ...],
  "stems": {
    "vocals": {
      "audio_key": "songs/<song_id>/stems/vocals.mp3",
      "regions": [
        {
          "start_sec": 12.0,
          "end_sec": 28.5,
          "envelope": [[0.0, 0.3], [0.04, 0.5], ...]
        }
      ]
    },
    "drums":  { "audio_key": "...", "regions": [...] },
    "bass":   { ... },
    "guitar": { ... },
    "piano":  { ... },
    "other":  { ... }
  }
}
```

**Removed from current shape:** `stems[name].envelope` (continuous) — replaced by per-region envelopes inside each region object.

**Note on mix audio:** original-mix playback uses the existing `songs.storage_key` column (pattern `songs/<uuid>.<ext>`). No new field needed; frontend hits `GET /api/songs/:id/audio` which presigns the existing `storage_key`.

**S3 key conventions:**
- Original mix (unchanged): `songs/<uuid>.<ext>` (set at upload by backend; stored in `songs.storage_key`)
- Stems (new): `songs/<song_id>/stems/<stem>.mp3` (set by audio-service after Demucs)

## Backend (Node)

- **`backend/src/domain/song.ts`**: update `AnalysisResult` to new shape. Add `StemRegion`, `StemAnalysis` types. Drop `stems[name].envelope` continuous field.
- **`backend/src/routes/songs.ts`**: add `GET /api/songs/:id/stems/:stem` — issues S3 presigned URL via existing S3 client, responds 302 redirect.
- **`backend/src/routes/songs.ts`**: add `GET /api/songs/:id/audio` — presigned URL for the original mix (resolves `songs.storage_key`).
- **`backend/src/services/storageService.ts`**: add `deleteFilesByPrefix(prefix: string)` — lists keys under prefix via `ListObjectsV2Command`, batch-deletes via `DeleteObjectsCommand`.
- **`backend/src/services/songService.ts`**: `deleteSong` extended to call `deleteFilesByPrefix("songs/<song_id>/")` in addition to deleting the original `storage_key`. The original key (`songs/<uuid>.<ext>`) is deleted as today; the new prefix cleans up stems.
- **`backend/src/workers/jobWorker.ts`**: no logic changes (response shape change is transparent to worker — it just stores `result_json`).

## Frontend

### New + rewritten components

```
SongTimeline.tsx        (rewrite — orchestrates everything)
├── TimeAxis.tsx        (keep — bars/seconds toggle stays)
├── SectionBar.tsx      (keep — section labels stay)
├── StemTrack.tsx       (NEW — one row per stem; replaces StemRow)
│   ├── RegionBlock.tsx (NEW — single block with mini-wave canvas)
│   └── StemControls.tsx (NEW — S/M solo/mute buttons)
└── PlaybackBar.tsx     (NEW — play/pause + scrub bar + cursor line)
```

### Region rendering

`RegionBlock` uses a `<canvas>` element per region to draw mini-envelope as RMS bars. Positioned via absolute CSS:
- `left: (region.start_sec / duration) * timelineWidth`
- `width: ((region.end_sec - region.start_sec) / duration) * timelineWidth`

Canvas avoids rendering thousands of SVG elements for long songs with many short regions.

### Playback

- Single hidden `<audio src={mix presigned URL}>` element managed via `useRef`
- `requestAnimationFrame` loop reads `audio.currentTime` → updates absolute-positioned cursor `left` style
- Click on `RegionBlock` → `audio.currentTime = region.start_sec; audio.play()`
- `PlaybackBar` exposes play/pause + scrub (clicking on the time axis seeks the audio)

### Solo/mute (P1 visual only)

`StemControls` renders S and M buttons. State stored in `SongTimeline` as `{ soloed: Set<string>, muted: Set<string> }`.

- Solo non-empty → only soloed stems render at full opacity; others dim to 30%
- Mute → that stem dims to 30%
- Audio playback still uses the original mix; per-stem-mix synthesis deferred to P2

### Removed

- `StemRow.tsx` (replaced by `StemTrack`)
- Continuous envelope code in the timeline

### Color palette

```ts
const STEM_COLORS = {
  vocals: '#6366f1', drums: '#f59e0b', bass:   '#10b981',
  guitar: '#a78bfa', piano: '#06b6d4', other:  '#94a3b8',
};
```

Bar grid drawn as faint vertical lines across the timeline. Region edges align to those lines.

## Runtime + storage budget

For a 5-minute song:
- Demucs htdemucs_6s on CPU: ~10-13 min (same class as current htdemucs)
- MP3 encoding 6 stems via ffmpeg: ~30 sec
- Region detection + envelopes: ~5 sec
- S3 upload 6 × ~5MB to local MinIO: ~10 sec
- **Total: ~12-15 min/song** (~10% slower than current)

Storage:
- Original mix (unchanged): ~5MB MP3 or ~50MB WAV
- 6 stems × MP3 128kbps × 5 min ≈ ~30MB extra/song
- Analysis JSON grows ~3× (per-region envelopes instead of one continuous envelope per stem) → still ~200KB

Memory peak in audio-service stays ~2GiB during Demucs. Container limit 3.83GiB adequate.

## Error handling

- **MP3 encode fails for one stem:** log; set that stem's `audio_key` to `null`; region detection still runs (envelope intact)
- **S3 upload fails:** fail the whole job (don't half-persist analysis); existing retry path handles up to `MAX_ATTEMPTS=3`
- **Region detection fails for one stem:** return `regions: []` for that stem; other stems unaffected
- **Demucs OOM:** container restarts (already fixed via `restart: unless-stopped` and no `--reload`); job re-queued, attempts++ until MAX
- **Frontend, old analysis shape encountered:** render "Re-analyze" button. Old `song_analysis` rows are dev-only; no migration path required.

## Backward compatibility

P1 ships the new analysis JSON shape only. Existing `song_analysis` rows from before P1 become unreadable by the new frontend; this is acceptable in the current dev environment. The frontend detects an old shape (missing `regions` field) and shows a "Re-analyze" button which re-enqueues a job.

## Testing

### audio-service (pytest)

- `test_regions.py`:
  - Synthetic stem `silence → 5s tone → silence → 8s tone → silence` → assert exactly 2 regions detected with start/end seconds within ±0.2s of truth
  - Bar-snap: given `beat_times` for 120 BPM 4/4 (downbeats at 0.0, 2.0, 4.0, ...) and a raw active span of `[0.3, 4.7]`, assert snapped to `[0.0, 4.0]`
  - Short-run filter: a 0.2-second active run is dropped
  - Gap-merge: two runs separated by a 0.3-second gap are merged into one region
- `test_demucs_runner.py`: model name = `"htdemucs_6s"`; stems list length = 6
- `test_storage.py`: `upload_from_path` round-trips a small file through MinIO

### backend (jest)

- `songService.test.ts`: type round-trip for the new analysis JSON shape (regions[], audio_key)
- `routes/songs.test.ts`: `GET /api/songs/:id/stems/:stem` returns 302 redirect; mock S3 client; bad stem name → 404
- `routes/songs.test.ts`: deleting a song calls `deleteFilesByPrefix("songs/<id>/")` and `deleteFile(originalKey)`
- `audioClient.test.ts`: parses new response shape into `AnalysisResult`

### frontend (vitest)

- `RegionBlock.test.tsx`: given a region `{start:10, end:20, envelope:[[0,0.5],[1,0.8]]}`, canvas renders at `left:10/duration*width`, `width:10/duration*width`
- `SongTimeline.test.tsx`: clicking a region sets `audio.currentTime` to `region.start_sec`
- `SongTimeline.test.tsx`: solo button on bass dims other stems' container opacity
- `PlaybackBar.test.tsx`: cursor position tracks `audio.currentTime` via rAF

### Manual smoke after merge

1. Upload a song. Wait for status `done`.
2. Verify 6 stem rows render with region blocks (vocals, drums, bass, guitar, piano, other).
3. Click on a region → audio jumps + plays from that point.
4. Toggle solo on bass → other rows dim.
5. Bars/seconds axis toggle still works.
6. Section labels still render correctly above the stems.

## Follow-up phases (sketched, not in scope)

- **P2** — Drum stem split into kick/snare/hi-hat/cymbals via onset+spectral classification (or DrumSep model). Web Audio API mix-on-the-fly for true solo/mute. Per-region click-to-solo-stem.
- **P3** — YAMNet (or similar) classifies segments inside the "other" stem into lead/pad/fx/strings labels. MSAF tuning + use stem-activity patterns to confirm section boundaries.
