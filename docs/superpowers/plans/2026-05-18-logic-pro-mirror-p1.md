# Logic-Pro-Mirror P1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the per-stem energy-ribbon view with a Logic-Pro-like view of bar-snapped region blocks containing mini-waveforms, with full-mix playback and per-stem solo/mute dimming.

**Architecture:** Audio-service runs Demucs `htdemucs_6s`, encodes each stem to MP3 and uploads to S3, then computes bar-snapped active regions per stem with per-region mini-envelopes. Backend exposes presigned URLs for the mix and per-stem audio. Frontend renders region blocks via absolute-positioned canvases on a single timeline, with a `<audio>` element playing the mix and a `requestAnimationFrame`-driven playhead cursor.

**Tech Stack:** Python 3.11 / FastAPI / Demucs / librosa / ffmpeg / boto3 / Node 20 / Express / pg / AWS SDK v3 / React 19 / Vite / TypeScript / Tailwind / Canvas2D.

**Spec:** [docs/superpowers/specs/2026-05-18-logic-pro-mirror-design.md](../specs/2026-05-18-logic-pro-mirror-design.md)

---

## Phase A — Audio-service pipeline

### Task A1: `upload_from_path` in `storage.py` (TDD)

**Files:**
- Modify: `audio-service/src/storage.py`
- Test: `audio-service/tests/test_storage_upload.py`

- [ ] **Step 1: Write the failing test**

Create `audio-service/tests/test_storage_upload.py`:

```python
import os
import tempfile
import pytest
from src.storage import upload_from_path, download_to_path, get_s3_client


@pytest.fixture(autouse=True)
def s3_env(monkeypatch):
    monkeypatch.setenv("S3_ENDPOINT", "http://localhost:9000")
    monkeypatch.setenv("S3_ACCESS_KEY", "minioadmin")
    monkeypatch.setenv("S3_SECRET_KEY", "minioadmin")
    monkeypatch.setenv("S3_BUCKET", "songs")
    monkeypatch.setenv("S3_REGION", "us-east-1")


@pytest.mark.integration
def test_upload_from_path_round_trip(tmp_path):
    client = get_s3_client()
    bucket = os.environ["S3_BUCKET"]
    try:
        client.create_bucket(Bucket=bucket)
    except Exception:
        pass

    src = tmp_path / "src.bin"
    src.write_bytes(b"hello despiece")
    key = "tests/upload_from_path.bin"
    upload_from_path(str(src), key)

    dst = tmp_path / "dst.bin"
    download_to_path(key, str(dst))
    assert dst.read_bytes() == b"hello despiece"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd audio-service && python -m pytest tests/test_storage_upload.py -v`
Expected: FAIL with `ImportError: cannot import name 'upload_from_path' from 'src.storage'`.

- [ ] **Step 3: Implement `upload_from_path`**

Append to `audio-service/src/storage.py`:

```python
def upload_from_path(local_path: str, storage_key: str) -> None:
    bucket = os.environ.get("S3_BUCKET")
    if not bucket:
        raise EnvironmentError("Missing required env var: S3_BUCKET")
    client = get_s3_client()
    client.upload_file(local_path, bucket, storage_key)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd audio-service && python -m pytest tests/test_storage_upload.py -v`
Expected: PASS (requires MinIO running on `localhost:9000`; skip with `-m "not integration"` if not).

- [ ] **Step 5: Commit**

```bash
git add audio-service/src/storage.py audio-service/tests/test_storage_upload.py
git commit -m "feat(audio-service): add upload_from_path to storage"
```

---

### Task A2: `regions.py` — `_find_runs` helper (TDD)

**Files:**
- Create: `audio-service/src/analysis/regions.py`
- Test: `audio-service/tests/test_regions.py`

- [ ] **Step 1: Write the failing test**

Create `audio-service/tests/test_regions.py`:

```python
import numpy as np
from src.analysis.regions import _find_runs


def test_find_runs_single_active_span():
    active = np.array([False, True, True, True, False, False])
    times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    runs = _find_runs(active, times)
    assert runs == [(0.1, 0.3)]


def test_find_runs_two_spans():
    active = np.array([True, True, False, False, True, True])
    times = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    runs = _find_runs(active, times)
    assert runs == [(0.0, 0.1), (0.4, 0.5)]


def test_find_runs_active_to_end():
    active = np.array([False, False, True, True])
    times = np.array([0.0, 0.1, 0.2, 0.3])
    runs = _find_runs(active, times)
    assert runs == [(0.2, 0.3)]


def test_find_runs_all_inactive():
    active = np.array([False, False, False])
    times = np.array([0.0, 0.1, 0.2])
    assert _find_runs(active, times) == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd audio-service && python -m pytest tests/test_regions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.analysis.regions'`.

- [ ] **Step 3: Implement helper in new file**

Create `audio-service/src/analysis/regions.py`:

```python
import numpy as np


def _find_runs(active, times):
    runs = []
    in_run = False
    start_idx = 0
    for i, a in enumerate(active):
        if a and not in_run:
            in_run = True
            start_idx = i
        elif not a and in_run:
            in_run = False
            runs.append((float(times[start_idx]), float(times[i - 1])))
    if in_run:
        runs.append((float(times[start_idx]), float(times[-1])))
    return runs
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd audio-service && python -m pytest tests/test_regions.py -v`
Expected: PASS, 4 tests.

- [ ] **Step 5: Commit**

```bash
git add audio-service/src/analysis/regions.py audio-service/tests/test_regions.py
git commit -m "feat(audio-service): regions module — _find_runs helper"
```

---

### Task A3: `regions.py` — `_merge_close` helper (TDD)

**Files:**
- Modify: `audio-service/src/analysis/regions.py`
- Modify: `audio-service/tests/test_regions.py`

- [ ] **Step 1: Add failing tests**

Append to `audio-service/tests/test_regions.py`:

```python
from src.analysis.regions import _merge_close


def test_merge_close_within_gap():
    runs = [(0.0, 1.0), (1.3, 2.0)]
    merged = _merge_close(runs, gap_sec=0.5)
    assert merged == [(0.0, 2.0)]


def test_merge_close_outside_gap_kept_separate():
    runs = [(0.0, 1.0), (2.0, 3.0)]
    merged = _merge_close(runs, gap_sec=0.5)
    assert merged == [(0.0, 1.0), (2.0, 3.0)]


def test_merge_close_empty():
    assert _merge_close([], gap_sec=0.5) == []


def test_merge_close_chain():
    runs = [(0.0, 1.0), (1.2, 2.0), (2.1, 3.0)]
    merged = _merge_close(runs, gap_sec=0.5)
    assert merged == [(0.0, 3.0)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd audio-service && python -m pytest tests/test_regions.py -v`
Expected: FAIL with `ImportError: cannot import name '_merge_close'`.

- [ ] **Step 3: Implement helper**

Append to `audio-service/src/analysis/regions.py`:

```python
def _merge_close(runs, gap_sec):
    if not runs:
        return []
    merged = [runs[0]]
    for s, e in runs[1:]:
        ps, pe = merged[-1]
        if s - pe <= gap_sec:
            merged[-1] = (ps, e)
        else:
            merged.append((s, e))
    return merged
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd audio-service && python -m pytest tests/test_regions.py -v`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add audio-service/src/analysis/regions.py audio-service/tests/test_regions.py
git commit -m "feat(audio-service): regions module — _merge_close helper"
```

---

### Task A4: `regions.py` — `_build_bar_grid` + `_snap` helpers (TDD)

**Files:**
- Modify: `audio-service/src/analysis/regions.py`
- Modify: `audio-service/tests/test_regions.py`

- [ ] **Step 1: Add failing tests**

Append to `audio-service/tests/test_regions.py`:

```python
from src.analysis.regions import _build_bar_grid, _snap


def test_build_bar_grid_120_bpm_4_4():
    # 120 BPM 4/4: beats at 0.5s spacing, downbeats every 4th beat (2.0s spacing)
    beats = [i * 0.5 for i in range(20)]
    grid = _build_bar_grid(beats, beats_per_bar=4, audio_duration=10.0)
    assert grid[0] == 0.0
    assert grid[1] == 0.0  # first beat
    assert grid[2] == 2.0
    assert grid[3] == 4.0
    assert grid[-1] == 10.0


def test_build_bar_grid_few_beats():
    grid = _build_bar_grid([0.0], beats_per_bar=4, audio_duration=10.0)
    assert grid == [0.0, 10.0]


def test_snap_picks_nearest():
    grid = [0.0, 2.0, 4.0, 6.0, 8.0]
    assert _snap(0.3, grid) == 0.0
    assert _snap(4.7, grid) == 4.0
    assert _snap(5.5, grid) == 6.0
    assert _snap(8.0, grid) == 8.0


def test_snap_empty_grid_returns_input():
    assert _snap(2.5, []) == 2.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd audio-service && python -m pytest tests/test_regions.py -v`
Expected: FAIL with `ImportError: cannot import name '_build_bar_grid'`.

- [ ] **Step 3: Implement helpers**

Append to `audio-service/src/analysis/regions.py`:

```python
def _build_bar_grid(beat_times, beats_per_bar, audio_duration):
    if len(beat_times) < 2:
        return [0.0, float(audio_duration)]
    bars = [float(beat_times[i]) for i in range(0, len(beat_times), beats_per_bar)]
    if bars[0] > 0.0:
        bars.insert(0, 0.0)
    if bars[-1] < audio_duration:
        bars.append(float(audio_duration))
    return bars


def _snap(t, bar_grid):
    if not bar_grid:
        return float(t)
    return float(min(bar_grid, key=lambda b: abs(b - t)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd audio-service && python -m pytest tests/test_regions.py -v`
Expected: PASS, 12 tests.

- [ ] **Step 5: Commit**

```bash
git add audio-service/src/analysis/regions.py audio-service/tests/test_regions.py
git commit -m "feat(audio-service): regions — _build_bar_grid and _snap"
```

---

### Task A5: `regions.py` — top-level `detect_regions` (TDD)

**Files:**
- Modify: `audio-service/src/analysis/regions.py`
- Modify: `audio-service/tests/test_regions.py`

- [ ] **Step 1: Add failing tests**

Append to `audio-service/tests/test_regions.py`:

```python
from src.analysis.regions import detect_regions


def _make_audio(sr, segments):
    """segments = [(start_sec, end_sec, amp)]; rest is silence."""
    total_sec = max(seg[1] for seg in segments)
    y = np.zeros(int(sr * total_sec), dtype=np.float32)
    for s, e, amp in segments:
        i0 = int(s * sr)
        i1 = int(e * sr)
        t = np.arange(i1 - i0) / sr
        y[i0:i1] = (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return y


def test_detect_regions_two_active_intervals():
    sr = 22050
    # 5s tone (1-6s), silence (6-8s), 8s tone (8-16s) -> total 16s
    y = _make_audio(sr, [(1.0, 6.0, 0.5), (8.0, 16.0, 0.5)])
    # 120 BPM 4/4: beats at 0.5s spacing, downbeats every 2s
    beats = [i * 0.5 for i in range(int(16 / 0.5))]
    regions = detect_regions(y, sr, beats, beats_per_bar=4)

    assert len(regions) == 2
    assert abs(regions[0]["start_sec"] - 0.0) < 0.5 or abs(regions[0]["start_sec"] - 2.0) < 0.5
    assert regions[0]["end_sec"] > regions[0]["start_sec"]
    assert regions[1]["start_sec"] > regions[0]["end_sec"]


def test_detect_regions_silence_returns_empty():
    sr = 22050
    y = np.zeros(sr * 4, dtype=np.float32)
    beats = [i * 0.5 for i in range(8)]
    assert detect_regions(y, sr, beats, beats_per_bar=4) == []


def test_detect_regions_short_active_filtered_out():
    sr = 22050
    # 0.2s active burst — should be dropped (< 0.5s min run)
    y = _make_audio(sr, [(1.0, 1.2, 0.5)])
    beats = [i * 0.5 for i in range(8)]
    # Pad to 4s
    pad = np.zeros(sr * 4 - len(y), dtype=np.float32)
    y = np.concatenate([y, pad])
    assert detect_regions(y, sr, beats, beats_per_bar=4) == []


def test_detect_regions_envelope_relative_to_region_start():
    sr = 22050
    y = _make_audio(sr, [(1.0, 6.0, 0.5)])
    beats = [i * 0.5 for i in range(12)]
    regions = detect_regions(y, sr, beats, beats_per_bar=4)
    assert len(regions) >= 1
    env = regions[0]["envelope"]
    assert env[0][0] == 0.0  # relative time starts at 0
    assert all(0.0 <= v <= 1.0 for _, v in env)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd audio-service && python -m pytest tests/test_regions.py -v`
Expected: FAIL with `ImportError: cannot import name 'detect_regions'`.

- [ ] **Step 3: Implement `detect_regions`**

Append to `audio-service/src/analysis/regions.py`:

```python
import librosa
from scipy.ndimage import median_filter


def detect_regions(audio, sr, beat_times, beats_per_bar=4):
    """
    Detect active intervals in a stem, snap edges to nearest bar, attach mini-envelopes.

    Returns: list of {"start_sec", "end_sec", "envelope": [[t_rel, e_0_1], ...]}.
    """
    if audio.size == 0:
        return []

    target_fps = 50.0
    hop_length = max(1, int(sr / target_fps))
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

    window_frames = max(3, int(target_fps * 0.3))
    if window_frames % 2 == 0:
        window_frames += 1
    smoothed = median_filter(rms, size=window_frames)

    peak = float(smoothed.max()) if smoothed.size else 0.0
    if peak <= 0:
        return []
    noise_floor = float(np.percentile(smoothed, 10))
    threshold = max(noise_floor * 2.0, 0.05 * peak)
    active = smoothed > threshold

    runs = _find_runs(active, times)
    runs = _merge_close(runs, gap_sec=0.5)
    runs = [r for r in runs if (r[1] - r[0]) >= 0.5]

    audio_duration = float(times[-1]) if times.size else 0.0
    bar_grid = _build_bar_grid(beat_times, beats_per_bar, audio_duration)

    out = []
    for s, e in runs:
        s_snap = _snap(s, bar_grid)
        e_snap = _snap(e, bar_grid)
        if e_snap <= s_snap:
            continue
        env = _region_envelope(smoothed, times, s_snap, e_snap, peak)
        out.append({
            "start_sec": round(s_snap, 3),
            "end_sec": round(e_snap, 3),
            "envelope": env,
        })
    return out


def _region_envelope(smoothed, times, start_sec, end_sec, peak):
    mask = (times >= start_sec) & (times <= end_sec)
    sub_rms = smoothed[mask]
    sub_times = times[mask]
    if len(sub_rms) == 0 or peak <= 0:
        return []
    norm = np.clip(sub_rms / peak, 0.0, 1.0)
    return [
        [round(float(t - start_sec), 4), round(float(v), 4)]
        for t, v in zip(sub_times, norm)
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd audio-service && python -m pytest tests/test_regions.py -v`
Expected: PASS, 16 tests.

- [ ] **Step 5: Commit**

```bash
git add audio-service/src/analysis/regions.py audio-service/tests/test_regions.py
git commit -m "feat(audio-service): detect_regions with bar-snap and per-region envelope"
```

---

### Task A6: Switch Demucs to `htdemucs_6s`

**Files:**
- Modify: `audio-service/src/analysis/demucs_runner.py`
- Modify: `audio-service/Dockerfile`
- Test: `audio-service/tests/test_demucs_runner.py`

- [ ] **Step 1: Write the failing test**

Create `audio-service/tests/test_demucs_runner.py`:

```python
from src.analysis.demucs_runner import STEMS


def test_stems_list_is_six_for_htdemucs_6s():
    assert len(STEMS) == 6
    assert set(STEMS) == {"drums", "bass", "other", "vocals", "guitar", "piano"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd audio-service && python -m pytest tests/test_demucs_runner.py -v`
Expected: FAIL — `STEMS` currently has 4 entries.

- [ ] **Step 3: Update model + stem list**

In `audio-service/src/analysis/demucs_runner.py`:

```python
# Stem order for htdemucs_6s
STEMS = ["drums", "bass", "other", "vocals", "guitar", "piano"]

_model = None


def _get_model():
    global _model
    if _model is None:
        import torch
        from demucs.pretrained import get_model

        _model = get_model("htdemucs_6s")
        _model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _model.to(device)
    return _model
```

In `audio-service/Dockerfile`, update the pre-download line:

```dockerfile
# Pre-download Demucs htdemucs_6s model so it's cached in the image layer
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs_6s')" || true
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd audio-service && python -m pytest tests/test_demucs_runner.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add audio-service/src/analysis/demucs_runner.py audio-service/Dockerfile audio-service/tests/test_demucs_runner.py
git commit -m "feat(audio-service): switch Demucs to htdemucs_6s (6 stems)"
```

---

### Task A7: MP3 encoder helper (TDD)

**Files:**
- Create: `audio-service/src/analysis/mp3_encoder.py`
- Test: `audio-service/tests/test_mp3_encoder.py`

- [ ] **Step 1: Write the failing test**

Create `audio-service/tests/test_mp3_encoder.py`:

```python
import os
import numpy as np
from src.analysis.mp3_encoder import encode_mp3


def test_encode_mp3_writes_valid_mp3(tmp_path):
    sr = 22050
    t = np.linspace(0, 2.0, int(sr * 2.0), endpoint=False)
    y = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    out_path = tmp_path / "test.mp3"
    encode_mp3(y, sr, str(out_path), bitrate_kbps=128)
    assert out_path.exists()
    # MP3 files start with ID3 tag or 0xFF 0xFB/0xFA/0xF3/0xF2 sync word
    header = out_path.read_bytes()[:3]
    assert header.startswith(b"ID3") or header[0] == 0xFF
    assert out_path.stat().st_size > 1000  # sanity: not empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd audio-service && python -m pytest tests/test_mp3_encoder.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement encoder**

Create `audio-service/src/analysis/mp3_encoder.py`:

```python
import subprocess
import tempfile
import numpy as np
import soundfile as sf


def encode_mp3(audio: np.ndarray, sr: int, out_path: str, bitrate_kbps: int = 128) -> None:
    """
    Write `audio` (float32 mono or stereo) to `out_path` as MP3 at `bitrate_kbps`.

    Uses ffmpeg via subprocess. ffmpeg is required to be installed.
    """
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name
    try:
        sf.write(wav_path, audio, sr)
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-i", wav_path,
            "-codec:a", "libmp3lame",
            "-b:a", f"{bitrate_kbps}k",
            out_path,
        ]
        subprocess.run(cmd, check=True)
    finally:
        try:
            import os
            os.unlink(wav_path)
        except OSError:
            pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd audio-service && python -m pytest tests/test_mp3_encoder.py -v`
Expected: PASS (requires ffmpeg installed locally; already present in container and on macOS via brew).

- [ ] **Step 5: Commit**

```bash
git add audio-service/src/analysis/mp3_encoder.py audio-service/tests/test_mp3_encoder.py
git commit -m "feat(audio-service): MP3 encoder via ffmpeg subprocess"
```

---

### Task A8: Wire new pipeline into `/analyze`

**Files:**
- Modify: `audio-service/src/main.py`

- [ ] **Step 1: Update `/analyze` endpoint**

Replace the body of `/analyze` in `audio-service/src/main.py`:

```python
import os
import tempfile
import traceback

import numpy as np
import librosa
from fastapi import FastAPI, HTTPException
from .models import AnalyzeRequest
from .storage import download_to_path, upload_from_path
from .analysis.demucs_runner import separate_stems, STEMS
from .analysis.beat_analysis import detect_bpm_and_beats, detect_key
from .analysis.segmentation import detect_sections
from .analysis.regions import detect_regions
from .analysis.mp3_encoder import encode_mp3

app = FastAPI(title="Audio Analysis Service")

BEATS_PER_BAR = 4
STEM_MP3_BITRATE_KBPS = 128


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    storage_key = req.storage_key
    song_id = _song_id_from_storage_key(storage_key)

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

            stems[stem_name] = {"audio_key": audio_key, "regions": regions}

        return {
            "bpm": bpm,
            "key": key,
            "duration_sec": round(duration_sec, 3),
            "beat_grid": beat_grid,
            "bar_grid": bar_grid,
            "sections": sections,
            "stems": stems,
        }


def _song_id_from_storage_key(storage_key: str) -> str:
    """`songs/<uuid>.<ext>` → `<uuid>`. Falls back to the raw key with `/` replaced."""
    base = os.path.basename(storage_key)
    return os.path.splitext(base)[0]


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

- [ ] **Step 2: Adjust existing `test_main.py` if it expects the old shape**

Run: `cd audio-service && python -m pytest tests/test_main.py -v`. If failures, update assertions to match new response shape (presence of `stems[x].regions`, `stems[x].audio_key`, top-level `bar_grid`). If `test_main.py` is purely an integration test against the live `/analyze` (which runs Demucs), it may be skipped — preserve current behavior.

- [ ] **Step 3: Commit**

```bash
git add audio-service/src/main.py audio-service/tests/test_main.py
git commit -m "feat(audio-service): wire htdemucs_6s + MP3 upload + per-stem regions"
```

---

## Phase B — Backend (Node)

### Task B1: Update `AnalysisResult` types

**Files:**
- Modify: `backend/src/domain/song.ts`

- [ ] **Step 1: Replace `AnalysisResult` and `StemData`**

Replace the bottom of `backend/src/domain/song.ts`:

```typescript
export interface AnalysisResult {
  bpm: number;
  key: string;
  duration_sec: number;
  beat_grid: number[];
  bar_grid: number[];
  sections: Section[];
  stems: Record<string, StemAnalysis>;
}

export interface Section {
  label: string;
  start_sec: number;
  end_sec: number;
}

export interface StemAnalysis {
  audio_key: string | null;
  regions: StemRegion[];
}

export interface StemRegion {
  start_sec: number;
  end_sec: number;
  envelope: [number, number][]; // [t_relative_sec, energy_0_1]
}
```

- [ ] **Step 2: Verify backend compiles**

Run: `cd backend && npm run build`
Expected: PASS. Any callers referencing `stems[x].envelope` (continuous) will now fail to compile — none exist in the backend today (frontend handles rendering).

- [ ] **Step 3: Commit**

```bash
git add backend/src/domain/song.ts
git commit -m "feat(backend): AnalysisResult types — regions + audio_key + bar_grid"
```

---

### Task B2: `deleteFilesByPrefix` in storageService (TDD)

**Files:**
- Modify: `backend/src/services/storageService.ts`
- Test: `backend/src/services/__tests__/storageService.test.ts`

- [ ] **Step 1: Add failing test**

Append to `backend/src/services/__tests__/storageService.test.ts`:

```typescript
import { ListObjectsV2Command, DeleteObjectsCommand } from '@aws-sdk/client-s3';
import { deleteFilesByPrefix } from '../storageService.js';

it('deleteFilesByPrefix lists then batch-deletes', async () => {
  const sendMock = (await import('@aws-sdk/client-s3')).S3Client.prototype.send as jest.Mock;
  sendMock.mockReset();
  sendMock
    .mockResolvedValueOnce({
      Contents: [{ Key: 'songs/abc/stems/vocals.mp3' }, { Key: 'songs/abc/stems/drums.mp3' }],
      IsTruncated: false,
    })
    .mockResolvedValueOnce({ Deleted: [{ Key: 'songs/abc/stems/vocals.mp3' }] });

  await deleteFilesByPrefix('songs/abc/');

  const calls = sendMock.mock.calls.map((c) => c[0].constructor.name);
  expect(calls).toContain('ListObjectsV2Command');
  expect(calls).toContain('DeleteObjectsCommand');
});
```

If the existing `storageService.test.ts` uses a different mock pattern, adapt to it (look at how `uploadFile` is tested) — the assertion of interest is that `ListObjectsV2Command` then `DeleteObjectsCommand` are sent.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && npm test -- storageService`
Expected: FAIL with import error.

- [ ] **Step 3: Implement `deleteFilesByPrefix`**

Update `backend/src/services/storageService.ts` imports and add a new function:

```typescript
import {
  S3Client,
  PutObjectCommand,
  DeleteObjectCommand,
  DeleteObjectsCommand,
  ListObjectsV2Command,
  HeadBucketCommand,
  CreateBucketCommand,
} from '@aws-sdk/client-s3';
```

Append to the file:

```typescript
export async function deleteFilesByPrefix(prefix: string): Promise<void> {
  const s3 = getS3();
  const bucket = getBucket();
  let continuationToken: string | undefined;

  do {
    const listResp = await s3.send(
      new ListObjectsV2Command({
        Bucket: bucket,
        Prefix: prefix,
        ContinuationToken: continuationToken,
      })
    );

    const objects = (listResp.Contents ?? [])
      .map((o) => o.Key)
      .filter((k): k is string => typeof k === 'string')
      .map((Key) => ({ Key }));

    if (objects.length > 0) {
      await s3.send(
        new DeleteObjectsCommand({
          Bucket: bucket,
          Delete: { Objects: objects },
        })
      );
    }

    continuationToken = listResp.IsTruncated ? listResp.NextContinuationToken : undefined;
  } while (continuationToken);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && npm test -- storageService`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/services/storageService.ts backend/src/services/__tests__/storageService.test.ts
git commit -m "feat(backend): deleteFilesByPrefix helper in storageService"
```

---

### Task B3: Presigned URL helper for stems + mix

**Files:**
- Modify: `backend/src/services/storageService.ts`
- Test: `backend/src/services/__tests__/storageService.test.ts`

- [ ] **Step 1: Add failing test**

Append to `backend/src/services/__tests__/storageService.test.ts`:

```typescript
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';
import { presignDownload } from '../storageService.js';

jest.mock('@aws-sdk/s3-request-presigner', () => ({
  getSignedUrl: jest.fn().mockResolvedValue('https://example.com/signed'),
}));

it('presignDownload returns a signed URL', async () => {
  const url = await presignDownload('songs/abc/stems/vocals.mp3', 60);
  expect(url).toBe('https://example.com/signed');
  expect((getSignedUrl as jest.Mock).mock.calls.length).toBeGreaterThan(0);
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && npm test -- storageService`
Expected: FAIL — `presignDownload` doesn't exist.

- [ ] **Step 3: Install dependency**

Run: `cd backend && npm install --save @aws-sdk/s3-request-presigner`

- [ ] **Step 4: Implement `presignDownload`**

Append to `backend/src/services/storageService.ts`:

```typescript
import { GetObjectCommand } from '@aws-sdk/client-s3';
import { getSignedUrl } from '@aws-sdk/s3-request-presigner';

export async function presignDownload(
  key: string,
  expiresInSec: number = 3600
): Promise<string> {
  const cmd = new GetObjectCommand({ Bucket: getBucket(), Key: key });
  return getSignedUrl(getS3(), cmd, { expiresIn: expiresInSec });
}
```

(Add `GetObjectCommand` to the existing import block at the top instead of re-importing if convenient.)

- [ ] **Step 5: Run test to verify it passes**

Run: `cd backend && npm test -- storageService`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/src/services/storageService.ts backend/src/services/__tests__/storageService.test.ts backend/package.json backend/package-lock.json
git commit -m "feat(backend): presignDownload helper using @aws-sdk/s3-request-presigner"
```

---

### Task B4: `GET /api/songs/:id/audio` (mix presigned URL)

**Files:**
- Modify: `backend/src/routes/songs.ts`
- Test: `backend/src/routes/__tests__/songs.test.ts`

- [ ] **Step 1: Add failing test**

Append to `backend/src/routes/__tests__/songs.test.ts`:

```typescript
describe('GET /api/songs/:id/audio', () => {
  it('returns 302 redirect to presigned URL', async () => {
    const songService = await import('../../services/songService.js');
    (songService.getSong as jest.Mock).mockResolvedValueOnce({
      id: 'song-1',
      storage_key: 'songs/song-1.mp3',
      original_name: 't.mp3',
      status: 'done',
      duration_sec: 1,
      bpm: 1,
      music_key: 'C major',
      error_message: null,
      created_at: new Date().toISOString(),
    });

    const storage = await import('../../services/storageService.js');
    (storage as unknown as { presignDownload: jest.Mock }).presignDownload = jest
      .fn()
      .mockResolvedValue('https://example.com/audio');

    const res = await request(app).get('/api/songs/song-1/audio');
    expect(res.status).toBe(302);
    expect(res.headers.location).toBe('https://example.com/audio');
  });

  it('returns 404 for unknown song', async () => {
    const songService = await import('../../services/songService.js');
    (songService.getSong as jest.Mock).mockResolvedValueOnce(null);
    const res = await request(app).get('/api/songs/nope/audio');
    expect(res.status).toBe(404);
  });
});
```

Update the existing `jest.mock('../../services/storageService.js'...)` block in this file to include:

```typescript
jest.mock('../../services/storageService.js', () => ({
  uploadFile: jest.fn().mockResolvedValue(undefined),
  deleteFile: jest.fn().mockResolvedValue(undefined),
  deleteFilesByPrefix: jest.fn().mockResolvedValue(undefined),
  ensureBucketExists: jest.fn().mockResolvedValue(undefined),
  presignDownload: jest.fn().mockResolvedValue('https://example.com/default'),
}));
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && npm test -- songs`
Expected: FAIL — route 404.

- [ ] **Step 3: Add the route**

Add to `backend/src/routes/songs.ts` (before `router.delete`):

```typescript
import { presignDownload } from '../services/storageService.js';

// GET /api/songs/:id/audio — redirect to presigned URL for original mix
router.get('/:id/audio', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const song = await songService.getSong(req.params.id);
    if (!song) {
      res.status(404).json({ error: 'Song not found' });
      return;
    }
    const url = await presignDownload(song.storage_key);
    res.redirect(302, url);
  } catch (err) {
    next(err);
  }
});
```

If `presignDownload` is not yet in the imports at the top of `songs.ts`, add it to the existing import line:

```typescript
import { uploadFile, deleteFile, presignDownload } from '../services/storageService.js';
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && npm test -- songs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/routes/songs.ts backend/src/routes/__tests__/songs.test.ts
git commit -m "feat(backend): GET /api/songs/:id/audio — presigned mix URL"
```

---

### Task B5: `GET /api/songs/:id/stems/:stem`

**Files:**
- Modify: `backend/src/routes/songs.ts`
- Test: `backend/src/routes/__tests__/songs.test.ts`

- [ ] **Step 1: Add failing tests**

Append to `backend/src/routes/__tests__/songs.test.ts`:

```typescript
const VALID_STEMS = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other'];

describe('GET /api/songs/:id/stems/:stem', () => {
  it('returns 302 to presigned stem URL', async () => {
    const songService = await import('../../services/songService.js');
    (songService.getSong as jest.Mock).mockResolvedValueOnce({
      id: 'song-1',
      storage_key: 'songs/song-1.mp3',
      original_name: 't.mp3',
      status: 'done',
      duration_sec: 1,
      bpm: 1,
      music_key: 'C major',
      error_message: null,
      created_at: new Date().toISOString(),
    });

    const storage = await import('../../services/storageService.js');
    (storage.presignDownload as jest.Mock).mockResolvedValueOnce(
      'https://example.com/stem'
    );

    const res = await request(app).get('/api/songs/song-1/stems/vocals');
    expect(res.status).toBe(302);
    expect(res.headers.location).toBe('https://example.com/stem');
    expect((storage.presignDownload as jest.Mock).mock.calls[0][0]).toBe(
      'songs/song-1/stems/vocals.mp3'
    );
  });

  it('returns 400 for invalid stem name', async () => {
    const res = await request(app).get('/api/songs/song-1/stems/banjo');
    expect(res.status).toBe(400);
  });

  it('returns 404 for unknown song', async () => {
    const songService = await import('../../services/songService.js');
    (songService.getSong as jest.Mock).mockResolvedValueOnce(null);
    const res = await request(app).get('/api/songs/missing/stems/vocals');
    expect(res.status).toBe(404);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && npm test -- songs`
Expected: FAIL — route doesn't exist.

- [ ] **Step 3: Add the route**

Add to `backend/src/routes/songs.ts` (after `:id/audio` route):

```typescript
const VALID_STEMS = new Set(['vocals', 'drums', 'bass', 'guitar', 'piano', 'other']);

// GET /api/songs/:id/stems/:stem — redirect to presigned URL for a stem MP3
router.get('/:id/stems/:stem', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const { id, stem } = req.params;
    if (!VALID_STEMS.has(stem)) {
      res.status(400).json({ error: 'Invalid stem name' });
      return;
    }
    const song = await songService.getSong(id);
    if (!song) {
      res.status(404).json({ error: 'Song not found' });
      return;
    }
    const key = `songs/${id}/stems/${stem}.mp3`;
    const url = await presignDownload(key);
    res.redirect(302, url);
  } catch (err) {
    next(err);
  }
});
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && npm test -- songs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/routes/songs.ts backend/src/routes/__tests__/songs.test.ts
git commit -m "feat(backend): GET /api/songs/:id/stems/:stem — presigned stem URL"
```

---

### Task B6: Extend `deleteSong` to clean up stems

**Files:**
- Modify: `backend/src/services/songService.ts`
- Modify: `backend/src/routes/songs.ts`
- Test: `backend/src/routes/__tests__/songs.test.ts`

- [ ] **Step 1: Add failing test**

Append to `backend/src/routes/__tests__/songs.test.ts`:

```typescript
describe('DELETE /api/songs/:id cleans up stems', () => {
  it('calls deleteFilesByPrefix and deleteFile', async () => {
    const songService = await import('../../services/songService.js');
    (songService.deleteSong as jest.Mock).mockResolvedValueOnce('songs/abc.mp3');

    const storage = await import('../../services/storageService.js');
    (storage.deleteFilesByPrefix as jest.Mock).mockClear();
    (storage.deleteFile as jest.Mock).mockClear();

    const res = await request(app).delete('/api/songs/abc');
    expect(res.status).toBe(204);
    expect((storage.deleteFile as jest.Mock).mock.calls[0][0]).toBe('songs/abc.mp3');
    expect((storage.deleteFilesByPrefix as jest.Mock).mock.calls[0][0]).toBe(
      'songs/abc/'
    );
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && npm test -- songs`
Expected: FAIL — DELETE route doesn't call `deleteFilesByPrefix`.

- [ ] **Step 3: Update DELETE route**

In `backend/src/routes/songs.ts`, update the DELETE handler:

```typescript
import { uploadFile, deleteFile, deleteFilesByPrefix, presignDownload } from '../services/storageService.js';

// DELETE /api/songs/:id
router.delete('/:id', async (req: Request, res: Response, next: NextFunction) => {
  try {
    const id = req.params.id;
    const storageKey = await songService.deleteSong(id);
    if (!storageKey) {
      res.status(404).json({ error: 'Song not found' });
      return;
    }
    await deleteFile(storageKey).catch((err) =>
      logger.warn({ err, storageKey }, 'Failed to delete original mix; continuing')
    );
    await deleteFilesByPrefix(`songs/${id}/`).catch((err) =>
      logger.warn({ err, id }, 'Failed to delete stem prefix; continuing')
    );
    res.status(204).send();
  } catch (err) {
    next(err);
  }
});
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && npm test -- songs`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/src/routes/songs.ts backend/src/routes/__tests__/songs.test.ts
git commit -m "feat(backend): delete song cleans up stems via deleteFilesByPrefix"
```

---

## Phase C — Frontend

### Task C1: Update frontend types

**Files:**
- Modify: `frontend/src/types/song.ts`

- [ ] **Step 1: Replace `AnalysisResult` and `StemData`**

Replace the body of `frontend/src/types/song.ts`:

```typescript
export type SongStatus = 'queued' | 'processing' | 'done' | 'error';

export interface Song {
  id: string;
  original_name: string;
  duration_sec: number | null;
  bpm: number | null;
  music_key: string | null;
  status: SongStatus;
  error_message: string | null;
  created_at: string;
  analysis?: AnalysisResult;
}

export interface AnalysisResult {
  bpm: number;
  key: string;
  duration_sec: number;
  beat_grid: number[];
  bar_grid: number[];
  sections: Section[];
  stems: Record<string, StemAnalysis>;
}

export interface Section {
  label: string;
  start_sec: number;
  end_sec: number;
}

export interface StemAnalysis {
  audio_key: string | null;
  regions: StemRegion[];
}

export interface StemRegion {
  start_sec: number;
  end_sec: number;
  envelope: [number, number][];
}
```

- [ ] **Step 2: Verify frontend compiles**

Run: `cd frontend && npm run build`
Expected: build errors in `SongTimeline.tsx` and `StemRow.tsx` referencing the removed `StemData.envelope` continuous field. Those callers will be rewritten in later tasks — for now the failure is expected. Skip the build until C7.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/types/song.ts
git commit -m "feat(frontend): AnalysisResult types — regions + audio_key + bar_grid"
```

---

### Task C2: API client helpers for stem/mix URLs

**Files:**
- Modify: `frontend/src/lib/api.ts`

- [ ] **Step 1: Inspect existing api.ts**

Read `frontend/src/lib/api.ts` to find the API base URL constant and existing helpers (`getSong`, `uploadSong`, etc.). Use the same `baseURL`/`API_URL` pattern.

- [ ] **Step 2: Add helpers**

Append to `frontend/src/lib/api.ts`:

```typescript
const API_BASE = import.meta.env.VITE_API_URL ?? '/api';

export function getMixAudioUrl(songId: string): string {
  return `${API_BASE}/songs/${songId}/audio`;
}

export function getStemAudioUrl(songId: string, stem: string): string {
  return `${API_BASE}/songs/${songId}/stems/${stem}`;
}
```

If `API_BASE` is already declared elsewhere in the file, reuse the existing identifier instead of redeclaring it.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/lib/api.ts
git commit -m "feat(frontend): API helpers for mix and stem audio URLs"
```

---

### Task C3: `RegionBlock.tsx` — canvas-based mini-wave block

**Files:**
- Create: `frontend/src/components/RegionBlock.tsx`
- Test: `frontend/src/components/__tests__/RegionBlock.test.tsx`

- [ ] **Step 1: Write the failing test**

Create `frontend/src/components/__tests__/RegionBlock.test.tsx`:

```typescript
import { render } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { RegionBlock } from '../RegionBlock';

describe('RegionBlock', () => {
  const region = {
    start_sec: 10,
    end_sec: 20,
    envelope: [[0, 0.5], [1, 0.8]] as [number, number][],
  };

  it('positions itself proportionally to duration on timeline', () => {
    const { container } = render(
      <RegionBlock
        region={region}
        durationSec={100}
        timelineWidth={1000}
        color="#6366f1"
        onClick={() => {}}
      />
    );
    const el = container.firstChild as HTMLElement;
    expect(el.style.left).toBe('100px'); // 10/100 * 1000
    expect(el.style.width).toBe('100px'); // (20-10)/100 * 1000
  });

  it('invokes onClick with region.start_sec', () => {
    const onClick = vi.fn();
    const { container } = render(
      <RegionBlock
        region={region}
        durationSec={100}
        timelineWidth={1000}
        color="#6366f1"
        onClick={onClick}
      />
    );
    (container.firstChild as HTMLElement).click();
    expect(onClick).toHaveBeenCalledWith(10);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- RegionBlock`
Expected: FAIL with module-not-found.

- [ ] **Step 3: Implement the component**

Create `frontend/src/components/RegionBlock.tsx`:

```typescript
import { useEffect, useRef } from 'react';
import { StemRegion } from '@/types/song';

interface Props {
  region: StemRegion;
  durationSec: number;
  timelineWidth: number;
  color: string;
  onClick: (startSec: number) => void;
  rowHeight?: number;
}

export function RegionBlock({
  region,
  durationSec,
  timelineWidth,
  color,
  onClick,
  rowHeight = 48,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const left = (region.start_sec / durationSec) * timelineWidth;
  const width = ((region.end_sec - region.start_sec) / durationSec) * timelineWidth;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(width * dpr));
    canvas.height = Math.max(1, Math.floor(rowHeight * dpr));
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.scale(dpr, dpr);

    ctx.fillStyle = color + '22'; // ~13% alpha background
    ctx.fillRect(0, 0, width, rowHeight);

    ctx.strokeStyle = color;
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, width - 1, rowHeight - 1);

    if (region.envelope.length === 0) return;
    const regionDur = region.end_sec - region.start_sec;
    ctx.fillStyle = color;
    for (const [t, v] of region.envelope) {
      const x = (t / regionDur) * width;
      const h = Math.max(1, v * (rowHeight - 4));
      ctx.fillRect(x, rowHeight - h - 2, 1, h);
    }
  }, [region, width, rowHeight, color]);

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
    >
      <canvas
        ref={canvasRef}
        style={{ width: '100%', height: '100%', display: 'block' }}
      />
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test -- RegionBlock`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/RegionBlock.tsx frontend/src/components/__tests__/RegionBlock.test.tsx
git commit -m "feat(frontend): RegionBlock — canvas mini-wave inside positioned block"
```

---

### Task C4: `StemControls.tsx` — solo/mute buttons

**Files:**
- Create: `frontend/src/components/StemControls.tsx`
- Test: `frontend/src/components/__tests__/StemControls.test.tsx`

- [ ] **Step 1: Write the failing test**

Create `frontend/src/components/__tests__/StemControls.test.tsx`:

```typescript
import { render, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { StemControls } from '../StemControls';

describe('StemControls', () => {
  it('renders S and M buttons and fires callbacks', () => {
    const onSolo = vi.fn();
    const onMute = vi.fn();
    const { getByRole } = render(
      <StemControls soloed={false} muted={false} onToggleSolo={onSolo} onToggleMute={onMute} />
    );
    fireEvent.click(getByRole('button', { name: /solo/i }));
    fireEvent.click(getByRole('button', { name: /mute/i }));
    expect(onSolo).toHaveBeenCalled();
    expect(onMute).toHaveBeenCalled();
  });

  it('highlights soloed state visually', () => {
    const { getByRole } = render(
      <StemControls soloed muted={false} onToggleSolo={() => {}} onToggleMute={() => {}} />
    );
    expect(getByRole('button', { name: /solo/i }).getAttribute('aria-pressed')).toBe('true');
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- StemControls`
Expected: FAIL.

- [ ] **Step 3: Implement**

Create `frontend/src/components/StemControls.tsx`:

```typescript
interface Props {
  soloed: boolean;
  muted: boolean;
  onToggleSolo: () => void;
  onToggleMute: () => void;
}

export function StemControls({ soloed, muted, onToggleSolo, onToggleMute }: Props) {
  return (
    <div className="flex gap-1">
      <button
        type="button"
        aria-label="solo"
        aria-pressed={soloed}
        onClick={onToggleSolo}
        className={
          'text-[10px] font-bold w-5 h-5 rounded border ' +
          (soloed ? 'bg-yellow-400 text-black border-yellow-400' : 'border-muted text-muted-foreground hover:bg-muted/40')
        }
      >
        S
      </button>
      <button
        type="button"
        aria-label="mute"
        aria-pressed={muted}
        onClick={onToggleMute}
        className={
          'text-[10px] font-bold w-5 h-5 rounded border ' +
          (muted ? 'bg-red-500 text-white border-red-500' : 'border-muted text-muted-foreground hover:bg-muted/40')
        }
      >
        M
      </button>
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test -- StemControls`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/StemControls.tsx frontend/src/components/__tests__/StemControls.test.tsx
git commit -m "feat(frontend): StemControls — solo/mute toggle buttons"
```

---

### Task C5: `StemTrack.tsx` — composes label + controls + region blocks

**Files:**
- Create: `frontend/src/components/StemTrack.tsx`
- Test: `frontend/src/components/__tests__/StemTrack.test.tsx`

- [ ] **Step 1: Write the failing test**

Create `frontend/src/components/__tests__/StemTrack.test.tsx`:

```typescript
import { render } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { StemTrack } from '../StemTrack';
import { StemAnalysis } from '@/types/song';

describe('StemTrack', () => {
  const stem: StemAnalysis = {
    audio_key: 'songs/abc/stems/vocals.mp3',
    regions: [
      { start_sec: 5, end_sec: 15, envelope: [[0, 0.5]] },
      { start_sec: 20, end_sec: 30, envelope: [[0, 0.3]] },
    ],
  };

  it('renders one block per region', () => {
    const { container } = render(
      <StemTrack
        name="vocals"
        stem={stem}
        durationSec={60}
        timelineWidth={600}
        color="#6366f1"
        dim={false}
        soloed={false}
        muted={false}
        onToggleSolo={() => {}}
        onToggleMute={() => {}}
        onRegionClick={() => {}}
      />
    );
    const blocks = container.querySelectorAll('canvas');
    expect(blocks.length).toBe(2);
  });

  it('dims container when dim=true', () => {
    const { container } = render(
      <StemTrack
        name="vocals"
        stem={stem}
        durationSec={60}
        timelineWidth={600}
        color="#6366f1"
        dim
        soloed={false}
        muted={false}
        onToggleSolo={() => {}}
        onToggleMute={() => {}}
        onRegionClick={() => {}}
      />
    );
    const row = container.querySelector('[data-stem-track]') as HTMLElement;
    expect(row.style.opacity).toBe('0.3');
  });

  it('passes start_sec into onRegionClick on click', () => {
    const handler = vi.fn();
    const { container } = render(
      <StemTrack
        name="vocals"
        stem={stem}
        durationSec={60}
        timelineWidth={600}
        color="#6366f1"
        dim={false}
        soloed={false}
        muted={false}
        onToggleSolo={() => {}}
        onToggleMute={() => {}}
        onRegionClick={handler}
      />
    );
    const block = container.querySelectorAll('[style*="position: absolute"]')[0] as HTMLElement;
    block.click();
    expect(handler).toHaveBeenCalledWith(5);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- StemTrack`
Expected: FAIL.

- [ ] **Step 3: Implement**

Create `frontend/src/components/StemTrack.tsx`:

```typescript
import { StemAnalysis } from '@/types/song';
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
        {stem.regions.map((region, i) => (
          <RegionBlock
            key={i}
            region={region}
            durationSec={durationSec}
            timelineWidth={timelineWidth}
            color={color}
            onClick={onRegionClick}
            rowHeight={rowHeight}
          />
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test -- StemTrack`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/StemTrack.tsx frontend/src/components/__tests__/StemTrack.test.tsx
git commit -m "feat(frontend): StemTrack — label, controls, positioned region blocks"
```

---

### Task C6: `PlaybackBar.tsx` — play/pause + cursor via rAF

**Files:**
- Create: `frontend/src/components/PlaybackBar.tsx`
- Test: `frontend/src/components/__tests__/PlaybackBar.test.tsx`

- [ ] **Step 1: Write the failing test**

Create `frontend/src/components/__tests__/PlaybackBar.test.tsx`:

```typescript
import { render, fireEvent, act } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import { PlaybackBar } from '../PlaybackBar';

describe('PlaybackBar', () => {
  it('renders a play button initially', () => {
    const audio = document.createElement('audio');
    const { getByRole } = render(
      <PlaybackBar audioRef={{ current: audio }} durationSec={60} timelineWidth={600} />
    );
    expect(getByRole('button', { name: /play/i })).toBeTruthy();
  });

  it('toggles to pause when clicked while playing', () => {
    const audio = document.createElement('audio');
    audio.play = vi.fn().mockResolvedValue(undefined);
    audio.pause = vi.fn();
    const { getByRole } = render(
      <PlaybackBar audioRef={{ current: audio }} durationSec={60} timelineWidth={600} />
    );
    fireEvent.click(getByRole('button', { name: /play/i }));
    act(() => {
      audio.dispatchEvent(new Event('play'));
    });
    expect(getByRole('button', { name: /pause/i })).toBeTruthy();
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npm test -- PlaybackBar`
Expected: FAIL.

- [ ] **Step 3: Implement**

Create `frontend/src/components/PlaybackBar.tsx`:

```typescript
import { RefObject, useEffect, useRef, useState } from 'react';

interface Props {
  audioRef: RefObject<HTMLAudioElement | null>;
  durationSec: number;
  timelineWidth: number;
  labelWidth?: number;
}

export function PlaybackBar({ audioRef, durationSec, timelineWidth, labelWidth = 96 }: Props) {
  const [playing, setPlaying] = useState(false);
  const [cursorPx, setCursorPx] = useState(0);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const audio = audioRef.current;
    if (!audio) return;
    const onPlay = () => setPlaying(true);
    const onPause = () => setPlaying(false);
    audio.addEventListener('play', onPlay);
    audio.addEventListener('pause', onPause);
    audio.addEventListener('ended', onPause);
    return () => {
      audio.removeEventListener('play', onPlay);
      audio.removeEventListener('pause', onPause);
      audio.removeEventListener('ended', onPause);
    };
  }, [audioRef]);

  useEffect(() => {
    if (!playing) {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
      rafRef.current = null;
      return;
    }
    const tick = () => {
      const audio = audioRef.current;
      if (audio && durationSec > 0) {
        setCursorPx((audio.currentTime / durationSec) * timelineWidth);
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, [playing, durationSec, timelineWidth, audioRef]);

  function toggle() {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) audio.play();
    else audio.pause();
  }

  function onScrub(e: React.MouseEvent<HTMLDivElement>) {
    const audio = audioRef.current;
    if (!audio || durationSec <= 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    audio.currentTime = (x / timelineWidth) * durationSec;
  }

  return (
    <div className="flex items-center mt-2">
      <div style={{ width: labelWidth }} className="pr-2 text-right">
        <button
          type="button"
          aria-label={playing ? 'pause' : 'play'}
          onClick={toggle}
          className="text-xs px-2 py-1 rounded border hover:bg-muted"
        >
          {playing ? 'Pause' : 'Play'}
        </button>
      </div>
      <div
        className="relative bg-muted/30 rounded cursor-pointer"
        style={{ width: timelineWidth, height: 8 }}
        onClick={onScrub}
      >
        <div
          className="absolute top-0 bottom-0 bg-primary"
          style={{ left: 0, width: `${cursorPx}px` }}
        />
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd frontend && npm test -- PlaybackBar`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/PlaybackBar.tsx frontend/src/components/__tests__/PlaybackBar.test.tsx
git commit -m "feat(frontend): PlaybackBar — play/pause + cursor via rAF"
```

---

### Task C7: Rewrite `SongTimeline.tsx`

**Files:**
- Rewrite: `frontend/src/components/SongTimeline.tsx`
- Modify: `frontend/src/pages/AnalysisPage.tsx`
- Delete: `frontend/src/components/StemRow.tsx`

- [ ] **Step 1: Replace `SongTimeline.tsx` entirely**

Replace `frontend/src/components/SongTimeline.tsx`:

```typescript
import { useMemo, useRef, useState } from 'react';
import { AnalysisResult } from '@/types/song';
import { StemTrack } from './StemTrack';
import { SectionBar } from './SectionBar';
import { TimeAxis } from './TimeAxis';
import { PlaybackBar } from './PlaybackBar';
import { getMixAudioUrl } from '@/lib/api';

const STEM_ORDER = ['vocals', 'drums', 'bass', 'guitar', 'piano', 'other'];

const STEM_COLORS: Record<string, string> = {
  vocals: '#6366f1',
  drums:  '#f59e0b',
  bass:   '#10b981',
  guitar: '#a78bfa',
  piano:  '#06b6d4',
  other:  '#94a3b8',
};

const ROW_HEIGHT = 48;
const SECTION_HEIGHT = 28;
const AXIS_HEIGHT = 24;
const LABEL_WIDTH = 96;

interface Props {
  songId: string;
  analysis: AnalysisResult;
}

export function SongTimeline({ songId, analysis }: Props) {
  const [axisMode, setAxisMode] = useState<'seconds' | 'bars'>('bars');
  const [sectionLabels, setSectionLabels] = useState<Record<string, string>>({});
  const [soloed, setSoloed] = useState<Set<string>>(new Set());
  const [muted, setMuted] = useState<Set<string>>(new Set());
  const audioRef = useRef<HTMLAudioElement | null>(null);

  const timelineWidth = useMemo(
    () => Math.max(600, Math.min(1400, window.innerWidth - LABEL_WIDTH - 80)),
    []
  );

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
    if (muted.has(name)) return true;
    if (soloed.size > 0 && !soloed.has(name)) return true;
    return false;
  }

  function handleRegionClick(startSec: number) {
    const audio = audioRef.current;
    if (!audio) return;
    audio.currentTime = startSec;
    audio.play();
  }

  function handleRename(label: string, name: string) {
    setSectionLabels((prev) => ({ ...prev, [label]: name }));
  }

  return (
    <div className="space-y-4">
      <audio ref={audioRef} src={getMixAudioUrl(songId)} preload="auto" />

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
            audioRef={audioRef}
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
    </div>
  );
}
```

- [ ] **Step 2: Update `AnalysisPage.tsx`**

In `frontend/src/pages/AnalysisPage.tsx`, change the `<SongTimeline />` invocation to pass `songId`:

```typescript
{song.status === 'done' && song.analysis && song.analysis.stems && (
  <SongTimeline songId={song.id} analysis={song.analysis} />
)}
```

If the analysis JSON is the old shape (no `regions` inside a stem), show a re-analyze hint. Add this conditional block above the `<SongTimeline />`:

```typescript
{song.status === 'done' &&
  song.analysis &&
  Object.values(song.analysis.stems).some(
    (s) => !('regions' in (s as object))
  ) && (
    <div className="rounded-lg border p-4 text-sm text-muted-foreground">
      Older analysis format detected. Re-upload this song to view the new region layout.
    </div>
  )}
```

- [ ] **Step 3: Delete `StemRow.tsx`**

```bash
git rm frontend/src/components/StemRow.tsx
```

- [ ] **Step 4: Verify build + tests**

Run:
```bash
cd frontend && npm run build && npm test -- --run
```
Expected: build PASS, tests PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/components/SongTimeline.tsx frontend/src/pages/AnalysisPage.tsx
git commit -m "feat(frontend): SongTimeline rewritten to render region blocks + playback"
```

---

## Phase D — End-to-end verification

### Task D1: Manual smoke test

**Files:** none (manual)

- [ ] **Step 1: Restart all services**

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml restart audio-service
npm run start:dev
```

Wait for backend "Server started", frontend Vite ready, and audio-service `/health` to return 200.

- [ ] **Step 2: Upload a known song**

Open `http://localhost:3000`, upload an MP3 or WAV (~3-5 min). Note the song id from the URL after upload.

- [ ] **Step 3: Wait for completion**

In a separate terminal:
```bash
docker exec despiece-postgres-1 psql -U user -d mydb -c \
  "SELECT id, status, bpm, music_key FROM songs ORDER BY created_at DESC LIMIT 1;"
```
Wait until `status = done` (10-15 min).

- [ ] **Step 4: Verify analysis JSON shape**

```bash
docker exec despiece-postgres-1 psql -U user -d mydb -c \
  "SELECT jsonb_pretty(result_json) FROM song_analysis ORDER BY created_at DESC LIMIT 1;"
```
Verify: 6 stems, each with `audio_key` and `regions[]`, top-level `bar_grid`.

- [ ] **Step 5: Verify S3 stems exist**

```bash
docker exec despiece-minio-1 mc ls local/songs/<song_id>/stems/
```
(If `mc` isn't aliased, use MinIO console at `http://localhost:9001`.) Expect 6 MP3 files.

- [ ] **Step 6: Verify UI rendering and interaction**

In browser:
1. Six rows visible (vocals, drums, bass, guitar, piano, other) with region blocks
2. Mini-waveform visible inside each block
3. Click "Play" — audio starts, cursor sweeps
4. Click a region — audio jumps + plays from that timestamp
5. Click solo (S) on bass — other rows dim
6. Click mute (M) on drums — drums row dims
7. Toggle Bars/Seconds axis
8. Section labels (A/B/C/D) render above stems

- [ ] **Step 7: Verify delete cleans up stems**

Delete the song via UI. Confirm:
```bash
docker exec despiece-minio-1 mc ls local/songs/<song_id>/
```
returns no objects (or MinIO console shows the prefix gone).

- [ ] **Step 8: If everything works, mark plan complete**

No commit. Plan is done.

---

## Self-review checklist

After implementation, sanity-check against the spec:

- [ ] Spec line 14: Demucs `htdemucs` → `htdemucs_6s` — covered by **A6**
- [ ] Spec line 15: per-stem region-detection — covered by **A2-A5**
- [ ] Spec line 16: per-region mini-envelope — covered by **A5** (`_region_envelope`)
- [ ] Spec line 17: per-stem MP3 in S3 at `songs/<song_id>/stems/<stem>.mp3` — covered by **A7-A8**
- [ ] Spec line 18: SongTimeline rewrite — covered by **C3-C7**
- [ ] Spec line 19: full-mix playback with cursor — covered by **C6-C7**
- [ ] Spec line 20: click region → seek audio — covered by **C7** `handleRegionClick`
- [ ] Spec line 21: solo/mute dimming — covered by **C4-C5, C7**
- [ ] Spec line 126: `GET /api/songs/:id/stems/:stem` — covered by **B5**
- [ ] Spec line 127: `GET /api/songs/:id/audio` — covered by **B4**
- [ ] Spec line 128: `deleteFilesByPrefix` — covered by **B2**
- [ ] Spec line 129: `deleteSong` extended — covered by **B6**

If any item is unchecked, add a task before continuing.
