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

app = FastAPI(title="Audio Analysis Service")

BEATS_PER_BAR = 4
STEM_MP3_BITRATE_KBPS = 128


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


def _bar_grid_from_beats(beat_times, beats_per_bar, audio_duration):
    if len(beat_times) < 2:
        return [0.0, round(float(audio_duration), 3)]
    bars = [round(float(beat_times[i]), 3) for i in range(0, len(beat_times), beats_per_bar)]
    if bars[0] > 0.0:
        bars.insert(0, 0.0)
    if bars[-1] < audio_duration:
        bars.append(round(float(audio_duration), 3))
    return bars
