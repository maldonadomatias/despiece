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

        if USE_RESIDUAL_SUBTRACT:
            try:
                bass_audio, bass_sr = stems_data["bass"]
                for h in ("guitar", "piano"):
                    h_audio, h_sr = stems_data[h]
                    if h_sr != bass_sr:
                        continue
                    h_cleaned = spectral_subtract(
                        h_audio, bass_audio, sr=h_sr, alpha=RESIDUAL_ALPHA
                    )
                    stems_data[h] = (h_cleaned, h_sr)
            except Exception:
                traceback.print_exc()

        all_audio = [audio for audio, _ in stems_data.values()]
        sr = next(iter(stems_data.values()))[1]
        mix_mono: np.ndarray = (
            np.mean(all_audio, axis=0) if all_audio else np.zeros(1)
        )
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
                regions = detect_regions(
                    audio, stem_sr, beat_grid, beats_per_bar=BEATS_PER_BAR
                )
            except Exception:
                traceback.print_exc()
                regions = []

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
    bars = [
        round(float(beat_times[i]), 3)
        for i in range(0, len(beat_times), beats_per_bar)
    ]
    if bars[0] > 0.0:
        bars.insert(0, 0.0)
    if bars[-1] < audio_duration:
        bars.append(round(float(audio_duration), 3))
    return bars
