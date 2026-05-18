import os
import tempfile
import traceback

import numpy as np
from fastapi import FastAPI, HTTPException
from .models import AnalyzeRequest
from .storage import download_to_path
from .analysis.demucs_runner import separate_stems
from .analysis.envelope import compute_envelope
from .analysis.beat_analysis import detect_bpm_and_beats, detect_key
from .analysis.segmentation import detect_sections

app = FastAPI(title="Audio Analysis Service")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze(req: AnalyzeRequest):
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input.audio")

        try:
            download_to_path(req.storage_key, input_path)
        except Exception as e:
            raise HTTPException(status_code=404, detail=f"Cannot fetch audio: {e}")

        try:
            stems_data = separate_stems(input_path, tmpdir)
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Demucs failed: {e}")

        # Mix all stems to mono for global analysis
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

        stems: dict = {}
        for stem_name, (audio, stem_sr) in stems_data.items():
            stems[stem_name] = {"envelope": compute_envelope(audio, stem_sr)}

        return {
            "bpm": bpm,
            "key": key,
            "duration_sec": round(duration_sec, 3),
            "beat_grid": beat_grid,
            "sections": sections,
            "stems": stems,
        }
