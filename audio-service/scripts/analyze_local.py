"""Run the Phase A pipeline on a local audio file, write JSON to stdout.

Usage:  python scripts/analyze_local.py path/to/song.mp3
"""
import json
import os
import sys
import tempfile

import numpy as np

# Force the module-level toggles to defaults before importing main.
os.environ.setdefault("DESPIECE_USE_ROFORMER_VOCALS", "true")
os.environ.setdefault("DESPIECE_USE_ROFORMER_BASS", "true")
os.environ.setdefault("DESPIECE_USE_RESIDUAL_SUBTRACT", "true")
os.environ.setdefault("DESPIECE_USE_TAGGER", "true")

from src.analysis.demucs_runner import separate_stems, STEMS
from src.analysis.beat_analysis import detect_bpm_and_beats
from src.analysis.regions import detect_regions
from src.analysis.ensemble import (
    separate_roformer_vocals,
    separate_roformer_bass,
    free_model as ensemble_free,
)
from src.analysis.residual import spectral_subtract
from src.analysis.tagging import tag_clip, free_model as tagging_free
from src.analysis.sub_label import collapse_to_sub_label


def main(audio_path: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        stems_data = separate_stems(audio_path, tmpdir)

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
            stems_data[h] = (
                spectral_subtract(ha, bass_a, sr=hsr, alpha=0.5),
                hsr,
            )

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

    print(
        json.dumps(
            {
                "analysis_version": 2,
                "bpm": bpm,
                "duration_sec": round(duration, 3),
                "beat_grid": beat_grid,
                "stems": out_stems,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python scripts/analyze_local.py path/to/audio", file=sys.stderr)
        sys.exit(2)
    main(sys.argv[1])
