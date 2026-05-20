import os

import numpy as np

# Stem order for htdemucs_6s
STEMS = ["drums", "bass", "other", "vocals", "guitar", "piano"]

# Demucs splits the mix into chunks of this length (seconds) before inference.
# Smaller value → lower peak RAM at slight runtime cost. Default htdemucs_6s
# segment is ~7.8s; 4.0s cuts the peak tensor footprint roughly in half so
# the pipeline fits inside the 3.83 GiB Docker Desktop container limit.
DEMUCS_SEGMENT_SEC = float(os.environ.get("DESPIECE_DEMUCS_SEGMENT_SEC", "4.0"))
DEMUCS_OVERLAP = float(os.environ.get("DESPIECE_DEMUCS_OVERLAP", "0.1"))

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


def separate_stems(
    input_path: str, out_dir: str
) -> dict[str, tuple[np.ndarray, int]]:
    """Run Demucs on input_path; return {stem_name: (mono_float32, sample_rate)}."""
    import torch
    from demucs.apply import apply_model
    from demucs.audio import AudioFile

    model = _get_model()
    device = next(model.parameters()).device

    wav = AudioFile(input_path).read(
        streams=0,
        samplerate=model.samplerate,
        channels=model.audio_channels,
    )
    wav = wav.unsqueeze(0).to(device)  # (1, channels, samples)

    with torch.no_grad():
        sources = apply_model(
            model,
            wav,
            device=device,
            segment=DEMUCS_SEGMENT_SEC,
            overlap=DEMUCS_OVERLAP,
        )[0]  # (4, channels, samples)

    results: dict[str, tuple[np.ndarray, int]] = {}
    for i, stem_name in enumerate(STEMS):
        stem_wav = sources[i]  # (channels, samples)
        mono = stem_wav.mean(dim=0).cpu().numpy()  # (samples,) float32
        results[stem_name] = (mono, model.samplerate)

    return results
