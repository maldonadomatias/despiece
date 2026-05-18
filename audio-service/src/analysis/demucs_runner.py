import numpy as np

# Stem order for htdemucs
STEMS = ["drums", "bass", "other", "vocals"]

_model = None


def _get_model():
    global _model
    if _model is None:
        import torch
        from demucs.pretrained import get_model

        _model = get_model("htdemucs")
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
        sources = apply_model(model, wav, device=device)[0]  # (4, channels, samples)

    results: dict[str, tuple[np.ndarray, int]] = {}
    for i, stem_name in enumerate(STEMS):
        stem_wav = sources[i]  # (channels, samples)
        mono = stem_wav.mean(dim=0).cpu().numpy()  # (samples,) float32
        results[stem_name] = (mono, model.samplerate)

    return results
