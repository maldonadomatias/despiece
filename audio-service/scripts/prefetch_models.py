"""Pre-pull model weights at Docker-build time so the running container
never blocks on a first-run download."""
import sys


def pull_demucs():
    from demucs.pretrained import get_model
    get_model("htdemucs_6s")


def pull_audio_separator():
    from audio_separator.separator import Separator
    sep = Separator(log_level=30)
    for name in (
        "model_bs_roformer_ep_368_sdr_12.9628.ckpt",
        "MDX23C-InstVoc_HQ.ckpt",
    ):
        try:
            sep.load_model(name)
        except Exception as e:
            print(f"[prefetch] could not pre-pull {name}: {e}", file=sys.stderr)


def pull_panns():
    from panns_inference import AudioTagging
    AudioTagging(checkpoint_path=None, device="cpu")


if __name__ == "__main__":
    for name, fn in (
        ("demucs", pull_demucs),
        ("audio-separator", pull_audio_separator),
        ("panns", pull_panns),
    ):
        print(f"[prefetch] {name}...", flush=True)
        try:
            fn()
            print(f"[prefetch] {name} ok", flush=True)
        except Exception as e:
            print(f"[prefetch] {name} failed: {e}", file=sys.stderr)
