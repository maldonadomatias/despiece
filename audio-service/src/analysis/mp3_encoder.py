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
