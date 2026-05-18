import string
import numpy as np


def _uniform_sections(duration_sec: float, n: int = 4) -> list[dict]:
    """Divide song into n equal sections labeled A, B, C…"""
    seg_dur = duration_sec / n
    labels = list(string.ascii_uppercase)
    return [
        {
            "label": labels[i],
            "start_sec": round(i * seg_dur, 3),
            "end_sec": round((i + 1) * seg_dur, 3),
        }
        for i in range(n)
    ]


def detect_sections(audio: np.ndarray, sr: int, duration_sec: float) -> list[dict]:
    """
    Attempt MSAF structural segmentation; fall back to uniform sections.
    MSAF is optional — import failure silently falls back.
    """
    try:
        import msaf
        import tempfile
        import soundfile as sf
        import os

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
        try:
            sf.write(tmp_path, audio, sr)
            boundaries, _ = msaf.process(tmp_path, boundaries_id="sf", labels_id=None)
            sections = []
            label_chars = list(string.ascii_uppercase)
            for i, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
                sections.append({
                    "label": label_chars[i % len(label_chars)],
                    "start_sec": round(float(start), 3),
                    "end_sec": round(float(end), 3),
                })
            return sections if sections else _uniform_sections(duration_sec)
        finally:
            os.unlink(tmp_path)
    except Exception:
        return _uniform_sections(duration_sec)
