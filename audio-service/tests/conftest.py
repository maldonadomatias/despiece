import numpy as np
import pytest


@pytest.fixture
def sample_audio():
    """440 Hz sine wave, 5 seconds, sr=22050."""
    sr = 22050
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return y, sr
