from src.analysis.demucs_runner import STEMS


def test_stems_list_is_six_for_htdemucs_6s():
    assert len(STEMS) == 6
    assert set(STEMS) == {"drums", "bass", "other", "vocals", "guitar", "piano"}
