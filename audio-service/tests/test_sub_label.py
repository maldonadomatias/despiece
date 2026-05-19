from src.analysis.sub_label import collapse_to_sub_label


def test_strings_dominant():
    label, conf = collapse_to_sub_label([
        ("Violin, fiddle", 0.6),
        ("Cello", 0.2),
        ("Music", 0.05),
    ])
    assert label == "strings"
    assert 0.79 <= conf <= 0.81


def test_synth_dominant():
    label, conf = collapse_to_sub_label([
        ("Synthesizer", 0.7),
        ("Music", 0.1),
    ])
    assert label == "synth"
    assert 0.69 <= conf <= 0.71


def test_unmapped_falls_back_to_other_misc():
    label, conf = collapse_to_sub_label([
        ("Music", 0.9),
        ("Speech", 0.05),
    ])
    assert label == "other_misc"
    assert 0.89 <= conf <= 0.91


def test_empty_input():
    label, conf = collapse_to_sub_label([])
    assert label == "other_misc"
    assert conf == 0.0


def test_confidence_clipped_to_one():
    label, conf = collapse_to_sub_label([
        ("Violin, fiddle", 0.7),
        ("Cello", 0.4),
    ])
    assert label == "strings"
    assert conf == 1.0
