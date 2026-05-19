"""Collapse PANNs CNN14 AudioSet (527-class) tags to a 6-class taxonomy.

The TAXONOMY dict is data, not algorithm logic. Iterate it freely without
touching `collapse_to_sub_label`. Unmapped tags fall through to
`other_misc` with the dominant unmapped probability.
"""
from __future__ import annotations

from typing import Literal

SubLabel = Literal["lead", "pad", "synth", "strings", "fx", "other_misc"]

TAXONOMY: dict[str, SubLabel] = {
    # strings
    "Violin, fiddle": "strings",
    "Cello": "strings",
    "Double bass": "strings",
    "Orchestra": "strings",
    "String section": "strings",
    "Pizzicato": "strings",
    # pad
    "Pad": "pad",
    "Choir": "pad",
    # synth
    "Synthesizer": "synth",
    "Sampler": "synth",
    "Electronic music": "synth",
    # lead
    "Lead (synth lead)": "lead",
    "Saxophone": "lead",
    "Trumpet": "lead",
    "Flute": "lead",
    "Whistle": "lead",
    # fx
    "Sound effect": "fx",
    "Noise": "fx",
    "Whoosh, swoosh, swish": "fx",
    "Boom": "fx",
    "Reverberation": "fx",
    "Echo": "fx",
}


def collapse_to_sub_label(
    audioset_tags: list[tuple[str, float]],
) -> tuple[SubLabel, float]:
    """Collapse top-k AudioSet labels into the 6-class taxonomy.

    Sums probabilities per sub_label across input tags. Returns the
    argmax sub_label and its clipped sum. Unmapped tags accumulate
    against `other_misc`.
    """
    if not audioset_tags:
        return ("other_misc", 0.0)

    sums: dict[SubLabel, float] = {}
    unmapped_max = 0.0
    for tag, prob in audioset_tags:
        sub = TAXONOMY.get(tag)
        if sub is None:
            if prob > unmapped_max:
                unmapped_max = prob
            continue
        sums[sub] = sums.get(sub, 0.0) + float(prob)

    if not sums:
        return ("other_misc", float(min(unmapped_max, 1.0)))

    best = max(sums.items(), key=lambda kv: kv[1])
    return (best[0], float(min(best[1], 1.0)))
