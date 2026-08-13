"""Confusion pairs shared with ml/confusions.py (kept in sync manually).

The backend needs the pair table at runtime to know which letters the char-LM
may swap, independent of the training code in ml/.
"""

TIER_A = [("b", "d"), ("p", "q"), ("b", "p"), ("d", "q")]
TIER_B = [("d", "t"), ("g", "k"), ("f", "v"), ("s", "z")]
TIER_C = [("m", "n")]


def all_pairs(include_tier_c: bool = False) -> list[tuple[str, str]]:
    pairs = list(TIER_A) + list(TIER_B)
    if include_tier_c:
        pairs += TIER_C
    return pairs


def neighbours(letter: str, include_tier_c: bool = False) -> list[str]:
    pairs = all_pairs(include_tier_c)
    out = []
    for a, b in pairs:
        if a == letter:
            out.append(b)
        elif b == letter:
            out.append(a)
    return out


CONFUSION_LETTERS = sorted({l for p in all_pairs() for l in p})
