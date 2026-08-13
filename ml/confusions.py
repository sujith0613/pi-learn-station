"""Evidence-based confusion pairs for the writing app.

Tiers from dyslexia literature (see docs/research.md):
  Tier A - mirror / reversible letters (Fernandes & Leite 2017; Perea 2011):
           the only lowercase Latin letters differing solely by orientation.
           left-right mirrors b<->d, p<->q; up-down mirrors b<->p, d<->q.
  Tier B - voiceless/voiced phonological cognates, the most frequent
           phonological spelling errors in children (PLOS One 2019;
           Bahr/Silliman/Berninger 2012). b<->p already covered by Tier A.
  Tier C - stretch (off by default): m<->n nasal confusion + letter-swap
           detection (Friedmann letter-position dyslexia).
"""

# Pairs define the CNNs/LM disambiguation. Each tuple is an unordered pair;
# the model scores both directions from context.
TIER_A = [("b", "d"), ("p", "q"), ("b", "p"), ("d", "q")]
TIER_B = [("d", "t"), ("g", "k"), ("f", "v"), ("s", "z")]
TIER_C = [("m", "n")]

# Letters that take part in at least one confusion relationship.
CONFUSION_LETTERS = sorted({l for p in TIER_A + TIER_B for l in p})

def all_pairs(include_tier_c: bool = False) -> list[tuple[str, str]]:
    pairs = list(TIER_A) + list(TIER_B)
    if include_tier_c:
        pairs += TIER_C
    return pairs


def neighbours(letter: str, include_tier_c: bool = False) -> list[str]:
    """All confusion partners for a given letter."""
    all_ = {l for p in TIER_A + TIER_B + (TIER_C if include_tier_c else [])
            for l in p}
    return sorted(s for p in (TIER_A + TIER_B
                              + (TIER_C if include_tier_c else []))
                  for s in p if letter in p and s != letter)