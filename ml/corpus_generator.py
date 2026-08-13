"""Corpus expansion + eval-set construction for the char-level LM.

Training corpus (expanded deterministically to ~30-40k lowercase sentences):
  * the Claude seed sentences (Sections 1-7) verbatim
  * grammar-driven template expansion so every confusion letter appears often

Eval set (the honest >=85% gate):
  * driven by the Section-8 minimal pairs (base_word/alt_word), where BOTH
    sides are real English words and differ in exactly one confusion letter.
  * For each pair, place the base word in a few natural sentence contexts,
    mask the confusion letter, and record
        {sentence, pos, correct_char, alt_char}
  * The LM passes the case iff it scores correct_char > alt_char at pos.
  * We do NOT include cases where the alt produces gibberish -- that would
    inflate accuracy. Every case here is a genuinely confusable, in-vocab word.

Outputs (data/corpus/):
    train_sentences.txt   expanded training corpus (one sentence/line, lowercase)
    eval_cases.json       {sentence, pos, correct, alt}
    min_pairs.json        raw section-8 pairs
"""

import json
import os
import random
import re

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "corpus"))
RAW = os.path.join(ROOT, "raw_claude_sentences.txt")
SEED = 42

WORD_BANKS = {
    "b": ["big", "bad", "bed", "bag", "ball", "bird", "book", "box", "boy",
          "bun", "bus", "blue", "black", "but", "back", "best", "bump",
          "barn", "bell", "bat", "bear", "boat", "bone", "boot", "bee"],
    "d": ["dog", "dig", "duck", "dad", "door", "down", "day", "do", "did",
          "doll", "dot", "dark", "deep", "dip", "drum", "dirt", "dust",
          "dime", "deer", "desk", "draw"],
    "p": ["pig", "pink", "pat", "pen", "pet", "pot", "pup", "pan", "peg",
          "park", "pond", "put", "pull", "plum", "pump", "pad", "pod", "pip"],
    "q": ["quack", "queen", "quick", "quit", "quiz", "quiet", "quill",
          "quilt"],
    "t": ["top", "ten", "toe", "tap", "tin", "tub", "tug", "tall", "tip",
          "tree", "truck", "train"],
    "g": ["go", "get", "got", "give", "good", "gate", "goat", "gum", "gun",
          "girl", "glad", "grab", "green", "grass", "grow"],
    "k": ["kick", "king", "kit", "kid", "keep", "key", "kiss", "kite",
          "kind", "look", "book", "cook", "bake", "make", "lake", "bike"],
    "f": ["fun", "fan", "far", "fast", "fish", "fit", "fox", "five", "foot",
          "frog", "from", "friend", "four"],
    "v": ["van", "vet", "very", "vase", "vest", "visit", "have", "live",
          "love", "give", "five", "of"],
    "s": ["sun", "sad", "sit", "six", "sock", "see", "so", "sand", "sell",
          "send", "sing", "small", "stop", "swim", "bus", "gas", "has",
          "is", "his", "this", "yes", "dress", "grass", "miss"],
    "z": ["zip", "zoo", "zero", "zigzag", "zebra", "buzz", "fizz", "fuzz"],
}


def _parse():
    sentences = []
    pairs = []
    sec = None
    with open(RAW) as f:
        for line in f:
            s = line.strip().lower()
            if s.startswith("=== section"):
                sec = s.replace("=", "").strip()
                continue
            if sec == "section 8" and "/" in s:
                a, b = s.split("/", 1)
                pairs.append((a.strip(), b.strip()))
            elif sec and sec != "section 8" and s:
                sentences.append(s)
    return sentences, pairs


def split(sentences, ratio=0.85):
    rng = random.Random(SEED)
    shuffled = sentences[:]
    rng.shuffle(shuffled)
    k = int(len(shuffled) * ratio)
    return shuffled[:k], shuffled[k:]


SUBJECTS = ["the", "a", "my", "your", "her", "our", "that", "this", "look at",
            "see the", "i see", "you see", "we see", "the big", "the red",
            "little", "my best", "the first", "our new"]
VERBS = ["sees", "likes", "finds", "has", "wants", "gets", "needs", "spots",
         "picks", "holds", "touches", "draws", "brings", "made", "found",
         "saw", "wants the", "put the", "keeps the"]
MOD = ["my", "his", "her", "our", "the", "a", "that", "this", "some", "the big",
       "every", "another", "the little"]
VERY = ["very", "so", "quite", "really", ""]


def expand_template(train, pairs):
    """Grammar-driven expansion so every confusion letter appears often enough
    for the model to learn word identity. Returns a superset of `train`.
    Also seeds the section-8 minimal-pair words into natural contexts so the
    model learns their identity (their distinguishing letter is what the gate
    tests)."""
    rng = random.Random(SEED + 1)
    extra = []

    # section-8 words: both sides of every minimal pair, in natural contexts
    sec8_words = set()
    for bw, aw in pairs:
        sec8_words.add(bw)
        sec8_words.add(aw)

    # guarantee every confusion-letter word bank word shows up many times
    bank_words = set(sum(WORD_BANKS.values(), [])) | sec8_words
    for w in sorted(bank_words):
        for _ in range(3):
            subj = rng.choice(SUBJECTS)
            punct = rng.choice([".", "!", "?"])
            pat = rng.randrange(6)
            adj = rng.choice(["big", "small", "red", "fast", "new", "old",
                              "good", "fun", ""])
            here = rng.choice(["here", "there", "near us", "in the box",
                               "on the desk", "under the tree"])
            if pat == 0:
                sv = rng.choice(VERBS)
                extra.append(f"{subj} {sv} {_art(w)} {w}{punct}")
            elif pat == 1:
                extra.append(f"the {w} is {here}{punct}")
            elif pat == 2:
                extra.append(f"{rng.choice(['this','that','it','here is','there is'])} {w}{punct}")
            elif pat == 3:
                extra.append(f"i like the {w} {rng.choice(['a lot','best','most','now'])}{punct}")
            elif pat == 4:
                extra.append(f"{subj} has {_art(w)} {w} {here}{punct}")
            else:
                extra.append(f"the {adj} {w} {rng.choice(['ran','fell','slept','stopped','sang','jumped'])}{punct}")
    # add a few longer structured sentences using multiple bank words
    bank = sorted(bank_words)
    for _ in range(16000):
        w1 = rng.choice(bank)
        w2 = rng.choice(bank)
        punct = rng.choice([".", ".", "!", "?"])
        extra.append(f"my {rng.choice(['best','little','old','new'])} friend "
                     f"{rng.choice(['saw','kept','drew','made','found'])} "
                     f"{_art(w1)} {w1} and {_art(w2)} {w2}{punct}")
    expanded = list(train) + extra
    return expanded


def _art(w):
    return "an" if w[0] in "aeiou" else "a"


def _masked_eval(test_sentences, pairs):
    """Eval from REAL held-out corpus sentences.

    For each section-8 minimal pair (base, alt) that differ in exactly one
    confusion letter, find held-out test sentences that contain `base`, mask
    that confusion char, and record {sentence, pos, correct, alt}.

    Contexts are authentic English (from the Claude corpus) rather than
    synthetic templates, so the gate measures real disambiguation instead of
    letter-prior fallback.
    """
    occ = {}
    for s in test_sentences:
        for m in re.finditer(r"[a-z]+", s):
            w = m.group(0)
            occ.setdefault(w, []).append((s, m.start()))

    cases = []
    for base, alt in pairs:
        diffs = [(i, x, y) for i, (x, y) in enumerate(zip(base, alt)) if x != y]
        if len(diffs) != 1 or len(base) != len(alt):
            continue
        i, correct, alt_ch = diffs[0]
        for sent, wstart in occ.get(base, []):
            pos = wstart + i
            cases.append({
                "sentence": sent,
                "pos": pos,
                "correct": correct,
                "alt": alt_ch,
            })
    return cases


def main():
    sentences, pairs = _parse()
    train, test = split(sentences)

    expanded = expand_template(train, pairs)
    train_out = os.path.join(ROOT, "train_sentences.txt")
    with open(train_out, "w") as f:
        for s in expanded:
            f.write(s.strip() + "\n")

    eval_cases = _masked_eval(test, pairs)
    with open(os.path.join(ROOT, "min_pairs.json"), "w") as f:
        json.dump(pairs, f, indent=2)
    with open(os.path.join(ROOT, "eval_cases.json"), "w") as f:
        json.dump(eval_cases, f, indent=2)

    from confusions import all_pairs
    per_pair = {}
    for a, b in all_pairs():
        per_pair[f"{a}/{b}"] = sum(
            1 for c in eval_cases
            if {c["correct"], c["alt"]} == {a, b}
            or {c["correct"], c["alt"]} == {b, a})

    print(f"parsed: sentences={len(sentences)} min_pairs={len(pairs)}")
    print(f"split: train={len(train)} test={len(test)}")
    print(f"expanded train={len(expanded)}")
    print(f"eval cases: {len(eval_cases)}")
    print("eval cases by confusion pair:")
    for k, v in per_pair.items():
        print(f"  {k}: {v}")
    print(f"wrote {train_out}")


if __name__ == "__main__":
    main()