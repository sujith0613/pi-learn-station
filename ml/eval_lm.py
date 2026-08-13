"""Evaluate the char-LM on the disambiguation gate (+confusion cases).

For each eval case {sentence, pos, correct, alt}: mask char at pos, feed the
sequence, and check P(correct) > P(alt) at that position.

Reports per-pair accuracy and the overall macro-mean gate (>=85% required).
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from train_lm import CharMLM  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LM = os.path.join(ROOT, "models", "lm")
EVAL = os.path.join(ROOT, "data", "corpus", "eval_cases.json")
MIN_PAIRS = os.path.join(ROOT, "data", "corpus", "min_pairs.json")

GATE = 0.85

PAIRS = [("b", "d"), ("b", "p"), ("d", "q"), ("d", "t"),
         ("f", "v"), ("g", "k"), ("p", "q"), ("s", "z")]


def load_model():
    import json as _j
    cfg = _j.load(open(os.path.join(LM, "config.json")))
    vocab = _j.load(open(os.path.join(LM, "vocab.json")))
    model = CharMLM(cfg)
    model.load_state_dict(torch.load(os.path.join(LM, "model.pt"),
                                     map_location="cpu"))
    model.eval()
    return model, cfg, vocab


def run():
    cases = json.load(open(EVAL))
    model, cfg, vocab = load_model()
    char2id = vocab["char2id"]
    MAXL = cfg["max_len"]

    stats = {p: [0, 0] for p in PAIRS}  # correct, total
    with torch.no_grad():
        for c in cases:
            sent = c["sentence"]
            pos, correct, alt = c["pos"], c["correct"], c["alt"]
            tok_pos = _token_pos(sent, pos)
            if tok_pos is None:
                continue
            ids = [char2id[ch] for ch in sent if ch in char2id]
            if tok_pos >= len(ids):
                continue
            ids_t = ids[:]
            ids_t[tok_pos] = 1  # mask id
            x = torch.tensor([ids_t])
            attn = torch.ones_like(x).bool()
            logits = model(x, attn)[0, tok_pos]
            p = torch.softmax(logits.float(), -1)
            pc = p[char2id[correct]].item()
            pa = p[char2id[alt]].item()
            if (correct, alt) in PAIRS:
                key = (correct, alt)
            elif (alt, correct) in PAIRS:
                key = (alt, correct)
            else:
                continue
            stats[key][1] += 1
            if pc > pa:
                stats[key][0] += 1

    correct_total = 0
    total = 0
    macro = []
    print(f"{'pair':<6}{'tok_ok':>6}{'acc':>8}")
    for p in PAIRS:
        ok, tot = stats[p]
        acc = ok / tot if tot else 0
        macro.append(acc)
        correct_total += ok
        total += tot
        print(f"{p[0]+'/'+p[1]:<6}{tot:>6}{acc:>8.1%}")
    macro_mean = sum(macro) / len(macro)
    overall = correct_total / total if total else 0
    print(f"\noverall={overall:.1%}  macro={macro_mean:.1%}  "
          f"gate={GATE:.0%} -> {'PASS' if macro_mean >= GATE else 'FAIL'}")

    best_worst = sorted(macro)
    print(f"worst pair acc={best_worst[0]:.1%}")


def _token_pos(sentence, char_pos):
    """Map a character index in sentence to token index (skips non-vocab)."""
    import random
    k = 0
    for i, ch in enumerate(sentence):
        if ch not in CHAR2ID_GLOBAL:
            continue
        if i == char_pos:
            return k
        k += 1
    return None


# global char set from train_lm vocabulary
from train_lm import CHAR2ID  # noqa: E402
CHAR2ID_GLOBAL = CHAR2ID


if __name__ == "__main__":
    run()