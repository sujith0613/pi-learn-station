"""Context disambiguation via the char-level masked transformer.

Given a sentence (left+right context) and a target word position, mask the
word's characters, score each candidate spelling with the char-LM, and combine
with the recognizer's per-letter probabilities.

    P_final(candidate) = P_rec(candidate) * P_lm(candidate)^alpha
"""

import json
import os

import numpy as np
import onnxruntime as ort

import app.confusions as conf

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "lm", "model.onnx"
)
VOCAB_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "lm", "vocab.json"
)
ALPHA = 1.0              # LM weight in the combine
SUGGEST_MARGIN = 0.15    # min log-likelihood margin to surface a suggestion
MAX_LEN = 64

_session = None
_vocab = None
_MASK_ID = 1
_PAD_ID = 0


def _get_session():
    global _session
    if _session is None:
        path = os.path.abspath(MODEL_PATH)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"charlm.onnx missing: {path} (train + export in ml/)"
            )
        _session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    return _session


def _get_vocab():
    global _vocab
    if _vocab is None:
        with open(VOCAB_PATH) as f:
            _vocab = json.load(f)["char2id"]
    return _vocab


def _tokenize(sentence: str):
    """Map sentence to (token ids, char_offset list) using the LM char vocab.

    Returns ids and a parallel list mapping each token index back to its
    character position in `sentence` (None for dropped/unknown chars).
    """
    v = _get_vocab()
    ids = []
    offsets = []
    for i, ch in enumerate(sentence):
        if ch in v:
            ids.append(v[ch])
            offsets.append(i)
    return ids, offsets


def score_candidate(sentence: str, word_start: int, word_end: int,
                    candidate: str) -> float:
    """Log-likelihood of `candidate` replacing [word_start, word_end) in
    sentence, from the char-LM.

    Mask every char in the word region, run the masked transformer ONCE, then
    sum the log-probs the model assigns to each of the candidate's chars at the
    corresponding positions.
    """
    ids, offsets = _tokenize(sentence)
    if not ids:
        return -1e9

    # token range covering [word_start, word_end)
    tok_lo = tok_hi = None
    for j, off in enumerate(offsets):
        if off >= word_start and tok_lo is None:
            tok_lo = j
        if off >= word_end:
            tok_hi = j
            break
    if tok_lo is None:
        return -1e9
    if tok_hi is None:
        tok_hi = len(ids)
    if len(candidate) != (tok_hi - tok_lo):
        # lengths must match (the recognizer returned a same-length candidate);
        # otherwise fall back to a flat prior
        return 0.0

    masked = ids[:]
    for j in range(tok_lo, tok_hi):
        masked[j] = _MASK_ID
    seq = np.array([masked], dtype=np.int64)

    sess = _get_session()
    logits = sess.run(None, {"token_ids": seq})[0][0]  # (L, V)
    logp = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True) + 1e-9)

    v = _get_vocab()
    score = 0.0
    for j, ch in enumerate(candidate):
        if ch in v:
            score += float(logp[tok_lo + j, v[ch]])
    return score


def disambiguate(sentence: str, word_start: int, word_end: int,
                 recognizer_candidates) -> dict:
    """Combine recognizer per-letter probabilities with the char-LM to pick the
    best spelling for the target word.

    recognizer_candidates: list/tuple of (letter, prob) in probability order
    for each char position of the word (outer = positions, inner = letters).

    Returns {"best": str, "alternatives": [str,...]} if the LM's top choice
    beats the recognizer's greedy spelling by SUGGEST_MARGIN, else {best} only.
    """
    n = word_end - word_start
    # greedy recognizer spelling: argmax prob at each position
    greedy = "".join(max(pos, key=lambda kv: kv[1])[0]
                     for pos in recognizer_candidates)
    if len(recognizer_candidates) != n:
        return {"best": greedy}

    # build candidate spellings that swap in confusion letters
    best_spelling = greedy
    best_score = score_candidate(sentence, word_start, word_end, greedy)
    alternatives = []

    for pos_idx, pos in enumerate(recognizer_candidates):
        letter = max(pos, key=lambda kv: kv[1])[0]
        for nb in conf.neighbours(letter):
            if nb not in {l for l, _ in pos}:
                continue
            cand = base[:pos_idx] + nb + base[pos_idx + 1:]
            sc = score_candidate(sentence, word_start, word_end, cand)
            if sc > best_score:
                best_score = sc
                best_spelling = cand
            if nb != letter:
                alternatives.append(cand)

    if best_spelling != greedy:
        margin = best_score - score_candidate(sentence, word_start, word_end,
                                              greedy)
        if margin >= SUGGEST_MARGIN:
            return {"best": best_spelling, "alternatives": alternatives}
    return {"best": greedy}