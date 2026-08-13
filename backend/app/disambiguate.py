"""Context disambiguation via the char-level masked transformer.

Given a sentence (left+right context) and a target word position, mask the
word's characters, score each candidate spelling with the char-LM, and combine
with the recognizer's per-letter probabilities.

    P_final(candidate) = P_rec(candidate) * P_lm(candidate)^alpha
"""

import os

import numpy as np
import onnxruntime as ort

MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "charlm.onnx"
)
ALPHA = 1.0              # LM weight in the combine
SUGGEST_MARGIN = 0.15    # min log-likelihood margin to surface a suggestion
MAX_LEN = 64

_session = None


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


def score_candidate(sentence: str, word_start: int, word_end: int,
                    candidate: str) -> float:
    """Log-likelihood of `candidate` replacing [word_start, word_end) in sentence.

    Filled in Phase 3: tokenize (char vocab), mask the word region, run the
    masked transformer, sum per-position log-probs for the candidate's chars.
    """
    raise NotImplementedError("Phase 3")


def disambiguate(sentence: str, word_start: int, word_end: int,
                 recognizer_candidates) -> dict:
    """Returns {best: str, alternatives: [...]} or {} if below margin."""
    raise NotImplementedError("Phase 3")