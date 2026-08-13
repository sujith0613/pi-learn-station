"""Export the char-LM to ONNX for on-device inference (Pi).

The dynamic input is (batch, seq_len) long tokens; ONNX needs fixed shapes or
dynamic axes. We use dynamic axes on the sequence dim so the backend can mask
arbitrary-length strokes.

Outputs models/lm/model.onnx
"""

import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from train_lm import CharMLM  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
LM = os.path.join(ROOT, "models", "lm")


def main():
    cfg = json.load(open(os.path.join(LM, "config.json")))
    model = CharMLM(cfg)
    model.load_state_dict(torch.load(os.path.join(LM, "model.pt"),
                                     map_location="cpu"))
    model.eval()
    dummy = torch.zeros(1, 16, dtype=torch.long)
    torch.onnx.export(
        model, (dummy,),
        os.path.join(LM, "model.onnx"),
        input_names=["token_ids"],
        output_names=["logits"],
        dynamic_axes={"token_ids": {0: "batch", 1: "seq"},
                      "logits": {0: "batch", 1: "seq"}},
        opset_version=12,
    )
    print("exported", os.path.join(LM, "model.onnx"))


if __name__ == "__main__":
    main()