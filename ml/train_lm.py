"""Character-level masked-transformer language model, pure PyTorch.

Purpose
-------
Learn letter/word co-occurrence so we can disambiguate the 8 confusing letter
pairs (b/d, b/p, d/q, d/t, f/v, g/k, p/q, s/z) using left+right context.

Architecture
------------
A small encoder-only transformer (BERT-style) over a char vocabulary.
MLM head over ~35 tokens. Trained on data/corpus/train_sentences.txt.

Output
------
models/lm/model.pt   model state dict
models/lm/vocab.json  char->id
models/lm/config.json
"""

import json
import os
import random
import time

import torch
import torch.nn as nn

SEED = 42
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data", "corpus", "train_sentences.txt")
OUT = os.path.join(ROOT, "models", "lm")

MAX_LEN = 64
VOCAB_CHARS = list("abcdefghijklmnopqrstuvwxyz .!?")  # 30 chars
PAD, MASK = 0, 1
SPECIALS = {"<pad>": PAD, "<mask>": MASK}
CHAR2ID = {c: i + 2 for i, c in enumerate(VOCAB_CHARS)}
ID2CHAR = {v: k for k, v in CHAR2ID.items()}
ID2CHAR[PAD], ID2CHAR[MASK] = "<pad>", "<mask>"
VOCAB_SIZE = len(CHAR2ID) + 2

# model hyperparams (small enough for CPU / Pi)
CFG = dict(
    vocab_size=VOCAB_SIZE,
    d_model=128,
    nhead=4,
    nlayer=4,
    d_ff=256,
    max_len=MAX_LEN,
    dropout=0.1,
)


class CharMLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["d_model"]
        self.tok = nn.Embedding(cfg["vocab_size"], d)
        self.pos = nn.Embedding(cfg["max_len"], d)
        self.norm = nn.LayerNorm(d)
        layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg["nhead"], dim_feedforward=cfg["d_ff"],
            dropout=cfg["dropout"], batch_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=cfg["nlayer"])
        self.lm_head = nn.Linear(d, cfg["vocab_size"])

    def forward(self, x, mask_2d=None):
        n, L = x.shape
        tok = self.tok(x)
        pos = self.pos(torch.arange(L, device=x.device))
        h = tok + pos
        if mask_2d is not None:
            key_pad = mask_2d.logical_not()
        else:
            key_pad = None
        h = self.enc(h, src_key_padding_mask=key_pad)
        h = self.norm(h)
        return self.lm_head(h)


def tokenize(sent):
    ids = [CHAR2ID[c] for c in sent if c in CHAR2ID]
    return ids[:MAX_LEN]


def collate_batch(sents, rng):
    # pad batch to max length
    toks = [tokenize(s) for s in sents]
    L = max(len(t) for t in toks)
    x = torch.full((len(toks), L), PAD, dtype=torch.long)
    for i, t in enumerate(toks):
        x[i, : len(t)] = torch.tensor(t)
    labels = x.clone()
    for i in range(x.shape[0]):
        Ln = len(toks[i])
        if Ln == 0:
            continue
        n_mask = max(1, int(0.15 * Ln))
        idx = rng.sample(range(Ln), n_mask)
        for j in idx:
            r = rng.random()
            if r < 0.8:
                x[i, j] = MASK
            elif r < 0.9:
                x[i, j] = rng.randrange(2, VOCAB_SIZE)
            # else keep original
    attn = (x != PAD)
    return x, labels, attn


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(DATA) as f:
        sents = [ln.strip() for ln in f if ln.strip()]
    rng = random.Random(SEED)
    torch.manual_seed(SEED)

    model = CharMLM(CFG)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    nll = nn.CrossEntropyLoss(ignore_index=PAD)

    B = 64
    steps_per_epoch = max(1, len(sents) // B)
    total = steps_per_epoch * 10
    print(f"corpus={len(sents)} vocab={VOCAB_SIZE} d={CFG['d_model']} "
          f"layers={CFG['nlayer']} steps={total}")
    t0 = time.time()
    step = 0
    for epoch in range(12):
        rng.shuffle(sents)
        running = 0.0
        for b in range(steps_per_epoch):
            batch = sents[b * B : (b + 1) * B]
            x, labels, attn = collate_batch(batch, rng)
            logits = model(x, attn)
            loss = nll(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
            step += 1
            if step % 500 == 0:
                dt = time.time() - t0
                print(f"  step {step}/{total} loss={running/500:.4f} "
                      f"({dt:.0f}s)")
                running = 0.0
        print(f"epoch {epoch+1}/12 loss={running/steps_per_epoch:.4f}")

    torch.save(model.state_dict(), os.path.join(OUT, "model.pt"))
    with open(os.path.join(OUT, "vocab.json"), "w") as f:
        json.dump({"char2id": CHAR2ID, "id2char": ID2CHAR}, f)
    with open(os.path.join(OUT, "config.json"), "w") as f:
        json.dump(CFG, f, indent=2)
    print(f"done {time.time()-t0:.0f}s -> {OUT}/model.pt")


if __name__ == "__main__":
    main()