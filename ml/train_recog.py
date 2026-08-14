"""Train a 62-class EMNIST ByClass recognizer ON the app's input distribution.

Input: data/processed/train62_uint8.npz (produced by prep_emnist_render.py) —
glyphs already rendered through the app's exact normalize path (crop ->
squash 20x20 -> 4px border -> ink=1), with handwriting augmentation.

Architecture: the ScriboGenie myCnn ResNet (3 residual blocks: 64/128/256 +
GAP + FC), ported to PyTorch. The exported ONNX accepts NHWC (28,28,1) input,
identical to the current backend contract.

Outputs (models/recog/):
  model.onnx   (NHWC dynamic batch, 62-class softmax)
  config.json  (class order = CHAR_LIST, input ["N",28,28,1])
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC = os.environ.get("EMNIST_PROC", os.path.join(ROOT, "data", "processed"))
OUT = os.environ.get("EMNIST_OUT", os.path.join(ROOT, "models", "recog"))
SEED = 42

CHAR_LIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
EPOCHS = int(os.environ.get("EMNIST_EPOCHS", "10"))
BATCH = int(os.environ.get("EMNIST_BATCH", "512"))
LR = 1e-3
THREADS = int(os.environ.get("EMNIST_THREADS", "4"))
HEARTBEAT_S = int(os.environ.get("EMNIST_HEARTBEAT", "30"))


class ResNet(nn.Module):
    """myCnn (ScriboGenie) ResNet port. Expects NCHW input in forward()."""

    def __init__(self, num_classes=62):
        super().__init__()
        # block1
        self.c1 = nn.Conv2d(1, 64, 3, padding=1, bias=False); self.b1 = nn.BatchNorm2d(64)
        self.c1b = nn.Conv2d(64, 64, 3, padding=1, bias=False); self.b1b = nn.BatchNorm2d(64)
        self.sc1 = nn.Conv2d(1, 64, 1, bias=False)
        # block2
        self.c2 = nn.Conv2d(64, 128, 3, padding=1, bias=False); self.b2 = nn.BatchNorm2d(128)
        self.c2b = nn.Conv2d(128, 128, 3, padding=1, bias=False); self.b2b = nn.BatchNorm2d(128)
        self.sc2 = nn.Conv2d(64, 128, 1, bias=False)
        # block3
        self.c3 = nn.Conv2d(128, 256, 3, padding=1, bias=False); self.b3 = nn.BatchNorm2d(256)
        self.c3b = nn.Conv2d(256, 256, 3, padding=1, bias=False); self.b3b = nn.BatchNorm2d(256)
        self.sc3 = nn.Conv2d(128, 256, 1, bias=False)
        self.fc1 = nn.Linear(256, 512); self.fc2 = nn.Linear(512, num_classes)
        self.pool = nn.MaxPool2d(2)
        self.dp = nn.Dropout(0.25)
        self.dp2 = nn.Dropout(0.5)

    def forward(self, x):  # NCHW
        # block1
        h = torch.relu(self.b1(self.c1(x)))
        h = torch.relu(self.b1b(self.c1b(h)))
        x = h + self.sc1(x)
        x = self.dp(self.pool(x))
        # block2
        h = torch.relu(self.b2(self.c2(x)))
        h = torch.relu(self.b2b(self.c2b(h)))
        x = h + self.sc2(x)
        x = self.dp(self.pool(x))
        # block3
        h = torch.relu(self.b3(self.c3(x)))
        h = torch.relu(self.b3b(self.c3b(h)))
        x = h + self.sc3(x)
        x = self.dp(self.pool(x))
        x = x.mean(dim=(2, 3))                # global average pool
        x = torch.relu(self.fc1(x))
        x = self.dp2(x)
        x = self.fc2(x)
        return x

    def forward_nhwc(self, x):  # NHWC -> NCHW (for ONNX export)
        return self.forward(x.permute(0, 3, 1, 2))


def load_split(path):
    d = np.load(path)
    X = torch.from_numpy(d["X"]).to(torch.uint8)   # (N,28,28) 0..255 ink
    y = torch.from_numpy(d["y"]).to(torch.long)
    return X, y


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        torch.set_num_threads(THREADS)
    print(f"device: {device}  cuda={torch.cuda.is_available()}", flush=True)

    X, y = load_split(os.path.join(PROC, "train62_uint8.npz"))
    Xv, yv = load_split(os.path.join(PROC, "val62_uint8.npz"))
    print(f"train {X.shape} val {Xv.shape}", flush=True)

    model = ResNet(62).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    nll = nn.CrossEntropyLoss()
    n = X.shape[0]
    nv = Xv.shape[0]
    best_val = 0.0
    wall0 = time.time()

    def hb(ep, b, loss):
        el = time.time() - wall0
        per = el / max(1, b)
        left = per * (n - b) + per * n * (EPOCHS - ep - 1)
        print(f"[ep{ep+1}/{EPOCHS}] batch {b}/{n} loss {loss:.4f} "
              f"elapsed {el/60:.0f}m eta {left/60:.0f}m", flush=True)

    for ep in range(EPOCHS):
        model.train()
        t0 = time.time()
        perm = torch.randperm(n)
        tl = 0.0; nb = 0; tlast = time.time()
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            xb = X[idx].float().unsqueeze(1).to(device) / 255.0   # (B,1,28,28)
            yb = y[idx].to(device)
            opt.zero_grad()
            loss = nll(model(xb), yb)
            loss.backward()
            opt.step()
            tl += loss.item() * len(idx); nb += 1
            now = time.time()
            if now - tlast >= HEARTBEAT_S or b + BATCH >= n:
                hb(ep, b + BATCH, tl / (b + BATCH))
                tlast = now
        # val
        model.eval()
        with torch.no_grad():
            corr1 = 0; corr5 = 0; tot = 0
            for b in range(0, nv, BATCH):
                xb = Xv[b:b+BATCH].float().unsqueeze(1).to(device) / 255.0
                yb = yv[b:b+BATCH].to(device)
                lg = model(xb)
                order = lg.topk(5, dim=1).indices
                corr1 += (order[:, 0] == yb).sum().item()
                corr5 += (order == yb.unsqueeze(1)).any(1).sum().item()
                tot += len(yb)
        va1 = corr1 / tot; va5 = corr5 / tot
        print(f"epoch {ep+1}/{EPOCHS} loss {tl/n:.4f} val_top1 {va1:.4f} "
              f"val_top5 {va5:.4f}  [{time.time()-t0:.0f}s]", flush=True)
        if va1 > best_val:
            best_val = va1
            best = model.state_dict()

    model.load_state_dict(best)
    print(f"best val_top1 {best_val:.4f}", flush=True)

    os.makedirs(OUT, exist_ok=True)
    torch.save({k: v.cpu() for k, v in best.items()},
               os.path.join(OUT, "model.pt"))

    class NHWCWrapper(nn.Module):
        """Export shell: accepts NHWC (N,28,28,1), converts to NCHW internally."""

        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            return self.model(x.permute(0, 3, 1, 2))

    model.cpu().eval()
    dummy = torch.zeros(1, 28, 28, 1)
    torch.onnx.export(
        NHWCWrapper(model), dummy, os.path.join(OUT, "model.onnx"),
        input_names=["input"], output_names=["logits"],
        opset_version=18,
        dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}},
    )
    with open(os.path.join(OUT, "config.json"), "w") as f:
        json.dump({"classes": list(CHAR_LIST), "input": ["N", 28, 28, 1]}, f, indent=2)
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()