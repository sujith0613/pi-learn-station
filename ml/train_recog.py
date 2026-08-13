"""Train a small CNN to recognize EMNIST lowercase letters (26 classes).

Data: data/raw/emnist_lowercase.npz (C,H,W) = (1,28,28) float32 [0,1].
Outputs:
  models/recog/model.pt
  models/recog/model.onnx
  models/recog/config.json   (export input shape / class order a..z)
"""

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(__file__))

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
NPZ = os.path.join(ROOT, "data", "raw", "emnist_lowercase.npz")
OUT = os.path.join(ROOT, "models", "recog")
SEED = 42

CLASSES = list("abcdefghijklmnopqrstuvwxyz")


class SmallCNN(nn.Module):
    def __init__(self, num_classes=26):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(),
            nn.MaxPool2d(2),                      # 14
            nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(),
            nn.MaxPool2d(2),                      # 7
            nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(),
        )                                        # 7x7x96
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 7 * 7, 256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.head(self.features(x))


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    d = np.load(NPZ)
    X_tr = torch.from_numpy(d["X_train"]).float()
    y_tr = torch.from_numpy(d["y_train"]).long()
    X_te = torch.from_numpy(d["X_test"]).float()
    y_te = torch.from_numpy(d["y_test"]).long()
    print(f"train {X_tr.shape} test {X_te.shape}")

    model = SmallCNN(26)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    nll = nn.CrossEntropyLoss()
    B = 128
    epochs = 12
    train_n = X_tr.shape[0]
    test_n = X_te.shape[0]
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(train_n)
        tl = 0.0
        for b in range(0, train_n, B):
            idx = perm[b : b + B]
            x, y = X_tr[idx], y_tr[idx]
            opt.zero_grad()
            loss = nll(model(x), y)
            loss.backward()
            opt.step()
            tl += loss.item() * len(idx)
        # test
        model.eval()
        with torch.no_grad():
            accs = []
            for b in range(0, test_n, B):
                x, y = X_te[b : b + B], y_te[b : b + B]
                accs.append((model(x).argmax(1) == y).float().mean().item())
        acc = float(np.mean(accs))
        print(f"epoch {ep+1}/{epochs} train_loss={tl/train_n:.4f} "
              f"test_acc={acc:.4f}")

    torch.save(model.state_dict(), os.path.join(OUT, "model.pt"))
    model.eval()
    dummy = torch.zeros(1, 1, 28, 28)
    torch.onnx.export(
        model, dummy, os.path.join(OUT, "model.onnx"),
        input_names=["input"], output_names=["logits"],
        opset_version=12, dynamic_axes=None,
    )
    with open(os.path.join(OUT, "config.json"), "w") as f:
        json.dump({"classes": CLASSES, "input": [1, 1, 28, 28]}, f, indent=2)
    print(f"saved {OUT}/model.pt + model.onnx")


if __name__ == "__main__":
    main()