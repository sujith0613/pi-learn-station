"""Download EMNIST ByClass and extract the 26 lowercase letter classes (36-61).

ByClass has 62 classes: 0-9 digits, 10-35 A-Z (uppercase), 36-61 a-z
(lowercase). We keep only 36-61 -> remap to 0-25 ('a'..'z').

Output (NPZ, saved under data/raw/):
    X_train, y_train, X_test, y_test   (X: float32 NxC x H x W in [0,1])
"""

import gzip
import json
import os
import sys
from urllib import request

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "raw"))
# Working mirror (anonymous, LFS). The official NIST/MIU endpoint
# (biometrics.cse.msu.edu) resets connections; HF Royc30ne/emnist-byclass
# hosts byte-identical per-split IDX gz files.
BASE = ("https://huggingface.co/datasets/Royc30ne/emnist-byclass/resolve/main"
        "/emnist-byclass-")
FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "test-images-idx3-ubyte.gz",
    "test-labels-idx1-ubyte.gz",
]


def _download(name: str):
    local = os.path.join(ROOT, name)
    if os.path.exists(local) and os.path.getsize(local) > 0:
        print(f"  cached {name}")
        return local
    os.makedirs(ROOT, exist_ok=True)
    url = BASE + name + "?download=true"
    print(f"  downloading {name} ...")
    headers = {"User-Agent": "Mozilla/5.0"}
    req = request.Request(url, headers=headers)
    with request.urlopen(req, timeout=120) as r, open(local, "wb") as f:
        f.write(r.read())
    return local


def _read_mnist_images(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic, n, rows, cols = np.frombuffer(raw[:16], dtype=">u4")
    imgs = np.frombuffer(raw[16:], dtype=np.uint8).reshape(int(n), int(rows), int(cols))
    return imgs


def _read_mnist_labels(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    return np.frombuffer(raw[8:], dtype=np.uint8)


def main():
    os.makedirs(ROOT, exist_ok=True)
    paths = {name: _download(name) for name in FILES}

    print("loading ...")
    X_tr = _read_mnist_images(paths["train-images-idx3-ubyte.gz"])
    y_tr = _read_mnist_labels(paths["train-labels-idx1-ubyte.gz"])
    X_te = _read_mnist_images(paths["test-images-idx3-ubyte.gz"])
    y_te = _read_mnist_labels(paths["test-labels-idx1-ubyte.gz"])

    # Rotate 90 deg (EMNIST is transposed) + flip to correct orientation.
    # NOTE: shipped model was trained with this k=1 transform; the backend
    # (recognition.recognize) rotates input 180 deg to compensate so upright
    # writing classifies correctly. If you change k here, update that too.
    def norm(imgs):
        imgs = np.rot90(imgs, 1, axes=(1, 2))
        imgs = imgs[:, :, ::-1]  # flip horizontally
        return imgs

    X_tr = norm(X_tr)
    X_te = norm(X_te)

    # Keep lowercase labels 36-61, remap to 0-25.
    def filt(X, y):
        m = (y >= 36) & (y <= 61)
        return X[m], (y[m] - 36)

    X_tr, y_tr = filt(X_tr, y_tr)
    X_te, y_te = filt(X_te, y_te)

    X_tr = X_tr.astype(np.float32) / 255.0
    X_te = X_te.astype(np.float32) / 255.0
    # C x H x W
    X_tr = X_tr[:, None, :, :]
    X_te = X_te[:, None, :, :]

    out = os.path.join(ROOT, "emnist_lowercase.npz")
    np.savez_compressed(out, X_train=X_tr, y_train=y_tr,
                        X_test=X_te, y_test=y_te)
    print(f"saved {out}")
    print(f"train {X_tr.shape[0]}  test {X_te.shape[0]}  "
          f"classes {len(np.unique(y_tr))}")
    # quick label histogram
    hist = np.bincount(y_tr, minlength=26)
    print("per-class train counts:", hist.tolist())


if __name__ == "__main__":
    sys.exit(main())