"""Render EMNIST ByClass train glyphs into the app's exact 28x28 input form.

The app normalizes a drawn letter as: crop letter bbox from gray -> squash to
20x20 -> place in 28x28 with a 4px border -> invert so ink=1, bg=0.

The current model was trained on thin, directly-fed EMNIST glyphs, so the
app's thick, squash-resized, bordered crops are far out of distribution
(rendered-mode lowercase accuracy is only ~47%). Here we rebuild the training
distribution to MATCH that exact path, with handwriting-style augmentation
(thickness, rotation, translation) so the retrained model generalizes.

Output: data/processed/train62_uint8.npz
  X (uint8, N,28,28): ink in 0..255 (0=bg, 255=ink)  == (255-resized)/255 *255
  y (uint8, N): labels 0..61 (EMNIST ByClass order = CHAR_LIST)
  S (uint8, N): sample ids (for splitting variants without leakage)
"""
import gzip
import os
import struct
import sys

import cv2
import numpy as np

import argparse

DATA = os.environ.get("EMNIST_DATA",
                      "/home/sujith/projects/pi-learn-station/data/raw")
OUT_DIR = os.environ.get("EMNIST_OUT",
                         "/home/sujith/projects/pi-learn-station/data/processed")
VARIANTS = int(os.environ.get("EMNIST_VARIANTS", "3"))  # clean, rotated, thick

ADAPTIVE_BLOCK = 15
ADAPTIVE_C = 8
MIN_AREA = 12


def load_idx_images(path, n=None):
    with gzip.open(path, "rb") as f:
        magic, num, r, c = struct.unpack(">IIII", f.read(16))
        num = min(num, n) if n else num
        return np.frombuffer(f.read(num * r * c), dtype=np.uint8).reshape(num, r, c)


def load_idx_labels(path, n=None):
    with gzip.open(path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        num = min(num, n) if n else num
        return np.frombuffer(f.read(num), dtype=np.uint8)


def upright(img):
    """EMNIST glyphs are stored rotated/mirrored; convert to upright."""
    return np.rot90(img, 3)[:, ::-1]


def make_array(img_28, rng, variant):
    """Produce one 28x28 ink=1..0 float array from an upright glyph."""
    # glyph -> [0,1] ink=1
    g = upright(img_28).astype(np.float32) / 255.0

    if variant == 1:  # rotated
        ang = float(rng.uniform(-16, 16))
        from scipy.ndimage import rotate
        g = rotate(g, ang, reshape=False, order=1, mode="constant", cval=0.0)
    elif variant == 2:  # thicker stroke (pen-width variation) + slight blur
        # dilate ink to simulate a thicker pen
        ink = (g > 0.1).astype(np.uint8)
        k = rng.randint(2, 4)
        kern = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        ink = cv2.dilate(ink, kern)
        g = np.where(ink > 0, np.maximum(g, 0.6), 0.0).astype(np.float32)
        g = cv2.GaussianBlur(g, (3, 3), 0)

    # translation jitter
    tx = int(rng.uniform(-2, 2))
    ty = int(rng.uniform(-2, 2))

    # threshold (same params as backend) -> find the ink bbox
    gray = (g * 255.0).astype(np.uint8)
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY_INV, ADAPTIVE_BLOCK, ADAPTIVE_C)
    num, _l, stats, _c = cv2.connectedComponentsWithStats(thr)
    cands = [(i, stats[i][4]) for i in range(1, num) if stats[i][4] >= MIN_AREA]
    if not cands:
        return np.zeros((28, 28), dtype=np.float32)
    i = max(cands, key=lambda t: t[1])[0]
    x, y, w, h = stats[i][:4]
    # margin so letters that touch the border still keep a 4px frame
    m = 2
    x = max(0, x - m); y = max(0, y - m)
    x1 = min(g.shape[1], x + w + m); y1 = min(g.shape[0], y + h + m)
    crop = g[y:y1, x:x1]

    # squash to 20x20 (cv2 default INTER_LINEAR, same as backend) + 4px border
    resized = cv2.resize(crop, (20, 20)).astype(np.float32)
    arr = np.zeros((28, 28), dtype=np.float32)
    cx = 4 + tx; cy = 4 + ty
    arr[cy:cy + 20, cx:cx + 20] = resized
    return arr


def main(argv=None):
    global DATA, OUT_DIR, VARIANTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA, help="dir with train-images/labels .gz")
    ap.add_argument("--out", default=OUT_DIR, help="output dir for npz")
    ap.add_argument("--variants", type=int, default=VARIANTS)
    a = ap.parse_args(argv)
    DATA, OUT_DIR, VARIANTS = a.data, a.out, a.variants
    os.makedirs(OUT_DIR, exist_ok=True)
    imgs = load_idx_images(f"{DATA}/train-images-idx3-ubyte.gz")
    lbls = load_idx_labels(f"{DATA}/train-labels-idx1-ubyte.gz")
    print(f"train {imgs.shape} labels max {lbls.max()}", flush=True)

    rng = np.random.RandomState(1234)
    n = len(lbls)
    val_n = 10000
    tr_n = n - val_n
    # preallocate: (variants * split) images
    X_tr = np.zeros((tr_n * VARIANTS, 28, 28), np.uint8)
    y_tr = np.zeros(tr_n * VARIANTS, np.uint8)
    S_tr = np.zeros(tr_n * VARIANTS, np.int32)
    X_va = np.zeros((val_n, 28, 28), np.uint8)
    y_va = np.zeros(val_n, np.uint8)

    def fill(Xb, yb, Sb, idxs, variants):
        ptr = 0
        for si in idxs:
            for v in variants:
                arr = make_array(imgs[si], rng, v)
                Xb[ptr] = (np.clip(arr, 0, 1) * 255.0).astype(np.uint8)
                yb[ptr] = lbls[si]
                Sb[ptr] = si
                ptr += 1
        return ptr

    # validation: one clean copy per val sample
    fill(X_va, y_va, np.zeros(val_n, np.int32), range(tr_n, n), [0])
    # train: all VARIANTS per train sample
    fill(X_tr, y_tr, S_tr, range(tr_n), range(VARIANTS))

    np.savez_compressed(f"{OUT_DIR}/train62_uint8.npz",
                        X=X_tr, y=y_tr, S=S_tr)
    np.savez_compressed(f"{OUT_DIR}/val62_uint8.npz",
                        X=X_va, y=y_va)
    print(f"train {X_tr.shape} val {X_va.shape}", flush=True)
    print("saved to", OUT_DIR, flush=True)


if __name__ == "__main__":
    main()