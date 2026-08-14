"""End-to-end recognition accuracy harness for the app's pipeline.

Feeds EMNIST ByClass test glyphs through the SAME path the app uses for a
drawn letter:

  render/app_gray -> adaptiveThreshold(15,8) -> connectedComponents
  -> crop largest component from GRAY -> resize(20,20) squash
  -> arr[4:24,4:24] = (255-x)/255 (4px border, ink=1) -> model softmax

Two render modes:
  native    - the raw 28x28 EMNIST glyph, inverted to dark-ink-on-light
              (thin, small -> what the model saw during its own training)
  rendered  - glyph upscaled onto an 800x370 canvas + dilation to simulate a
              thick pen stroke (approximates real browser handwriting)

Reports per-group accuracy (digits / A-Z / a-z) and top-K coverage, plus a
per-letter confusion matrix. This is the baseline that every retraining change
is validated against.

Usage:
  python ml/eval_pipeline.py --n 40000 --mode rendered --out /tmp/opencode/base_new.json
"""

import argparse
import gzip
import json
import struct

import cv2
import numpy as np

DATA = "/home/sujith/projects/pi-learn-station/data/raw"
MODEL = "/home/sujith/projects/pi-learn-station/models/recog/model.onnx"

CHAR_LIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

# backend normalization constants (must match segmentation.py/recognition.py)
ADAPTIVE_BLOCK = 15
ADAPTIVE_C = 8
MIN_AREA = 12
LOGICAL_W, LOGICAL_H = 800, 370


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
    """EMNIST ByClass glyphs are stored rotated/mirrored; convert to the upright
    orientation the app's drawn strokes naturally have (rot90 k=3 + flipH)."""
    return np.rot90(img, 3)[:, ::-1]


def render_glyph(img, mode):
    """Return an 'app_gray' uint8 (dark ink ~0, light paper ~255) for a glyph."""
    img = upright(img)
    if mode == "native":
        # EMNIST ink is 255; invert so ink is dark like the app's gray.
        return (255 - img).astype(np.uint8)
    # rendered: place upscaled on 800x370 and thicken to simulate a pen.
    s = 4  # upscale factor -> ~112px tall letter
    big = cv2.resize(img, (28 * s, 28 * s), interpolation=cv2.INTER_NEAREST)
    canvas = np.full((LOGICAL_H, LOGICAL_W), 255, np.uint8)
    h, w = big.shape
    y0 = (LOGICAL_H - h) // 2
    x0 = (LOGICAL_W - w) // 2
    # big has ink=255; subtract from white paper => ink dark
    canvas[y0:y0 + h, x0:x0 + w] = 255 - big
    # dilate the dark ink to simulate pen thickness (round-ish stroke)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    ink = (canvas < 128).astype(np.uint8) * 255
    dilated = cv2.dilate(ink, k)
    canvas = 255 - dilated
    return canvas


def normalize_crop(crop):
    """Exactly the backend's normalize_component (squash to 20x20, 4px border)."""
    resized = cv2.resize(crop, (20, 20)).astype(np.float32)
    arr = np.zeros((28, 28), dtype=np.float32)
    arr[4:24, 4:24] = (255.0 - resized) / 255.0
    return arr


def run_pipeline(gray):
    """Segment one glyph (largest component) and return its normalized 28x28 arr,
    or None if nothing survives segmentation."""
    thr = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, ADAPTIVE_BLOCK, ADAPTIVE_C)
    num, _l, stats, _c = cv2.connectedComponentsWithStats(thr)
    cands = [(i, stats[i][4]) for i in range(1, num) if stats[i][4] >= MIN_AREA]
    if not cands:
        return None
    # largest component = the glyph (one letter per canvas in the harness)
    i = max(cands, key=lambda t: t[1])[0]
    x, y, w, h = stats[i][:4]
    crop = gray[y:y + h, x:x + w]
    return normalize_crop(crop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000)
    ap.add_argument("--mode", choices=["native", "rendered"], default="rendered")
    ap.add_argument("--out", default=None, help="write JSON summary")
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    import onnxruntime as ort
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name

    imgs = load_idx_images(f"{DATA}/test-images-idx3-ubyte.gz", args.n)
    lbls = load_idx_labels(f"{DATA}/test-labels-idx1-ubyte.gz", args.n)
    print(f"loaded {imgs.shape}", flush=True)

    n = len(lbls)
    top1 = np.zeros(n, bool)
    topk = np.zeros(n, bool)
    pred_idx = np.empty(n, np.int64)
    conf = np.empty(n, np.float32)
    K = 5

    for i in range(n):
        arr = run_pipeline(render_glyph(imgs[i], args.mode))
        if arr is None:
            pred_idx[i] = -1
            continue
        logits = sess.run(None, {inp: arr[None, ..., None].astype(np.float32)})[0][0]
        order = np.argsort(logits)[::-1]
        top1[i] = (order[0] == lbls[i])
        topk[i] = (lbls[i] in order[:K])
        pred_idx[i] = order[0]
        conf[i] = float(np.exp(logits[order[0]]) / np.exp(logits).sum())

    skip = pred_idx == -1
    valid = ~skip
    print(f"segmentation failures: {skip.sum()} / {n}", flush=True)

    overall = top1[valid].mean()
    topk_all = topk[valid].mean()
    print(f"\n== mode={args.mode}  n(valid)={valid.sum()} ==", flush=True)
    print(f"OVERALL top-1 {overall:.4f}   top-{K} coverage {topk_all:.4f}", flush=True)
    groups = {"digits(0-9)": (0, 10), "A-Z": (10, 36), "a-z": (36, 62)}
    summary = {"mode": args.mode, "n": int(valid.sum()),
               "top1": float(overall), "top5": float(topk_all),
               "groups": {}}
    for name, (lo, hi) in groups.items():
        m = (lbls >= lo) & (lbls < hi) & valid
        a1 = top1[m].mean()
        a5 = topk[m].mean()
        print(f"  {name}: top-1 {a1:.4f}  top-{K} {a5:.4f}  n={m.sum()}", flush=True)
        summary["groups"][name] = {"top1": float(a1), "top5": float(a5), "n": int(m.sum())}

    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
        print("wrote", args.out, flush=True)


if __name__ == "__main__":
    main()