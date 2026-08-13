# Pi Learning Station

A Raspberry-Pi "learning station" that helps dyslexic children with letter
confusions (b/d, p/q, d/t, …). A child writes a word on a touchscreen; a CNN
recognizes the letters and a masked language model disambiguates confusing
letters using the sentence context, then reads the word aloud (TTS).

```
handwriting  ->  CNN (EMNIST lowercase)  ->  per-letter candidates
sentence ctx ->  char-LM (masked)        ->  best spelling suggestion
             ->  Svelte 5 SPA (word tiles, correction bubble, speech)
```

## Layout

```
backend/   FastAPI app (torch-free: onnxruntime only) + tests
ml/        training + eval + ONNX export (torch, build machine only)
web/       Svelte 5 SPA (built to web/dist, served by backend)
models/    exported ONNX models committed for the Pi deploy
deploy/    Containerfile + Pi setup/hotspot scripts
scripts/   build / dev / e2e helpers
dist/      podman image tarball for the Pi (gitignored)
```

## Disambiguation gate

The LM must reach ≥85% macro-mean per-pair accuracy over the 8 confusion pairs.
Current run: **macro 86.9% PASS** (worst pair b/d 72.9%). Eval set lives in
`data/corpus/eval_cases.json`.

## Build

```bash
# train + export models (build machine, has torch)
cd ml && uv venv && uv pip install -r requirements.txt   # torch, etc.
python download_emnist.py && python train_recog.py
python train_lm.py && python export_lm_onnx.py && python eval_lm.py

# backend (torch-free)
cd ../backend && uv venv && uv pip install -r requirements.txt onnxruntime
uvicorn app.main:app --host 0.0.0.0 --port 8000

# frontend
cd ../web && npm install && npm run build
```

## Containerize for the Pi (arm64)

Cross-build the arm64 image on an x86_64 machine with podman + QEMU:

```bash
podman build --platform linux/arm64/v8 -t pi-learn-station:arm64 -f deploy/Containerfile .
podman save pi-learn-station:arm64 | gzip > dist/pi-learn-station-arm64.tar.gz
```

## Deploy to the Pi

```bash
scp dist/pi-learn-station-arm64.tar.gz pi@<ip>:~/
bash deploy/pi_setup.sh     # installs podman, loads image, runs app on :8000
bash deploy/pi_hotspot.sh   # WiFi hotspot + Chromium kiosk autostart
```

Children connect to the hotspot SSID and the app opens automatically.