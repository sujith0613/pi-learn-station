# Pi Learning Station — Plan (v4, writing + context-brain)

## Product summary
A Raspberry Pi touchscreen learning station where children (~5-7) write
sentences freehand on a canvas. A small **character-level masked transformer**
uses the sentence context accumulated so far to disambiguate confusion letters
(b/d, p/q, and voicing pairs) and offer gentle corrections/suggestions. Spoken
feedback via browser TTS. A Pi-hosted hotspot lets phones/tablets open the same
app in the browser — no installs.

> v4 scope change: the v3 mic/speech-recognition pipeline is **deferred**.
> Handwriting + a context-aware confusion-char brain is now the core.

## Architecture (one Svelte 5 codebase → three surfaces)

```
                 Svelte 5 + Vite SPA (web/)
                    Vite build → dist/
        ┌──────────────┼─────────────────┐
        ▼              ▼                 ▼
  Pi kiosk        Mobile (hotspot)   Laptop dev
  Chromium        phone browser      Vite HMR
  localhost:8000  http://192.168.4.1  :5173
        └──────────────┬─────────────────┘
                 Python backend (FastAPI)
            app.frontend() serves dist/
            GET  /api/health
            GET  /generate_204   → 204 (captive portal)
            POST /api/segment-recognize  strokes → letters (top-K, onnx)
            POST /api/suggest            char-LM context disambiguation
```

## Data flow

```
canvas strokes ──► [segmentation] ──► per-letter CNN (26 lowercase, EMNIST
                   x-gaps + pen-lift  ByClass 36-61) ──► top-K per letter
                   cluster/valley
                                         │
                     [char-LM masked transformer]  mask uncertain letter,
                                         │  score P(b|ctx) vs P(d|ctx) etc.
                                         ▼
                   combine P_rec × P_lm^α ──► suggestion / correction UI + TTS
```

## Models
- **Letter recognizer** — 2-conv-block CNN, 26 lowercase classes, trained on
  EMNIST **ByClass filtered to labels 36-61** (a-z; b/d/p/q included).
- **Char-level masked transformer** — BERT-style, token vocab `a-z + space +
  . , ! ? ' - + [PAD][CLS][SEP][MASK]`, embed 128, 4 heads, 2-4 layers,
  max_len 64 (~2-4M params). Trained via MLM on a generated corpus of simple
  kid sentences (data/corpus/raw_claude_sentences.txt + template expansion).
- Both exported to **ONNX**, run via `onnxruntime` on the Pi (arm64 wheel).
  Committed under `models/` so Pi deploys via `git pull`.

## Confusion set (evidence-based — docs/research.md)
- **Tier A — mirror/reversible letters** (Fernandes & Leite 2017; Perea 2011):
  `b↔d`, `p↔q` (left-right), `b↔p`, `d↔q` (up-down). The only lowercase Latin
  letters differing solely by orientation.
- **Tier B — voiceless/voiced cognates** (PLOS One 2019; Bahr 2012): `d↔t`,
  `g↔k`, `f↔v`, `s↔z` (`b↔p` already in A).
- **Tier C — stretch** (flag off): `m↔n`, letter-swap detection.

## Tech decisions (why — from research)

| Part | Choice | Why |
|---|---|---|
| Serving | FastAPI `app.frontend("/", directory="dist", fallback="index.html")` | SPA mount, 404s missing assets, never shadows `/api`. No nginx. |
| HTTP/WS | Uvicorn on :8000; iptables DNAT 80→8000 | URL is just `http://192.168.4.1`; captive-portal probes hit it. |
| Frontend | Plain Vite + Svelte 5 (no SvelteKit) | Single-page canvas app; no SSR needed. |
| Drawing | Pointer Events + setPointerCapture + touch-action:none + HiDPI (ResizeObserver × DPR) | One handler for mouse/touch/pen; crisp strokes on phone screens. |
| Recognition | onnxruntime + 26-cls lowercase CNN | Offline, fast on Pi 4/5; small committed model. |
| Context brain | Custom char-level masked transformer (ONNX) | Disambiguates confusion letters from sentence context; trained on generated kid corpus. |
| TTS | Browser speechSynthesis (offline voices) | Zero server audio deps. |
| Hotspot | NetworkManager-native AP (`nmcli`, `ipv4.method shared`) | Reboot-safe; no manual hostapd/dnsmasq. |
| Captive portal | `/generate_204` → 204 | Auto-opens app on connect; no DNS hijack. |
| Mobile/desktop shell | None — the served web page IS the app | No Capacitor/Tauri; native adds build chains with zero benefit on a hotspot-only deployment. |
| Envs | uv — `ml/` py3.12 (torch CPU), `backend/` py3.12 torch-free (fastapi+onnxruntime) | Keeps torch off the runtime; IaC-style env. |

## Phases
0. **Repo/layout** *(done)*: repo at `~/projects/pi-learn-station`; data/, ml/,
   models/, backend/tests/, scripts/, deploy/; corpus ingested.
1. **Backend skeleton** *(done)*: FastAPI + `app.frontend()`, `/api/health`,
   `/generate_204`, WS echo, serves SPA.
2. **Training (Thu)**: EMNIST ByClass → lowercase filter; corpus expansion
   ~30-40k; train char-LM + recognizer; **eval gate ≥85% context
   disambiguation per pair**; export ONNX.
3. **App (Fri)**: segmentation + recognition + disambiguate endpoints; Svelte
   canvas word-strip UI, correction bubble, TTS; laptop e2e green.
4. **Pi (Sat)**: onnxruntime arm64, model copy, hotspot + kiosk autostart,
   touchscreen tuning.
5. **Buffer (Sun)**: refine corpus if gates short; rehearsal; harden autostart.

## Risks & fallbacks
- **Freehand segmentation** is the weak link → kid taps a highlighted letter to
  pick a candidate (still the pedagogical moment); tune on real finger data.
- **LM weak on novel sentences** → expand corpus; default to no suggestion when
  uncertain.
- **Pi aarch64 install** → onnxruntime arm64 wheel; laptop demo is the safety net.