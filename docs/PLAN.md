# Pi Learning Station — Plan (v3, research-backed)

## Product summary
A Raspberry Pi learning station: kids write an answer on a touchscreen
(canvas) and the Pi *listens* to their spoken answer via a microphone,
recognizes it offline (TensorFlow Lite), and gives spoken feedback via TTS.
A Pi-hosted hotspot lets phones/tablets open the same app in the browser —
no installs, no app stores.

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
            GET /api/health
            WS   /ws  (state broadcast)
            POST /api/recognize  → tflite worker
            GET  /generate_204   → 204 (captive portal)
```

## Tech decisions (why — from research)

| Part | Choice | Why |
|---|---|---|
| Serving | FastAPI `app.frontend("/", directory="dist", fallback="index.html")` | New first-class SPA mount (v0.138+). Correctly 404s missing assets, never shadows `/api`. Avoids the classic `StaticFiles(html=True)` catch-all bug that eats API routes. No nginx. |
| HTTP/WS | Uvicorn on :8000; iptables DNAT 80→8000 | Port 80 so URL is just `http://192.168.4.1`, and captive-portal probes hit it. |
| Frontend | Plain Vite + Svelte 5 (no SvelteKit) | Single-page canvas app; no SSR/routing needed. |
| Drawing | Pointer Events + setPointerCapture + touch-action:none + getCoalescedEvents + HiDPI (ResizeObserver × devicePixelRatio) | One handler for mouse/touch/pen; pointer capture fixes drag-outside; DPR scale keeps strokes crisp on phone screens. |
| Input | Canvas → strokes → compare to target answer | Stroke fingerprinting in Svelte, sent over WS. |
| Recognition | tflite-runtime + keyword-spotting (Google Speech Commands), sounddevice circular buffer + VAD + confidence gate | Offline, real-time on Pi 4/5; model from research repos. |
| TTS | Browser speechSynthesis (offline voices), filter `localService===true`, ~300-char chunks, app-level queue | Zero server audio deps; cross-platform voices. |
| WS client | Svelte store; exponential backoff **with jitter** (base 1s, cap 30s), reconnect only in `onclose`, heartbeat, UI state machine | Industry standard (websocket.org, 137Foundry). |
| Hotspot | NetworkManager-native AP (`nmcli`, `ipv4.method shared`, autoconnect on) | Bookworm conflict-free, reboot-safe; manual hostapd+dnsmasq is the failure source. |
| Captive portal | FastAPI `/generate_204` → 204 (Android shows "Sign in to network") | Auto-opens app on connect without URL typing; no DNS hijack needed. |
| Mobile | No Capacitor/Tauri — the served web page IS the app | Research: PWA/self-host is the proven Pi path; native adds build chains with zero benefit on a hotspot-only deployment. iOS/Android identical via browser. |

## Phases

1. **Phase 0 — Pi foundation**: Pi OS Lite 64-bit, Python 3.11 venv, git, ssh key. *(weeks ago partially; re-document)*
2. **Phase 1 — Backend skeleton**: FastAPI + `app.frontend()`, `/api/health`, `/generate_204`, Uvicorn on :8000, iptables 80→8000. Verify SPA serves.
3. **Phase 2 — Svelte 5 UI**: canvas (HiDPI Pointer Events), responsive portrait/landscape, settings screen (WS URL), minimal styling.
4. **Phase 3 — Link UI↔WS**: reconnecting WS store; server broadcasts state; connection indicator.
5. **Phase 4 — Kiosk**: Chromium autostart on boot, kiosk flags, `localhost:8000`.
6. **Phase 5 — Recognition**: sounddevice capture → circular buffer → VAD → tflite inference → confidence → answer comparison. Feedback loop.
7. **Phase 6 — Hotspot + PWA**: NM AP (SSID/pass), manifest+icons (`display: standalone`), `/generate_204`, reboot-safe.
8. **Phase 7 — Hardening**: systemd services, log rotation, docs/run scripts, optional stroke-similarity scoring.

## Open questions
- [ ] Target answers: numbers vs words vs arithmetic — decide for model label set.
- [ ] Single- vs multi-client on one Pi (WS broadcast currently assumes shared state).
- [ ] Voice feedback language/locale (TTS voices are OS-dependent).
