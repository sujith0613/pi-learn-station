from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response

BACKEND_DIR = Path(__file__).resolve().parent.parent
DIST = BACKEND_DIR.parent / "web" / "dist"

app = FastAPI(title="Pi Learning Station")


@app.get("/api/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/generate_204")
async def captive_portal_probe() -> Response:
    # Android/iOS captive-portal probes expect 204 -> shows "Sign in to network".
    return Response(status_code=204)


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            # Echo for now; Phase 3 wires real state sync.
            await websocket.send_json({"type": "echo", "data": data})
    except WebSocketDisconnect:
        return


# Serve the Svelte SPA last; only used when no path operation matched.
if DIST.exists():
    app.frontend("/", directory=str(DIST), fallback="index.html")
else:
    print(f"[warn] {DIST} not built yet; web UI not served")