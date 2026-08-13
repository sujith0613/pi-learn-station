from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from pydantic import BaseModel

from app import disambiguate, recognition, segmentation

BACKEND_DIR = Path(__file__).resolve().parent.parent
DIST = BACKEND_DIR.parent / "web" / "dist"

app = FastAPI(title="Pi Learning Station")


class StrokePoint(BaseModel):
    x: float
    y: float
    t: float


class StrokesIn(BaseModel):
    strokes: list[list[list[StrokePoint]]]  # [letter][stroke][point]
    sentence: str = ""                # partial sentence so far (for context)
    word_start: int = 0
    word_end: int = 0


def _letters_to_bitmaps(strokes: list[list[StrokePoint]]):
    """Turn grouped pointer strokes into (letter, bitmap) list."""
    out = []
    for group in strokes:
        seg = [segmentation.Stroke([(p.x, p.y, p.t) for p in s])
               for s in group]
        bmp = recognition.normalize_strokes_to_bitmap(seg)
        out.append(bmp)
    return out


def _recognize_bitmap_list(bitmaps):
    """Return per-letter candidate lists: [[(c,p),...], ...]."""
    per_letter = []
    for bmp in bitmaps:
        top = recognition.recognize(bmp)
        per_letter.append(list(top.items()))
    return per_letter


@app.post("/api/recognize")
async def recognize(payload: StrokesIn) -> dict:
    """Recognize grouped strokes into a word.

    Returns per-letter candidate probs plus any LM-assisted suggestion.
    """
    bitmaps = _letters_to_bitmaps(payload.strokes)
    per_letter = _recognize_bitmap_list(bitmaps)

    greedy = "".join(max(pos, key=lambda kv: kv[1])[0] for pos in per_letter)

    result: dict[str, Any] = {
        "word": greedy,
        "letters": per_letter,
    }

    # disambiguate using the char-LM when we have context
    if payload.sentence and payload.word_end > payload.word_start:
        ws = payload.word_start
        we = payload.word_end
        if we - ws == len(greedy) and payload.sentence[ws:we]:
            sugg = disambiguate.disambiguate(
                payload.sentence, ws, we, per_letter)
            if sugg["best"] != greedy:
                result["suggestion"] = sugg

    return result


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
            # Stream-ready; minimal sync for now.
            await websocket.send_json({"type": "echo", "data": data})
    except WebSocketDisconnect:
        return


# Serve the Svelte SPA last; only used when no path operation matched.
if DIST.exists():
    app.frontend("/", directory=str(DIST), fallback="index.html")
else:
    print(f"[warn] {DIST} not built yet; web UI not served")