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
    # Flat list of strokes; each stroke is a list of {x, y, t} points.
    # Letter AND word boundaries are detected server-side (auto-detect).
    strokes: list[list[StrokePoint]]


def _to_strokes(points: list[StrokePoint]) -> segmentation.Stroke:
    return segmentation.Stroke([(p.x, p.y, p.t) for p in points])


def _recognize_letter(strokes: list[segmentation.Stroke]) -> list[tuple[str, float]]:
    """Recognize one letter-group into an ordered (letter, prob) candidate list."""
    bmp = recognition.normalize_strokes_to_bitmap(strokes)
    return list(recognition.recognize(bmp).items())


@app.post("/api/recognize")
async def recognize(payload: StrokesIn) -> dict:
    """Auto-segment flat strokes into letters and words, then return the
    greedy spelling of each word plus any LM-assisted suggestion.

    Response: {"sentence": str, "words": [{start, end, greedy, letters,
    suggestion?}]}. `start`/`end` are char offsets into `sentence` so the
    frontend can map suggestions back to words.
    """
    strokes = [_to_strokes(s) for s in payload.strokes]
    letters = segmentation.group_strokes(strokes)
    words = segmentation.segment_words(letters)

    # per-word recognition
    word_cands = [
        [_recognize_letter(g) for g in word]  # [letter_group][(letter, prob)]
        for word in words
    ]
    greedy_words = [
        "".join(max(pos, key=lambda kv: kv[1])[0] for pos in word)
        for word in word_cands
    ]

    # build the running sentence (single spaces between words)
    parts: list[str] = []
    offsets: list[tuple[int, int]] = []  # (start, end) per word in `sentence`
    cur = 0
    for gw in greedy_words:
        parts.append(gw)
        start = cur
        end = cur + len(gw)
        offsets.append((start, end))
        cur = end + 1  # trailing space slot
    sentence = " ".join(parts)

    out_words: list[dict[str, Any]] = []
    for idx, gw in enumerate(greedy_words):
        ws, we = offsets[idx]
        letters_list = [[list(c) for c in pos] for pos in word_cands[idx]]
        entry: dict[str, Any] = {
            "start": ws,
            "end": we,
            "greedy": gw,
            "letters": letters_list,
        }
        # disambiguate this word using the whole-sentence context
        sugg = disambiguate.disambiguate(sentence, ws, we, word_cands[idx])
        if sugg["best"] != gw:
            entry["suggestion"] = sugg
        out_words.append(entry)

    return {"sentence": sentence, "words": out_words}


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