"""FastAPI server that exposes the ARC-RAIDERS WikiBot agent over HTTP.

PRODUCTION vs DEVELOPMENT:
    In development, the React frontend runs on its own Vite dev server
    (port 5173) which proxies /api requests to this backend (port 8000).
    Two separate processes, hot-reload on both sides.

    In production (Docker), the frontend is pre-built into static files
    (HTML/JS/CSS) and this server serves them directly alongside the API.
    One container, one port, simpler deployment.

    The FRONTEND_DIST_DIR env var controls this. If it's set and the
    directory exists, we mount it. Otherwise we skip it (dev mode).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_backend_dir = str(Path(__file__).resolve().parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from arc_raider_bot.agent import ask
from arc_raider_bot.models import AskRequest, AskResponse

_executor = ThreadPoolExecutor(max_workers=4)

app = FastAPI(title="ARC-RAIDERS WikiBot API")

# CORS is needed in development when the frontend (port 5173) and backend
# (port 8000) run on different origins. In production (Docker) they're on
# the same origin so CORS doesn't apply, but keeping it doesn't hurt.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse)
async def ask_endpoint(req: AskRequest):
    loop = asyncio.get_event_loop()
    response, session_id = await loop.run_in_executor(
        _executor, ask, req.question, req.session_id
    )
    return AskResponse(
        answer=response.answer,
        sources=response.sources,
        session_id=session_id,
    )


# ---------------------------------------------------------------------------
# Static file serving (production / Docker only)
# ---------------------------------------------------------------------------
# When FRONTEND_DIST_DIR is set and exists, serve the built React app from /.
# This must be mounted AFTER the /api routes so API routes take priority.
# In dev mode this block is skipped — Vite handles the frontend.
_dist_dir = os.getenv("FRONTEND_DIST_DIR", "")
if _dist_dir and Path(_dist_dir).is_dir():
    app.mount("/", StaticFiles(directory=_dist_dir, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
