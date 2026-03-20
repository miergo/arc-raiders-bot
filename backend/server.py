"""FastAPI server that exposes the ARC-RAIDERS WikiBot agent over HTTP."""

from __future__ import annotations

import sys
from pathlib import Path

_backend_dir = str(Path(__file__).resolve().parent)
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from arc_raider_bot.agent import ask
from arc_raider_bot.models import AskRequest, AskResponse

_executor = ThreadPoolExecutor(max_workers=4)

app = FastAPI(title="ARC-RAIDERS WikiBot API")

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
