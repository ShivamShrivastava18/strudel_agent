"""FastAPI playback server for the Strudel agent.

Routes:
    GET  /                  -> serves web/index.html
    GET  /static/*          -> static assets
    POST /api/generate      -> SSE stream of agent events for a prompt
    POST /api/validate      -> re-validate edited code (for the Live edit pane)

Run:
    uvicorn server:app --reload
or:
    python server.py
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

import agent as agent_mod
import tools as toolset
from llm import get_llm

load_dotenv()

ROOT = Path(__file__).parent
WEB_DIR = ROOT / "web"

# ---------------------------------------------------------------------------
# Config (env-overridable)
# ---------------------------------------------------------------------------

PROVIDER = os.environ.get("STRUDEL_AGENT_PROVIDER", "ollama")
MODEL = os.environ.get(
    "STRUDEL_AGENT_MODEL",
    "gpt-oss:20b" if PROVIDER == "ollama" else "gemini-1.5-pro",
)
MAX_ITERS = int(os.environ.get("STRUDEL_AGENT_MAX_ITERS", "12"))


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="Strudel Agent", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


# Lazy-built singleton LLM (loaded once on first request)
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm(PROVIDER, model=MODEL)
    return _llm


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    prompt: str
    max_iters: int | None = None


class ValidateRequest(BaseModel):
    code: str


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
async def index():
    html_path = WEB_DIR / "index.html"
    if not html_path.exists():
        raise HTTPException(404, "web/index.html not found")
    return FileResponse(str(html_path))


@app.get("/api/config")
async def get_config():
    return {
        "provider": PROVIDER,
        "model": MODEL,
        "max_iters": MAX_ITERS,
    }


@app.post("/api/validate")
async def validate(req: ValidateRequest):
    return toolset.validate_code(req.code)


@app.post("/api/generate")
async def generate(req: GenerateRequest):
    """Stream agent events as Server-Sent Events.

    Each event is `data: {json}\n\n`. The stream ends when {type: "done"}
    is yielded.
    """
    if not req.prompt.strip():
        raise HTTPException(400, "prompt is empty")

    max_iters = req.max_iters or MAX_ITERS
    llm = _get_llm()

    async def event_gen() -> AsyncIterator[dict]:
        # Immediate starting event so the browser confirms the SSE pipe is live
        yield {"data": json.dumps({
            "type": "starting",
            "provider": PROVIDER,
            "model": MODEL,
        })}

        loop = asyncio.get_running_loop()
        # Run the synchronous generator in a thread, marshal events back via a queue.
        queue: asyncio.Queue = asyncio.Queue()
        SENTINEL = object()

        def producer():
            try:
                for ev in agent_mod.run_agent_stream(
                    req.prompt, llm, max_iters=max_iters
                ):
                    asyncio.run_coroutine_threadsafe(queue.put(ev), loop)
            except Exception as e:
                asyncio.run_coroutine_threadsafe(
                    queue.put({"type": "error", "message": f"{type(e).__name__}: {e}"}),
                    loop,
                )
                asyncio.run_coroutine_threadsafe(
                    queue.put({"type": "done"}), loop
                )
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(SENTINEL), loop)

        loop.run_in_executor(None, producer)

        # Periodically emit a heartbeat so the UI can show "still working" while
        # the LLM is generating its first response (gpt-oss:20b cold-start can
        # take 30-60s before the first thinking event arrives).
        while True:
            try:
                ev = await asyncio.wait_for(queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                yield {"data": json.dumps({"type": "heartbeat"})}
                continue
            if ev is SENTINEL:
                break
            yield {"data": json.dumps(ev, ensure_ascii=False)}
            if ev.get("type") == "done":
                break

    return EventSourceResponse(event_gen())


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Strudel agent web app.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    import uvicorn
    uvicorn.run(
        "server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
