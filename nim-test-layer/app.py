"""Local test layer for an NVIDIA NIM (build.nvidia.com) API key.

Run:  uvicorn app:app --reload --port 8000
Open: http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

# nim_client loads .env itself, at import time, before it reads NVIDIA_BASE_URL and
# NVIDIA_MODEL — doing it here instead would run too late to affect those constants.
import nim_client as nim
import octopus

ENV_FILE = Path(__file__).parent / ".env"
STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    # flush=True: stdout is block-buffered when uvicorn's output is piped or captured, and
    # a startup banner that only appears once the buffer fills is worse than none.
    def say(msg: str) -> None:
        print(f"[nim-test-layer] {msg}", flush=True)

    say(f".env {'found' if ENV_FILE.exists() else 'MISSING'} at {ENV_FILE}")
    try:
        say(f"key loaded: {nim.key_fingerprint(nim.resolve_key())}")
    except nim.NimError as err:
        say(err.detail)
    say(f"base url: {nim.BASE_URL}")
    yield


app = FastAPI(title="NIM Test Layer", version="1.1.0", lifespan=lifespan)

# This service is meant to bind to localhost only, so a permissive policy here just
# means the console still works if you open index.html straight from the filesystem.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model: str = nim.DEFAULT_MODEL
    system: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)


def _key(override: str | None) -> str:
    try:
        return nim.resolve_key(override)
    except nim.NimError as err:
        raise HTTPException(status_code=err.status, detail=err.detail) from err


def _page(name: str) -> FileResponse:
    path = STATIC / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"{path} is missing.")
    return FileResponse(path)


@app.get("/", include_in_schema=False)
def octopus_ui() -> FileResponse:
    return _page("octopus.html")


@app.get("/console", include_in_schema=False)
def console() -> FileResponse:
    """The original key-testing console."""
    return _page("index.html")


@app.get("/api/catalog")
async def api_catalog(force: bool = False, x_nim_key: str | None = Header(default=None)) -> dict:
    """What this key can actually reach, grouped into families, with tentacle bindings."""
    key = _key(x_nim_key)
    try:
        return await octopus.catalog(key, force=force)
    except nim.NimError as err:
        raise HTTPException(status_code=err.status, detail=err.detail) from err


class DispatchRequest(BaseModel):
    task: str = Field(min_length=1)
    # Debugging escape hatch only. The UI never sends it: working out what kind of task
    # this is, and how many agents it deserves, is the planner's job.
    roles: list[str] | None = None


@app.post("/api/dispatch")
async def api_dispatch(req: DispatchRequest, x_nim_key: str | None = Header(default=None)):
    key = _key(x_nim_key)
    return StreamingResponse(
        octopus.dispatch(key, req.task, req.roles),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
def health(x_nim_key: str | None = Header(default=None)) -> dict:
    """Is a key loaded at all, and where did it come from?"""
    runtime = bool(x_nim_key)
    try:
        key = nim.resolve_key(x_nim_key)
    except nim.NimError:
        return {"ok": False, "source": None, "base_url": nim.BASE_URL,
                "detail": "No key found. Add NVIDIA_API_KEY to .env or paste one in the console."}
    return {
        "ok": True,
        "source": "console" if runtime else ".env",
        "fingerprint": nim.key_fingerprint(key),
        "looks_like_nvapi": key.startswith("nvapi-"),
        "base_url": nim.BASE_URL,
        "default_model": nim.DEFAULT_MODEL,
    }


@app.get("/api/models")
async def models(x_nim_key: str | None = Header(default=None)) -> dict:
    key = _key(x_nim_key)
    try:
        result = await nim.list_models(key)
    except nim.NimError as err:
        raise HTTPException(status_code=err.status, detail=err.detail) from err
    ids = sorted(m.get("id", "") for m in result.data.get("data", []))
    return {"count": len(ids), "latency_ms": result.latency_ms, "models": ids}


@app.post("/api/chat")
async def chat(req: ChatRequest, x_nim_key: str | None = Header(default=None)) -> dict:
    key = _key(x_nim_key)
    try:
        result = await nim.chat(
            key, req.prompt, req.model, req.system, req.temperature, req.max_tokens
        )
    except nim.NimError as err:
        raise HTTPException(status_code=err.status, detail=err.detail) from err
    return {
        "text": nim.extract_text(result.data),
        "latency_ms": result.latency_ms,
        "usage": result.data.get("usage", {}),
        "model": result.data.get("model", req.model),
        "finish_reason": (result.data.get("choices") or [{}])[0].get("finish_reason"),
    }


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, x_nim_key: str | None = Header(default=None)):
    key = _key(x_nim_key)

    async def relay():
        try:
            async for line in nim.chat_stream(
                key, req.prompt, req.model, req.system, req.temperature, req.max_tokens
            ):
                yield line + "\n\n"
        except nim.NimError as err:
            yield f"data: {json.dumps({'error': err.detail, 'status': err.status})}\n\n"

    return StreamingResponse(relay(), media_type="text/event-stream")


TOTAL_CHECKS = 5


@app.post("/api/selftest")
async def selftest(x_nim_key: str | None = Header(default=None)) -> dict:
    """Five checks, in the order things usually break."""
    checks: list[dict] = []

    def record(name: str, ok: bool, note: str, ms: int | None = None) -> None:
        checks.append({"name": name, "ok": ok, "note": note, "latency_ms": ms})

    try:
        key = nim.resolve_key(x_nim_key)
    except nim.NimError as err:
        record("Key loaded", False, err.detail)
        return {"passed": 0, "total": TOTAL_CHECKS, "checks": checks}

    record("Key loaded", True, nim.key_fingerprint(key))

    try:
        listed = await nim.list_models(key)
        ids = [m.get("id", "") for m in listed.data.get("data", [])]
        record("Key authenticates", True, f"{len(ids)} models listed", listed.latency_ms)
    except nim.NimError as err:
        record("Key authenticates", False, err.detail)
        return {"passed": 1, "total": TOTAL_CHECKS, "checks": checks}

    # Being listed does not mean being served: most listed models 404 and some accept the
    # request then hang. Reuse the octopus catalog, which has already proven which ones
    # answer, so the round-trip checks below test the key rather than a dead model.
    try:
        cat = await octopus.catalog(key)
        working = [t["model"] for t in cat["tentacles"] if t["model"]]
        record("Models actually serving", bool(working),
               f"{cat['verified']} of {cat['total']} listed models answered a test request")
    except nim.NimError as err:
        record("Models actually serving", False, err.detail)
        working = []

    probe_model = (nim.DEFAULT_MODEL if nim.DEFAULT_MODEL in working
                   else (working[0] if working else nim.DEFAULT_MODEL))

    try:
        answer = await nim.chat(key, "Reply with the single word: pong", probe_model, max_tokens=16)
        text = nim.extract_text(answer.data).strip()
        record("Completion round trip", bool(text), f"{probe_model} → {text[:60] or 'empty response'}", answer.latency_ms)
    except nim.NimError as err:
        record("Completion round trip", False, err.detail)

    try:
        started = time.perf_counter()
        chunks, first_token_ms = 0, None
        async for line in nim.chat_stream(key, "Count from one to five.", probe_model, max_tokens=48):
            if line.startswith("data:") and "[DONE]" not in line:
                chunks += 1
                if first_token_ms is None:
                    first_token_ms = int((time.perf_counter() - started) * 1000)
        record("Streaming", chunks > 0, f"{chunks} chunks, first token in {first_token_ms} ms",
               int((time.perf_counter() - started) * 1000))
    except nim.NimError as err:
        record("Streaming", False, err.detail)

    return {"passed": sum(1 for c in checks if c["ok"]), "total": TOTAL_CHECKS, "checks": checks}
