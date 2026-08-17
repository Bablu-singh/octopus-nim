"""Local test layer and multi-provider agent console.

Run:  uvicorn app:app --reload --port 8000
Open: http://127.0.0.1:8000

Models come from providers (see providers.py): a local Ollama server, NVIDIA NIM, or
both. A NIM key is no longer required to use the app — with Ollama running, every
endpoint here works with no key and no network.
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

# providers loads .env itself, at import time, before it reads the base URLs and
# timeouts — doing it here instead would run too late to affect those constants.
import nim_client as nim
import octopus
import providers
import routing

ENV_FILE = Path(__file__).parent / ".env"
STATIC = Path(__file__).parent / "static"


@asynccontextmanager
async def lifespan(_: FastAPI):
    # flush=True: stdout is block-buffered when uvicorn's output is piped or captured, and
    # a startup banner that only appears once the buffer fills is worse than none.
    def say(msg: str) -> None:
        print(f"[octopus] {msg}", flush=True)

    say(f".env {'found' if ENV_FILE.exists() else 'MISSING'} at {ENV_FILE}")
    for p in await providers.survey():
        mark = "ok  " if p["usable"] else "--  "
        say(f"{mark}{p['label']:<12} {p['state']:<12} {p['base_url']}")
    say(f"routing: {routing.MODE} (threshold {routing.THRESHOLD})")
    yield


app = FastAPI(title="Octopus", version="2.0.0", lifespan=lifespan)

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
    # Qualified ids ('local:qwen2.5:7b') pick their provider; a bare id means NIM, which
    # is what every id in this app meant before providers existed.
    model: str = nim.DEFAULT_MODEL
    system: str | None = None
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)


def _key(override: str | None) -> str:
    """A NIM key, or a 401. For endpoints that genuinely cannot work without one."""
    try:
        return nim.resolve_key(override)
    except providers.ProviderError as err:
        raise HTTPException(status_code=err.status, detail=err.detail) from err


def _optional_key(override: str | None) -> str | None:
    """A NIM key if there is one, else None.

    Most of the app no longer needs a key: with Ollama up, the catalog, the router and a
    whole dispatch run happen locally. Demanding a key here would turn 'works offline'
    into a 401.
    """
    if override and override.strip():
        return override.strip()
    nvidia = providers.get("nvidia")
    return nvidia.key() if nvidia.has_key() else None


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


@app.get("/api/providers")
async def api_providers(x_nim_key: str | None = Header(default=None)) -> dict:
    """Which providers are registered, which are up, and why the others are not."""
    survey = await providers.survey(_optional_key(x_nim_key))
    return {
        "providers": survey,
        "usable": [p["name"] for p in survey if p["usable"]],
        "routing": {
            "mode": routing.MODE,
            "threshold": routing.THRESHOLD,
            "role_weight": routing.ROLE_WEIGHT,
        },
    }


class RouteRequest(BaseModel):
    subtask: str = Field(min_length=1)
    role: str = "analyst"
    mode: str | None = None


@app.post("/api/route")
async def api_route(req: RouteRequest, x_nim_key: str | None = Header(default=None)) -> dict:
    """Where would this piece of work go, and why? Dry run — nothing is dispatched.

    Exists because a router you cannot interrogate is a router you cannot tune. Post a
    subtask, see the score and the provider, adjust OCTOPUS_ROUTE_THRESHOLD if you
    disagree with it.
    """
    survey = await providers.survey(_optional_key(x_nim_key))
    usable = {p["name"] for p in survey if p["usable"]}
    score, why = routing.weigh(req.subtask, req.role)
    provider, reason = routing.choose(req.role, req.subtask, usable, mode=req.mode)
    return {
        "role": req.role, "score": round(score, 3), "signals": why,
        "threshold": routing.THRESHOLD, "heavy": score >= routing.THRESHOLD,
        "provider": provider, "reason": reason, "usable": sorted(usable),
    }


@app.get("/api/catalog")
async def api_catalog(force: bool = False, x_nim_key: str | None = Header(default=None)) -> dict:
    """Everything reachable right now, across providers, with per-provider bindings."""
    try:
        return await octopus.catalog(_optional_key(x_nim_key), force=force)
    except providers.ProviderError as err:
        raise HTTPException(status_code=err.status, detail=err.detail) from err


class DispatchRequest(BaseModel):
    task: str = Field(min_length=1)
    # Debugging escape hatch only. The UI never sends it: working out what kind of task
    # this is, and how many agents it deserves, is the planner's job.
    roles: list[str] | None = None
    # 'auto' (default), 'local', or 'nvidia'. Pinning is for working offline or for
    # comparing the two; auto is the point of the router.
    mode: str | None = None


@app.post("/api/dispatch")
async def api_dispatch(req: DispatchRequest, x_nim_key: str | None = Header(default=None)):
    return StreamingResponse(
        octopus.dispatch(_optional_key(x_nim_key), req.task, req.roles, req.mode),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/health")
async def health(x_nim_key: str | None = Header(default=None)) -> dict:
    """Is anything usable at all, and where would work go right now?"""
    runtime = bool(x_nim_key)
    key = _optional_key(x_nim_key)
    survey = await providers.survey(key)
    usable = [p["name"] for p in survey if p["usable"]]
    return {
        # 'ok' now means 'this app can do work', not 'a NIM key exists' — with Ollama up
        # and no key at all, the answer is yes.
        "ok": bool(usable),
        "usable": usable,
        "providers": survey,
        "route_mode": routing.MODE,
        "source": ("console" if runtime else ".env") if key else None,
        "fingerprint": providers.key_fingerprint(key or ""),
        "looks_like_nvapi": bool(key and key.startswith("nvapi-")),
        "base_url": nim.BASE_URL,
        "default_model": nim.DEFAULT_MODEL,
        "detail": None if usable else
                  "Nothing reachable. Start Ollama (`ollama serve`) or add NVIDIA_API_KEY to .env.",
    }


@app.get("/api/models")
async def models(provider: str | None = None,
                 x_nim_key: str | None = Header(default=None)) -> dict:
    """Models one provider lists. Defaults to NIM, for compatibility with the console."""
    name = provider or "nvidia"
    try:
        p = providers.get(name)
        if p.needs_key:
            p = providers.with_key(p, _key(x_nim_key))
        result = await providers.list_models(p)
    except providers.ProviderError as err:
        raise HTTPException(status_code=err.status, detail=err.detail) from err
    ids = sorted(m.get("id", "") for m in result.data.get("data", []))
    return {"provider": name, "count": len(ids), "latency_ms": result.latency_ms,
            "models": ids, "qualified": [p.qualify(i) for i in ids]}


@app.post("/api/chat")
async def chat(req: ChatRequest, x_nim_key: str | None = Header(default=None)) -> dict:
    try:
        result = await providers.chat(req.model, req.prompt, req.system, req.temperature,
                                      req.max_tokens, key=_optional_key(x_nim_key))
    except providers.ProviderError as err:
        raise HTTPException(status_code=err.status, detail=err.detail) from err
    return {
        "text": providers.extract_text(result.data),
        "latency_ms": result.latency_ms,
        "usage": result.data.get("usage", {}),
        "model": result.data.get("model", req.model),
        "provider": providers.provider_of(req.model),
        "finish_reason": (result.data.get("choices") or [{}])[0].get("finish_reason"),
    }


@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest, x_nim_key: str | None = Header(default=None)):
    key = _optional_key(x_nim_key)

    async def relay():
        try:
            async for line in providers.chat_stream(
                req.model, req.prompt, req.system, req.temperature, req.max_tokens, key=key
            ):
                yield line + "\n\n"
        except providers.ProviderError as err:
            yield f"data: {json.dumps({'error': err.detail, 'status': err.status})}\n\n"

    return StreamingResponse(relay(), media_type="text/event-stream")


TOTAL_CHECKS = 5


@app.post("/api/selftest")
async def selftest(x_nim_key: str | None = Header(default=None)) -> dict:
    """Five checks, in the order things usually break.

    Still five, and still in the same order, because the console paints them into a fixed
    strip. What changed is that none of them are NIM-specific any more: with only Ollama
    running they all pass, and the notes say which provider answered.
    """
    checks: list[dict] = []

    def record(name: str, ok: bool, note: str, ms: int | None = None) -> None:
        checks.append({"name": name, "ok": ok, "note": note, "latency_ms": ms})

    key = _optional_key(x_nim_key)
    survey = await providers.survey(key)
    usable = [p for p in survey if p["usable"]]

    if not usable:
        record("Key loaded", False,
               "No provider is reachable. Start Ollama, or add NVIDIA_API_KEY to .env.")
        return {"passed": 0, "total": TOTAL_CHECKS, "checks": checks}

    record("Key loaded", True,
           f"{', '.join(p['label'] for p in usable)} — "
           + (providers.key_fingerprint(key) if key else "no key needed"))

    try:
        cat = await octopus.catalog(key)
    except providers.ProviderError as err:
        record("Key authenticates", False, err.detail)
        return {"passed": 1, "total": TOTAL_CHECKS, "checks": checks}

    record("Key authenticates", cat["total"] > 0,
           ", ".join(f"{n}: {d['total']} listed" for n, d in cat["by_provider"].items())
           or "nothing listed",
           cat["latency_ms"])

    # Being listed does not mean being served: most models NIM lists 404 and some accept
    # the request then hang. The catalog has already proven which ones answer, so the
    # round-trip checks below test the plumbing rather than a dead model.
    working = [t["model"] for t in cat["tentacles"] if t["model"]]
    per = ", ".join(f"{name}: {d['verified']}" for name, d in cat["by_provider"].items())
    record("Models actually serving", bool(working),
           f"{cat['verified']} of {cat['total']} models answered ({per})"
           if working else "no model answered a test request")

    probe_model = working[0] if working else None
    if probe_model is None:
        record("Completion round trip", False, "no working model to try")
        record("Streaming", False, "no working model to try")
        return {"passed": sum(1 for c in checks if c["ok"]), "total": TOTAL_CHECKS,
                "checks": checks}

    try:
        answer = await providers.chat(probe_model, "Reply with the single word: pong",
                                      max_tokens=16, key=key)
        text = providers.extract_text(answer.data).strip()
        record("Completion round trip", bool(text),
               f"{probe_model} → {text[:60] or 'empty response'}", answer.latency_ms)
    except providers.ProviderError as err:
        record("Completion round trip", False, err.detail)

    try:
        started = time.perf_counter()
        chunks, first_token_ms = 0, None
        async for line in providers.chat_stream(probe_model, "Count from one to five.",
                                                max_tokens=48, key=key):
            if line.startswith("data:") and "[DONE]" not in line:
                chunks += 1
                if first_token_ms is None:
                    first_token_ms = int((time.perf_counter() - started) * 1000)
        record("Streaming", chunks > 0, f"{chunks} chunks, first token in {first_token_ms} ms",
               int((time.perf_counter() - started) * 1000))
    except providers.ProviderError as err:
        record("Streaming", False, err.detail)

    return {"passed": sum(1 for c in checks if c["ok"]), "total": TOTAL_CHECKS, "checks": checks}
