"""Local test layer and multi-provider agent console.

Run:  uvicorn app:app --reload --port 8000
Open: http://127.0.0.1:8000

Models come from providers (see providers.py): a local Ollama server, NVIDIA NIM, or
both. A NIM key is no longer required to use the app — with Ollama running, every
endpoint here works with no key and no network.
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response as RawResponse, StreamingResponse
from pydantic import BaseModel, Field

# providers loads .env itself, at import time, before it reads the base URLs and
# timeouts — doing it here instead would run too late to affect those constants.
import nim_client as nim
import octopus
import discordbot
import providers
import routing
import chat_bridge
import wa_bridge
import voice
import wa_qr
import web
import whatsapp

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
    say(f"web access: {'on' if web.ENABLED else 'off'}")

    # Warm the catalog in the background. Verifying which models actually answer costs a
    # probe per candidate against each provider's rate limit; paying that during startup,
    # while the user is still opening the page, is free. Paying it on the first dispatch
    # makes the app look hung for a minute before the first agent appears.
    async def warm() -> None:
        try:
            cat = await octopus.catalog(_optional_key(None))
            say(f"catalog warm: {cat['verified']} of {cat['total']} models answered")
        except Exception as err:                      # never let this break startup
            say(f"catalog warm failed ({type(err).__name__}) — it will be read on demand")

    task = asyncio.create_task(warm())

    # Load the speech models now rather than on the first thing anyone asks to hear.
    # Kokoro takes about five seconds to load; paying that while nobody is listening is
    # free, paying it on the first answer is the difference between instant and broken.
    async def warm_voice() -> None:
        ok, why = voice.available()
        say(f"voice: {'warming' if ok else 'off'} — {why}")
        if ok:
            await asyncio.get_running_loop().run_in_executor(None, voice.warm)
            say("voice: models ready")

    voice_task = asyncio.create_task(warm_voice())

    # The Discord bot lives in this process and this event loop. Its gateway is an
    # outbound WebSocket, so unlike the WhatsApp webhook it needs no public URL, no
    # tunnel and no inbound port — which is why it is the front door worth reaching for.
    ok, why = discordbot.configured()
    say(f"discord: {'starting' if ok else 'off'} — {why}")
    discordbot.launch()

    # QR-linked WhatsApp. Off unless two separate switches are set, because the risk it
    # carries is to the user's own phone number rather than to a revocable key.
    ok, why = wa_qr.configured()
    say(f"whatsapp-qr: {'starting' if ok else 'off'} — {why}")
    if ok:
        say(f"whatsapp-qr: {wa_qr.WARNING}")
    wa_qr.launch()

    try:
        yield
    finally:
        task.cancel()
        voice_task.cancel()
        await discordbot.shutdown()


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


class WebRequest(BaseModel):
    query: str | None = None
    url: str | None = None
    results: int = Field(default=5, ge=1, le=10)
    pages: int = Field(default=3, ge=0, le=6)


@app.post("/api/web")
async def api_web(req: WebRequest) -> dict:
    """Search the web, or read one page. Dry run — no model is involved.

    Here so the fetching can be checked on its own: when a research agent answers oddly
    the first question is always whether it got sensible sources, and that should be
    answerable without spending a completion to find out.
    """
    if not web.ENABLED:
        raise HTTPException(status_code=503, detail="Web access is off. Set ENABLE_WEB=1.")
    try:
        if req.url:
            page = await web.fetch(req.url)
            if not page.ok:
                raise HTTPException(status_code=502, detail=page.error)
            return {"url": page.url, "title": page.title, "chars": len(page.text),
                    "text": page.text}
        if not req.query:
            raise HTTPException(status_code=422, detail="Give either a query or a url.")
        bundle = await web.research(req.query, req.results, req.pages)
        return {
            "query": bundle["query"],
            "results": bundle["results"],
            "sources": web.sources(bundle),
            "failed": bundle["failed"],
            "context_chars": len(web.as_context(bundle)),
        }
    except web.WebError as err:
        raise HTTPException(status_code=502, detail=str(err)) from err


# --- WhatsApp ---------------------------------------------------------------
# A public webhook that runs work on this machine. Everything about these two handlers
# is shaped by that: verify the signature before parsing, check the allowlist before
# acting, and never confirm to a stranger that the endpoint does anything.


@app.get("/api/whatsapp", include_in_schema=False)
def whatsapp_verify(request: Request) -> Response:
    """Meta's one-time handshake when the webhook URL is saved."""
    q = request.query_params
    try:
        challenge = whatsapp.verify_handshake(
            q.get("hub.mode"), q.get("hub.verify_token"), q.get("hub.challenge"))
    except PermissionError:
        raise HTTPException(status_code=403, detail="verify token mismatch")
    return Response(content=challenge, media_type="text/plain")


@app.post("/api/whatsapp", include_in_schema=False)
async def whatsapp_inbound(request: Request) -> Response:
    """Inbound messages. Always answers 200 — quickly.

    Meta retries anything that is slow or non-2xx, and a dispatch takes minutes. So the
    work is started in the background and the delivery acknowledged straight away;
    replying only when the agents finish would guarantee duplicate deliveries and, with
    them, duplicate runs.
    """
    raw = await request.body()

    if not whatsapp.signed_by_meta(raw, request.headers.get("X-Hub-Signature-256")):
        # 403 rather than 401: there is no authentication to retry with.
        raise HTTPException(status_code=403, detail="bad signature")

    ok, why = whatsapp.configured()
    if not ok:
        print(f"[whatsapp] ignoring delivery — {why}", flush=True)
        return Response(status_code=200)

    try:
        payload = json.loads(raw.decode("utf-8", "replace"))
    except ValueError:
        return Response(status_code=200)

    for msg in whatsapp.parse(payload):
        if not whatsapp.permitted(msg.sender):
            # Silence, not an error. An unknown number learns nothing about what is here.
            print(f"[whatsapp] refused message from {msg.sender[:4]}…", flush=True)
            continue
        if whatsapp.already_handled(msg.message_id):
            continue
        asyncio.create_task(wa_bridge.handle(msg))

    return Response(status_code=200)


@app.get("/api/whatsapp/health")
async def whatsapp_health() -> dict:
    """Is the WhatsApp side wired up? Never returns the token or the secret."""
    ok, why = whatsapp.configured()
    return {
        "enabled": whatsapp.ENABLED,
        "ready": ok,
        "detail": why,
        "phone_id_set": bool(whatsapp.PHONE_ID),
        "app_secret_set": bool(whatsapp.APP_SECRET),
        "allowed_numbers": len(whatsapp.ALLOWED),
        "active_runs": wa_bridge.active_runs(),
    }


@app.get("/api/whatsapp-qr/health")
async def wa_qr_health() -> dict:
    """Pairing state for the QR-linked WhatsApp. Never returns the session."""
    return wa_qr.status()


@app.get("/api/whatsapp-qr/image", include_in_schema=False)
def wa_qr_image() -> FileResponse:
    """The pairing QR as a PNG, for scanning off a screen rather than a console.

    A Windows console at the wrong font size renders a QR as unreadable mush; the image
    always works. It exists only while pairing is pending and is deleted on connect —
    a live pairing code is a credential.
    """
    if not wa_qr.QR_PNG.is_file():
        raise HTTPException(status_code=404,
                            detail="No pairing code pending. Either it is already linked, "
                                   "or WhatsApp QR is not enabled.")
    return FileResponse(wa_qr.QR_PNG, media_type="image/png")


@app.post("/api/whatsapp-qr/test", include_in_schema=False)
async def wa_qr_test() -> dict:
    """Send one diagnostic message to the allowed number. Proves both directions."""
    return await wa_qr.send_test()


class SpeakRequest(BaseModel):
    text: str = Field(min_length=1)


@app.post("/api/voice/speak")
async def api_speak(req: SpeakRequest):
    """Text in, WAV out. Used by the page's play buttons."""
    clip = await voice.speak(req.text)
    if not clip:
        ok, why = voice.available()
        raise HTTPException(status_code=503, detail=why if not ok else "synthesis failed")
    return RawResponse(content=clip, media_type="audio/wav",
                       headers={"Cache-Control": "no-store"})


@app.post("/api/voice/speak/stream")
async def api_speak_stream(req: SpeakRequest):
    """One clip per sentence, streamed as each is ready.

    Waiting for a whole answer to synthesise means several seconds of silence before the
    first word; a sentence takes under one. The client plays them in order as they land.
    Base64 over SSE because the browser already speaks that protocol here and binary
    multipart would need a second parser for no gain.
    """
    async def gen():
        try:
            async for clip in voice.speak_stream(req.text):
                yield "data: " + base64.b64encode(clip).decode() + "\n\n"
        except Exception as err:                      # never leave the stream hanging
            print(f"[voice] stream failed: {type(err).__name__}: {err}", flush=True)
        yield "data: [DONE]\n\n"


    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/voice/listen")
async def api_listen(audio: UploadFile = File(...)) -> dict:
    """Audio in, text out. Whatever the browser's MediaRecorder produced is fine.

    No format negotiation: faster-whisper decodes through PyAV, which handles the webm,
    ogg and mp4 containers browsers actually record into.
    """
    data = await audio.read()
    text = await voice.listen(data)
    if not text:
        ok, why = voice.available()
        raise HTTPException(status_code=503 if not ok else 422,
                            detail=why if not ok else "nothing recognisable in that audio")
    return {"text": text, "bytes": len(data)}


@app.get("/api/voice/health")
async def api_voice_health() -> dict:
    return voice.status()


@app.get("/api/discord/health")
async def discord_health() -> dict:
    """Is the bot connected? Never returns the token."""
    return discordbot.status()


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
    # Conversation id. When present the task is dispatched with this session's earlier
    # turns as context, and the result is recorded back into it — the same store the
    # Discord and WhatsApp doors use, so a session is a session wherever it is driven
    # from. Absent means a one-off with no memory, which is what this used to be.
    session: str | None = None


@app.post("/api/dispatch")
async def api_dispatch(req: DispatchRequest, x_nim_key: str | None = Header(default=None)):
    key = _optional_key(x_nim_key)

    if not req.session:
        return StreamingResponse(
            octopus.dispatch(key, req.task, req.roles, req.mode),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    async def with_memory():
        """Stream the run through untouched, and keep a digest of it afterwards.

        The browser wants raw dispatch events, not chat messages, so it cannot go through
        the Transport path — but it should still share the session store. The events are
        passed along verbatim and only observed in passing, so the UI sees exactly what it
        saw before.
        """
        outputs: dict[str, str] = {}
        labels: dict[str, str] = {}
        try:
            async for line in octopus.dispatch(
                key, chat_bridge.context_for(req.session, req.task), req.roles, req.mode,
                weigh_as=req.task,
            ):
                yield line
                if not line.startswith("data:"):
                    continue
                try:
                    e = json.loads(line[5:])
                except ValueError:
                    continue
                if e.get("event") == "wave":
                    for a in e.get("agents", []):
                        labels[a["id"]] = a.get("label", a["id"])
                elif e.get("event") == "chunk":
                    outputs[e["id"]] = outputs.get(e["id"], "") + e.get("text", "")
        finally:
            digest = " | ".join(f"{labels.get(aid, aid)}: {text[:160]}"
                                for aid, text in outputs.items() if text)
            chat_bridge.remember(req.session, req.task, digest)

    return StreamingResponse(
        with_memory(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/session/{convo}")
async def api_session(convo: str) -> dict:
    """What this session remembers, and what it is doing."""
    return chat_bridge.summary(convo)


@app.post("/api/session/{convo}/new")
async def api_session_new(convo: str) -> dict:
    """End a session: forget the thread and drop anything queued."""
    return {"cleared": chat_bridge.forget(convo)}


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
        "web": {"enabled": web.ENABLED, "results": web.RESULTS,
                "rpm": web.REQUESTS_PER_MIN},
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
