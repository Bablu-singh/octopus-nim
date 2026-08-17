"""Thin async wrapper around the NVIDIA NIM OpenAI-compatible API."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import httpx
from dotenv import load_dotenv

# Load .env here rather than in each entrypoint: the constants below are read at import
# time, and an entrypoint that imports this module before calling load_dotenv would
# silently get the defaults instead of its own overrides.
load_dotenv(Path(__file__).parent / ".env", override=True)

BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
DEFAULT_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")

# NIM lists models it cannot actually serve. Some of those 404 immediately, but others
# accept the connection and then never send a byte, so every timeout here has to be short
# enough that a dead model fails fast instead of stalling the whole request.
CONNECT_TIMEOUT = float(os.getenv("NIM_CONNECT_TIMEOUT", "10"))
READ_TIMEOUT = float(os.getenv("NIM_READ_TIMEOUT", "45"))
PROBE_TIMEOUT = float(os.getenv("NIM_PROBE_TIMEOUT", "12"))

TIMEOUT = httpx.Timeout(connect=CONNECT_TIMEOUT, read=READ_TIMEOUT, write=30.0, pool=10.0)
PROBE = httpx.Timeout(connect=CONNECT_TIMEOUT, read=PROBE_TIMEOUT, write=15.0, pool=10.0)


class NimError(RuntimeError):
    """Raised when the upstream API returns a non-2xx response."""

    def __init__(self, status: int, detail: str):
        super().__init__(f"NIM API returned {status}: {detail}")
        self.status = status
        self.detail = detail


@dataclass
class Timed:
    """A payload plus how long the round trip took."""

    data: dict
    latency_ms: int


def resolve_key(override: str | None = None) -> str:
    key = (override or os.getenv("NVIDIA_API_KEY") or "").strip()
    if not key:
        raise NimError(401, "No API key. Set NVIDIA_API_KEY in .env or pass one in the console.")
    return key


def key_fingerprint(key: str) -> str:
    """Safe-to-log identifier: prefix plus last four characters."""
    if len(key) < 10:
        return "too short to fingerprint"
    return f"{key[:8]}…{key[-4:]} ({len(key)} chars)"


def _headers(key: str, stream: bool = False) -> dict:
    return {
        "Authorization": f"Bearer {key}",
        "Accept": "text/event-stream" if stream else "application/json",
        "Content-Type": "application/json",
    }


def _raise_for_status(resp: httpx.Response, body: str) -> None:
    if resp.status_code >= 400:
        detail = body.strip()[:600] or resp.reason_phrase
        raise NimError(resp.status_code, detail)


def transport_error(err: Exception, model: str | None = None) -> NimError:
    """Turn an httpx failure into the same NimError every caller already handles."""
    what = f"'{model}' " if model else ""
    if isinstance(err, httpx.ReadTimeout):
        return NimError(504, f"Model {what}accepted the request but sent nothing back "
                             f"within {READ_TIMEOUT:.0f}s — it is listed but not actually serving.")
    if isinstance(err, httpx.ConnectTimeout):
        return NimError(504, f"Could not connect to {BASE_URL} within {CONNECT_TIMEOUT:.0f}s.")
    if isinstance(err, httpx.TimeoutException):
        return NimError(504, f"Request to {what}timed out ({type(err).__name__}).")
    return NimError(502, f"Network error talking to {BASE_URL}: {type(err).__name__}: {err}")


async def list_models(key: str) -> Timed:
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.get(f"{BASE_URL}/models", headers=_headers(key))
            _raise_for_status(resp, resp.text)
            payload = resp.json()
    except httpx.HTTPError as err:
        raise transport_error(err) from err
    return Timed(payload, int((time.perf_counter() - started) * 1000))


async def is_alive(key: str, model: str, client: httpx.AsyncClient | None = None) -> bool:
    """Does this model actually answer? Cheapest possible completion, short timeout.

    The catalog is not a promise: on a typical key most listed models 404 and a handful
    hang forever. Binding a tentacle to one of those is what turns a normal request into
    a read timeout, so every model gets proven before it is used.

    A 200 alone is not proof. Completion-only and capacity-exhausted models answer 200 and
    return an empty message, so the probe carries a system prompt — the shape a real agent
    call has — and insists on actual content coming back.
    """
    body = {
        "model": model,
        "messages": [{"role": "system", "content": "Answer with one word."},
                     {"role": "user", "content": "Say: ready"}],
        "max_tokens": 8,
        "stream": False,
    }
    owned = client is None
    client = client or httpx.AsyncClient(timeout=PROBE)
    try:
        resp = await client.post(
            f"{BASE_URL}/chat/completions", headers=_headers(key), json=body, timeout=PROBE
        )
        return resp.status_code == 200 and bool(extract_text(resp.json()).strip())
    except (httpx.HTTPError, ValueError):
        return False
    finally:
        if owned:
            await client.aclose()


async def chat(
    key: str,
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> Timed:
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                f"{BASE_URL}/chat/completions", headers=_headers(key), json=body
            )
            _raise_for_status(resp, resp.text)
            payload = resp.json()
    except httpx.HTTPError as err:
        raise transport_error(err, model) from err
    return Timed(payload, int((time.perf_counter() - started) * 1000))


async def chat_stream(
    key: str,
    prompt: str,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> AsyncIterator[str]:
    """Yield raw SSE lines from the upstream so the caller can pass them straight through."""
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    body = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            async with client.stream(
                "POST", f"{BASE_URL}/chat/completions",
                headers=_headers(key, stream=True), json=body,
            ) as resp:
                if resp.status_code >= 400:
                    text = (await resp.aread()).decode("utf-8", "replace")
                    _raise_for_status(resp, text)
                async for line in resp.aiter_lines():
                    if line:
                        yield line
    except httpx.HTTPError as err:
        raise transport_error(err, model) from err


def extract_text(payload: dict) -> str:
    """Pull the assistant message out of a chat completion payload."""
    try:
        return payload["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""
