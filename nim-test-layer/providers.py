"""Where a model can actually come from.

Octopus was bound to one API. It is not any more. A provider is a base URL, an optional
key, and a wire format; everything above this file addresses models as `provider:model`
rather than by bare name. That is the whole extension seam — adding GPT or Claude later
is a row in `PROVIDERS` plus, for Claude, an adapter for its non-OpenAI wire format.
Nothing in the octopus, the tentacles, or the UI has to learn about it.

Two providers are live today, and they are deliberately the two that cost nothing:

  local   Ollama — or any OpenAI-compatible server — on this machine. No key, no
          network, no quota, no catalog that lies. Slow, because it is CPU inference,
          but always entitled.
  nvidia  build.nvidia.com. Free tier, models far larger than anything that fits in
          local RAM, but a catalog that advertises models it cannot serve.

openai and anthropic are registered and disabled. They are here so the shape is visible
and so enabling one later is a config change rather than a refactor — not because the
app can use them. Keeping to free resources is a deliberate choice, not an oversight.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import AsyncIterator

import httpx
from dotenv import load_dotenv

# Load .env here rather than in each entrypoint: the constants below are read at import
# time, and an entrypoint that imports this module before calling load_dotenv would
# silently get the defaults instead of its own overrides.
load_dotenv(Path(__file__).parent / ".env", override=True)

# A qualified id is 'provider:model'. Ollama tags contain colons of their own
# ('llama3.2:3b'), so the split is always exactly once, from the left.
SEP = ":"


class ProviderError(RuntimeError):
    """Raised when an upstream returns a non-2xx response, or the transport fails.

    One error type across every provider on purpose: callers branch on `status`, which
    means the same 404-means-rebind logic works whether the model was missing from
    Ollama or unentitled on NIM.
    """

    def __init__(self, status: int, detail: str, provider: str = ""):
        who = provider or "upstream"
        super().__init__(f"{who} returned {status}: {detail}")
        self.status = status
        self.detail = detail
        self.provider = provider


@dataclass
class Timed:
    """A payload plus how long the round trip took."""

    data: dict
    latency_ms: int


@dataclass
class Provider:
    """One place models can be reached.

    `trust_catalog` is the important field. NIM lists roughly twenty models for every
    one it will actually serve, so its catalog has to be probed model by model before
    anything is bound. Ollama only lists what is already on disk, so probing it is pure
    latency — a cold 7B can take a minute to load, and paying that during a catalog read
    makes the UI look broken. Local models are therefore trusted on sight and dropped
    the normal way, by `_mark_dead`, if they ever fail for real.
    """

    name: str
    label: str
    base_url: str
    key_env: str | None            # None means the provider needs no credential
    tier: str                      # 'local' | 'free-api' | 'paid'
    blurb: str
    wire: str = "openai"           # request/response shape; see `_require_openai_wire`
    enabled: bool = True
    disabled_note: str = ""
    trust_catalog: bool = False
    # Local inference is an order of magnitude slower than a hosted GPU and has to load
    # weights from disk on first use, so timeouts are per-provider rather than global.
    connect_timeout: float = 10.0
    read_timeout: float = 45.0
    probe_timeout: float = 12.0
    # Every token costs wall-clock time on a CPU. Capping generation length locally is
    # what keeps a small task genuinely faster than the network round trip to NIM.
    max_tokens_cap: int | None = None
    # Ranked hint for the router, not a hard rule: see routing.py.
    prefers: str = "any"           # 'small' | 'large' | 'any'
    # One cheap model used only to prove a credential works; see `_key_works`.
    probe_model: str = ""
    key_override: str | None = field(default=None, repr=False)

    # -- credentials ---------------------------------------------------------

    @property
    def needs_key(self) -> bool:
        return self.key_env is not None

    def key(self) -> str:
        """The credential for this provider, or '' if it does not take one."""
        if not self.needs_key:
            return ""
        key = (self.key_override or os.getenv(self.key_env or "") or "").strip()
        if not key:
            raise ProviderError(
                401, f"No API key. Set {self.key_env} in .env or paste one in the console.",
                self.name,
            )
        return key

    def has_key(self) -> bool:
        if not self.needs_key:
            return True
        return bool((self.key_override or os.getenv(self.key_env or "") or "").strip())

    # -- transport -----------------------------------------------------------

    @property
    def timeout(self) -> httpx.Timeout:
        return httpx.Timeout(connect=self.connect_timeout, read=self.read_timeout,
                             write=30.0, pool=10.0)

    @property
    def probe(self) -> httpx.Timeout:
        return httpx.Timeout(connect=self.connect_timeout, read=self.probe_timeout,
                             write=15.0, pool=10.0)

    def headers(self, stream: bool = False) -> dict:
        head = {
            "Accept": "text/event-stream" if stream else "application/json",
            "Content-Type": "application/json",
        }
        # Ollama ignores Authorization entirely; sending an empty bearer would be noise.
        if self.needs_key:
            head["Authorization"] = f"Bearer {self.key()}"
        return head

    def cap(self, max_tokens: int) -> int:
        return min(max_tokens, self.max_tokens_cap) if self.max_tokens_cap else max_tokens

    def qualify(self, model: str) -> str:
        return f"{self.name}{SEP}{model}"


# --- registry ---------------------------------------------------------------

def _f(env: str, default: float) -> float:
    try:
        return float(os.getenv(env, default))
    except ValueError:
        return default


PROVIDERS: list[Provider] = [
    Provider(
        name="local",
        label="Local",
        base_url=os.getenv("LOCAL_BASE_URL", "http://127.0.0.1:11434/v1"),
        key_env=None,
        tier="local",
        blurb="Runs on this machine. Free, offline, unlimited — and slow.",
        trust_catalog=os.getenv("LOCAL_TRUST_CATALOG", "1") != "0",
        # A cold model is read off disk before the first token appears; on a 7B that is
        # tens of seconds during which the socket is silent. Anything under a minute
        # here fails healthy models purely for being cold.
        connect_timeout=_f("LOCAL_CONNECT_TIMEOUT", 5),
        read_timeout=_f("LOCAL_READ_TIMEOUT", 300),
        probe_timeout=_f("LOCAL_PROBE_TIMEOUT", 180),
        max_tokens_cap=int(os.getenv("LOCAL_MAX_TOKENS", "700")),
        prefers="small",
    ),
    Provider(
        name="nvidia",
        label="NVIDIA NIM",
        base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        key_env="NVIDIA_API_KEY",
        tier="free-api",
        blurb="build.nvidia.com. Free tier, large models, a catalog that overpromises.",
        trust_catalog=False,
        # NIM lists models it cannot serve. Some 404 immediately, but others accept the
        # connection and then never send a byte, so every timeout has to be short enough
        # that a dead model fails fast instead of stalling the whole request.
        connect_timeout=_f("NIM_CONNECT_TIMEOUT", 10),
        read_timeout=_f("NIM_READ_TIMEOUT", 45),
        probe_timeout=_f("NIM_PROBE_TIMEOUT", 12),
        probe_model=os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct"),
        prefers="large",
    ),
    Provider(
        name="openai",
        label="OpenAI",
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        key_env="OPENAI_API_KEY",
        tier="paid",
        blurb="Same wire format as the two above — needs only a key and this row enabled.",
        enabled=os.getenv("ENABLE_OPENAI", "0") == "1",
        disabled_note="Disabled: paid. Octopus is deliberately free-resources-only for now.",
        prefers="large",
    ),
    Provider(
        name="anthropic",
        label="Anthropic",
        base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"),
        key_env="ANTHROPIC_API_KEY",
        tier="paid",
        blurb="Different wire format (/messages, x-api-key). Needs an adapter, not just a key.",
        wire="anthropic",
        enabled=os.getenv("ENABLE_ANTHROPIC", "0") == "1",
        disabled_note="Disabled: paid, and its wire format needs an adapter first.",
        prefers="large",
    ),
]

BY_NAME: dict[str, Provider] = {p.name: p for p in PROVIDERS}


def get(name: str) -> Provider:
    p = BY_NAME.get(name)
    if p is None:
        raise ProviderError(400, f"Unknown provider '{name}'.", name)
    return p


def split(qualified: str) -> tuple[Provider, str]:
    """'local:llama3.2:3b' -> (local provider, 'llama3.2:3b').

    An unqualified id is read as NIM, because that is what every id in this codebase
    meant before providers existed — old .env values and saved requests keep working.
    """
    if SEP in qualified:
        head, _, tail = qualified.partition(SEP)
        if head in BY_NAME:
            return BY_NAME[head], tail
    return BY_NAME["nvidia"], qualified


def provider_of(qualified: str) -> str:
    return split(qualified)[0].name


def bare(qualified: str) -> str:
    """The model id as its own provider knows it, with any prefix stripped."""
    return split(qualified)[1]


def _require_openai_wire(p: Provider) -> None:
    """The seam where a non-OpenAI provider would plug in.

    Everything below speaks OpenAI's /chat/completions. Anthropic's /messages differs in
    request shape, streaming events and auth header, so it gets its own adapter here
    rather than a pile of conditionals threaded through the transport. Until someone
    writes that adapter, saying so plainly beats failing with a confusing 404.
    """
    if p.wire != "openai":
        raise ProviderError(
            501, f"{p.label} speaks the '{p.wire}' wire format, which has no adapter yet.",
            p.name,
        )


def transport_error(err: Exception, p: Provider, model: str | None = None) -> ProviderError:
    """Turn an httpx failure into the same ProviderError every caller already handles."""
    what = f"'{model}' " if model else ""
    if isinstance(err, httpx.ConnectError):
        hint = (" Is `ollama serve` running?" if p.tier == "local" else "")
        return ProviderError(503, f"Could not reach {p.label} at {p.base_url}.{hint}", p.name)
    if isinstance(err, httpx.ReadTimeout):
        return ProviderError(504, f"Model {what}accepted the request but sent nothing back "
                                  f"within {p.read_timeout:.0f}s.", p.name)
    if isinstance(err, httpx.ConnectTimeout):
        return ProviderError(504, f"Could not connect to {p.base_url} within "
                                  f"{p.connect_timeout:.0f}s.", p.name)
    if isinstance(err, httpx.TimeoutException):
        return ProviderError(504, f"Request to {what}timed out ({type(err).__name__}).", p.name)
    return ProviderError(502, f"Network error talking to {p.base_url}: "
                              f"{type(err).__name__}: {err}", p.name)


def _raise_for_status(resp: httpx.Response, body: str, p: Provider) -> None:
    if resp.status_code >= 400:
        detail = body.strip()[:600] or resp.reason_phrase
        raise ProviderError(resp.status_code, detail, p.name)


# --- calls ------------------------------------------------------------------


async def _key_works(p: Provider, client: httpx.AsyncClient) -> bool:
    """Prove the credential, because a successful catalog read does not.

    NIM serves `/v1/models` to anyone: a garbage key gets a clean 200 and a list of a
    hundred models, and the first sign of trouble is every agent in a dispatch failing
    403 several seconds later. So a provider that takes a credential spends one
    one-token completion proving it.

    Only 401 and 403 count against the key. A 404 means the probe model is not entitled
    to this account, which says nothing at all about whether the key is valid.
    """
    if not p.probe_model:
        return True
    body = {"model": bare(p.probe_model),
            "messages": [{"role": "user", "content": "ok"}],
            "max_tokens": 1, "stream": False}
    try:
        resp = await client.post(f"{p.base_url}/chat/completions",
                                 headers=p.headers(), json=body)
        return resp.status_code not in (401, 403)
    except (httpx.HTTPError, ProviderError):
        return False


async def reachable(p: Provider) -> str:
    """'up', 'unreachable', or 'bad key'. One cheap GET, plus an auth probe if keyed.

    Local is the case that matters for 'unreachable': Ollama may simply not be running,
    and the whole point of the router is to notice that and send the work elsewhere
    rather than fail.
    """
    if not p.enabled or not p.has_key() or p.wire != "openai":
        return "unreachable"
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=p.connect_timeout,
                                                           read=15.0, write=8.0,
                                                           pool=5.0)) as c:
            resp = await c.get(f"{p.base_url}/models", headers=p.headers())
            if resp.status_code in (401, 403):
                return "bad key"
            if resp.status_code >= 400:
                return "unreachable"
            if not p.needs_key:
                return "up"
            return "up" if await _key_works(p, c) else "bad key"
    except (httpx.HTTPError, ProviderError):
        return "unreachable"


async def list_models(p: Provider) -> Timed:
    _require_openai_wire(p)
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=p.timeout) as client:
            resp = await client.get(f"{p.base_url}/models", headers=p.headers())
            _raise_for_status(resp, resp.text, p)
            payload = resp.json()
    except httpx.HTTPError as err:
        raise transport_error(err, p) from err
    return Timed(payload, int((time.perf_counter() - started) * 1000))


async def model_ids(p: Provider) -> list[str]:
    """Qualified ids for everything this provider lists."""
    result = await list_models(p)
    return sorted(p.qualify(m["id"]) for m in result.data.get("data", []) if m.get("id"))


async def is_alive(p: Provider, model: str, client: httpx.AsyncClient | None = None) -> bool:
    """Does this model actually answer? Cheapest possible completion, short timeout.

    A 200 alone is not proof. Completion-only and capacity-exhausted models answer 200
    and return an empty message, so the probe carries a system prompt — the shape a real
    agent call has — and insists on actual content coming back.
    """
    body = {
        "model": bare(model),
        "messages": [{"role": "system", "content": "Answer with one word."},
                     {"role": "user", "content": "Say: ready"}],
        "max_tokens": 8,
        "stream": False,
    }
    owned = client is None
    client = client or httpx.AsyncClient(timeout=p.probe)
    try:
        resp = await client.post(f"{p.base_url}/chat/completions", headers=p.headers(),
                                 json=body, timeout=p.probe)
        return resp.status_code == 200 and bool(extract_text(resp.json()).strip())
    except (httpx.HTTPError, ValueError, ProviderError):
        return False
    finally:
        if owned:
            await client.aclose()


def _body(p: Provider, model: str, prompt: str, system: str | None,
          temperature: float, max_tokens: int, stream: bool) -> dict:
    messages = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": prompt}
    ]
    return {
        "model": bare(model),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": p.cap(max_tokens),
        "stream": stream,
    }


async def chat(model: str, prompt: str, system: str | None = None,
               temperature: float = 0.2, max_tokens: int = 512,
               key: str | None = None) -> Timed:
    """One non-streaming completion. `model` is a qualified id; the provider follows."""
    p = with_key(split(model)[0], key)
    _require_openai_wire(p)
    body = _body(p, model, prompt, system, temperature, max_tokens, stream=False)
    started = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=p.timeout) as client:
            resp = await client.post(f"{p.base_url}/chat/completions",
                                     headers=p.headers(), json=body)
            _raise_for_status(resp, resp.text, p)
            payload = resp.json()
    except httpx.HTTPError as err:
        raise transport_error(err, p, model) from err
    return Timed(payload, int((time.perf_counter() - started) * 1000))


async def chat_stream(model: str, prompt: str, system: str | None = None,
                      temperature: float = 0.2, max_tokens: int = 512,
                      key: str | None = None) -> AsyncIterator[str]:
    """Yield raw SSE lines from the upstream so the caller can pass them straight through."""
    p = with_key(split(model)[0], key)
    _require_openai_wire(p)
    body = _body(p, model, prompt, system, temperature, max_tokens, stream=True)
    try:
        async with httpx.AsyncClient(timeout=p.timeout) as client:
            async with client.stream("POST", f"{p.base_url}/chat/completions",
                                     headers=p.headers(stream=True), json=body) as resp:
                if resp.status_code >= 400:
                    text = (await resp.aread()).decode("utf-8", "replace")
                    _raise_for_status(resp, text, p)
                async for line in resp.aiter_lines():
                    if line:
                        yield line
    except httpx.HTTPError as err:
        raise transport_error(err, p, model) from err


def with_key(p: Provider, key: str | None) -> Provider:
    """A copy of a provider using a caller-supplied key, for keys pasted into the console.

    Copied rather than mutated: the registry is module-level and shared across requests,
    so writing a key into it would leak one browser session's credential into every
    other request in the process.
    """
    if not key or not p.needs_key:
        return p
    return replace(p, key_override=key)


def extract_text(payload: dict) -> str:
    """Pull the assistant message out of a chat completion payload."""
    try:
        return payload["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


def key_fingerprint(key: str) -> str:
    """Safe-to-log identifier: prefix plus last four characters."""
    if not key:
        return "no key needed"
    if len(key) < 10:
        return "too short to fingerprint"
    return f"{key[:8]}…{key[-4:]} ({len(key)} chars)"


# Surveying costs a round trip per provider, and health, catalog and dispatch all want
# the answer. Short TTL rather than none: long enough that a page load does not probe
# three times, short enough that starting Ollama shows up without a restart.
SURVEY_TTL = float(os.getenv("PROVIDER_SURVEY_TTL", "45"))
_survey: dict[str, tuple[float, list[dict]]] = {}


async def survey(key_override: str | None = None, force: bool = False) -> list[dict]:
    """Status of every registered provider, for the UI and for routing.

    Reachability is checked concurrently: a local server that is not running takes a
    full connect timeout to establish that, and doing it in series behind a healthy
    remote provider would make every catalog read wait for it.
    """
    slot = _survey.get(key_fingerprint(key_override or ""))
    if slot and not force and time.time() - slot[0] < SURVEY_TTL:
        return slot[1]

    def scoped(p: Provider) -> Provider:
        return with_key(p, key_override)

    def has_key(p: Provider) -> bool:
        return scoped(p).has_key()

    testable = [p for p in PROVIDERS if p.enabled and has_key(p) and p.wire == "openai"]
    checks = await asyncio.gather(*(reachable(scoped(p)) for p in testable),
                                  return_exceptions=True)
    state_of = {p.name: (c if isinstance(c, str) else "unreachable")
                for p, c in zip(testable, checks)}

    out = []
    for p in PROVIDERS:
        if not p.enabled:
            state, note = "disabled", p.disabled_note
        elif not has_key(p):
            state, note = "no key", f"Set {p.key_env} in .env."
        elif p.wire != "openai":
            state, note = "no adapter", f"'{p.wire}' wire format is not implemented."
        else:
            state = state_of.get(p.name, "unreachable")
            if state == "up":
                note = p.blurb
            elif state == "bad key":
                note = (f"{p.key_env} was rejected — it is present but not valid for "
                        f"{p.label}.")
            else:
                note = ("Nothing is listening at " + p.base_url
                        + (". Start it with `ollama serve`." if p.tier == "local" else "."))
        out.append({
            "name": p.name, "label": p.label, "tier": p.tier, "base_url": p.base_url,
            "state": state, "note": note, "usable": state == "up", "prefers": p.prefers,
        })

    _survey[key_fingerprint(key_override or "")] = (time.time(), out)
    return out
