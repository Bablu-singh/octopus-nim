"""Where a model can actually come from.

Octopus was bound to one API. It is not any more. A provider is a base URL, an optional
key, and a wire format; everything above this file addresses models as `provider:model`
rather than by bare name. That is the whole extension seam — adding GPT or Claude later
is a row in `PROVIDERS` plus, for Claude, an adapter for its non-OpenAI wire format.
Nothing in the octopus, the tentacles, or the UI has to learn about it.

Three providers are live today, and they are deliberately the three that cost nothing:

  local   Ollama — or any OpenAI-compatible server — on this machine. No key, no
          network, no quota, no catalog that lies. Slow, because it is CPU inference,
          but always entitled.
  nvidia  build.nvidia.com. Free tier, models far larger than anything that fits in
          local RAM, but a catalog that advertises models it cannot serve.
  gemini  aistudio.google.com. Free tier, large models, and an OpenAI-compatible
          surface — which is why it is a ten-line row rather than an adapter.

A second local slot, `localalt`, is registered and off. Nothing in this file is specific
to Ollama; it speaks OpenAI-compatible HTTP and nothing else, so llama.cpp's llama-server,
LM Studio, vLLM or LocalAI all drop in by pointing LOCAL_ALT_BASE_URL at them.

Each becomes usable only when its credential is present, so the set of providers is
whatever the machine can actually reach today. Missing keys are a state, not an error.

openai and anthropic are registered and disabled. They are here so the shape is visible
and so enabling one later is a config change rather than a refactor — not because the
app can use them. Keeping to free resources is a deliberate choice, not an oversight,
and it is why Gemini is enabled above while those two are not.
"""

from __future__ import annotations

import asyncio
import os
import re
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
    # Tie-break within a `prefers` class, lowest first. Two providers can both be the
    # right *kind* of place for a piece of work; this says which to try first.
    priority: int = 50
    # Stripped from ids this provider lists. Gemini's OpenAI-compatible endpoint returns
    # 'models/gemini-2.5-flash' but accepts the bare name, and carrying the prefix around
    # would put a second slash-path inside an id that is already 'provider:model'.
    id_prefix: str = ""
    # Price per million tokens, for the ledger. Zero is the honest number for a free
    # tier and for local inference — the real cost there is quota and wall-clock, which
    # is what `free_tier` and the token counts are for. Set these when enabling a paid
    # provider; leaving them at zero would silently report a paid run as free.
    usd_in: float = 0.0
    usd_out: float = 0.0
    free_tier: bool = True
    # Requests per minute this provider is willing to take. 0 means "no limit worth
    # enforcing" — that is local, where the only real constraint is the CPU and the
    # semaphore in octopus.py already handles it. Free tiers publish small numbers and
    # answer 429 when you exceed them, so the gate is set below the published figure:
    # being throttled costs a whole retry, waiting 200ms costs 200ms.
    rpm: int = 0
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

    def normalize(self, model: str) -> str:
        """A listed id as this provider's own /chat/completions wants to receive it."""
        if self.id_prefix and model.startswith(self.id_prefix):
            return model[len(self.id_prefix):]
        return model

    def qualify(self, model: str) -> str:
        return f"{self.name}{SEP}{self.normalize(model)}"


# --- registry ---------------------------------------------------------------

def _f(env: str, default: float) -> float:
    try:
        return float(os.getenv(env, default))
    except ValueError:
        return default


def _on(name: str, default: str = "1") -> bool:
    """Is this provider switched on? `ENABLE_NVIDIA=0` turns one off without code changes.

    The point is the day a free tier stops being free. Flip the flag and that provider
    leaves `usable`; the router already treats `usable` as the only truth, so every agent
    silently goes somewhere else and the app keeps working exactly as before. Nothing
    needs to be uninstalled, no key needs deleting, and flipping it back is one restart.
    """
    return os.getenv(f"ENABLE_{name.upper()}", default) != "0"


PROVIDERS: list[Provider] = [
    Provider(
        name="local",
        enabled=_on("local", "1"),
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
        priority=0,
        rpm=int(os.getenv("LOCAL_RPM", "0")),
    ),
    Provider(
        name="localalt",
        label="Local (alt)",
        enabled=_on("localalt", "0"),
        # A second local engine alongside Ollama. Nothing here is Ollama-specific — the
        # provider layer speaks OpenAI-compatible HTTP and nothing else — so this row
        # takes llama.cpp's llama-server (port 8080), LM Studio (1234), vLLM or LocalAI
        # without a line of code. Two local providers is the useful shape: the router
        # already fails over between providers, so one engine going down is survivable,
        # and two engines can be compared on the same task by pinning each in turn.
        base_url=os.getenv("LOCAL_ALT_BASE_URL", "http://127.0.0.1:8080/v1"),
        key_env=None,
        tier="local",
        blurb="A second local engine — llama.cpp, LM Studio, vLLM. Off unless enabled.",
        disabled_note=("Off. Start another OpenAI-compatible server, set "
                       "LOCAL_ALT_BASE_URL and ENABLE_LOCALALT=1."),
        trust_catalog=os.getenv("LOCAL_ALT_TRUST_CATALOG", "1") != "0",
        connect_timeout=_f("LOCAL_ALT_CONNECT_TIMEOUT", 5),
        read_timeout=_f("LOCAL_ALT_READ_TIMEOUT", 300),
        probe_timeout=_f("LOCAL_ALT_PROBE_TIMEOUT", 180),
        max_tokens_cap=int(os.getenv("LOCAL_ALT_MAX_TOKENS", "700")),
        rpm=int(os.getenv("LOCAL_ALT_RPM", "0")),
        prefers="small",
        priority=1,
    ),
    Provider(
        name="nvidia",
        enabled=_on("nvidia", "1"),
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
        priority=10,
        rpm=int(os.getenv("NVIDIA_RPM", "36")),
    ),
    Provider(
        name="gemini",
        enabled=_on("gemini", "1"),
        label="Google Gemini",
        # Google publishes an OpenAI-compatible surface alongside its native API. That is
        # the whole reason this row is ten lines and not an adapter: same /models, same
        # /chat/completions, same bearer token.
        base_url=os.getenv("GEMINI_BASE_URL",
                           "https://generativelanguage.googleapis.com/v1beta/openai"),
        key_env="GEMINI_API_KEY",
        tier="free-api",
        blurb="Free tier at aistudio.google.com. Large models, honest catalog, quota'd.",
        # Unlike NIM, Gemini lists what it will actually serve, so probing every candidate
        # would spend free-tier requests confirming something already true — and a 429
        # part-way through would mark healthy models dead. NIM is the anomaly here, not
        # the rule. A model that does fail is still dropped mid-run by `_mark_dead`.
        trust_catalog=os.getenv("GEMINI_TRUST_CATALOG", "1") != "0",
        connect_timeout=_f("GEMINI_CONNECT_TIMEOUT", 10),
        read_timeout=_f("GEMINI_READ_TIMEOUT", 60),
        probe_timeout=_f("GEMINI_PROBE_TIMEOUT", 20),
        probe_model=os.getenv("GEMINI_MODEL", "gemini-flash-latest"),
        id_prefix="models/",
        prefers="large",
        priority=20,
        rpm=int(os.getenv("GEMINI_RPM", "5")),
    ),
    Provider(
        name="openai",
        label="OpenAI",
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        key_env="OPENAI_API_KEY",
        tier="paid",
        blurb="Same wire format as the two above — needs only a key and this row enabled.",
        enabled=_on("openai", "0"),
        free_tier=False,
        disabled_note="Disabled: paid. Octopus is deliberately free-resources-only for now.",
        prefers="large",
        priority=30,
    ),
    Provider(
        name="anthropic",
        label="Anthropic",
        base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1"),
        key_env="ANTHROPIC_API_KEY",
        tier="paid",
        blurb="Different wire format (/messages, x-api-key). Needs an adapter, not just a key.",
        wire="anthropic",
        enabled=_on("anthropic", "0"),
        free_tier=False,
        disabled_note="Disabled: paid, and its wire format needs an adapter first.",
        prefers="large",
        priority=40,
    ),
]

BY_NAME: dict[str, Provider] = {p.name: p for p in PROVIDERS}


class RateLimiter:
    """Spaces requests to one provider so a free tier is never the thing that breaks.

    A minimum interval rather than a burst bucket, deliberately. A burst bucket lets a
    wave of eight agents fire at once and then sit out the rest of the minute, which is
    exactly the shape that trips a per-minute quota; spacing them keeps every one of
    them served. The wait is short enough to disappear next to a completion.
    """

    def __init__(self, rpm: int) -> None:
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def acquire(self) -> None:
        if not self.interval:
            return
        async with self._lock:
            now = time.monotonic()
            wait = self._next - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next = now + self.interval


# Keyed by provider name, not held on the Provider itself: `with_key` copies a provider
# per request, and a limiter living on the copy would gate nothing.
_limiters: dict[str, RateLimiter] = {}
# When a provider told us to back off, and why. Honouring this is the difference between
# a 429 costing one wasted request and a 429 costing every request in the wave.
_cooldown: dict[str, tuple[float, str]] = {}


def limiter(p: Provider) -> RateLimiter:
    lim = _limiters.get(p.name)
    if lim is None or lim.interval != (60.0 / p.rpm if p.rpm > 0 else 0.0):
        lim = _limiters[p.name] = RateLimiter(p.rpm)
    return lim


# How many agents each provider has been handed. Used to spread heavy work across the
# free tiers instead of draining one: two providers at 50% each hit a per-minute quota
# far less often than one at 100%, and the free capacity available is the sum, not the
# maximum. Reset per process, which is the right scope — it is a load balancer, not a
# billing record.
_load: dict[str, int] = {}


def note_use(name: str | None, n: int = 1) -> None:
    if name:
        _load[name] = _load.get(name, 0) + n


def load(name: str) -> int:
    return _load.get(name, 0)


# Consecutive throttles per provider, so repeated 429s back off instead of retrying into
# the same wall. Cleared by any successful call.
_strikes: dict[str, int] = {}


def cool(name: str, seconds: float, why: str = "") -> None:
    """Stand a provider down. The router simply stops seeing it until it recovers.

    Backs off exponentially on repeated throttling. A provider's own `Retry-After` is a
    floor, not a ceiling: honouring 30s literally and then retrying got a second 429
    immediately, because the limit that bit was a per-minute one and 30s had not cleared
    it. Doubling per strike costs one slow run and then stops wasting requests entirely.
    """
    _strikes[name] = _strikes.get(name, 0) + 1
    backoff = seconds * (2 ** (_strikes[name] - 1))
    seconds = max(1.0, min(backoff, 3600.0))
    until = time.time() + seconds
    have = _cooldown.get(name)
    if not have or until > have[0]:
        _cooldown[name] = (until, f"{why or 'rate limited'} (strike {_strikes[name]})")


def note_ok(name: str | None) -> None:
    """A call got through, so the provider is healthy again."""
    if name:
        _strikes.pop(name, None)


def cooldown_left(name: str) -> float:
    slot = _cooldown.get(name)
    if not slot:
        return 0.0
    left = slot[0] - time.time()
    if left <= 0:
        _cooldown.pop(name, None)
        return 0.0
    return left


def cooldown_why(name: str) -> str:
    slot = _cooldown.get(name)
    return slot[1] if slot else ""


def _retry_after(resp: httpx.Response) -> float:
    """Seconds the provider asked us to wait, or a sane default.

    Retry-After may be seconds or an HTTP date; Google also returns the delay inside the
    error body. Anything unparseable falls back to a fixed pause, because guessing low
    here means walking straight back into the same 429.
    """
    raw = (resp.headers.get("Retry-After") or "").strip()
    if raw.isdigit():
        return float(raw)
    body = resp.text[:600]
    m = re.search(r'"?retryDelay"?[":\s]+"?(\d+(?:\.\d+)?)s', body)
    if m:
        return float(m.group(1))
    return 30.0


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
    if resp.status_code == 429:
        wait = _retry_after(resp)
        cool(p.name, wait, f"429 from {p.label}; asked for {wait:.0f}s")
    elif resp.status_code < 400:
        note_ok(p.name)
    if resp.status_code >= 400:
        detail = body.strip()[:600] or resp.reason_phrase
        raise ProviderError(resp.status_code, detail, p.name)


# --- calls ------------------------------------------------------------------


# Providers disagree about which status a rejected credential deserves. NIM and OpenAI
# send 401/403; Google's OpenAI-compatible surface sends 400 "Please pass a valid API
# key". Reading that 400 as "the host is down" would tell the user to check their network
# when the real problem is the key, so the body gets a look before that conclusion.
_AUTH_WORDS = re.compile(
    r"api[ _-]?key|credential|unauthenticated|unauthorized|permission denied|"
    r"invalid.{0,20}key|expired.{0,20}token", re.I)


def _is_auth_failure(status: int, body: str) -> bool:
    if status in (401, 403):
        return True
    # A 400 is ambiguous in general, but on a bare GET /models there is no request body
    # to have malformed, so an auth-shaped message is the only sensible reading.
    return status == 400 and bool(_AUTH_WORDS.search(body or ""))


async def _key_works(p: Provider, client: httpx.AsyncClient) -> bool:
    """Prove the credential, because a successful catalog read does not.

    NIM serves `/v1/models` to anyone: a garbage key gets a clean 200 and a list of a
    hundred models, and the first sign of trouble is every agent in a dispatch failing
    403 several seconds later. So a provider that takes a credential spends one
    one-token completion proving it.

    Only auth-shaped failures count against the key — see `_is_auth_failure`, which has
    to cope with providers disagreeing about the status code. A 404 means the probe model
    is not entitled to this account, which says nothing about whether the key is valid.
    """
    if not p.probe_model:
        return True
    body = {"model": bare(p.probe_model),
            "messages": [{"role": "user", "content": "ok"}],
            "max_tokens": 1, "stream": False}
    try:
        resp = await client.post(f"{p.base_url}/chat/completions",
                                 headers=p.headers(), json=body)
        return not _is_auth_failure(resp.status_code, resp.text)
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
            if _is_auth_failure(resp.status_code, resp.text):
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
    await limiter(p).acquire()
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
    await limiter(p).acquire()
    try:
        resp = await client.post(f"{p.base_url}/chat/completions", headers=p.headers(),
                                 json=body, timeout=p.probe)
        if resp.status_code == 429:
            # Being throttled says nothing about whether this model works. Treating it
            # as dead would drop a perfectly good model for the rest of the cache TTL.
            cool(p.name, _retry_after(resp), f"429 while probing {p.label}")
            return False
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
    await limiter(p).acquire()
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
    await limiter(p).acquire()
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


# The console override arrives on the X-NIM-Key header, so it is NVIDIA's credential and
# nobody else's. Before this was pinned down, a key meant for NIM was being applied to
# every keyed provider — which handed Google an nvapi- string and got back "bad key" for
# a provider the user had never configured.
KEY_OWNER = "nvidia"


def with_key(p: Provider, key: str | None, owner: str = KEY_OWNER) -> Provider:
    """A copy of a provider using a caller-supplied key, for keys pasted into the console.

    Copied rather than mutated: the registry is module-level and shared across requests,
    so writing a key into it would leak one browser session's credential into every
    other request in the process — and, now that there is more than one keyed provider,
    into a provider it was never meant for. Hence `owner`: an override applies to exactly
    one provider, and every other provider keeps reading its own environment variable.
    """
    if not key or not p.needs_key or p.name != owner:
        return p
    return replace(p, key_override=key)


def extract_text(payload: dict) -> str:
    """Pull the assistant message out of a chat completion payload."""
    try:
        return payload["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""


# Rough, and deliberately so. Providers report real usage only on non-streaming calls,
# and every agent here streams. Four characters per token is the usual English rule of
# thumb and is close enough to compare providers within one run — which is what the
# ledger is for. It is labelled an estimate everywhere it surfaces.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return max(1, len(text or "") // CHARS_PER_TOKEN)


def price(p: Provider, tokens_in: int, tokens_out: int) -> float:
    return (tokens_in * p.usd_in + tokens_out * p.usd_out) / 1_000_000


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
    cooling = any(cooldown_left(p.name) > 0 for p in PROVIDERS)
    # A cached "up" for a provider that has since been throttled would send a whole wave
    # straight back into the 429 it was standing down from.
    if slot and not force and not cooling and time.time() - slot[0] < SURVEY_TTL:
        return slot[1]

    def scoped(p: Provider) -> Provider:
        return with_key(p, key_override)

    def has_key(p: Provider) -> bool:
        return scoped(p).has_key()

    testable = [p for p in PROVIDERS if p.enabled and has_key(p) and p.wire == "openai"
                and cooldown_left(p.name) <= 0]
    checks = await asyncio.gather(*(reachable(scoped(p)) for p in testable),
                                  return_exceptions=True)
    state_of = {p.name: (c if isinstance(c, str) else "unreachable")
                for p, c in zip(testable, checks)}

    out = []
    for p in PROVIDERS:
        if not p.enabled:
            state = "disabled"
            note = p.disabled_note or f"Switched off — set ENABLE_{p.name.upper()}=1 to use it."
        elif not has_key(p):
            state, note = "no key", f"Set {p.key_env} in .env."
        elif p.wire != "openai":
            state, note = "no adapter", f"'{p.wire}' wire format is not implemented."
        elif cooldown_left(p.name) > 0:
            left = cooldown_left(p.name)
            state = "cooling down"
            note = f"{cooldown_why(p.name)} — back in {left:.0f}s."
        else:
            state = state_of.get(p.name, "unreachable")
            if state == "up":
                note = p.blurb
                if p.rpm:
                    note += f" Gated to {p.rpm} req/min."
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
