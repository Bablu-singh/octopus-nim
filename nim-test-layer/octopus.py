"""Octopus orchestrator — routes one task across several models at once.

Mounted onto the existing test layer by app.py. Everything streams over a single SSE
connection so the UI can light tentacles up as they start, finish, or fail independently.

Models come from more than one place now. `catalog()` merges every reachable provider
into a single pool of qualified ids, and `dispatch()` asks `routing` where each agent's
work should go before binding a model to it — small work to the local machine, heavy work
to NIM, everything to whichever one is actually up. See providers.py and routing.py.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import time
from typing import AsyncIterator

import httpx

import agents
import providers
import routing
import web

IMAGE_BASE = os.getenv("NVIDIA_IMAGE_BASE", "https://ai.api.nvidia.com/v1/genai")
CATALOG_TTL = 900          # seconds; verification is expensive, so cache it for a while
PROBE_CONCURRENCY = 16
# How many candidates per role get probed on a cold catalog. Every probe is a real
# request against the provider's rate limit, so this is the difference between a first
# dispatch that starts in seconds and one that sits silent for a minute. Four is enough
# to find a working model for a role in practice; the fallback pool covers the rest.
PROBE_PER_ROLE = int(os.getenv("OCTOPUS_PROBE_PER_ROLE", "4"))
STALL_TIMEOUT = 120        # seconds of silence from every agent before giving up

# The agent pool is sized by the task, not by the number of roles. A role is a template
# that can be instantiated as often as the work divides; these are the only ceilings.
MAX_AGENTS = int(os.getenv("OCTOPUS_MAX_AGENTS", "24"))      # total, across all waves
FIRST_WAVE = int(os.getenv("OCTOPUS_FIRST_WAVE", "8"))       # cap on the opening plan
MAX_WAVES = int(os.getenv("OCTOPUS_MAX_WAVES", "4"))         # planner + 3 growth rounds
MAX_PARALLEL = int(os.getenv("OCTOPUS_MAX_PARALLEL", "6"))   # concurrent upstream streams
# Local inference competes for the same cores, so parallelism there is negative value past
# a point: two 7Bs on eight cores each run at less than half speed. Hosted models have no
# such problem, which is why the limit is per provider rather than global.
LOCAL_PARALLEL = int(os.getenv("OCTOPUS_LOCAL_PARALLEL", "2"))
DIGEST_CHARS = 700         # per agent, when summarising a wave for the supervisor

# --- cost control ---------------------------------------------------------------
# Everything reachable today is free, so "cost" here means quota and wall-clock rather
# than money. The cheapest call is still the one that never happens, which is what these
# three knobs are for; the ledger then reports what the run actually spent.
#
# Below this task weight, wave 1 is planned by keyword instead of by a model. A one-line
# question does not need a planning completion to work out that it is one agent — and on
# a trivial task that call is a large fraction of the whole run's cost.
PLANNER_SKIP_BELOW = float(os.getenv("OCTOPUS_PLANNER_SKIP_BELOW", "0.20"))
# Hard ceiling on estimated tokens for one dispatch. 0 disables it. Counts every agent
# plus the planner and supervisor; when it trips, no further agents are enlisted and the
# ones already running finish normally.
TOKEN_BUDGET = int(os.getenv("OCTOPUS_TOKEN_BUDGET", "0"))

# Keyed by fingerprint: two different keys have different entitlements, and sharing one
# cache between them would bind tentacles to models the current key cannot reach.
_catalog: dict[str, dict] = {}


def _slot_key(key: str | None) -> str:
    return providers.key_fingerprint(key or "") or "no-key"


async def _verify(p: providers.Provider, ids: list[str]) -> set[str]:
    """Probe every model any tentacle might want, and return the ones that answer.

    NIM's catalog is aspirational — on a typical key most listed models 404 and several
    accept a connection then never respond. Probing here, once, is what stops a dispatch
    from stalling on a dead model.

    A local catalog is not aspirational: Ollama lists what is on disk, and a cold 7B can
    take a minute to load, so probing it would add minutes to a catalog read to confirm
    something already known. Those providers are trusted on sight and dropped by
    `_mark_dead` if they ever actually fail.
    """
    if p.trust_catalog:
        return set(ids)

    wanted: list[str] = []
    for t in agents.TENTACLES:
        wanted += [m for m, _ in agents.shortlist(t, ids, limit=PROBE_PER_ROLE)]
    wanted = list(dict.fromkeys(wanted))  # dedupe, preserve order

    sem = asyncio.Semaphore(PROBE_CONCURRENCY)

    async def check(client: httpx.AsyncClient, mid: str) -> tuple[str, bool]:
        async with sem:
            return mid, await providers.is_alive(p, mid, client)

    async with httpx.AsyncClient(timeout=p.probe) as client:
        results = await asyncio.gather(*(check(client, m) for m in wanted))
    return {mid for mid, ok in results if ok}


async def _read_provider(p: providers.Provider) -> tuple[list[str], set[str], str]:
    """One provider's catalog, or empty lists plus the reason it gave nothing."""
    try:
        ids = await providers.model_ids(p)
    except providers.ProviderError as err:
        return [], set(), f"HTTP {err.status}: {err.detail[:160]}"
    if not ids:
        return [], set(), "listed no models"
    return ids, await _verify(p, ids), ""


async def catalog(key: str | None = None, force: bool = False) -> dict:
    """Everything reachable right now, from every provider, as one pool.

    Cached, because verifying NIM costs a probe per candidate model and the UI polls this.
    """
    slot = _catalog.get(_slot_key(key))
    if slot and not force and time.time() - slot["fetched_at"] < CATALOG_TTL:
        ids, live, per, latency = slot["ids"], slot["live"], slot["per"], None
    else:
        started = time.perf_counter()
        survey = await providers.survey(key)
        up = [s["name"] for s in survey if s["usable"]]

        # Concurrently: a slow local load must not hold up the hosted catalog, and a
        # hosted timeout must not hold up local.
        reads = await asyncio.gather(*(
            _read_provider(providers.with_key(providers.get(n), key)) for n in up
        ), return_exceptions=True)

        ids, live, per = [], set(), {}
        for name, read in zip(up, reads):
            if isinstance(read, BaseException):
                per[name] = {"total": 0, "verified": 0, "note": f"{type(read).__name__}"}
                continue
            pids, plive, note = read
            ids += pids
            live |= plive
            per[name] = {"total": len(pids), "verified": len(plive), "note": note}
        ids = sorted(ids)

        _catalog[_slot_key(key)] = {
            "ids": ids, "live": live, "per": per, "survey": survey,
            "fetched_at": time.time(),
        }
        latency = int((time.perf_counter() - started) * 1000)

    survey = _catalog[_slot_key(key)]["survey"]
    grouped = agents.classify(ids)
    usable = {s["name"] for s in survey if s["usable"]}
    return {
        "total": len(ids),
        "verified": len(live),
        # Raw material for re-binding: a dispatch that loses a model mid-run needs to pick
        # a replacement without paying for another full probe.
        "ids": ids,
        "live": sorted(live),
        "latency_ms": latency,
        "cached": latency is None,
        "families": {k: grouped[k] for k in sorted(grouped, key=lambda f: -len(grouped[f]))},
        "providers": survey,
        "by_provider": per,
        "route_mode": routing.MODE,
        # What each role would bind to, per provider. The router picks the provider at
        # dispatch time from the subtask, so showing only one binding per role would be
        # showing a decision that has not been made yet.
        "bindings": {name: agents.bind(ids, live=live, provider=name) for name in usable},
        "tentacles": agents.bind(ids, live=live),
    }


def _mark_dead(key: str | None, model: str | None) -> None:
    """Forget a model that failed for real, and force a re-probe on the next catalog read.

    Entitlements and cold-start behaviour drift, so a binding that worked when the cache
    was filled can break inside the TTL. Without this the same dead model would be handed
    out for the rest of the window.
    """
    slot = _catalog.get(_slot_key(key))
    if slot and model and model in slot["live"]:
        slot["live"].discard(model)
        slot["fetched_at"] = 0.0


class Ledger:
    """What a dispatch spent, per provider.

    Token counts are estimates — see `providers.estimate_tokens`; agents stream, and
    streaming responses carry no usage block. They are still the only way to compare
    where a run's budget went, and they are what `TOKEN_BUDGET` is enforced against.
    """

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}

    def add(self, provider: str | None, prompt: str, output: str, calls: int = 1) -> None:
        if not provider:
            return
        p = providers.BY_NAME.get(provider)
        tin, tout = providers.estimate_tokens(prompt), providers.estimate_tokens(output)
        row = self.rows.setdefault(
            provider, {"calls": 0, "tokens_in": 0, "tokens_out": 0, "usd": 0.0,
                       "free": p.free_tier if p else True})
        row["calls"] += calls
        row["tokens_in"] += tin
        row["tokens_out"] += tout
        if p:
            row["usd"] = round(row["usd"] + providers.price(p, tin, tout), 6)

    @property
    def tokens(self) -> int:
        return sum(r["tokens_in"] + r["tokens_out"] for r in self.rows.values())

    def report(self) -> dict:
        return {
            "by_provider": self.rows,
            "tokens": self.tokens,
            "usd": round(sum(r["usd"] for r in self.rows.values()), 6),
            "billable": any(not r["free"] for r in self.rows.values()),
            "estimated": True,
        }


def _event(kind: str, **payload) -> str:
    return f"data: {json.dumps({'event': kind, **payload})}\n\n"


def _slug(label: str) -> str:
    """Normalise a deliverable name so near-identical labels collide.

    A supervisor asking again for work it already has rarely reuses the exact wording —
    'IBAN function' comes back as 'iban_function'. Punctuation and case have to go before
    the two can be recognised as the same deliverable.
    """
    return re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()


def _parse_agents(raw: str, budget: int, task: str) -> tuple[list[dict], str]:
    """Pull an agent list out of a planner or supervisor reply. Raises on unusable JSON."""
    trimmed = raw[raw.find("{"): raw.rfind("}") + 1]
    parsed = json.loads(trimmed)
    out = []
    for a in parsed.get("agents", []):
        if not isinstance(a, dict):
            continue
        subtask = (a.get("subtask") or "").strip() or task
        label = (a.get("label") or "").strip()[:60]
        role = a.get("role")
        if role not in agents.BY_ID:
            # Planners invent roles. Map to the nearest real one rather than dropping the
            # agent — the work it describes is usually valid even when the label is not.
            role = agents.role_for_text(f"{role or ''} {label} {subtask}")
        out.append({"role": role, "label": label or agents.BY_ID[role].name,
                    "subtask": subtask})
    return out[:budget], str(parsed.get("why", "") or "")


async def plan(key: str | None, task: str, model: str | None,
               budget: int) -> tuple[list[dict], str]:
    """Decompose the task into a first wave of agents. Falls back to keywords on failure."""
    if not model:
        return agents.keyword_plan(task, budget), "keyword planning (no planner model available)"
    try:
        result = await providers.chat(
            model, f"Task:\n{task}",
            system=agents.planner_system(budget), temperature=0.0, max_tokens=1500, key=key,
        )
        picks, why = _parse_agents(providers.extract_text(result.data), budget, task)
        if picks:
            return picks, why or "planned by model"
        why = "planner returned no usable agents"
    except providers.ProviderError as err:
        why = f"planner unavailable (HTTP {err.status})"
    except (ValueError, KeyError, TypeError):
        why = "planner did not return usable JSON"
    return agents.keyword_plan(task, budget), f"keyword planning ({why})"


async def supervise(key: str | None, task: str, model: str | None, digest: str,
                    budget: int) -> tuple[list[dict], str]:
    """Having seen the wave that just finished, decide whether the task needs more agents.

    This is what lets the pool grow with the work instead of being fixed at dispatch time.
    An empty list is the normal answer and ends the run.
    """
    if not model or budget <= 0:
        return [], ""
    try:
        result = await providers.chat(
            model, f"Task:\n{task}\n\nProduced so far:\n{digest}",
            system=agents.supervisor_system(budget), temperature=0.0, max_tokens=1200, key=key,
        )
        return _parse_agents(providers.extract_text(result.data), budget, task)
    except (providers.ProviderError, ValueError, KeyError, TypeError):
        # A supervisor that cannot answer means "no more agents", never a failed dispatch.
        return [], ""


async def _generate_image(key: str, model: str, prompt: str) -> dict:
    """NIM image models use the genai endpoint, not the OpenAI-compatible one."""
    url = f"{IMAGE_BASE}/{providers.bare(model)}"
    body = {"prompt": prompt, "cfg_scale": 5, "mode": "base", "steps": 30, "seed": 0}
    nvidia = providers.get("nvidia")
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=120.0,
                                                           write=30.0, pool=10.0)) as client:
            resp = await client.post(
                url, json=body,
                headers={"Authorization": f"Bearer {key or nvidia.key()}",
                         "Accept": "application/json"},
            )
            if resp.status_code >= 400:
                raise providers.ProviderError(resp.status_code, resp.text[:400], "nvidia")
            data = resp.json()
    except httpx.HTTPError as err:
        raise providers.transport_error(err, nvidia, model) from err
    # Response shape varies by model family; check the documented spots in turn.
    b64 = (data.get("image")
           or (data.get("artifacts") or [{}])[0].get("base64")
           or (data.get("data") or [{}])[0].get("b64_json"))
    if not b64:
        raise providers.ProviderError(502, f"No image field in response. Keys: {list(data)[:6]}",
                                      "nvidia")
    return {"image": f"data:image/png;base64,{b64}"}


async def _run_agent(key: str | None, agent: dict, queue: asyncio.Queue,
                     sems: dict[str, asyncio.Semaphore], outputs: dict[str, str],
                     ledger: "Ledger | None" = None) -> None:
    """Run one agent to completion, streaming as it goes.

    The model is passed in on the agent dict rather than read off the shared role template,
    so any number of agents — including several sharing a role — can run at once without
    treading on each other. The semaphore comes from `sems` by provider: local work queues
    behind a much narrower limit than hosted work, because it is contending for this
    machine's cores rather than someone else's GPUs.
    """
    aid, role, model = agent["id"], agent["role"], agent["model"]
    t = agents.BY_ID[role]
    outputs[aid] = ""
    prov = agent.get("provider") or (providers.provider_of(model) if model else None)

    await queue.put(_event("queued", id=aid, role=role, label=agent["label"],
                           model=model, name=t.name, color=t.color,
                           provider=prov, route_why=agent.get("route_why", "")))

    if not model:
        await queue.put(_event("error", id=aid,
                               detail=agent.get("route_why")
                                      or "No reachable model fits this role."))
        return

    async with sems.get(prov, sems["nvidia"]):
        started = time.perf_counter()
        ms = lambda: int((time.perf_counter() - started) * 1000)

        # Eyes, for the roles that need them. A researcher always looks things up; any
        # other agent does too if its subtask names a URL, because a subtask that quotes
        # a link plainly wants that page read rather than guessed at.
        system, prompt = t.system, agent["subtask"]
        urls = web.URL_IN_TEXT.findall(agent["subtask"])
        if web.ENABLED and (role == "researcher" or urls):
            await queue.put(_event("searching", id=aid,
                                   detail=f"reading {len(urls)} link(s)" if urls
                                          else "searching the web"))
            try:
                if urls:
                    pages = await asyncio.gather(*(web.fetch(u) for u in urls[:4]))
                    bundle = {"query": agent["subtask"],
                              "results": [],
                              "pages": [p for p in pages if p.ok],
                              "failed": [{"url": p.url, "error": p.error}
                                         for p in pages if not p.ok]}
                else:
                    bundle = await web.research(agent["subtask"])
                context = web.as_context(bundle)
                if context:
                    # The guardrail goes in the system prompt, not the user turn: it has
                    # to outrank anything the fetched page tries to say.
                    system = f"{t.system}\n\n{web.GUARDRAIL}"
                    prompt = (f"{context}\n\nUsing only the sources above, do this:\n"
                              f"{agent['subtask']}")
                    await queue.put(_event("sources", id=aid,
                                           sources=web.sources(bundle),
                                           query=bundle.get("query", ""),
                                           hits=len(bundle.get("results", [])),
                                           failed=bundle.get("failed", [])))
                elif role == "researcher":
                    # Silence here is what produces invented citations: asked for sources
                    # and given none, a model helpfully supplies plausible-looking URLs.
                    # Saying so explicitly, in the prompt, is what stops it.
                    prompt = f"""NO SOURCES WERE RETRIEVED. The web search returned
nothing readable for this question.

Say plainly that you could not retrieve sources and therefore cannot verify an answer.
You may add what you know from training, clearly labelled as unverified and possibly out
of date. Do NOT invent, guess at, or reconstruct any URL — cite nothing rather than
citing something you did not read.

The question:
{agent['subtask']}"""
                    await queue.put(_event("sources", id=aid, sources=[],
                                           query=bundle.get("query", ""),
                                           hits=len(bundle.get("results", [])),
                                           failed=bundle.get("failed", []),
                                           detail="nothing readable came back — "
                                                  "answering without sources"))
            except web.WebError as err:
                await queue.put(_event("sources", id=aid, sources=[], failed=[],
                                       detail=f"web lookup failed: {err}"))
        await queue.put(_event("start", id=aid, role=role, label=agent["label"],
                               model=model, name=t.name, color=t.color,
                               provider=prov, route_why=agent.get("route_why", "")))
        try:
            if t.kind == "image" and agents.family_of(model) == "image":
                out = await _generate_image(key or "", model, agent["subtask"])
                await queue.put(_event("image", id=aid, **out, latency_ms=ms()))
                await queue.put(_event("done", id=aid, model=model, provider=prov,
                                       latency_ms=ms()))
                return

            chunks = 0
            async for line in providers.chat_stream(model, prompt, system=system,
                                                    temperature=t.temperature,
                                                    max_tokens=t.max_tokens, key=key):
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    delta = json.loads(payload)["choices"][0]["delta"].get("content") or ""
                except (KeyError, IndexError, ValueError):
                    continue
                if delta:
                    chunks += 1
                    outputs[aid] += delta
                    await queue.put(_event("chunk", id=aid, text=delta))

            if ledger is not None:
                # `prompt` rather than the subtask: fetched context is most of what was
                # sent on a research agent, and a ledger that ignored it would under-report
                # the expensive calls by an order of magnitude.
                ledger.add(prov, system + prompt, outputs[aid])

            if chunks == 0:
                _mark_dead(key, model)
                await queue.put(_event("error", id=aid, model=model, provider=prov,
                                       model_at_fault=True,
                                       detail=f"'{model}' streamed no content — listed, but "
                                              f"not serving this request."))
                return
            await queue.put(_event("done", id=aid, model=model, provider=prov,
                                   latency_ms=ms()))
        except asyncio.CancelledError:
            raise
        except providers.ProviderError as err:
            # These say the binding is bad rather than the prompt, so the caller should
            # swap models rather than retry the same one.
            #
            # 429 is the exception: it means we asked too often, not that this model is
            # broken. Forgetting the model would drop a working one for the rest of the
            # cache TTL and, worse, send the retry at a *different* model on the same
            # throttled provider. `providers` has already put that provider on cooldown,
            # so the right move is to leave the model alone and let the router pick a
            # provider that is not currently standing down.
            throttled = err.status == 429
            at_fault = err.status in (404, 500, 502, 503, 504)
            if at_fault:
                _mark_dead(key, model)
            await queue.put(_event(
                "error", id=aid, model=model, provider=prov,
                model_at_fault=at_fault, throttled=throttled,
                detail=(f"{prov} is rate limited — standing it down for "
                        f"{providers.cooldown_left(prov or ''):.0f}s and re-routing."
                        if throttled else f"HTTP {err.status}: {err.detail[:300]}")))
        except Exception as err:  # keep one failed agent from killing the others
            await queue.put(_event("error", id=aid, model=model, provider=prov,
                                   model_at_fault=False,
                                   detail=f"{type(err).__name__}: {err}"))


async def dispatch(key: str | None, task: str, roles: list[str] | None = None,
                   mode: str | None = None) -> AsyncIterator[str]:
    """Grow a pool of agents around one task and stream all of them at once.

    The task decides how many agents run — a one-line request gets one, a broad one gets a
    dozen. Wave 1 comes from the planner; after each wave the supervisor sees what came
    back and adds more if the work is not covered, until it says stop or a budget runs out.

    It also decides *where* each of them runs. Every agent's subtask is weighed before a
    model is bound to it, and light work goes to the local machine while heavy work goes
    to NIM — so an eight-agent dispatch can be half local and half hosted, and a run with
    one provider down still completes on the other.

    `roles` is a debugging escape hatch that seeds wave 1 by hand. The UI never sends it:
    identifying the kind of work is the planner's job, not the user's.
    """
    # Emitted before the catalog is read, not after. On a cold cache that read probes
    # every candidate model against a rate limit and can take the best part of a minute,
    # during which the UI previously showed nothing at all — which reads as a hung app,
    # or as agents running one at a time.
    yield _event("status", detail="reading model catalogs and verifying which models answer…")
    try:
        cat = await catalog(key)
    except providers.ProviderError as err:
        yield _event("fatal", detail=f"Could not read the model catalog — "
                                     f"HTTP {err.status}: {err.detail[:200]}")
        yield _event("complete", agents=0, waves=0)
        return

    ids, live = cat["ids"], set(cat["live"])
    usable = {p["name"] for p in cat["providers"] if p["usable"]}
    dead: set[str] = set()

    yield _event("providers", providers=cat["providers"], mode=mode or routing.MODE)

    def rebind(provider: str) -> dict[str, str | None]:
        return {t["id"]: t["model"]
                for t in agents.bind(ids, live=live - dead, provider=provider)}

    bound: dict[str, dict[str, str | None]] = {p: rebind(p) for p in usable}
    if not any(m for table in bound.values() for m in table.values()):
        reachable = ", ".join(sorted(usable)) or "nothing"
        yield _event("fatal", detail=(
            f"Reached {reachable}, but no model there answered a test request. "
            "Check entitlements at build.nvidia.com/settings/api-keys, or that "
            "`ollama list` shows a pulled model."))
        yield _event("complete", agents=0, waves=0)
        return

    yield _event("status", detail="planning the agent pool…")
    planner_provider, planner_why = routing.choose_planner(usable, mode)
    thinker = agents.pick_planner(live - dead, planner_provider)

    # Scored once, against the whole task rather than a subtask, and used only to decide
    # whether growing the pool past wave 1 can possibly be justified. See the wave loop.
    task_weight, _ = routing.weigh(task, "analyst")

    queue: asyncio.Queue = asyncio.Queue()
    sems = {"local": asyncio.Semaphore(LOCAL_PARALLEL)}
    for name in usable | {"nvidia"}:
        sems.setdefault(name, asyncio.Semaphore(MAX_PARALLEL))
    outputs: dict[str, str] = {}
    ledger = Ledger()
    workers: list[asyncio.Task] = []
    spawned: list[dict] = []
    counts: dict[str, int] = {}
    attempts: dict[str, int] = {}
    slices: dict[str, str] = {}                  # deliverable -> running|ok|failed
    owner: dict[str, str] = {}                   # agent id -> deliverable
    errors: dict[str, str] = {}
    ok = failed = waves_run = 0

    def route(a: dict) -> tuple[str | None, str | None, str]:
        """Where this agent runs, which model it gets there, and why.

        Provider first, model second. An imager only counts as wanting image generation
        if a provider that can do it is actually up; otherwise it is a text agent writing
        a generation prompt, and routes on the weight of that text like anything else.
        """
        t = agents.BY_ID[a["role"]]
        # Only NIM's genai endpoint is implemented in `_generate_image`, so an image
        # model listed by anyone else does not make this agent a drawing agent. Gemini
        # lists 'gemini-2.5-flash-image'; binding to it and then posting to NVIDIA's
        # endpoint would fail, and treating the agent as image-capable when NIM is down
        # would strand it with no provider at all instead of falling back to prose.
        wants_image = t.kind == "image" and "nvidia" in usable and any(
            agents.family_of(m) == "image" and providers.provider_of(m) == "nvidia"
            for m in (live - dead)
        )
        provider, why = routing.choose(a["role"], a["subtask"], usable - _exhausted(),
                                       wants_image=wants_image, mode=mode)
        if not provider:
            return None, None, why
        return provider, bound.get(provider, {}).get(a["role"]), why

    def _exhausted() -> set[str]:
        """Providers the router should stop offering: nothing left to bind, or throttled.

        Re-read every time rather than cached, because a cooldown set part-way through a
        wave has to affect the very next agent enlisted, not the next dispatch.
        """
        return ({p for p in usable if not any(bound.get(p, {}).values())}
                | {p for p in usable if providers.cooldown_left(p) > 0})

    def enlist(batch: list[dict]) -> list[dict]:
        """Give each planned agent a unique id, a provider, and a model, and start it.

        A slice is the label, which names a deliverable. Keying on the label alone rather
        than on (role, label) is deliberate: planners routinely hand the same deliverable
        to two roles, and one 'edge cases' document is enough.

        Work already done, or in flight, is never started again —
        that alone removes most of the padding a supervisor produces. A slice that failed
        gets one retry, since the usual cause is a model that has since been swapped out;
        a slice that fails twice is not going to work and is left alone.
        """
        made = []
        for a in batch:
            if len(spawned) >= MAX_AGENTS:
                break
            # Agents already running are left alone — cancelling them would spend the
            # tokens and throw away the answer, which is the opposite of the point.
            if TOKEN_BUDGET and ledger.tokens >= TOKEN_BUDGET:
                break
            sid = _slug(a["label"])
            if slices.get(sid) in ("ok", "running") or attempts.get(sid, 0) >= 2:
                continue
            attempts[sid] = attempts.get(sid, 0) + 1
            slices[sid] = "running"
            counts[a["role"]] = counts.get(a["role"], 0) + 1
            provider, model, why = route(a)
            # Counted at enlist time, not at completion: the next agent in this same wave
            # has to see the load this one just added, or a wave of six would all be
            # routed to whichever provider happened to be idle when the wave began.
            providers.note_use(provider)
            agent = {**a, "id": f"{a['role']}-{counts[a['role']]}",
                     "provider": provider, "model": model, "route_why": why}
            owner[agent["id"]] = sid
            spawned.append(agent)
            made.append(agent)
            workers.append(asyncio.create_task(
                _run_agent(key, agent, queue, sems, outputs, ledger)))
        return made

    try:
        for wave in range(1, MAX_WAVES + 1):
            remaining = MAX_AGENTS - len(spawned)
            if remaining <= 0:
                break

            if wave == 1:
                if roles:
                    batch = [{"role": r, "label": agents.BY_ID[r].name, "subtask": task}
                             for r in roles if r in agents.BY_ID]
                    why = "roles supplied directly"
                elif task_weight < PLANNER_SKIP_BELOW:
                    # The cheapest completion is the one never made. A task this light is
                    # one agent's work, and paying a planner call to be told so costs more
                    # than the answer does.
                    batch = agents.keyword_plan(task, min(FIRST_WAVE, remaining))
                    why = (f"keyword planning · task weight {task_weight:.2f} below "
                           f"{PLANNER_SKIP_BELOW:.2f}, planner call skipped to save a completion")
                else:
                    batch, why = await plan(key, task, thinker, min(FIRST_WAVE, remaining))
                    if thinker:
                        why = f"{why} · planner on {planner_provider} ({planner_why})"
                    ledger.add(planner_provider, task, "", calls=1)
            else:
                # A task the planner sized at a single agent is by definition not
                # multi-part, so if that agent succeeded there is nothing to grow into.
                # Without this the supervisor invents busywork for one-line questions.
                if len(spawned) == 1 and not errors:
                    break

                # The same reasoning one level up, and the guard that actually bites.
                # The rule above only fires when the planner produced exactly one agent,
                # which a weak planner rarely does: asked for a one-line thank-you note,
                # a local 7B returned two agents, and the supervisor then grew that to
                # five across four waves — 150s of CPU for a sentence. The task's own
                # weight is the honest signal here, and it does not depend on the planner
                # having been any good.
                if task_weight < routing.THRESHOLD and not errors:
                    break

                # Failures go in too. A supervisor shown only successes reads a failed
                # agent as untouched work and asks for it again, every wave.
                def entry(a: dict) -> str:
                    head = f"[{a['label']} · {a['role']}] "
                    if a["id"] in errors:
                        return head + f"FAILED, do not request again: {errors[a['id']]}"
                    text = outputs.get(a["id"], "")
                    # Flag the cut explicitly. An excerpt that stops mid-sentence otherwise
                    # reads as an unfinished agent, and the supervisor orders it redone.
                    if len(text) > DIGEST_CHARS:
                        return (head + text[:DIGEST_CHARS]
                                + " …[EXCERPT ENDS — this agent finished in full]")
                    return head + text

                digest = "\n\n".join(
                    entry(a) for a in spawned
                    if outputs.get(a["id"]) or a["id"] in errors
                )
                batch, why = await supervise(key, task, thinker, digest, remaining)
                ledger.add(planner_provider, digest, "", calls=1)
                if not batch:
                    break
                why = why or "supervisor added follow-up agents"

            started = enlist(batch)
            if not started:
                break
            waves_run = wave

            yield _event("wave", n=wave, why=why, agents=[
                {"id": a["id"], "role": a["role"], "label": a["label"], "model": a["model"],
                 "name": agents.BY_ID[a["role"]].name, "color": agents.BY_ID[a["role"]].color,
                 "provider": a["provider"], "route_why": a["route_why"],
                 "subtask": a["subtask"][:400]}
                for a in started
            ])

            # Drain until this whole wave is finished — the supervisor needs the results
            # before it can judge what is still missing.
            finished = 0
            while finished < len(started):
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=STALL_TIMEOUT)
                except asyncio.TimeoutError:
                    running = [a["id"] for a, w in zip(spawned, workers) if not w.done()]
                    yield _event("fatal", detail=(
                        f"No output for {STALL_TIMEOUT}s from: "
                        f"{', '.join(running[:6]) or 'any agent'}."))
                    raise
                yield item
                # Parse rather than string-match: a chunk of model text could otherwise
                # contain the literal marker and miscount the wave as finished.
                try:
                    ev = json.loads(item[5:])
                except ValueError:
                    continue
                kind = ev.get("event")
                if kind == "done":
                    ok += 1
                    finished += 1
                    slices[owner[ev["id"]]] = "ok"
                elif kind == "error":
                    failed += 1
                    finished += 1
                    slices[owner[ev["id"]]] = "failed"
                    errors[ev["id"]] = ev.get("detail", "")[:200]
                    # The model let us down, not the prompt — drop it and re-bind, so the
                    # next wave reaches for a different one instead of the same dead end.
                    # Only that model's own provider is re-bound; a NIM model going dark
                    # says nothing about what Ollama can serve.
                    if ev.get("model_at_fault") and ev.get("model"):
                        dead.add(ev["model"])
                        hurt = providers.provider_of(ev["model"])
                        if hurt in bound:
                            bound[hurt] = rebind(hurt)
                        thinker = agents.pick_planner(live - dead, planner_provider) or thinker
                        swapped = {f"{p}/{r}": m for p, table in bound.items()
                                   for r, m in table.items() if m and m not in dead}
                        yield _event("rebind", dropped=ev["model"], provider=hurt,
                                     bindings=swapped)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
    finally:
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    used: dict[str, int] = {}
    for a in spawned:
        if a.get("provider"):
            used[a["provider"]] = used.get(a["provider"], 0) + 1
    yield _event("complete", agents=len(spawned), waves=waves_run, ok=ok, failed=failed,
                 by_provider=used, cost=ledger.report())
