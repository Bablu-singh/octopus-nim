"""Octopus orchestrator — routes one task across several NIM models at once.

Mounted onto the existing test layer by app.py. Everything streams over a single SSE
connection so the UI can light tentacles up as they start, finish, or fail independently.
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
import nim_client as nim

IMAGE_BASE = os.getenv("NVIDIA_IMAGE_BASE", "https://ai.api.nvidia.com/v1/genai")
CATALOG_TTL = 900          # seconds; verification is expensive, so cache it for a while
PROBE_CONCURRENCY = 16
STALL_TIMEOUT = 120        # seconds of silence from every agent before giving up

# The agent pool is sized by the task, not by the number of roles. A role is a template
# that can be instantiated as often as the work divides; these are the only ceilings.
MAX_AGENTS = int(os.getenv("OCTOPUS_MAX_AGENTS", "24"))      # total, across all waves
FIRST_WAVE = int(os.getenv("OCTOPUS_FIRST_WAVE", "8"))       # cap on the opening plan
MAX_WAVES = int(os.getenv("OCTOPUS_MAX_WAVES", "4"))         # planner + 3 growth rounds
MAX_PARALLEL = int(os.getenv("OCTOPUS_MAX_PARALLEL", "6"))   # concurrent upstream streams
DIGEST_CHARS = 700         # per agent, when summarising a wave for the supervisor

# Keyed by fingerprint: two different keys have different entitlements, and sharing one
# cache between them would bind tentacles to models the current key cannot reach.
_catalog: dict[str, dict] = {}


async def _verify(key: str, ids: list[str]) -> set[str]:
    """Probe every model any tentacle might want, and return the ones that answer.

    The catalog is aspirational — on a typical key most listed models 404 and several
    accept a connection then never respond. Probing here, once, is what stops a dispatch
    from stalling on a dead model.
    """
    wanted: list[str] = []
    for t in agents.TENTACLES:
        wanted += [m for m, _ in agents.shortlist(t, ids)]
    wanted = list(dict.fromkeys(wanted))  # dedupe, preserve order

    sem = asyncio.Semaphore(PROBE_CONCURRENCY)

    async def check(client: httpx.AsyncClient, mid: str) -> tuple[str, bool]:
        async with sem:
            return mid, await nim.is_alive(key, mid, client)

    async with httpx.AsyncClient(timeout=nim.PROBE) as client:
        results = await asyncio.gather(*(check(client, m) for m in wanted))
    return {mid for mid, ok in results if ok}


async def catalog(key: str, force: bool = False) -> dict:
    """Live model list, cached so the UI can poll without re-probing the whole catalog."""
    slot = _catalog.get(nim.key_fingerprint(key))
    if slot and not force and time.time() - slot["fetched_at"] < CATALOG_TTL:
        ids, live, latency = slot["ids"], slot["live"], None
    else:
        result = await nim.list_models(key)
        ids = sorted(m.get("id", "") for m in result.data.get("data", []) if m.get("id"))
        live = await _verify(key, ids)
        _catalog[nim.key_fingerprint(key)] = {
            "ids": ids, "live": live, "fetched_at": time.time()
        }

        latency = result.latency_ms

    grouped = agents.classify(ids)
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
        "tentacles": agents.bind(ids, live=live),
    }


def _mark_dead(key: str, model: str | None) -> None:
    """Forget a model that failed for real, and force a re-probe on the next catalog read.

    Entitlements and cold-start behaviour drift, so a binding that worked when the cache
    was filled can break inside the TTL. Without this the same dead model would be handed
    out for the rest of the window.
    """
    slot = _catalog.get(nim.key_fingerprint(key))
    if slot and model and model in slot["live"]:
        slot["live"].discard(model)
        slot["fetched_at"] = 0.0


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


async def plan(key: str, task: str, model: str | None, budget: int) -> tuple[list[dict], str]:
    """Decompose the task into a first wave of agents. Falls back to keywords on failure."""
    if not model:
        return agents.keyword_plan(task, budget), "keyword planning (no planner model available)"
    try:
        result = await nim.chat(
            key, f"Task:\n{task}", model,
            system=agents.planner_system(budget), temperature=0.0, max_tokens=1500,
        )
        picks, why = _parse_agents(nim.extract_text(result.data), budget, task)
        if picks:
            return picks, why or "planned by model"
        why = "planner returned no usable agents"
    except nim.NimError as err:
        why = f"planner unavailable (HTTP {err.status})"
    except (ValueError, KeyError, TypeError):
        why = "planner did not return usable JSON"
    return agents.keyword_plan(task, budget), f"keyword planning ({why})"


async def supervise(key: str, task: str, model: str | None, digest: str,
                    budget: int) -> tuple[list[dict], str]:
    """Having seen the wave that just finished, decide whether the task needs more agents.

    This is what lets the pool grow with the work instead of being fixed at dispatch time.
    An empty list is the normal answer and ends the run.
    """
    if not model or budget <= 0:
        return [], ""
    try:
        result = await nim.chat(
            key, f"Task:\n{task}\n\nProduced so far:\n{digest}", model,
            system=agents.supervisor_system(budget), temperature=0.0, max_tokens=1200,
        )
        return _parse_agents(nim.extract_text(result.data), budget, task)
    except (nim.NimError, ValueError, KeyError, TypeError):
        # A supervisor that cannot answer means "no more agents", never a failed dispatch.
        return [], ""


async def _generate_image(key: str, model: str, prompt: str) -> dict:
    """NIM image models use the genai endpoint, not the OpenAI-compatible one."""
    url = f"{IMAGE_BASE}/{model}"
    body = {"prompt": prompt, "cfg_scale": 5, "mode": "base", "steps": 30, "seed": 0}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=120.0,
                                                           write=30.0, pool=10.0)) as client:
            resp = await client.post(
                url, json=body,
                headers={"Authorization": f"Bearer {key}", "Accept": "application/json"},
            )
            if resp.status_code >= 400:
                raise nim.NimError(resp.status_code, resp.text[:400])
            data = resp.json()
    except httpx.HTTPError as err:
        raise nim.transport_error(err, model) from err
    # Response shape varies by model family; check the documented spots in turn.
    b64 = (data.get("image")
           or (data.get("artifacts") or [{}])[0].get("base64")
           or (data.get("data") or [{}])[0].get("b64_json"))
    if not b64:
        raise nim.NimError(502, f"No image field in response. Keys: {list(data)[:6]}")
    return {"image": f"data:image/png;base64,{b64}"}


async def _run_agent(key: str, agent: dict, queue: asyncio.Queue, sem: asyncio.Semaphore,
                     outputs: dict[str, str]) -> None:
    """Run one agent to completion, streaming as it goes.

    The model is passed in on the agent dict rather than read off the shared role template,
    so any number of agents — including several sharing a role — can run at once without
    treading on each other. `sem` bounds how many actually talk upstream at the same time;
    everything else here is fully concurrent.
    """
    aid, role, model = agent["id"], agent["role"], agent["model"]
    t = agents.BY_ID[role]
    outputs[aid] = ""

    await queue.put(_event("queued", id=aid, role=role, label=agent["label"],
                           model=model, name=t.name, color=t.color))

    if not model:
        await queue.put(_event("error", id=aid,
                               detail="No model this key can actually reach fits this role."))
        return

    async with sem:
        started = time.perf_counter()
        ms = lambda: int((time.perf_counter() - started) * 1000)
        await queue.put(_event("start", id=aid, role=role, label=agent["label"],
                               model=model, name=t.name, color=t.color))
        try:
            if t.kind == "image" and agents.family_of(model) == "image":
                out = await _generate_image(key, model, agent["subtask"])
                await queue.put(_event("image", id=aid, **out, latency_ms=ms()))
                await queue.put(_event("done", id=aid, model=model, latency_ms=ms()))
                return

            chunks = 0
            async for line in nim.chat_stream(key, agent["subtask"], model, system=t.system,
                                              temperature=t.temperature,
                                              max_tokens=t.max_tokens):
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

            if chunks == 0:
                _mark_dead(key, model)
                await queue.put(_event("error", id=aid, model=model, model_at_fault=True,
                                       detail=f"'{model}' streamed no content — listed, but "
                                              f"not serving this request."))
                return
            await queue.put(_event("done", id=aid, model=model, latency_ms=ms()))
        except asyncio.CancelledError:
            raise
        except nim.NimError as err:
            # These say the binding is bad rather than the prompt, so the caller should
            # swap models rather than retry the same one. 503 is capacity, not breakage.
            at_fault = err.status in (404, 429, 500, 502, 503, 504)
            if at_fault:
                _mark_dead(key, model)
            await queue.put(_event("error", id=aid, model=model, model_at_fault=at_fault,
                                   detail=f"HTTP {err.status}: {err.detail[:300]}"))
        except Exception as err:  # keep one failed agent from killing the others
            await queue.put(_event("error", id=aid, model=model, model_at_fault=False,
                                   detail=f"{type(err).__name__}: {err}"))


async def dispatch(key: str, task: str, roles: list[str] | None = None) -> AsyncIterator[str]:
    """Grow a pool of agents around one task and stream all of them at once.

    The task decides how many agents run — a one-line request gets one, a broad one gets a
    dozen. Wave 1 comes from the planner; after each wave the supervisor sees what came
    back and adds more if the work is not covered, until it says stop or a budget runs out.
    Everything inside a wave runs concurrently, `MAX_PARALLEL` of them talking upstream at
    any moment.

    `roles` is a debugging escape hatch that seeds wave 1 by hand. The UI never sends it:
    identifying the kind of work is the planner's job, not the user's.
    """
    try:
        cat = await catalog(key)
    except nim.NimError as err:
        yield _event("fatal", detail=f"Could not read the model catalog — "
                                     f"HTTP {err.status}: {err.detail[:200]}")
        yield _event("complete", agents=0, waves=0)
        return

    ids, live = cat["ids"], set(cat["live"])
    dead: set[str] = set()

    def rebind() -> dict[str, str | None]:
        return {t["id"]: t["model"] for t in agents.bind(ids, live=live - dead)}

    bound = rebind()
    if not any(bound.values()):
        yield _event("fatal", detail=(
            f"This key lists {cat['total']} models but none of them answered a test request. "
            "Check entitlements at build.nvidia.com/settings/api-keys."))
        yield _event("complete", agents=0, waves=0)
        return

    thinker = agents.pick_planner(live - dead)

    queue: asyncio.Queue = asyncio.Queue()
    sem = asyncio.Semaphore(MAX_PARALLEL)
    outputs: dict[str, str] = {}
    workers: list[asyncio.Task] = []
    spawned: list[dict] = []
    counts: dict[str, int] = {}
    attempts: dict[str, int] = {}
    slices: dict[str, str] = {}                  # deliverable -> running|ok|failed
    owner: dict[str, str] = {}                   # agent id -> deliverable
    errors: dict[str, str] = {}
    ok = failed = waves_run = 0

    def enlist(batch: list[dict]) -> list[dict]:
        """Give each planned agent a unique id and a model, and start it.

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
            sid = _slug(a["label"])
            if slices.get(sid) in ("ok", "running") or attempts.get(sid, 0) >= 2:
                continue
            attempts[sid] = attempts.get(sid, 0) + 1
            slices[sid] = "running"
            counts[a["role"]] = counts.get(a["role"], 0) + 1
            agent = {**a, "id": f"{a['role']}-{counts[a['role']]}",
                     "model": bound.get(a["role"])}
            owner[agent["id"]] = sid
            spawned.append(agent)
            made.append(agent)
            workers.append(asyncio.create_task(_run_agent(key, agent, queue, sem, outputs)))
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
                else:
                    batch, why = await plan(key, task, thinker, min(FIRST_WAVE, remaining))
            else:
                # A task the planner sized at a single agent is by definition not
                # multi-part, so if that agent succeeded there is nothing to grow into.
                # Without this the supervisor invents busywork for one-line questions.
                if len(spawned) == 1 and not errors:
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
                    if ev.get("model_at_fault") and ev.get("model"):
                        dead.add(ev["model"])
                        bound = rebind()
                        thinker = agents.pick_planner(live - dead) or thinker
                        swapped = {r: m for r, m in bound.items() if m and m not in dead}
                        yield _event("rebind", dropped=ev["model"], bindings=swapped)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        pass
    finally:
        for w in workers:
            w.cancel()
        await asyncio.gather(*workers, return_exceptions=True)

    yield _event("complete", agents=len(spawned), waves=waves_run, ok=ok, failed=failed)
