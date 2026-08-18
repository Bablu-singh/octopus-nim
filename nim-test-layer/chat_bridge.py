"""Turning chat messages into dispatches, and dispatch events back into chat.

Transport-agnostic on purpose. WhatsApp and Discord differ in three ways that matter —
how long a message may be, how bold and italic are spelled, and how a message is actually
sent — and in no other way worth duplicating a few hundred lines over. So a transport is
those three things plus a `send`, and everything else here is shared.

`octopus.py` knows agents and nothing about chat. The transports know their API and
nothing about agents. This is the only module that knows both, which is what keeps adding
a third front door a small job.

The shape of the problem is that a dispatch takes minutes and streams, while chat is a
series of discrete messages. So a run reports as it goes — the plan when it is made, each
agent's answer as it lands, a summary at the end — rather than going quiet and arriving as
one wall of text. It also means a failure is visible when it happens, not four minutes
later.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Awaitable, Callable

import octopus
import providers
import routing
import web


@dataclass
class Transport:
    """One way of talking to a person.

    `send` takes already-split text and is responsible only for delivery. `limit` is the
    platform's hard cap on one message: Discord refuses anything over 2000 characters,
    WhatsApp over 4096, and both refuse silently enough to be confusing.
    """

    name: str
    limit: int
    send: Callable[[str], Awaitable[bool]]
    bold: str = "*"          # WhatsApp uses *one*; Discord uses **two**
    italic: str = "_"

    def b(self, text: str) -> str:
        return f"{self.bold}{text}{self.bold}"

    def i(self, text: str) -> str:
        return f"{self.italic}{text}{self.italic}"


def split(text: str, limit: int) -> list[str]:
    """Break a long answer into sendable pieces, on a boundary a reader would choose.

    Paragraph first, then line, then a hard cut. Agent answers are usually Markdown, and
    slicing mid-sentence every N characters makes them unreadable on a phone.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    parts, buf = [], ""
    for para in text.split("\n\n"):
        if len(para) > limit:
            # One paragraph over the limit: fall back to lines, then to a hard cut.
            for line in para.split("\n"):
                while len(line) > limit:
                    parts.append(line[:limit])
                    line = line[limit:]
                if len(buf) + len(line) + 1 > limit:
                    parts.append(buf.strip())
                    buf = ""
                buf += line + "\n"
            continue
        if len(buf) + len(para) + 2 > limit:
            parts.append(buf.strip())
            buf = ""
        buf += para + "\n\n"
    if buf.strip():
        parts.append(buf.strip())

    parts = [p for p in parts if p]
    if len(parts) > 1:
        total = len(parts)
        # The counter costs characters, so it is budgeted for by the caller's limit.
        parts = [f"({i}/{total}) {p}" for i, p in enumerate(parts, 1)]
    return parts


# One run per conversation. A second task while one is going would interleave two sets of
# results with no way to tell them apart.
_runs: dict[str, asyncio.Task] = {}
# Per-conversation routing pin, set with /mode. Not global: the browser, a phone and a
# Discord channel should be able to disagree about where work goes.
_mode: dict[str, str] = {}
_last_cost: dict[str, dict] = {}

BUSY = "⏳ A run is already going. Send /stop to cancel it, or wait for it to finish."


def help_text(t: Transport) -> str:
    return f"""🐙 {t.b('Octopus')}

Send any task and the agent pool runs it on the host machine:
{t.i('draft a release note for v2 and list the migration risks')}

{t.b('Commands')} (either / or ! works)
/help — this message
/status — providers, routing mode, web access
/models — what each role is bound to, per provider
/cost — what the last run spent
/mode auto|local|nvidia|gemini — pin where work goes
/web <query> — search the web, no model involved
/stop — cancel the run in progress

Anything not starting with / is treated as a task."""


def _key() -> str | None:
    nvidia = providers.get("nvidia")
    return nvidia.key() if nvidia.has_key() else None


async def _status(t: Transport) -> str:
    survey = await providers.survey(_key())
    lines = [f"🐙 {t.b('Octopus status')}", ""]
    for p in survey:
        mark = {"up": "🟢", "cooling down": "🟡", "disabled": "⚫"}.get(p["state"], "🔴")
        lines.append(f"{mark} {t.b(p['label'])} — {p['state']}")
    lines += ["", f"Routing: {t.b(routing.MODE)} (threshold {routing.THRESHOLD})",
              f"Web access: {t.b('on' if web.ENABLED else 'off')}"]
    return "\n".join(lines)


async def _models(t: Transport) -> str:
    cat = await octopus.catalog(_key())
    headline = f"{cat['verified']} of {cat['total']} models answered"
    lines = [f"🐙 {t.b(headline)}", ""]
    for name, rows in (cat.get("bindings") or {}).items():
        lines.append(t.b(name))
        for row in rows:
            model = providers.bare(row["model"]) if row["model"] else "—"
            lines.append(f"  {row['name']}: {model}")
        lines.append("")
    return "\n".join(lines).strip()


def _cost(convo: str, t: Transport) -> str:
    c = _last_cost.get(convo)
    if not c:
        return "No run recorded here yet. Send a task first."
    lines = [f"🐙 {t.b('Last run')}", ""]
    for name, row in (c.get("by_provider") or {}).items():
        lines.append(f"{t.b(name)} — {row['calls']} calls, "
                     f"{row['tokens_in'] + row['tokens_out']} tokens")
    tail = f"\n~{c.get('tokens', 0)} tokens estimated"
    tail += f"\nCost: ${c.get('usd', 0):.4f}" if c.get("billable") else "\nCost: free"
    return "\n".join(lines) + tail


async def _web(query: str, t: Transport) -> str:
    if not query:
        return "Give me something to search: /web ollama structured outputs"
    if not web.ENABLED:
        return "Web access is off (ENABLE_WEB=0)."
    try:
        bundle = await web.research(query, 5, 0)
    except web.WebError as err:
        return f"Search failed: {err}"
    if not bundle["results"]:
        return f"Nothing came back for {t.i(query)}."
    lines = [f"🔎 {t.b(bundle['query'])}", ""]
    for r in bundle["results"][:5]:
        lines.append(f"• {t.b(r['title'][:70])}\n{r['url']}")
    return "\n".join(lines)


def _set_mode(convo: str, arg: str, t: Transport) -> str:
    arg = (arg or "").strip().lower()
    valid = {"auto"} | set(providers.BY_NAME)
    if arg not in valid:
        return (f"Pick one of: {', '.join(sorted(valid))}. "
                f"Currently {t.b(_mode.get(convo, 'auto'))}.")
    _mode[convo] = arg
    return (f"Routing pinned to {t.b(arg)} here."
            if arg != "auto" else f"Routing back to {t.b('auto')} — weighed per subtask.")


async def _run_task(convo: str, task: str, t: Transport) -> None:
    """Stream one dispatch into a conversation."""
    outputs: dict[str, str] = {}
    labels: dict[str, str] = {}
    sent_any = False
    try:
        mode = _mode.get(convo)
        async for line in octopus.dispatch(_key(), task, None,
                                           mode if mode and mode != "auto" else None):
            if not line.startswith("data:"):
                continue
            try:
                e = json.loads(line[5:])
            except ValueError:
                continue
            kind = e.get("event")

            if kind == "wave":
                head = [f"{t.b('Wave ' + str(e['n']))} — "
                        f"{len(e['agents'])} agent(s) in parallel"]
                for a in e["agents"]:
                    model = providers.bare(a["model"]) if a["model"] else "—"
                    labels[a["id"]] = a["label"]
                    head.append(f"• {t.b(a['label'])} — {a['provider']} / {model}")
                await t.send("\n".join(head))

            elif kind == "chunk":
                outputs[e["id"]] = outputs.get(e["id"], "") + e["text"]

            elif kind == "sources":
                srcs = e.get("sources") or []
                if srcs:
                    await t.send(f"🔎 {t.b('Sources read')}\n"
                                 + "\n".join(s["url"] for s in srcs[:5]))

            elif kind == "done":
                text = outputs.get(e["id"], "").strip()
                if text:
                    sent_any = True
                    await t.send(f"✅ {t.b(labels.get(e['id'], e['id']))}\n\n{text}")

            elif kind == "error":
                title = labels.get(e.get("id", ""), e.get("id", "an agent"))
                await t.send(f"⚠️ {t.b(title)} failed\n{e.get('detail', '')[:400]}")

            elif kind == "fatal":
                await t.send(f"🛑 {e.get('detail', 'dispatch failed')}")

            elif kind == "complete":
                _last_cost[convo] = e.get("cost", {})
                split_by = ", ".join(f"{v} {k}" for k, v in (e.get("by_provider") or {}).items())
                cost = e.get("cost") or {}
                money = "$%.4f" % cost["usd"] if cost.get("billable") else "free"
                await t.send(
                    f"🐙 {t.b('Done')} — {e['agents']} agent(s), {e['waves']} wave(s), "
                    f"{e.get('ok', 0)} ok, {e.get('failed', 0)} failed"
                    + (f"\n{split_by}" if split_by else "")
                    + f"\n~{cost.get('tokens', 0)} tokens · {money}")
                if not sent_any:
                    await t.send("No agent produced any text.")
    except asyncio.CancelledError:
        await t.send("🛑 Stopped.")
        raise
    except Exception as err:                          # never leave the chat hanging
        await t.send(f"🛑 The run broke: {type(err).__name__}: {err}")
    finally:
        _runs.pop(convo, None)


# Discord's client swallows anything beginning with '/' and tries to match it against
# registered application commands — so a '/help' typed there never arrives as a message,
# and the user sees "the application did not respond". '!' is accepted for exactly that
# reason, and both are treated identically everywhere else.
PREFIXES = ("/", "!")


def is_command(text: str) -> bool:
    return text.strip().startswith(PREFIXES)


def command_of(text: str) -> tuple[str, str]:
    """'/mode local' or '!mode local' -> ('mode', 'local')"""
    head, _, rest = text.strip()[1:].strip().partition(" ")
    return head.lower(), rest.strip()


async def handle(convo: str, text: str, t: Transport) -> None:
    """One inbound message, already authenticated by its transport."""
    text = text.strip()
    if not text:
        return

    if is_command(text):
        cmd, arg = command_of(text)
        if cmd in ("help", "start", "?"):
            await t.send(help_text(t))
        elif cmd == "status":
            await t.send(await _status(t))
        elif cmd == "models":
            await t.send(await _models(t))
        elif cmd == "cost":
            await t.send(_cost(convo, t))
        elif cmd == "mode":
            await t.send(_set_mode(convo, arg, t))
        elif cmd == "web":
            await t.send(await _web(arg, t))
        elif cmd == "stop":
            task = _runs.get(convo)
            if task and not task.done():
                task.cancel()
            else:
                await t.send("Nothing running.")
        else:
            await t.send(f"Unknown command /{cmd}.\n\n{help_text(t)}")
        return

    if convo in _runs and not _runs[convo].done():
        await t.send(BUSY)
        return

    pin = _mode.get(convo, "auto")
    await t.send(f"🐙 Working on it — routing {t.b(pin)}.\n"
                 "I'll send each agent's answer as it lands.")
    _runs[convo] = asyncio.create_task(_run_task(convo, task=text, t=t))
