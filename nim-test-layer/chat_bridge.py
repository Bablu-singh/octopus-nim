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
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Awaitable, Callable

import octopus
import providers
import routing
import voice
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
    # Optional richer surfaces. A platform that has them gets structure; one that does
    # not falls back to plain text, so nothing here has to know which is which.
    #   card   — one titled, coloured block (a Discord embed)
    #   status — a single line that is *edited* in place as the run progresses, rather
    #            than a new message per wave
    card: Callable[..., Awaitable[bool]] | None = None
    status: Callable[[str], Awaitable[bool]] | None = None
    # Embeds hold far more than a plain message, so the split budget differs by surface.
    card_limit: int = 0
    # Returns a fresh Transport for a concurrent run. Discord's status line is one message
    # that gets edited, so two runs sharing a transport would overwrite each other's
    # progress; each run takes its own. Platforms without an editable status can hand back
    # the same object, since there is nothing to collide over.
    fork: Callable[[], "Transport"] | None = None
    # Sends an audio clip. Present only where the platform can play one back; without it
    # `voice_note` quietly does nothing and the text answer stands on its own.
    audio: Callable[[bytes, str], Awaitable[bool]] | None = None

    def branch(self) -> "Transport":
        return self.fork() if self.fork else self

    async def voice_note(self, text: str) -> bool:
        """Speak an answer, if this platform can carry audio at all."""
        if not self.audio:
            return False
        clip = await voice.speak(text)
        if not clip:
            return False
        return await self.audio(clip, text[:60])

    def b(self, text: str) -> str:
        return f"{self.bold}{text}{self.bold}"

    def i(self, text: str) -> str:
        return f"{self.italic}{text}{self.italic}"

    async def show(self, title: str, body: str, colour: str = "", footer: str = "") -> bool:
        """One agent's answer, as structured as this platform allows."""
        if self.card:
            return await self.card(title, body, colour, footer)
        head = f"✅ {self.b(title)}"
        if footer:
            head += f"\n{self.i(footer)}"
        return await self.send(f"{head}\n\n{body}")

    async def progress(self, text: str) -> bool:
        """Run progress. Edited in place where possible, appended where not."""
        if self.status:
            return await self.status(text)
        return await self.send(text)


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

    # Re-balance code fences. A part that ends inside a ``` block renders as broken
    # Markdown and swallows everything after it, so the block is closed at the cut and
    # reopened — with its language — at the top of the next part. Coder agents produce
    # exactly this shape often enough to be worth the bookkeeping.
    fixed, carry = [], ""
    for part in parts:
        body = carry + part
        fences = re.findall(r"^```([A-Za-z0-9_+-]*)", body, re.M)
        if len(fences) % 2:
            lang = fences[-1]
            body += "\n```"
            carry = f"```{lang}\n"
        else:
            carry = ""
        fixed.append(body)
    return fixed


# --- sessions ----------------------------------------------------------------
# A conversation is not a series of unrelated requests. Two things follow from that, and
# both used to be missing:
#
#   Queueing. A dispatch takes minutes, and refusing new work for that whole time turns
#   the chat into a stop-and-wait terminal. Tasks are now accepted whenever they arrive
#   and run one after another — sequentially rather than concurrently, because two pools
#   at once would interleave their answers in one chat and fight over the same rate
#   limits.
#
#   Memory. "now make it shorter" means nothing without the previous turn. Each finished
#   task leaves a short digest behind, and later tasks are dispatched with those digests
#   prepended, so a follow-up can refer to what came before.
#
# Both live until the session is cleared with /new — deliberately, since the user decides
# when a train of thought is over, not the app.

MAX_QUEUE = int(os.getenv("CHAT_MAX_QUEUE", "12"))
# Tasks that may run at once in one conversation. More than one because waiting for a
# long run before starting an unrelated task is the complaint this exists to answer; not
# unbounded because every agent in every run competes for the same provider rate limits,
# and past a point more concurrency just buys more 429s.
MAX_CONCURRENT = int(os.getenv("CHAT_MAX_CONCURRENT", "3"))
# How many past turns a new task is told about. Enough to follow a thread, few enough
# that the planner is not reading an essay before it starts.
HISTORY_TURNS = int(os.getenv("CHAT_HISTORY_TURNS", "6"))
# Per-turn budget in the context block. A whole answer would swamp the new request.
DIGEST_CHARS = int(os.getenv("CHAT_DIGEST_CHARS", "500"))


@dataclass
class Turn:
    """One finished task and a compressed trace of what it produced."""

    task: str
    digest: str
    at: float = field(default_factory=time.time)


@dataclass
class Session:
    """Everything remembered about one conversation."""

    convo: str
    queue: deque = field(default_factory=deque)
    turns: list[Turn] = field(default_factory=list)
    worker: asyncio.Task | None = None
    # Tasks in flight right now, mapped to their text. Several, not one: a conversation
    # can have more than one tangle going.
    live: dict = field(default_factory=dict)
    mode: str = "auto"
    # 'auto' speaks only when the task itself arrived as speech, which is the rule people
    # expect from a conversation: answer in the medium you were addressed in. 'on' always
    # speaks, 'off' never does.
    voice_mode: str = "auto"
    spoke_last: bool = False        # was the task in flight dictated?
    last_cost: dict = field(default_factory=dict)
    started: float = field(default_factory=time.time)

    @property
    def busy(self) -> bool:
        return bool(self.live) or (self.worker is not None and not self.worker.done())

    @property
    def current(self) -> str:
        return "; ".join(list(self.live.values())[:2])


_sessions: dict[str, Session] = {}


def session(convo: str) -> Session:
    if convo not in _sessions:
        _sessions[convo] = Session(convo=convo)
    return _sessions[convo]


def active_runs() -> int:
    """How many conversations are mid-dispatch. For the health endpoints."""
    return sum(1 for s in _sessions.values() if s.busy)


def context_for(convo: str, task: str) -> str:
    """The context-prefixed task for a conversation. Public so the browser shares it.

    The web UI streams raw dispatch events rather than chat messages, so it cannot use
    the Transport path — but there is no reason for it to have a second, separate memory.
    One session store, three front doors.
    """
    return _context(session(convo), task)


def remember(convo: str, task: str, digest: str) -> None:
    """Record a finished turn from a caller that ran the dispatch itself."""
    sess = session(convo)
    sess.turns.append(Turn(task=task, digest=digest or "(no output)"))


def forget(convo: str) -> dict:
    """End a session. Returns what was dropped, for reporting."""
    sess = session(convo)
    dropped = {"turns": len(sess.turns), "queued": len(sess.queue)}
    sess.queue.clear()
    sess.turns.clear()
    sess.last_cost = {}
    sess.started = time.time()
    return dropped


def summary(convo: str) -> dict:
    sess = session(convo)
    return {
        "convo": convo,
        "turns": [{"task": t.task, "digest": t.digest, "at": t.at} for t in sess.turns],
        "queued": list(sess.queue),
        "current": sess.current,
        "busy": sess.busy,
        "mode": sess.mode,
        "open_seconds": int(time.time() - sess.started),
    }


def _context(sess: Session, task: str) -> str:
    """The task as the planner should see it: what came before, then what to do now.

    Marked as context and explicitly not as work, because a planner handed a paragraph of
    previous results will otherwise cheerfully plan agents to redo them.
    """
    if not sess.turns:
        return task
    lines = ["Context from earlier in this conversation. It is background only — it has "
             "already been produced and must NOT be redone:"]
    for i, turn in enumerate(sess.turns[-HISTORY_TURNS:], 1):
        lines.append(f"{i}. Asked: {turn.task[:200]}")
        lines.append(f"   Produced: {turn.digest[:DIGEST_CHARS]}")
    lines.append("")
    lines.append("THE ACTUAL TASK, which may refer back to the above:")
    lines.append(task)
    return "\n".join(lines)


def help_text(t: Transport) -> str:
    return f"""🐙 {t.b('Octopus')}

Send any task and the agent pool runs it on the host machine:
{t.i('draft a release note for v2 and list the migration risks')}

Send as many as you like — they queue up and run one after another, and each
one remembers what came before, so {t.i('now make that shorter')} works.

{t.b('Commands')} (either / or ! works)
/help — this message
/status — providers, routing mode, web access
/models — what each role is bound to, per provider
/cost — what the last run spent
/mode auto|local|nvidia|gemini — pin where work goes
/web <query> — search the web, no model involved
/voice on|off|auto — speak answers aloud
/queue — what is running and what is waiting
/history — what this session remembers
/stop — cancel the run in progress
/new — end the session: forget the thread, drop the queue

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


def _cost(sess: Session, t: Transport) -> str:
    c = sess.last_cost
    if not c:
        return "No run recorded here yet. Send a task first."
    lines = [f"🐙 {t.b('Last run')}", ""]
    for name, row in (c.get("by_provider") or {}).items():
        lines.append(f"{t.b(name)} — {row['calls']} calls, "
                     f"{row['tokens_in'] + row['tokens_out']} tokens")
    tail = f"\n~{c.get('tokens', 0)} tokens estimated"
    tail += f"\nCost: ${c.get('usd', 0):.4f}" if c.get("billable") else "\nCost: free"
    return "\n".join(lines) + tail


def _queue_view(sess: Session, t: Transport) -> str:
    lines = [f"🐙 {t.b('Session')}"]
    mins = (time.time() - sess.started) / 60
    lines.append(f"open {mins:.0f} min · {len(sess.turns)} task(s) remembered")
    lines.append("")
    for task in list(sess.live.values()):
        lines.append(f"▶ {t.b('running')}: {task[:80]}")
    if sess.queue:
        for i, (q, _) in enumerate(sess.queue, 1):
            lines.append(f"{i}. {q[:90]}")
    elif not sess.live:
        lines.append("Nothing running, nothing queued.")
    return "\n".join(lines)


def _history_view(sess: Session, t: Transport) -> str:
    if not sess.turns:
        return "Nothing remembered yet in this session."
    lines = [f"🐙 {t.b('What I remember')} ({len(sess.turns)} task(s))", ""]
    for i, turn in enumerate(sess.turns[-HISTORY_TURNS:], 1):
        lines.append(f"{i}. {t.b(turn.task[:90])}")
        lines.append(f"   {turn.digest[:200]}")
    return "\n".join(lines)


def _set_mode(sess: Session, arg: str, t: Transport) -> str:
    arg = (arg or "").strip().lower()
    valid = {"auto"} | set(providers.BY_NAME)
    if arg not in valid:
        return (f"Pick one of: {', '.join(sorted(valid))}. "
                f"Currently {t.b(sess.mode)}.")
    sess.mode = arg
    return (f"Routing pinned to {t.b(arg)} here."
            if arg != "auto" else f"Routing back to {t.b('auto')} — weighed per subtask.")


async def _run_task(sess: Session, task: str, t: Transport, spoken: bool = False) -> None:
    """Run one task to completion and remember what it produced.

    Progress is one line that gets rewritten — wave, who is working, then the summary —
    rather than a new message per event. On a phone a dozen status messages bury the
    thing you actually wanted, which is the answers.
    """
    outputs: dict[str, str] = {}
    agents_seen: dict[str, dict] = {}
    sent_any = False
    waves = 0

    def board(note: str = "") -> str:
        """The live picture of the run, rebuilt from scratch each time."""
        icon = {"running": "⏳", "done": "✅", "failed": "⚠️"}
        rows = [f"{icon.get(a.get('state', 'running'), '•')} {t.b(a['label'])} — "
                f"{a['provider']} / {a['model']}"
                for a in agents_seen.values()]
        head = f"🐙 {t.b(task[:120])}"
        mid = f"Wave {waves} · {len(agents_seen)} agent(s)" if waves else "Planning…"
        if sess.queue:
            mid += f" · {len(sess.queue)} queued after this"
        return "\n".join([head, mid] + rows + ([note] if note else []))

    try:
        mode = sess.mode if sess.mode != "auto" else None
        async for line in octopus.dispatch(_key(), _context(sess, task), None, mode,
                                           weigh_as=task):
            if not line.startswith("data:"):
                continue
            try:
                e = json.loads(line[5:])
            except ValueError:
                continue
            kind = e.get("event")

            if kind == "status":
                await t.progress(f"🐙 {t.i(e.get('detail', ''))}")

            elif kind == "wave":
                waves = e["n"]
                for a in e["agents"]:
                    agents_seen[a["id"]] = {
                        "label": a["label"], "provider": a["provider"],
                        "model": providers.bare(a["model"]) if a["model"] else "—",
                        "colour": a.get("color", ""), "state": "running",
                    }
                await t.progress(board())

            elif kind == "chunk":
                outputs[e["id"]] = outputs.get(e["id"], "") + e["text"]

            elif kind == "sources":
                srcs = e.get("sources") or []
                if srcs and e["id"] in agents_seen:
                    agents_seen[e["id"]]["sources"] = [s["url"] for s in srcs[:4]]

            elif kind == "done":
                info = agents_seen.get(e["id"], {})
                info["state"] = "done"
                text = outputs.get(e["id"], "").strip()
                await t.progress(board())
                if text:
                    sent_any = True
                    secs = (e.get("latency_ms") or 0) / 1000
                    foot = f"{info.get('provider', '')} · {info.get('model', '')} · {secs:.0f}s"
                    body = text
                    if info.get("sources"):
                        body += "\n\n" + t.b("Sources") + "\n" + "\n".join(info["sources"])
                    await t.show(info.get("label", e["id"]), body,
                                 info.get("colour", ""), foot)
                    # Spoken after the text, never instead of it. Audio cannot be
                    # skimmed, searched or copied, so it is an addition to the answer
                    # rather than the answer itself.
                    if sess.voice_mode == "on" or (sess.voice_mode == "auto" and spoken):
                        await t.voice_note(text)

            elif kind == "error":
                info = agents_seen.get(e.get("id", ""), {})
                info["state"] = "failed"
                await t.progress(board(f"⚠️ {e.get('detail', '')[:200]}"))

            elif kind == "fatal":
                await t.progress(f"🛑 {e.get('detail', 'dispatch failed')}")

            elif kind == "complete":
                sess.last_cost = e.get("cost", {})
                by = ", ".join(f"{v} {k}" for k, v in (e.get("by_provider") or {}).items())
                cost = e.get("cost") or {}
                money = "$%.4f" % cost["usd"] if cost.get("billable") else "free"
                summary = (f"🐙 {t.b('Done')} — {e['agents']} agent(s), {e['waves']} wave(s), "
                           f"{e.get('ok', 0)} ok, {e.get('failed', 0)} failed"
                           + (f" · {by}" if by else "")
                           + f" · ~{cost.get('tokens', 0)} tokens · {money}")
                await t.progress(board(summary))
                if not sent_any:
                    await t.send("No agent produced any text.")

        # Remember it, whatever happened. A task that failed is still context: without it
        # a follow-up like "try that again" has nothing to refer to.
        digest = " | ".join(
            f"{a['label']}: {outputs.get(aid, '')[:160]}"
            for aid, a in agents_seen.items() if outputs.get(aid))
        sess.turns.append(Turn(task=task, digest=digest or "(no output)"))

    except asyncio.CancelledError:
        sess.turns.append(Turn(task=task, digest="(cancelled)"))
        await t.progress(board("🛑 Stopped."))
        raise
    except Exception as err:
        await t.send(f"🛑 The run broke: {type(err).__name__}: {err}")


async def _drain(sess: Session, t: Transport) -> None:
    """Keep up to MAX_CONCURRENT tasks in flight until the queue is empty.

    Each task gets its own tangle and its own transport branch, so their progress lines
    and answers stay separable in one chat rather than overwriting each other.
    """
    try:
        while sess.queue or sess.live:
            while sess.queue and len(sess.live) < MAX_CONCURRENT:
                task, spoken = sess.queue.popleft()
                run = asyncio.create_task(_run_task(sess, task, t.branch(), spoken))
                sess.live[run] = task
                run.add_done_callback(lambda r: sess.live.pop(r, None))
            if not sess.live:
                break
            # Wake as soon as any one finishes, so a freed slot is refilled immediately
            # rather than after the slowest of the batch.
            await asyncio.wait(set(sess.live), return_when=asyncio.FIRST_COMPLETED)
    finally:
        sess.worker = None


# Discord's client swallows anything beginning with '/' and tries to match it against
# registered application commands, so a '/help' typed there never arrives as a message
# and the user sees "the application did not respond". '!' is accepted for that reason,
# and both are treated identically everywhere else.
PREFIXES = ("/", "!")


def is_command(text: str) -> bool:
    return text.strip().startswith(PREFIXES)


def command_of(text: str) -> tuple[str, str]:
    """'/mode local' or '!mode local' -> ('mode', 'local')"""
    head, _, rest = text.strip()[1:].strip().partition(" ")
    return head.lower(), rest.strip()


async def handle(convo: str, text: str, t: Transport, spoken: bool = False) -> None:
    """One inbound message, already authenticated by its transport.

    `spoken` says the text arrived as a voice note rather than typed, which is what the
    'auto' voice mode keys on.
    """
    text = text.strip()
    if not text:
        return
    sess = session(convo)

    if is_command(text):
        cmd, arg = command_of(text)
        if cmd in ("help", "start", "?"):
            await t.send(help_text(t))
        elif cmd == "status":
            await t.send(await _status(t))
        elif cmd == "models":
            await t.send(await _models(t))
        elif cmd == "cost":
            await t.send(_cost(sess, t))
        elif cmd == "mode":
            await t.send(_set_mode(sess, arg, t))
        elif cmd == "web":
            await t.send(await _web(arg, t))
        elif cmd == "voice":
            want = (arg or "").strip().lower()
            if want not in ("on", "off", "auto"):
                ok, why = voice.available()
                await t.send(
                    f"{t.b('Voice')} is {t.b(sess.voice_mode)}. Use /voice on|off|auto.\n"
                    f"auto = speak only when you send a voice note.\n"
                    f"engine: {'ready' if ok else why}")
            else:
                sess.voice_mode = want
                tail = {"on": " I'll speak every answer.",
                        "auto": " I'll speak only when you do.",
                        "off": " Text only."}[want]
                await t.send(f"Voice {t.b(want)}.{tail}")
        elif cmd in ("queue", "q"):
            await t.send(_queue_view(sess, t))
        elif cmd in ("history", "context"):
            await t.send(_history_view(sess, t))
        elif cmd in ("new", "reset", "clear"):
            # Ends the session: forgets the thread and drops anything not yet started.
            # The run in flight is left alone — it is already being paid for.
            dropped = len(sess.queue)
            remembered = len(sess.turns)
            sess.queue.clear()
            sess.turns.clear()
            sess.last_cost = {}
            sess.started = time.time()
            await t.send(f"🐙 {t.b('New session')} — forgot {remembered} task(s)"
                         + (f", dropped {dropped} queued" if dropped else "")
                         + (f". {len(sess.live)} run(s) in progress will finish."
                            if sess.live else "."))
        elif cmd == "stop":
            if sess.live:
                # Cancels every tangle in flight. /stop has always meant "stop what is
                # happening"; with several running that is all of them.
                n = len(sess.live)
                for run in list(sess.live):
                    run.cancel()
                await t.send(f"Stopping {n} run(s).")
            elif sess.queue:
                n = len(sess.queue)
                sess.queue.clear()
                await t.send(f"Cleared {n} queued task(s). Nothing was running.")
            else:
                await t.send("Nothing running.")
        else:
            await t.send(f"Unknown command /{cmd}.\n\n{help_text(t)}")
        return

    # Not a command, so it is work. It is always accepted — the whole point of a queue is
    # that you can keep thinking while something runs.
    if len(sess.queue) >= MAX_QUEUE:
        await t.send(f"Queue is full ({MAX_QUEUE}). Send /queue to see it, "
                     f"or /stop to clear it.")
        return

    sess.queue.append((text, spoken))

    # Counted after the append, and including anything the drainer has not picked up yet:
    # reading `live` alone reports 0 for tasks submitted faster than the loop starts them.
    pending = len(sess.queue) + len(sess.live)
    if sess.busy:
        if pending > MAX_CONCURRENT:
            await t.send(f"➕ {t.b('Queued')} — {MAX_CONCURRENT} tangles run at once, "
                         f"{pending} in this session.")
        else:
            await t.send(f"🐙 {t.b('Starting')} — {pending} tangles now running in parallel.")
        return

    pin = sess.mode
    await t.progress(f"🐙 Working on it — routing {t.b(pin)}…")
    sess.worker = asyncio.create_task(_drain(sess, t))
