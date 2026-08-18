"""Turning WhatsApp messages into dispatches, and dispatch events back into messages.

`whatsapp.py` knows the Cloud API and nothing about agents; `octopus.py` knows agents and
nothing about phones. This is the seam between them, and the only place that knows both.

The shape of the problem is that a dispatch takes minutes and streams, while WhatsApp is
a series of discrete messages. So a run reports as it goes — the plan when it is made,
each agent's answer as it finishes, a summary at the end — rather than going quiet and
arriving as one wall of text. That also means a failure is visible when it happens.
"""

from __future__ import annotations

import asyncio
import json

import octopus
import providers
import routing
import web
import whatsapp

# One run per sender. A second task while one is going would interleave two sets of
# results in a single chat with no way to tell them apart.
_runs: dict[str, asyncio.Task] = {}
# Per-sender routing pin, set with /mode. Not global: the browser and the phone should be
# able to disagree about where work goes.
_mode: dict[str, str] = {}
_last_cost: dict[str, dict] = {}

BUSY = ("⏳ A run is already going. Send /stop to cancel it, or wait for it to finish.")


def _key(key: str | None = None) -> str | None:
    nvidia = providers.get("nvidia")
    return nvidia.key() if nvidia.has_key() else None


async def _status() -> str:
    survey = await providers.survey(_key())
    lines = ["🐙 *Octopus status*", ""]
    for p in survey:
        mark = {"up": "🟢", "cooling down": "🟡", "disabled": "⚫"}.get(p["state"], "🔴")
        lines.append(f"{mark} *{p['label']}* — {p['state']}")
    lines += ["", f"Routing: *{routing.MODE}* (threshold {routing.THRESHOLD})",
              f"Web access: *{'on' if web.ENABLED else 'off'}*"]
    return "\n".join(lines)


async def _models() -> str:
    cat = await octopus.catalog(_key())
    lines = [f"🐙 *{cat['verified']} of {cat['total']} models answered*", ""]
    for name, rows in (cat.get("bindings") or {}).items():
        lines.append(f"*{name}*")
        for t in rows:
            model = providers.bare(t["model"]) if t["model"] else "—"
            lines.append(f"  {t['name']}: {model}")
        lines.append("")
    return "\n".join(lines).strip()


def _cost(sender: str) -> str:
    c = _last_cost.get(sender)
    if not c:
        return "No run recorded yet in this chat. Send a task first."
    lines = ["🐙 *Last run*", ""]
    for name, row in (c.get("by_provider") or {}).items():
        lines.append(f"*{name}* — {row['calls']} calls, "
                     f"{row['tokens_in'] + row['tokens_out']} tokens")
    total = f"\n~{c.get('tokens', 0)} tokens estimated"
    total += f"\nCost: ${c.get('usd', 0):.4f}" if c.get("billable") else "\nCost: free"
    return "\n".join(lines) + total


async def _web(query: str) -> str:
    if not query:
        return "Give me something to search: /web ollama structured outputs"
    if not web.ENABLED:
        return "Web access is off (ENABLE_WEB=0)."
    try:
        bundle = await web.research(query, 5, 0)
    except web.WebError as err:
        return f"Search failed: {err}"
    if not bundle["results"]:
        return f"Nothing came back for _{query}_."
    lines = [f"🔎 *{bundle['query']}*", ""]
    for r in bundle["results"][:5]:
        lines.append(f"• *{r['title'][:70]}*\n{r['url']}")
    return "\n".join(lines)


def _set_mode(sender: str, arg: str) -> str:
    arg = (arg or "").strip().lower()
    valid = {"auto"} | set(providers.BY_NAME)
    if arg not in valid:
        return f"Pick one of: {', '.join(sorted(valid))}. Currently *{_mode.get(sender, 'auto')}*."
    _mode[sender] = arg
    return (f"Routing pinned to *{arg}* for this chat."
            if arg != "auto" else "Routing back to *auto* — weighed per subtask.")


async def _run_task(sender: str, task: str) -> None:
    """Stream one dispatch to a phone.

    Sends the plan, then each agent's answer as it lands, then a summary. Errors are sent
    when they happen rather than being folded into the summary, because on a phone the
    useful moment for 'the coder failed' is immediately, not four minutes later.
    """
    outputs: dict[str, str] = {}
    labels: dict[str, str] = {}
    sent_any = False
    try:
        mode = _mode.get(sender)
        async for line in octopus.dispatch(_key(), task, None, mode if mode != "auto" else None):
            if not line.startswith("data:"):
                continue
            try:
                e = json.loads(line[5:])
            except ValueError:
                continue
            kind = e.get("event")

            if kind == "wave":
                head = [f"*Wave {e['n']}* — {len(e['agents'])} agent(s) in parallel"]
                for a in e["agents"]:
                    model = providers.bare(a["model"]) if a["model"] else "—"
                    labels[a["id"]] = a["label"]
                    head.append(f"• *{a['label']}* — {a['provider']} / {model}")
                await whatsapp.send(sender, "\n".join(head))

            elif kind == "chunk":
                outputs[e["id"]] = outputs.get(e["id"], "") + e["text"]

            elif kind == "sources":
                srcs = e.get("sources") or []
                if srcs:
                    body = "🔎 *Sources read*\n" + "\n".join(s["url"] for s in srcs[:5])
                    await whatsapp.send(sender, body)

            elif kind == "done":
                text = outputs.get(e["id"], "").strip()
                if text:
                    sent_any = True
                    title = labels.get(e["id"], e["id"])
                    await whatsapp.send(sender, f"✅ *{title}*\n\n{text}")

            elif kind == "error":
                title = labels.get(e.get("id", ""), e.get("id", "an agent"))
                await whatsapp.send(sender, f"⚠️ *{title}* failed\n{e.get('detail', '')[:400]}")

            elif kind == "fatal":
                await whatsapp.send(sender, f"🛑 {e.get('detail', 'dispatch failed')}")

            elif kind == "complete":
                _last_cost[sender] = e.get("cost", {})
                split = ", ".join(f"{v} {k}" for k, v in (e.get("by_provider") or {}).items())
                cost = e.get("cost") or {}
                tail = (f"\n~{cost.get('tokens', 0)} tokens · "
                        f"{'$%.4f' % cost['usd'] if cost.get('billable') else 'free'}")
                await whatsapp.send(
                    sender,
                    f"🐙 *Done* — {e['agents']} agent(s), {e['waves']} wave(s), "
                    f"{e.get('ok', 0)} ok, {e.get('failed', 0)} failed"
                    + (f"\n{split}" if split else "") + tail)
                if not sent_any:
                    await whatsapp.send(sender, "No agent produced any text.")
    except asyncio.CancelledError:
        await whatsapp.send(sender, "🛑 Stopped.")
        raise
    except Exception as err:                          # never leave the phone hanging
        await whatsapp.send(sender, f"🛑 The run broke: {type(err).__name__}: {err}")
    finally:
        _runs.pop(sender, None)


async def handle(msg: whatsapp.Inbound) -> None:
    """One inbound message, already authenticated and de-duplicated."""
    text = msg.text.strip()
    sender = msg.sender

    if whatsapp.is_command(text):
        cmd, arg = whatsapp.command_of(text)
        if cmd in ("help", "start", "?"):
            await whatsapp.send(sender, whatsapp.HELP)
        elif cmd == "status":
            await whatsapp.send(sender, await _status())
        elif cmd == "models":
            await whatsapp.send(sender, await _models())
        elif cmd == "cost":
            await whatsapp.send(sender, _cost(sender))
        elif cmd == "mode":
            await whatsapp.send(sender, _set_mode(sender, arg))
        elif cmd == "web":
            await whatsapp.send(sender, await _web(arg))
        elif cmd == "stop":
            task = _runs.get(sender)
            if task and not task.done():
                task.cancel()
            else:
                await whatsapp.send(sender, "Nothing running.")
        else:
            await whatsapp.send(sender, f"Unknown command /{cmd}.\n\n{whatsapp.HELP}")
        return

    if sender in _runs and not _runs[sender].done():
        await whatsapp.send(sender, BUSY)
        return

    pin = _mode.get(sender, "auto")
    await whatsapp.send(
        sender, f"🐙 Working on it — routing *{pin}*.\nI'll send each agent's answer as it lands.")
    _runs[sender] = asyncio.create_task(_run_task(sender, text))
