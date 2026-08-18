"""Driving the octopus from Discord.

The reason this exists alongside the WhatsApp door is that Discord asks for almost
nothing. Its gateway is an outbound WebSocket, so there is no public webhook, no tunnel,
no signature to verify and no reverse proxy — this process dials out and stays connected.
Setup is a bot token and an invite link, and it works from the phone app.

Contrast WhatsApp, which needs a Meta app, a publicly reachable HTTPS callback, a signed
webhook and a 24-hour messaging window. Both doors are kept; this is the one to reach for.

`discord.py` rather than a hand-rolled gateway client. The gateway itself is easy; the
parts that are not are resume-after-disconnect, heartbeat drift, and the rate limit
buckets — and getting those wrong produces a bot that goes quiet at 3am rather than one
that fails loudly. This is the one place in the codebase where a dependency earns its
keep.

--- Who is allowed to use it ------------------------------------------------

The bot runs work on the host machine, so it answers nobody by default. `DISCORD_ALLOWED`
is a list of Discord user IDs; an empty list means the bot connects, reports itself
ready, and ignores every message. That is deliberately useless rather than deliberately
open — the same choice made everywhere else here.
"""

from __future__ import annotations

import asyncio
import os

import chat_bridge

ENABLED = os.getenv("ENABLE_DISCORD", "0") == "1"
TOKEN = os.getenv("DISCORD_TOKEN", "").strip()

# Numeric Discord user IDs, comma separated. Enable Developer Mode in Discord, then
# right-click your name -> Copy User ID.
ALLOWED = {u.strip() for u in os.getenv("DISCORD_ALLOWED", "").split(",") if u.strip()}
# Optional: restrict to particular channel ids as well as particular people.
CHANNELS = {c.strip() for c in os.getenv("DISCORD_CHANNELS", "").split(",") if c.strip()}

# Discord refuses any message body over 2000 characters. Splitting at 1900 leaves room
# for the part counter that `chat_bridge.split` prepends.
LIMIT = int(os.getenv("DISCORD_MAX_BODY", "1900"))

_client = None
_task: asyncio.Task | None = None


def configured() -> tuple[bool, str]:
    if not ENABLED:
        return False, "ENABLE_DISCORD is not 1."
    if not TOKEN:
        return False, "Missing DISCORD_TOKEN."
    if not ALLOWED:
        return False, "DISCORD_ALLOWED is empty — the bot would ignore everyone."
    return True, "ready"


def _transport(channel) -> chat_bridge.Transport:
    async def send(text: str) -> bool:
        for part in chat_bridge.split(text, LIMIT):
            try:
                await channel.send(part)
            except Exception as err:                  # a dead channel must not kill the run
                print(f"[discord] send failed: {type(err).__name__}: {err}", flush=True)
                return False
        return True

    # Discord spells bold with two asterisks and italic with one; WhatsApp uses one and
    # an underscore. Getting this wrong shows up as literal asterisks in the chat.
    return chat_bridge.Transport(name="discord", limit=LIMIT, send=send,
                                 bold="**", italic="*")


async def _build():
    """Construct the client. Imported lazily so the app runs without discord.py."""
    import discord

    intents = discord.Intents.default()
    # Reading ordinary message text is a privileged intent and has to be switched on in
    # the Developer Portal too. Without it the bot connects and sees empty message bodies,
    # which looks exactly like being ignored.
    intents.message_content = True

    client = discord.Client(intents=intents)

    @client.event
    async def on_ready() -> None:
        print(f"[discord] connected as {client.user} — "
              f"{len(ALLOWED)} allowed user(s)", flush=True)

    @client.event
    async def on_message(message) -> None:
        if message.author.bot or message.author == client.user:
            return
        if str(message.author.id) not in ALLOWED:
            return                                    # silence, not an error
        if CHANNELS and str(message.channel.id) not in CHANNELS:
            return
        text = (message.content or "").strip()
        if not text:
            return
        # Keyed by channel so a DM and a server channel are separate conversations with
        # separate routing pins and separate in-flight runs.
        convo = f"discord:{message.channel.id}"
        await chat_bridge.handle(convo, text, _transport(message.channel))

    return client


async def start() -> None:
    """Connect and stay connected. Safe to call when unconfigured — it just returns."""
    global _client
    ok, why = configured()
    if not ok:
        print(f"[discord] not starting — {why}", flush=True)
        return
    try:
        _client = await _build()
    except ImportError:
        print("[discord] discord.py is not installed — pip install discord.py", flush=True)
        return
    try:
        await _client.start(TOKEN)
    except asyncio.CancelledError:
        raise
    except Exception as err:
        # A bad token or a missing intent should be one loud line at startup, not a
        # traceback that takes the whole web app down with it.
        print(f"[discord] stopped: {type(err).__name__}: {err}", flush=True)


def launch(loop: asyncio.AbstractEventLoop | None = None) -> asyncio.Task | None:
    """Run the bot beside the web app, in the same process and event loop."""
    global _task
    ok, _ = configured()
    if not ok:
        return None
    _task = asyncio.create_task(start())
    return _task


async def shutdown() -> None:
    global _client, _task
    if _client is not None:
        try:
            await _client.close()
        except Exception:
            pass
        _client = None
    if _task is not None:
        _task.cancel()
        _task = None


def status() -> dict:
    ok, why = configured()
    connected = bool(_client is not None and getattr(_client, "user", None))
    return {
        "enabled": ENABLED,
        "ready": ok,
        "detail": why,
        "connected": connected,
        "user": str(getattr(_client, "user", "")) if connected else None,
        "allowed_users": len(ALLOWED),
        "restricted_channels": len(CHANNELS),
        "active_runs": sum(1 for t in chat_bridge._runs.values() if not t.done()),
    }
