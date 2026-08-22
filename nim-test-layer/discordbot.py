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
import voice

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


# Discord embeds hold 4096 characters of description against 2000 for a plain message,
# and carry a title, a colour and a footer. That is most of the difference between an
# answer that reads as a document and one that reads as three fragments in a row.
EMBED_LIMIT = int(os.getenv("DISCORD_EMBED_LIMIT", "4000"))


def _colour(css: str):
    """'#f2b56b' -> a discord.Colour. Each role already has one; reuse it."""
    import discord
    try:
        return discord.Colour(int((css or "").lstrip("#"), 16))
    except (ValueError, AttributeError):
        return discord.Colour(0x5FD8D1)


def _transport(channel) -> chat_bridge.Transport:
    """A Discord conversation.

    Three surfaces rather than one: a status line that is edited in place as the run
    progresses, embed cards for the answers, and plain sends for everything else. The
    status line is what stops a five-agent run producing fifteen messages.
    """
    import discord

    state = {"status": None}

    async def send(text: str) -> bool:
        for part in chat_bridge.split(text, LIMIT):
            try:
                await channel.send(part)
            except Exception as err:
                print(f"[discord] send failed: {type(err).__name__}: {err}", flush=True)
                return False
        return True

    async def status(text: str) -> bool:
        """Rewrite the run's progress line, or post it the first time."""
        body = text if len(text) <= EMBED_LIMIT else text[:EMBED_LIMIT]
        try:
            embed = discord.Embed(description=body, colour=discord.Colour(0x1ABC9C))
            msg = state.get("status")
            if msg is None:
                state["status"] = await channel.send(embed=embed)
            else:
                await msg.edit(embed=embed)
            return True
        except Exception as err:
            # An edit can fail if the message was deleted; falling back to a new one
            # keeps the run legible rather than silently losing its progress.
            print(f"[discord] status failed: {type(err).__name__}: {err}", flush=True)
            state["status"] = None
            return await send(text)

    async def card(title: str, body: str, colour: str = "", footer: str = "") -> bool:
        """One agent's answer as an embed, split across several only when it must be."""
        try:
            parts = chat_bridge.split(body, EMBED_LIMIT)
            total = len(parts)
            for i, part in enumerate(parts, 1):
                embed = discord.Embed(
                    title=title[:250] + (f"  ({i}/{total})" if total > 1 else ""),
                    description=part, colour=_colour(colour))
                if footer:
                    embed.set_footer(text=footer[:2040])
                await channel.send(embed=embed)
            return True
        except Exception as err:
            print(f"[discord] card failed: {type(err).__name__}: {err}", flush=True)
            return await send(f"**{title}**\n\n{body}")

    async def audio(clip: bytes, caption: str = "") -> bool:
        """Attach a spoken answer. Discord renders a WAV with an inline player."""
        try:
            import io
            await channel.send(
                content=f"🔊 {caption[:80]}" if caption else None,
                file=discord.File(io.BytesIO(clip), filename="answer.wav"))
            return True
        except Exception as err:
            print(f"[discord] audio failed: {type(err).__name__}: {err}", flush=True)
            return False

    # Discord spells bold with two asterisks and italic with one; WhatsApp uses one and
    # an underscore. Getting this wrong shows up as literal asterisks in the chat.
    return chat_bridge.Transport(name="discord", limit=LIMIT, send=send, audio=audio,
                                 bold="**", italic="*",
                                 card=card, status=status, card_limit=EMBED_LIMIT,
                                 # A fresh transport per concurrent run: `state` above
                                 # holds the one message this run edits, and two runs
                                 # sharing it would overwrite each other's progress.
                                 fork=lambda: _transport(channel))


async def _build():
    """Construct the client. Imported lazily so the app runs without discord.py."""
    import discord
    from discord import app_commands

    intents = discord.Intents.default()
    # Reading ordinary message text is a privileged intent and has to be switched on in
    # the Developer Portal too. Without it the bot connects and sees empty message bodies,
    # which looks exactly like being ignored.
    intents.message_content = True

    client = discord.Client(intents=intents)
    tree = app_commands.CommandTree(client)

    def allowed(user_id: int) -> bool:
        return str(user_id) in ALLOWED

    async def run(interaction, text: str) -> None:
        """Answer a slash command through the shared bridge.

        Discord gives an interaction three seconds to be acknowledged or it shows "the
        application did not respond", and a dispatch takes minutes. So the interaction is
        acknowledged immediately and everything after that goes to the channel through the
        ordinary transport — which means slash commands get the same embeds and the same
        edited status line as plain messages, instead of a second, worse rendering path.
        Follow-up tokens also expire after fifteen minutes, which a long run would outlive.
        """
        if not allowed(interaction.user.id):
            await interaction.response.send_message(
                "Not on this bot's allowlist.", ephemeral=True)
            return
        try:
            await interaction.response.send_message("On it.", ephemeral=True)
        except Exception:
            pass
        convo = f"discord:{interaction.channel_id}"
        await chat_bridge.handle(convo, text, _transport(interaction.channel))

    @tree.command(name="help", description="What this bot can do")
    async def _help(interaction) -> None:
        await run(interaction, "/help")

    @tree.command(name="status", description="Providers, routing mode, web access")
    async def _status(interaction) -> None:
        await run(interaction, "/status")

    @tree.command(name="models", description="What each role is bound to, per provider")
    async def _models(interaction) -> None:
        await run(interaction, "/models")

    @tree.command(name="cost", description="What the last run spent")
    async def _cost(interaction) -> None:
        await run(interaction, "/cost")

    @tree.command(name="stop", description="Cancel the run in progress")
    async def _stop(interaction) -> None:
        await run(interaction, "/stop")

    @tree.command(name="mode", description="Pin where work goes")
    @app_commands.describe(where="auto, local, nvidia or gemini")
    async def _mode(interaction, where: str) -> None:
        await run(interaction, f"/mode {where}")

    @tree.command(name="web", description="Search the web (no model involved)")
    @app_commands.describe(query="what to search for")
    async def _web(interaction, query: str) -> None:
        await run(interaction, f"/web {query}")

    @tree.command(name="task", description="Run a task across the agent pool")
    @app_commands.describe(task="what you want done")
    async def _task(interaction, task: str) -> None:
        await run(interaction, task)

    @client.event
    async def on_ready() -> None:
        print(f"[discord] connected as {client.user} — "
              f"{len(ALLOWED)} allowed user(s)", flush=True)
        # Per-guild sync appears immediately; a global sync can take an hour to propagate.
        for guild in client.guilds:
            try:
                tree.copy_global_to(guild=guild)
                synced = await tree.sync(guild=guild)
                print(f"[discord] {len(synced)} slash commands in {guild.name}", flush=True)
            except Exception as err:
                # Almost always a missing 'applications.commands' scope on the invite.
                # Not fatal: '!' commands and plain text still work.
                print(f"[discord] could not register slash commands in {guild.name}: "
                      f"{type(err).__name__}: {err}", flush=True)
                print("[discord] re-invite with scope=bot+applications.commands to fix; "
                      "meanwhile use !help", flush=True)

    @client.event
    async def on_message(message) -> None:
        if message.author.bot or message.author == client.user:
            return
        if not allowed(message.author.id):
            return                                    # silence, not an error
        if CHANNELS and str(message.channel.id) not in CHANNELS:
            return
        text = (message.content or "").strip()

        # A voice message is a task too. Discord sends them as ordinary attachments, so
        # anything audio-shaped is transcribed and used as the message body — and the
        # answer comes back spoken, because that is the medium it arrived in.
        spoken = False
        for att in message.attachments:
            if (att.content_type or "").startswith("audio") or                     att.filename.lower().endswith((".ogg", ".mp3", ".m4a", ".wav", ".webm")):
                async with message.channel.typing():
                    heard = await voice.listen(await att.read())
                if heard:
                    text, spoken = heard, True
                    await message.channel.send(f"🎤 heard: *{heard[:300]}*")
                else:
                    await message.channel.send("🎤 I could not make that out.")
                    return
                break

        if not text:
            return
        # Keyed by channel so a DM and a server channel are separate conversations with
        # separate routing pins and separate in-flight runs.
        convo = f"discord:{message.channel.id}"
        await chat_bridge.handle(convo, text, _transport(message.channel), spoken=spoken)

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
        "active_runs": chat_bridge.active_runs(),
    }
