"""WhatsApp by QR scan — no Meta account, and no official blessing either.

This links to WhatsApp the way WhatsApp Web does: you scan a QR code once from the phone
app and the session persists. No Meta developer account, no app review, no public webhook,
no tunnel, no 24-hour messaging window.

╔══════════════════════════════════════════════════════════════════════════════╗
║  READ THIS BEFORE ENABLING                                                   ║
║                                                                              ║
║  This drives the WhatsApp Web protocol through an unofficial client          ║
║  (whatsmeow, via neonize). WhatsApp's Terms of Service prohibit unauthorised ║
║  clients, and enforcement is real rather than theoretical.                   ║
║                                                                              ║
║  A ban lands on the PHONE NUMBER, not on an API key. The thing at risk is    ║
║  the WhatsApp account itself — its chats, its groups, its contacts.          ║
║                                                                              ║
║  Use a spare number. Not the one that matters.                               ║
║                                                                              ║
║  The officially supported route is whatsapp.py (Meta Cloud API), and the     ║
║  route with no account risk at all is discordbot.py.                         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Because that risk is taken on rather than discovered, enabling this needs two separate
switches: `ENABLE_WHATSAPP_QR=1` turns it on and `WHATSAPP_QR_ACCEPT_RISK=1` says the
paragraph above was read. One without the other refuses to start and says why.

--- How it fits ------------------------------------------------------------

neonize is synchronous and carries a Go runtime, so the client runs on its own thread and
hands work back to the app's event loop. Everything above that is the same
`chat_bridge` the browser, Discord and the Cloud API all use.
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

import chat_bridge

HERE = Path(__file__).parent

ENABLED = os.getenv("ENABLE_WHATSAPP_QR", "0") == "1"
ACCEPTED_RISK = os.getenv("WHATSAPP_QR_ACCEPT_RISK", "0") == "1"

# Numbers allowed to drive it, international format without '+'. Empty means nobody,
# which is the same closed default every other door here uses.
ALLOWED = {n.strip().lstrip("+") for n in os.getenv("WHATSAPP_QR_ALLOWED", "").split(",")
           if n.strip()}

# The linked-device credentials live here. It is a secret: anyone holding this file can
# act as your WhatsApp. Gitignored, and kept out of the repo directory listing.
SESSION = os.getenv("WHATSAPP_QR_SESSION", str(HERE / "wa_session.sqlite3"))
QR_PNG = HERE / "wa_qr.png"

MAX_BODY = int(os.getenv("WHATSAPP_QR_MAX_BODY", "3500"))

_client = None
_thread: threading.Thread | None = None
_loop: asyncio.AbstractEventLoop | None = None
_state = {"connected": False, "qr": None, "me": None, "error": None}

# Ids of messages this process sent. When the linked account is also the account being
# messaged — the "message yourself" pattern, which is the natural way to use a bot that
# *is* your own WhatsApp — every reply comes back in as an inbound message from you. Left
# alone that is an infinite loop: reply, read own reply, reply again. Bounded because the
# process runs for days.
_sent_ids: "OrderedDict[str, float]" = OrderedDict()
SENT_MAX = 300


def _remember_sent(msg_id: str) -> None:
    if not msg_id:
        return
    _sent_ids[msg_id] = time.time()
    while len(_sent_ids) > SENT_MAX:
        _sent_ids.popitem(last=False)

WARNING = (
    "WhatsApp QR uses an unofficial client. WhatsApp's terms prohibit this and bans "
    "apply to the phone number, not to a key. Use a spare number."
)


def configured() -> tuple[bool, str]:
    if not ENABLED:
        return False, "ENABLE_WHATSAPP_QR is not 1."
    if not ACCEPTED_RISK:
        return False, ("WHATSAPP_QR_ACCEPT_RISK is not 1. " + WARNING)
    if not ALLOWED:
        return False, "WHATSAPP_QR_ALLOWED is empty — it would ignore everyone."
    try:
        import neonize  # noqa: F401
    except ImportError:
        return False, "neonize is not installed — pip install neonize"
    return True, "ready"


def _digits(jid) -> str:
    """The plain number out of whatever shape the JID arrives in."""
    raw = getattr(jid, "User", None) or str(jid)
    return "".join(ch for ch in str(raw).split("@")[0].split(":")[0] if ch.isdigit())


def _transport(chat_jid) -> chat_bridge.Transport:
    """A conversation on WhatsApp.

    `send_message` is a blocking call into the Go runtime, so it goes to a worker thread
    rather than stalling the event loop that every agent is streaming through.
    """
    async def send(text: str) -> bool:
        if _client is None:
            return False
        loop = asyncio.get_running_loop()
        for part in chat_bridge.split(text, MAX_BODY):
            try:
                resp = await loop.run_in_executor(
                    None, _client.send_message, chat_jid, part)
                _remember_sent(str(getattr(resp, "ID", "") or ""))
            except Exception as err:
                print(f"[wa-qr] send failed: {type(err).__name__}: {err}", flush=True)
                return False
        return True

    # WhatsApp has no embeds and no editable messages, so it takes the plain-text
    # fallbacks in chat_bridge — the same ones the Cloud API door uses.
    return chat_bridge.Transport(name="whatsapp-qr", limit=MAX_BODY, send=send,
                                 bold="*", italic="_")


def _render_qr(code: str) -> None:
    """Show the pairing code as something scannable, twice over.

    Terminal first, because that is where someone running this is looking. A PNG as well,
    because a Windows console at the wrong font size renders a QR as unreadable mush and
    the image always works.
    """
    _state["qr"] = code
    try:
        import segno
        qr = segno.make(code)
        print("\n[wa-qr] scan this from WhatsApp > Linked devices:\n", flush=True)
        qr.terminal(compact=True)
        qr.save(str(QR_PNG), scale=6, border=2)
        print(f"\n[wa-qr] also saved to {QR_PNG}", flush=True)
        print("[wa-qr] or open http://127.0.0.1:8000/api/whatsapp-qr/image", flush=True)
    except Exception as err:
        print(f"[wa-qr] could not render QR ({type(err).__name__}) — raw code below",
              flush=True)
        print(code, flush=True)


def _run_client() -> None:
    """Own thread: neonize is synchronous and `connect()` never returns."""
    global _client
    from neonize.client import NewClient
    from neonize.events import ConnectedEv, MessageEv, PairStatusEv, LoggedOutEv

    client = NewClient(SESSION)
    _client = client

    # Registered as `event.qr`, not `event(QREv)`. The latter silently never fires and
    # neonize's own printer runs instead — which looks like it works, right up until the
    # PNG and the health endpoint are both empty.
    @client.event.qr
    def _on_qr(_c, data) -> None:
        code = data.decode() if isinstance(data, (bytes, bytearray)) else str(data)
        _render_qr(code)

    @client.event(ConnectedEv)
    def _on_connected(_c, _e) -> None:
        _state["connected"] = True
        _state["qr"] = None
        # PairStatusEv only fires the first time a device is linked, so on every later
        # start — reconnecting from the stored session, which is the normal case — the
        # number would otherwise read as unknown. Ask the client who it is instead.
        try:
            _state["me"] = _digits(client.get_me().JID)
        except Exception:
            pass
        try:
            QR_PNG.unlink(missing_ok=True)      # the code is spent; do not leave it lying
        except OSError:
            pass
        print("[wa-qr] linked and connected", flush=True)

    @client.event(PairStatusEv)
    def _on_pair(_c, e) -> None:
        _state["me"] = _digits(getattr(e, "ID", ""))
        print(f"[wa-qr] paired as {_state['me']}", flush=True)

    @client.event(LoggedOutEv)
    def _on_logout(_c, _e) -> None:
        _state["connected"] = False
        _state["error"] = "logged out — the device was unlinked from the phone"
        print("[wa-qr] logged out", flush=True)

    @client.event(MessageEv)
    def _on_message(_c, message) -> None:
        try:
            info = message.Info
            msg_id = str(getattr(info, "ID", "") or "")
            if msg_id in _sent_ids:
                return                                 # our own reply coming back

            sender = _digits(info.MessageSource.Sender)
            # A message from the linked account is normally this bot talking. It is also
            # how someone drives a bot that *is* their own WhatsApp: they message
            # themselves. Both look identical here, so the id check above is what
            # separates them — an echo of our own send is dropped, anything else the
            # account typed is real input.
            if getattr(info, "IsFromMe", False) and sender not in ALLOWED:
                return
            if sender not in ALLOWED:
                return                                 # silence, not an error
            body = (message.Message.conversation
                    or message.Message.extendedTextMessage.text or "").strip()
            if not body:
                return
            chat = info.MessageSource.Chat
            # Hop from this thread onto the app's loop, where everything else lives.
            if _loop is not None:
                asyncio.run_coroutine_threadsafe(
                    chat_bridge.handle(f"waqr:{sender}", body, _transport(chat)), _loop)
        except Exception as err:
            print(f"[wa-qr] inbound error: {type(err).__name__}: {err}", flush=True)

    try:
        client.connect()
    except Exception as err:
        _state["error"] = f"{type(err).__name__}: {err}"
        print(f"[wa-qr] client stopped: {err}", flush=True)


def launch() -> bool:
    """Start the client thread. Safe to call when unconfigured — it just says no."""
    global _thread, _loop
    ok, why = configured()
    if not ok:
        print(f"[wa-qr] not starting — {why}", flush=True)
        return False
    print(f"[wa-qr] WARNING: {WARNING}", flush=True)
    _loop = asyncio.get_running_loop()
    _thread = threading.Thread(target=_run_client, name="wa-qr", daemon=True)
    _thread.start()
    return True


def status() -> dict:
    ok, why = configured()
    return {
        "enabled": ENABLED,
        "risk_accepted": ACCEPTED_RISK,
        "ready": ok,
        "detail": why,
        "connected": _state["connected"],
        "paired_as": _state["me"],
        "awaiting_scan": bool(_state["qr"]),
        "qr_image": "/api/whatsapp-qr/image" if _state["qr"] else None,
        "allowed_numbers": len(ALLOWED),
        "error": _state["error"],
        "warning": WARNING,
        "session_file": SESSION,
    }
