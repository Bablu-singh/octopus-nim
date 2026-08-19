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
# Logs every inbound message and the reason it was or was not acted on.
DEBUG = os.getenv("WHATSAPP_QR_DEBUG", "1") == "1"

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

# LID -> phone number, once resolved. The mapping does not change.
_lid_cache: dict[str, str] = {}


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


def _resolve_lid(jid) -> str:
    """Ask the client which phone number a LID belongs to, or '' if it cannot say.

    WhatsApp increasingly addresses people only by LID: the message carries the LID and
    `SenderAlt` is empty, so there is nothing in the event itself to compare against an
    allowlist of phone numbers. whatsmeow keeps the mapping, so the client is asked.
    Cached, because this is a round trip and every inbound message would otherwise pay
    for it.
    """
    key = _digits(jid)
    if not key:
        return ""
    if key in _lid_cache:
        return _lid_cache[key]
    number = ""
    try:
        if _client is not None:
            number = _digits(_client.get_pn_from_lid(jid))
    except Exception as err:
        if DEBUG:
            print(f"[wa-qr]   lid lookup failed: {type(err).__name__}: {err}", flush=True)
    _lid_cache[key] = number
    return number


def _identities(source) -> tuple[set[str], str]:
    """Every number this sender could be known by, and the phone one if we can tell.

    WhatsApp now addresses people by LID as well as by phone number. In LID mode `Sender`
    is a @lid identifier — a completely different number — and the phone number lives in
    `SenderAlt`. Matching only `Sender` against an allowlist of phone numbers therefore
    silently rejects the owner of the account, which is exactly the shape of "the bot
    connects fine and ignores every message".
    """
    ids, phone = set(), ""
    for attr in ("Sender", "SenderAlt"):
        jid = getattr(source, attr, None)
        if jid is None:
            continue
        num = _digits(jid)
        if not num:
            continue
        ids.add(num)
        # s.whatsapp.net is the phone-number namespace; lid is not.
        if str(getattr(jid, "Server", "")).startswith("s.whatsapp"):
            phone = phone or num
        elif str(getattr(jid, "Server", "")).startswith("lid"):
            resolved = _resolve_lid(jid)
            if resolved:
                ids.add(resolved)
                phone = phone or resolved
    return ids, phone


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
        """Inbound. Every early return says why when DEBUG is on.

        A door that silently ignores you is indistinguishable from a broken one, and the
        allowlist path is deliberately silent, so the only way to tell them apart is to
        say so in the log.
        """
        try:
            info = message.Info
            msg_id = str(getattr(info, "ID", "") or "")
            src = info.MessageSource
            ids, phone = _identities(src)
            sender = phone or (sorted(ids)[0] if ids else "")
            from_me = bool(getattr(info, "IsFromMe", False))
            msg = message.Message
            body = (getattr(msg, "conversation", "")
                    or getattr(getattr(msg, "extendedTextMessage", None), "text", "")
                    or "").strip()

            if DEBUG:
                mode = getattr(src, "AddressingMode", "")
                print(f"[wa-qr] inbound id={msg_id[:12]} ids={sorted(ids)} "
                      f"phone={phone or '-'} mode={mode} from_me={from_me} "
                      f"chars={len(body)} allowed={bool(ids & ALLOWED)}", flush=True)

            if msg_id in _sent_ids:
                if DEBUG:
                    print("[wa-qr]   dropped: our own reply echoing back", flush=True)
                return
            if not (ids & ALLOWED):
                if DEBUG:
                    print(f"[wa-qr]   dropped: none of {sorted(ids)} in "
                          f"WHATSAPP_QR_ALLOWED ({sorted(ALLOWED)})", flush=True)
                return
            if not body:
                if DEBUG:
                    kinds = [f.name for f, _ in msg.ListFields()]
                    print(f"[wa-qr]   dropped: no text; message carries {kinds}", flush=True)
                return

            chat = src.Chat
            if _loop is None:
                print("[wa-qr]   dropped: no event loop bound", flush=True)
                return
            if DEBUG:
                print(f"[wa-qr]   dispatching: {body[:60]!r}", flush=True)
            asyncio.run_coroutine_threadsafe(
                chat_bridge.handle(f"waqr:{sender}", body, _transport(chat)), _loop)
        except Exception as err:
            import traceback
            print(f"[wa-qr] inbound error: {type(err).__name__}: {err}", flush=True)
            traceback.print_exc()

    if DEBUG:
        # Trace every event the Go layer hands up, before any of our filtering. Without
        # this there is no way to tell "the message never arrived" from "the message
        # arrived and our handler rejected it" — and those need completely different
        # fixes. Patched before connect(), because the ctypes callback is bound there.
        from neonize.events import INT_TO_EVENT
        _orig_execute = client.event.execute

        def _traced(uuid, binary, size, code):
            name = INT_TO_EVENT.get(code)
            print(f"[wa-qr] raw event code={code} "
                  f"{getattr(name, '__name__', name)}", flush=True)
            return _orig_execute(uuid, binary, size, code)

        client.event.execute = _traced

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


async def send_test(text: str = "octopus round-trip test") -> dict:
    """Send one short message to the first allowed number.

    A diagnostic, and a deliberately explicit one: proving the send path and the receive
    path separately takes several rounds of a human typing on a phone, whereas one message
    to yourself exercises both — it arrives, and its echo comes back in as an inbound
    event. Only ever addressed to a number already on the allowlist.
    """
    if _client is None or not _state["connected"]:
        return {"sent": False, "detail": "client is not connected"}
    if not ALLOWED:
        return {"sent": False, "detail": "no allowed number to send to"}
    number = sorted(ALLOWED)[0]
    try:
        from neonize.utils import build_jid
        loop = asyncio.get_running_loop()
        resp = await loop.run_in_executor(
            None, _client.send_message, build_jid(number), text)
        msg_id = str(getattr(resp, "ID", "") or "")
        _remember_sent(msg_id)
        return {"sent": True, "to": number, "message_id": msg_id}
    except Exception as err:
        return {"sent": False, "detail": f"{type(err).__name__}: {err}"}


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
