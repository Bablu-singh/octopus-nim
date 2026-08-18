"""Driving the octopus from a phone, over WhatsApp.

The app is a localhost service with a browser UI. This module gives it a second front
door: send it a task from WhatsApp and the agent pool runs on your machine, streaming
each agent's answer back as messages. Nothing about the octopus changes — this is another
caller of `octopus.dispatch()`, the same one the browser uses.

Meta's WhatsApp Cloud API, deliberately. It is the official route, it has a free tier,
and it does not require pretending to be a phone. The unofficial libraries that drive
WhatsApp Web get accounts banned and are not worth suggesting.

--- What this exposes -------------------------------------------------------

A webhook is a public HTTPS endpoint that runs work on your machine. That is a serious
thing to open, so three gates stand in front of it and all three are on by default:

  1. Signature. Meta signs every delivery with your app secret; a body whose HMAC does
     not match is dropped before it is parsed.
  2. Allowlist. Only numbers in WHATSAPP_ALLOWED can make it do anything. Anyone else
     gets silence — not an error, which would confirm the endpoint exists.
  3. Replay. Meta retries deliveries it thinks failed, and a retried task would dispatch
     the agent pool a second time. Message ids are remembered and repeats ignored.

Without an app secret configured the signature check cannot run, and this module refuses
to act on anything rather than falling back to trusting the caller.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import os
import time
from collections import OrderedDict
from dataclasses import dataclass

import httpx

GRAPH = os.getenv("WHATSAPP_GRAPH", "https://graph.facebook.com/v21.0")

TOKEN = os.getenv("WHATSAPP_TOKEN", "").strip()
PHONE_ID = os.getenv("WHATSAPP_PHONE_ID", "").strip()
VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN", "").strip()
APP_SECRET = os.getenv("WHATSAPP_APP_SECRET", "").strip()

# Comma-separated numbers in international format without '+', e.g. 919876543210.
ALLOWED = {n.strip().lstrip("+") for n in os.getenv("WHATSAPP_ALLOWED", "").split(",")
           if n.strip()}

ENABLED = os.getenv("ENABLE_WHATSAPP", "0") == "1"

# WhatsApp rejects a text body over 4096 characters. Splitting at 3500 leaves room for
# the part counter and avoids a message that is exactly at the edge being refused.
MAX_BODY = int(os.getenv("WHATSAPP_MAX_BODY", "3500"))
# Sends are spaced: a finished wave can produce half a dozen messages at once, and a
# burst is what gets an application rate limited.
SEND_RPM = int(os.getenv("WHATSAPP_SEND_RPM", "40"))

_send_gate = asyncio.Lock()
_last_send = 0.0

# Bounded, because this process runs for days and an unbounded set of ids is a slow leak.
_seen: "OrderedDict[str, float]" = OrderedDict()
SEEN_MAX = 400


def configured() -> tuple[bool, str]:
    """Is this usable, and if not, which piece is missing?"""
    if not ENABLED:
        return False, "ENABLE_WHATSAPP is not 1."
    missing = [name for name, val in (
        ("WHATSAPP_TOKEN", TOKEN), ("WHATSAPP_PHONE_ID", PHONE_ID),
        ("WHATSAPP_VERIFY_TOKEN", VERIFY_TOKEN), ("WHATSAPP_APP_SECRET", APP_SECRET),
    ) if not val]
    if missing:
        return False, "Missing: " + ", ".join(missing)
    if not ALLOWED:
        # An open webhook that dispatches agents is not a default anyone should get by
        # forgetting a variable.
        return False, "WHATSAPP_ALLOWED is empty — refusing to accept messages from anyone."
    return True, "ready"


# --- inbound ----------------------------------------------------------------


def verify_handshake(mode: str | None, token: str | None, challenge: str | None) -> str:
    """Meta's GET handshake when you save the webhook URL. Echo the challenge back."""
    if mode == "subscribe" and token and VERIFY_TOKEN and \
            hmac.compare_digest(token, VERIFY_TOKEN):
        return challenge or ""
    raise PermissionError("verify token mismatch")


def signed_by_meta(raw_body: bytes, header: str | None) -> bool:
    """Constant-time check of X-Hub-Signature-256 against the app secret.

    Returns False when no secret is configured. That is deliberate: an unsigned webhook
    is an open remote-execution endpoint, and defaulting to 'allow' when the operator
    forgot a variable is how that happens by accident.
    """
    if not APP_SECRET or not header:
        return False
    prefix, _, sent = header.partition("=")
    if prefix != "sha256" or not sent:
        return False
    expected = hmac.new(APP_SECRET.encode(), raw_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, sent)


def already_handled(message_id: str) -> bool:
    """Meta re-delivers anything it thinks failed. A retry must not run the task twice."""
    if not message_id:
        return False
    if message_id in _seen:
        return True
    _seen[message_id] = time.time()
    while len(_seen) > SEEN_MAX:
        _seen.popitem(last=False)
    return False


@dataclass
class Inbound:
    sender: str
    text: str
    message_id: str


def parse(payload: dict) -> list[Inbound]:
    """Pull user text messages out of a webhook body, ignoring everything else.

    Meta delivers status callbacks (delivered, read), reactions and media through the
    same endpoint. Only text messages are work; the rest are noise and must not be
    mistaken for an empty task.
    """
    out: list[Inbound] = []
    for entry in payload.get("entry", []) or []:
        for change in entry.get("changes", []) or []:
            value = change.get("value") or {}
            for msg in value.get("messages", []) or []:
                if msg.get("type") != "text":
                    continue
                body = ((msg.get("text") or {}).get("body") or "").strip()
                if body:
                    out.append(Inbound(sender=str(msg.get("from", "")).lstrip("+"),
                                       text=body, message_id=str(msg.get("id", ""))))
    return out


def permitted(sender: str) -> bool:
    return sender.lstrip("+") in ALLOWED


# --- outbound ---------------------------------------------------------------


def split(text: str, limit: int = MAX_BODY) -> list[str]:
    """Break a long answer into sendable pieces, on a boundary a reader would choose.

    Paragraph first, then line, then a hard cut. An agent's answer is usually Markdown,
    and slicing mid-sentence every 3500 characters makes it unreadable on a phone.
    """
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    parts, buf = [], ""
    for para in text.split("\n\n"):
        block = para if len(para) <= limit else ""
        if not block:
            # A single paragraph over the limit: fall back to lines, then to a hard cut.
            for line in para.split("\n"):
                while len(line) > limit:
                    parts.append(line[:limit])
                    line = line[limit:]
                if len(buf) + len(line) + 1 > limit:
                    parts.append(buf.strip())
                    buf = ""
                buf += line + "\n"
            continue
        if len(buf) + len(block) + 2 > limit:
            parts.append(buf.strip())
            buf = ""
        buf += block + "\n\n"
    if buf.strip():
        parts.append(buf.strip())

    if len(parts) > 1:
        total = len(parts)
        parts = [f"({i}/{total}) {p}" for i, p in enumerate(parts, 1)]
    return parts


async def _gate() -> None:
    global _last_send
    interval = 60.0 / SEND_RPM if SEND_RPM > 0 else 0.0
    if not interval:
        return
    async with _send_gate:
        wait = (_last_send + interval) - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait)
        _last_send = time.monotonic()


async def send(to: str, text: str) -> bool:
    """Send one message, splitting it if needed. False if anything did not go out."""
    ok, _ = configured()
    if not ok:
        return False
    sent_all = True
    for part in split(text):
        await _gate()
        body = {"messaging_product": "whatsapp", "to": to, "type": "text",
                "text": {"preview_url": False, "body": part}}
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=8, read=25,
                                                               write=15, pool=5)) as c:
                r = await c.post(f"{GRAPH}/{PHONE_ID}/messages",
                                 headers={"Authorization": f"Bearer {TOKEN}",
                                          "Content-Type": "application/json"},
                                 json=body)
                if r.status_code >= 400:
                    print(f"[whatsapp] send failed {r.status_code}: {r.text[:200]}",
                          flush=True)
                    sent_all = False
        except httpx.HTTPError as err:
            print(f"[whatsapp] send error: {type(err).__name__}: {err}", flush=True)
            sent_all = False
    return sent_all


# --- commands ---------------------------------------------------------------

HELP = """🐙 *Octopus on WhatsApp*

Send any task and the agent pool runs it on your machine:
_"draft a release note for v2 and list the migration risks"_

*Commands*
/help — this message
/status — providers, routing mode, web access
/models — what each role is bound to, per provider
/cost — what the last run spent
/mode auto|local|nvidia|gemini — pin where work goes
/web <query> — search the web, no model involved
/stop — cancel the run in progress

Anything not starting with / is treated as a task."""


def is_command(text: str) -> bool:
    return text.strip().startswith("/")


def command_of(text: str) -> tuple[str, str]:
    """('/mode local') -> ('mode', 'local')"""
    body = text.strip()[1:].strip()
    head, _, rest = body.partition(" ")
    return head.lower(), rest.strip()
