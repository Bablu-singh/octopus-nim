"""WhatsApp's adapter onto the shared chat bridge.

Everything interesting — commands, streaming a dispatch into messages, the busy and stop
handling — lives in `chat_bridge`. All that is WhatsApp-specific is how a message is sent,
how long it may be, and how bold is spelled.
"""

from __future__ import annotations

import chat_bridge
import whatsapp

# Kept as a module-level alias so /api/whatsapp/health can report in-flight runs without
# reaching across into chat_bridge itself.
_runs = chat_bridge._runs


def transport(sender: str) -> chat_bridge.Transport:
    async def send(text: str) -> bool:
        return await whatsapp.send(sender, text)

    return chat_bridge.Transport(name="whatsapp", limit=whatsapp.MAX_BODY, send=send,
                                 bold="*", italic="_")


async def handle(msg: whatsapp.Inbound) -> None:
    """One inbound WhatsApp message, already authenticated and de-duplicated."""
    await chat_bridge.handle(f"whatsapp:{msg.sender}", msg.text, transport(msg.sender))
