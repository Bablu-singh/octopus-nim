"""NVIDIA NIM, as a named provider.

This module used to be the only way out of the process. It is now a thin facade over
`providers`, kept because its signatures are what the CLI self-test and the original
key-testing console are written against, and because 'the NIM client' is still a useful
thing to name even once it is one provider among several.

New code should call `providers` directly and address models as `nvidia:meta/llama-...`.
Everything here forwards, and any bare model id is read as a NIM id — which is what it
meant before providers existed, so old .env values keep working.
"""

from __future__ import annotations

import os
from typing import AsyncIterator

import httpx

import providers
from providers import ProviderError as NimError  # noqa: F401  (name kept for callers)
from providers import Timed, extract_text, key_fingerprint  # noqa: F401

NVIDIA = providers.get("nvidia")

BASE_URL = NVIDIA.base_url
DEFAULT_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")

TIMEOUT = NVIDIA.timeout
PROBE = NVIDIA.probe
CONNECT_TIMEOUT = NVIDIA.connect_timeout
READ_TIMEOUT = NVIDIA.read_timeout
PROBE_TIMEOUT = NVIDIA.probe_timeout


def resolve_key(override: str | None = None) -> str:
    if override and override.strip():
        return override.strip()
    return NVIDIA.key()


def _p(key: str | None) -> providers.Provider:
    """The NIM provider, carrying a caller-supplied key if there is one."""
    return providers.with_key(NVIDIA, key)


def transport_error(err: Exception, model: str | None = None) -> NimError:
    return providers.transport_error(err, NVIDIA, model)


async def list_models(key: str) -> Timed:
    return await providers.list_models(_p(key))


async def is_alive(key: str, model: str, client: httpx.AsyncClient | None = None) -> bool:
    return await providers.is_alive(_p(key), model, client)


async def chat(key: str, prompt: str, model: str = DEFAULT_MODEL, system: str | None = None,
               temperature: float = 0.2, max_tokens: int = 512) -> Timed:
    return await providers.chat(providers.get("nvidia").qualify(providers.bare(model)),
                                prompt, system, temperature, max_tokens, key=key)


async def chat_stream(key: str, prompt: str, model: str = DEFAULT_MODEL,
                      system: str | None = None, temperature: float = 0.2,
                      max_tokens: int = 512) -> AsyncIterator[str]:
    async for line in providers.chat_stream(
        providers.get("nvidia").qualify(providers.bare(model)),
        prompt, system, temperature, max_tokens, key=key,
    ):
        yield line
