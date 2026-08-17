"""Terminal smoke test — same checks as the console, no server needed.

Usage:  python cli_test.py [prompt]
Exits 0 if every check passes, 1 otherwise (safe to drop into CI).
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import nim_client as nim  # loads .env at import time
import octopus

ENV_FILE = Path(__file__).parent / ".env"

PASS, FAIL = "\033[92m✓\033[0m", "\033[91m✗\033[0m"


def line(ok: bool, name: str, note: str, ms: int | None = None) -> None:
    tail = f"  [{ms} ms]" if ms is not None else ""
    print(f"{PASS if ok else FAIL} {name:<24} {note}{tail}")


async def main() -> int:
    prompt = " ".join(sys.argv[1:]) or "Reply with the single word: pong"
    failures = 0

    try:
        key = nim.resolve_key()
        line(True, "Key loaded", nim.key_fingerprint(key))
    except nim.NimError as err:
        line(False, "Key loaded", err.detail)
        print(f"\n  running:  {Path(__file__).resolve()}")
        print(f"  .env:     {ENV_FILE}  [{'found' if ENV_FILE.exists() else 'MISSING'}]")
        if ENV_FILE.exists():
            keys = [ln.split("=", 1)[0].strip() for ln in ENV_FILE.read_text().splitlines()
                    if "=" in ln and not ln.strip().startswith("#")]
            print(f"  defines:  {keys or 'nothing — every line is commented out'}")
        return 1

    try:
        listed = await nim.list_models(key)
        ids = [m.get("id", "") for m in listed.data.get("data", [])]
        line(True, "Key authenticates", f"{len(ids)} models listed", listed.latency_ms)
    except nim.NimError as err:
        line(False, "Key authenticates", err.detail)
        return 1

    # Listed is not the same as served — most of these 404 and a few hang forever, so
    # find one that actually answers rather than trusting the catalog's first entry.
    try:
        cat = await octopus.catalog(key)
        working = [t["model"] for t in cat["tentacles"] if t["model"]]
        line(bool(working), "Models actually serving",
             f"{cat['verified']} of {cat['total']} listed models answered")
        failures += 0 if working else 1
    except nim.NimError as err:
        line(False, "Models actually serving", err.detail)
        working = []
        failures += 1

    model = (nim.DEFAULT_MODEL if nim.DEFAULT_MODEL in working
             else (working[0] if working else nim.DEFAULT_MODEL))

    try:
        answer = await nim.chat(key, prompt, model, max_tokens=128)
        text = nim.extract_text(answer.data).strip()
        line(bool(text), "Completion round trip", f"{model} → {text[:70] or 'empty response'}", answer.latency_ms)
        failures += 0 if text else 1
    except nim.NimError as err:
        line(False, "Completion round trip", err.detail)
        failures += 1

    try:
        started = time.perf_counter()
        chunks, first = 0, None
        async for raw in nim.chat_stream(key, "Count from one to five.", model, max_tokens=48):
            if raw.startswith("data:") and "[DONE]" not in raw:
                chunks += 1
                if first is None:
                    first = int((time.perf_counter() - started) * 1000)
        line(chunks > 0, "Streaming", f"{chunks} chunks, first token in {first} ms",
             int((time.perf_counter() - started) * 1000))
        failures += 0 if chunks else 1
    except nim.NimError as err:
        line(False, "Streaming", err.detail)
        failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
