#!/usr/bin/env python3
"""Record one real dispatch as a timestamped event log for the static Pages demo.

    python3 docs/record.py            # needs a working key in nim-test-layer/.env

Writes docs/demo.json: the catalog snapshot the page loads, plus every SSE event with the
millisecond it arrived, so the demo replays the run at its true pacing. Then rebuild the
page with docs/build.py.
"""

from __future__ import annotations

import asyncio
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "nim-test-layer"))

import octopus            # noqa: E402
import nim_client as nim  # noqa: E402

TASK = ("Launch our new payments product: assess the competitive risks, write the Stripe "
        "integration code, write the refund handler code, plan the four-week rollout, "
        "draft the launch announcement, and design a hero image for the landing page.")
OUT = ROOT / "docs" / "demo.json"

# Chunks arrive far faster than anyone can see. Each one costs ~45 bytes of JSON
# scaffolding, so coalescing them into buckets this wide cuts the file to a third with no
# visible difference on replay.
BUCKET_MS = 60


def compact(events: list) -> list:
    out, pending = [], {}

    def flush(aid: str) -> None:
        if aid in pending:
            t, txt = pending.pop(aid)
            out.append([t, {"event": "chunk", "id": aid, "text": txt}])

    for ms, e in events:
        if e["event"] == "chunk":
            aid = e["id"]
            if aid in pending and ms - pending[aid][0] < BUCKET_MS:
                pending[aid][1] += e["text"]
            else:
                flush(aid)
                pending[aid] = [ms, e["text"]]
        else:
            if e.get("id"):
                flush(e["id"])
            else:
                for aid in list(pending):
                    flush(aid)
            out.append([ms, e])
    for aid in list(pending):
        flush(aid)
    out.sort(key=lambda row: row[0])
    return out


async def main() -> int:
    try:
        key = nim.resolve_key()
    except nim.NimError as err:
        print(err.detail)
        return 1

    print("probing the catalog…")
    cat = await octopus.catalog(key)          # warmed here so it is not in the timeline
    print(f"  {cat['verified']} of {cat['total']} listed models answered")

    print("dispatching…")
    log, t0 = [], time.perf_counter()
    async for raw in octopus.dispatch(key, TASK):
        log.append([int((time.perf_counter() - t0) * 1000), json.loads(raw[5:])])

    events = compact(log)
    OUT.write_text(json.dumps(
        {"task": TASK, "catalog": cat, "events": events, "duration_ms": events[-1][0]},
        separators=(",", ":")))

    kinds: dict[str, int] = {}
    for _, e in events:
        kinds[e["event"]] = kinds.get(e["event"], 0) + 1
    print(f"  {kinds}")
    print(f"  {events[-1][0] / 1000:.0f}s of real time, {OUT.stat().st_size / 1024:.0f} KB")
    print("now run: python3 docs/build.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
