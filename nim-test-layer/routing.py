"""Which provider should do this particular piece of work.

The router exists because the two free providers fail in opposite directions. Local is
free, private, always entitled and never rate-limited, but it is CPU inference: a 7B
produces maybe ten tokens a second, so a long answer costs minutes. NIM is fast and far
larger, but it is a shared free tier with quotas, a catalog that lies, and a network
round trip. Neither is the right default for everything, and asking the user to pick per
task is asking them to do the router's job.

So each agent's subtask is weighed before it is bound. Light work — a short email, a
definition, a quick read on some numbers, the strict-JSON planning calls — goes local,
where it costs nothing and no quota is spent. Heavy work — architecture, migrations,
anything asking for depth or length — goes to NIM, where a 70B can actually do it. If
only one provider is up, everything goes there; that is the fallback, not a failure.

Scoring is deterministic keyword-and-length work on purpose. Asking a model which model
should answer would add a network round trip to every agent, which on a one-line
question costs more than simply answering it.
"""

from __future__ import annotations

import os
import re

# Above this, work is heavy enough to be worth a big remote model. Tuned so that plain
# prose and everyday analysis stay local — that is the common case and the one where
# local latency actually beats the round trip — while anything asking for depth leaves.
THRESHOLD = float(os.getenv("OCTOPUS_ROUTE_THRESHOLD", "0.55"))

# auto | local | nvidia. 'auto' is the point of this module; the other two are for
# pinning a run while debugging, or working offline on a plane.
MODE = os.getenv("OCTOPUS_ROUTE", "auto").strip().lower()

# Where each role sits before the subtask is read. Coding and analysis reward a bigger
# model more than prose does; scheduling emits a short strict-JSON object, which small
# models handle fine and produce fastest.
ROLE_WEIGHT: dict[str, float] = {
    "writer": 0.30,
    "analyst": 0.40,
    "scheduler": 0.25,
    "coder": 0.50,
    "imager": 1.00,   # no local image model exists here; see `choose`
}

# Words that mean the deliverable is substantial, not just that the topic is technical.
HEAVY = re.compile(
    r"comprehensive|in.depth|thorough|detailed|exhaustive|production.ready|production.grade|"
    r"architect|architecture|migration|end.to.end|full[- ]stack|whitepaper|rfc\b|design doc|"
    r"strategy|roadmap|distributed|scalab|threat model|security review|benchmark|"
    r"test suite|edge cases|trade.?off|enterprise|multi.tenant|step.by.step|"
    r"at least \d{2,}|\b\d{3,}\s*(?:words|lines)|every|all of the|robust"
)

# Words that mean the deliverable is small, whatever the subject matter is.
LIGHT = re.compile(
    r"\bquick\b|\bbrief\b|\bshort\b|one.liner|one line|single line|tl;?dr|\bsimple\b|"
    r"\bjust\b|rename|typo|grammar|spell|rephrase|reword|rewrite this|tweak|"
    r"headline|subject line|tweet|caption|slogan|tagline|\bname\b|\btitle\b|"
    r"what is|what's|define\b|definition of|convert|format|\blist\b|bullet"
)


def weigh(subtask: str, role: str) -> tuple[float, str]:
    """Score a subtask from 0 (trivial) to 1 (heavyweight), with the reason why.

    The reason is carried all the way to the UI. A router that silently sends work
    somewhere is impossible to trust or tune; one that says 'short prose task' is both.
    """
    text = (subtask or "").lower()
    words = len(text.split())
    score = ROLE_WEIGHT.get(role, 0.4)
    reasons: list[str] = []

    heavy_hits = len(HEAVY.findall(text))
    light_hits = len(LIGHT.findall(text))

    if heavy_hits:
        score += min(0.15 * heavy_hits, 0.40)
        reasons.append(f"{heavy_hits} depth cue{'s' if heavy_hits > 1 else ''}")
    if light_hits:
        score -= min(0.12 * light_hits, 0.30)
        reasons.append(f"{light_hits} brevity cue{'s' if light_hits > 1 else ''}")

    # Length of the *request* is a decent proxy for the size of the answer: a subtask
    # someone spent eighty words specifying is rarely satisfied by three sentences.
    if words <= 12:
        score -= 0.15
        reasons.append("one-line request")
    elif words >= 90:
        score += 0.20
        reasons.append("long, detailed request")
    elif words >= 45:
        score += 0.10
        reasons.append("multi-part request")

    # A numbered or bulleted list of requirements means several deliverables in one
    # subtask, which is exactly the shape small models drop half of.
    bullets = len(re.findall(r"(?m)^\s*(?:[-*•]|\d+[.)])\s+", subtask or ""))
    if bullets >= 3:
        score += 0.15
        reasons.append(f"{bullets} listed requirements")

    score = max(0.0, min(1.0, score))
    if not reasons:
        reasons.append(f"plain {role} work")
    return score, ", ".join(reasons)


def choose(role: str, subtask: str, usable: set[str], *,
           wants_image: bool = False, mode: str | None = None) -> tuple[str | None, str]:
    """Pick a provider for one agent. Returns (provider name, why) — or (None, why not).

    `usable` is the set of provider names that are up right now. Everything here degrades
    to it: the router expresses a preference, availability decides. A machine with Ollama
    stopped routes everything to NIM without noticing; a key that has run out of quota
    routes everything local. Neither case needs a code path of its own.
    """
    mode = (mode or MODE).strip().lower()

    if not usable:
        return None, "no provider is reachable"

    # Image generation is not a preference, it is a capability. Nothing local here
    # produces pixels, so an imager that is actually going to draw must go remote —
    # and if it cannot, the tentacle falls back to writing prompts, which is text work
    # and routes like anything else.
    if wants_image:
        if "nvidia" in usable:
            return "nvidia", "image generation — only NIM has image models"
        return None, "no image-capable provider is reachable"

    if mode in ("local", "nvidia"):
        if mode in usable:
            return mode, f"pinned to {mode} by OCTOPUS_ROUTE"
        other = next(iter(sorted(usable)))
        return other, f"{mode} pinned but unreachable — fell back to {other}"

    score, why = weigh(subtask, role)
    heavy = score >= THRESHOLD
    order = ["nvidia", "local"] if heavy else ["local", "nvidia"]
    # Providers the registry knows about but that this ordering does not name — a future
    # openai row, say — sit behind the two named ones rather than being unreachable.
    order += sorted(usable - set(order))

    for name in order:
        if name in usable:
            first = order[0]
            size = "heavy" if heavy else "light"
            if name == first:
                return name, f"{size} ({score:.2f}): {why}"
            return name, f"{size} ({score:.2f}): {why} — {first} unavailable"
    return None, "no provider is reachable"


def choose_planner(usable: set[str], mode: str | None = None) -> tuple[str | None, str]:
    """Provider for the planner and supervisor calls.

    Deliberately not weighed. Planning is the one call whose quality sets the shape of
    everything after it, and small models size a pool badly — they split a two-part task
    into eight, which then costs eight completions. So it takes the biggest thing
    available and only drops local when nothing else is up.
    """
    mode = (mode or MODE).strip().lower()
    if mode in ("local", "nvidia") and mode in usable:
        return mode, f"pinned to {mode}"
    for name in ("nvidia", "local"):
        if name in usable:
            return name, ("strongest available — planning quality sets the whole run"
                          if name == "nvidia" else "only provider up")
    return (next(iter(sorted(usable))), "only provider up") if usable else (None, "nothing up")
