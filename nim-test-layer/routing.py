"""Which provider should do this particular piece of work.

The router exists because the free providers fail in opposite directions. Local is free,
private, always entitled and never rate-limited, but it is CPU inference: a 7B produces
maybe ten tokens a second, so a long answer costs minutes. The hosted free tiers — NIM,
Gemini — are fast and far larger, but they are shared, quota'd, a network round trip
away, and in NIM's case advertise a catalog they cannot serve. Neither kind is the right
default for everything, and asking the user to pick per task is asking them to do the
router's job.

So each agent's subtask is weighed before it is bound. Light work — a short email, a
definition, a quick read on some numbers, the strict-JSON planning calls — goes local,
where it costs nothing and no quota is spent. Heavy work — architecture, migrations,
anything asking for depth or length — goes to a hosted model that can actually do it.
If only one provider is up, everything goes there; that is the fallback, not a failure.

Which hosted provider is not hardcoded: `rank()` orders whatever is reachable by what
each declares in the registry, so adding a key adds a destination and nothing else.

Scoring is deterministic keyword-and-length work on purpose. Asking a model which model
should answer would add a network round trip to every agent, which on a one-line
question costs more than simply answering it.
"""

from __future__ import annotations

import os
import re

import providers

# Above this, work is heavy enough to be worth a big remote model. Tuned so that plain
# prose and everyday analysis stay local — that is the common case and the one where
# local latency actually beats the round trip — while anything asking for depth leaves.
THRESHOLD = float(os.getenv("OCTOPUS_ROUTE_THRESHOLD", "0.55"))

# 'auto' is the point of this module. Any registered provider name also works as a pin,
# for working offline on a plane, or for comparing two of them on the same task.
MODE = os.getenv("OCTOPUS_ROUTE", "auto").strip().lower()

# Where each role sits before the subtask is read. Coding and analysis reward a bigger
# model more than prose does; scheduling emits a short strict-JSON object, which small
# models handle fine and produce fastest.
ROLE_WEIGHT: dict[str, float] = {
    "writer": 0.30,
    "analyst": 0.40,
    "scheduler": 0.25,
    "coder": 0.50,
    # Reads several thousand characters of fetched page before answering, so the prompt
    # is long even when the question is short — which is where a small model struggles.
    "researcher": 0.50,
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
    r"\bquick\b|\bbrief\b|\bshort\b|one[- ]?liner?|single[- ]?line|tl;?dr|\bsimple\b|"
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


def rank(usable: set[str], heavy: bool) -> list[str]:
    """Reachable providers, best first, for a decision of this weight.

    Derived from the registry rather than named here. This used to be the literal list
    ['nvidia', 'local'], which was fine while those were the only two but silently sent
    every later provider to the back — a free Gemini key would have ranked below a
    stopped Ollama. A provider now declares what kind of work it wants (`prefers`), what
    it charges, and how it ties against its peers (`priority`), and the router reads that.

    Order of tie-breaks: right size, then free before paid, then cheaper before dearer,
    then least-loaded so far, then declared priority.
    """
    def key(name: str) -> tuple[int, int, float, int, str]:
        p = providers.BY_NAME.get(name)
        pref = p.prefers if p else "any"
        want = "large" if heavy else "small"
        # Exact match first, then the undeclared middle, then the opposite end — which
        # is still reachable, because a wrong-sized provider beats no provider at all.
        klass = 0 if pref == want else (1 if pref == "any" else 2)
        # Then cost. Free before paid, and cheaper before dearer, so enabling a paid
        # provider adds a fallback rather than quietly becoming the default destination
        # for every heavy agent. Among today's providers this is a no-op — they are all
        # free — which is exactly why it has to be encoded rather than assumed.
        paid = 0 if (p is None or p.free_tier) else 1
        rate = p.usd_out if p else 0.0
        # Then spread. Two equally-suitable free tiers should share the work rather than
        # one being drained until it 429s while the other sits idle — the free capacity
        # on offer is the sum of them, not the larger of them. `priority` still decides
        # when the load is level, so the declared preference wins every tie it used to.
        return (klass, paid, rate, providers.load(name), p.priority if p else 50, name)

    return sorted(usable, key=key)


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

    if mode and mode != "auto":
        if mode in usable:
            return mode, f"pinned to {mode} by OCTOPUS_ROUTE"
        other = rank(usable, heavy=False)[0]
        return other, f"{mode} pinned but unreachable — fell back to {other}"

    score, why = weigh(subtask, role)
    heavy = score >= THRESHOLD
    order = rank(usable, heavy)

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
    if mode and mode != "auto" and mode in usable:
        return mode, f"pinned to {mode}"
    order = rank(usable, heavy=True)
    if not order:
        return None, "nothing up"
    return order[0], ("strongest available — planning quality sets the whole run"
                      if len(order) > 1 else "only provider up")
