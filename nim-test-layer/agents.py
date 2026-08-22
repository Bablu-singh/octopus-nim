"""The octopus: five specialist roles, grown into as many parallel agents as a task needs.

A role is a template — a system prompt, a temperature, and a model preference list. It is
not an agent and it is not a limit. One dispatch can run six coders on six files and two
analysts on two datasets; the planner decides how many, and the supervisor adds more once
it sees what came back. `octopus.dispatch()` owns that lifecycle.


Nothing here hardcodes a model. `bind()` takes the catalog from your key plus the set of
models that were proven to answer, and walks each tentacle's preference list until it
finds one of those — so the same code works on a key with 102 models and a key with three.

Being listed is not the same as being served: most catalogued models 404, and some accept
a request and never reply. That is why `bind()` wants a verified set rather than trusting
the catalog; `octopus.catalog()` produces it.

Model ids arriving here are qualified — 'nvidia:meta/llama-3.3-70b-instruct',
'local:qwen2.5:7b', 'gemini:gemini-2.5-flash'. Every tentacle carries candidate names for
every provider in one list, because binding is always scoped to a provider the router has
already chosen (`bind(..., provider=...)`), so names belonging to the others cannot match
and their ordering relative to each other never matters.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import providers

# --- catalog classification -------------------------------------------------
# Ordered: the first pattern that matches an ID wins, so 'coder' beats generic 'chat'.

FAMILY_PATTERNS: list[tuple[str, str]] = [
    ("image", r"stable-diffusion|sdxl|flux|sana|consistory|kandinsky|image|shuttle"),
    ("embedding", r"embed|embedqa|nv-embed"),
    ("rerank", r"rerank"),
    ("vision", r"vila|neva|llava|kosmos|vision|maverick|scout|vlm|ocdr|florence"),
    ("code", r"coder|codestral|codegemma|starcoder|codellama|code-"),
    ("reasoning", r"deepseek-r1|nemotron|qwq|magistral|reasoning|thinker|phi-4-reasoning"),
    ("speech", r"whisper|parakeet|riva|tts|asr|canary"),
    ("chat", r".*"),
]


# Models that answer /chat/completions but are not general-purpose assistants: safety
# classifiers, reward models, retrievers, OCR/parse and translation heads. They look like
# 'chat' to the patterns above, so a fallback would happily bind one and get back a
# one-word verdict instead of an answer. Never auto-bind these.
UTILITY_PATTERN = re.compile(
    r"guard|safety|content-safety|topic-control|reward|embed|rerank|retriev|"
    r"parse|deplot|nvclip|translate|detector|calibration|"
    # Gemini lists music, image, video and specialised agent heads alongside its chat
    # models. They answer /models but are not assistants, and a fallback would happily
    # bind 'lyria-3-pro' to the analyst and get back silence.
    r"lyria|imagen|veo-|nano-banana|antigravity|deep-research|robotics|"
    r"computer-use|native-audio|live-preview|-tts|aqa"
)


def family_of(model_id: str) -> str:
    lowered = model_id.lower()
    for family, pattern in FAMILY_PATTERNS:
        if re.search(pattern, lowered):
            return family
    return "chat"


def is_utility(model_id: str) -> bool:
    """True for classifiers and other single-purpose heads that must not act as agents."""
    return bool(UTILITY_PATTERN.search(model_id.lower()))


def classify(model_ids: list[str]) -> dict[str, list[str]]:
    """Group every model your key can reach into families."""
    grouped: dict[str, list[str]] = {}
    for mid in sorted(model_ids):
        grouped.setdefault(family_of(mid), []).append(mid)
    return grouped


def for_provider(model_ids: list[str], name: str | None) -> list[str]:
    """Narrow a qualified-id pool to one provider. `None` means no narrowing."""
    if not name:
        return list(model_ids)
    return [m for m in model_ids if providers.provider_of(m) == name]


# --- tentacles --------------------------------------------------------------


@dataclass
class Tentacle:
    id: str
    name: str
    color: str
    blurb: str
    keywords: list[str]
    candidates: list[str]          # substrings, best first
    fallback_families: list[str]   # if no candidate matches, take the first model from these
    system: str
    temperature: float = 0.4
    max_tokens: int = 1400
    kind: str = "chat"             # 'chat' or 'image'


TENTACLES: list[Tentacle] = [
    Tentacle(
        id="writer",
        name="Writer",
        color="#f2b56b",
        blurb="Drafts, edits, and rewrites prose",
        keywords=["write", "draft", "email", "blog", "post", "summar", "rewrite", "edit",
                  "essay", "copy", "story", "letter", "announce", "release note"],
        candidates=["palmyra-creative", "gpt-oss-120b", "nemotron-3-super", "step-3.7",
                    "muse-glimmer", "inkling", "kimi-k2", "glm-5", "llama-3.3-70b",
                    "llama-3.1-405b", "llama-3.1-70b", "mixtral-8x22b", "nemotron-4-340b",
                    "qwen2.5-72b", "llama-3.1-8b",
                    # Gemini: the '-latest' aliases first, because Google retires dated
                    # names — 'gemini-2.0-flash' already 404s pointing at its successor.
                    # Flash before pro: the free tier gives flash several times the pro
                    # quota, and on these agents being throttled costs far more than the
                    # quality difference is worth.
                    "gemini-flash-latest", "gemini-pro-latest", "gemini-2.5-flash",
                    # Local. aya-expanse first: it covers 23 languages properly, where
                    # qwen2.5 writes decent English and thin Hindi. Only reachable when
                    # the pool is scoped to local, so this ordering costs nothing hosted.
                    "aya-expanse", "aya", "gemma2", "qwen2.5", "llama3.2", "mistral", "phi"],
        fallback_families=["chat", "reasoning"],
        system=("You are a writing specialist. Produce finished prose, not an outline of prose. "
                "Match the register the request implies. No preamble, no 'here is'; open with the "
                "work itself. Use Markdown only where structure genuinely helps."),
        temperature=0.7,
    ),
    Tentacle(
        id="coder",
        name="Coder",
        color="#5fd8d1",
        blurb="Writes, reviews, and debugs code",
        keywords=["code", "function", "bug", "refactor", "api", "script", "sql", "regex",
                  "python", "java", "react", "test", "deploy", "jenkins", "pipeline", "error",
                  "stack trace", "class", "endpoint"],
        candidates=["qwen3-coder", "qwen2.5-coder-32b", "codestral", "deepseek-coder",
                    "laguna", "codellama-70b", "granite-34b-code", "starcoder2",
                    "gpt-oss-120b", "nemotron-3-super", "deepseek-r1", "llama-3.3-70b",
                    "gemini-flash-latest", "gemini-pro-latest", "gemini-2.5-flash",
                    # Local: a pulled coder model wins if there is one, else the general
                    # 7B, which handles a short function better than the 3B does.
                    "qwen2.5-coder", "deepseek-coder-v2", "codellama", "qwen2.5", "llama3.2"],
        fallback_families=["code", "reasoning", "chat"],
        system=("You are a software engineer. Give working code first, then a short note on the "
                "non-obvious parts only. Always name the language in the fence. State assumptions "
                "about versions or environment explicitly rather than guessing silently."),
        temperature=0.15,
        max_tokens=2000,
    ),
    Tentacle(
        id="scheduler",
        name="Scheduler",
        color="#a98cff",
        blurb="Turns intent into a dated, sequenced plan",
        keywords=["schedule", "plan", "calendar", "sprint", "deadline", "timeline", "roadmap",
                  "meeting", "milestone", "agenda", "when", "sequence", "release", "cutover"],
        # Strict JSON out, so plain instruction-following models beat reasoning models
        # here — the latter wrap their answer in <think> and break the parse.
        candidates=["llama-3.1-8b", "gpt-oss-20b", "step-3.7-flash", "llama-3.3-70b",
                    "llama-3.1-70b", "mistral-large", "nemotron-3.5-lightning",
                    "nemotron-mini",
                    # Gemini: flash over pro here for the same reason — this wants a
                    # short strict-JSON object, not reasoning narrated before the answer.
                    "gemini-flash-lite-latest", "gemini-flash-latest", "gemini-2.5-flash",
                    # Local: same reasoning as above — a small instruction-follower emits
                    # cleaner JSON here than a bigger model that likes to explain itself.
                    "llama3.2", "qwen2.5", "mistral"],
        fallback_families=["chat", "reasoning"],
        system=("You are a planning specialist. Respond with ONLY a JSON object, no prose and no "
                "Markdown fences, shaped: "
                '{"title": str, "assumptions": [str], "items": [{"when": str, "what": str, '
                '"duration": str, "depends_on": str|null, "owner": str|null}]}. '
                "Use relative dates ('Day 1', 'Week 2') unless the request gives real ones. "
                "Put anything you had to guess in assumptions."),
        temperature=0.2,
    ),
    Tentacle(
        id="imager",
        name="Imager",
        color="#ff8a7a",
        blurb="Generates images, or writes the prompt if no image model is entitled",
        keywords=["image", "picture", "logo", "illustration", "diagram", "poster", "banner",
                  "photo", "render", "visual", "icon", "thumbnail", "mockup", "art"],
        candidates=["flux.1-dev", "flux", "stable-diffusion-3-5-large", "stable-diffusion-3",
                    "sdxl", "sana", "shuttle", "consistory", "stable-diffusion"],
        fallback_families=["image"],
        # Only reached when no image model is entitled — an entitled one goes straight to
        # the genai endpoint and never sees a system prompt.
        system=("You are an art director. The account has no image model entitled, so instead of "
                "an image, deliver a production-ready generation prompt: subject, composition, "
                "lighting, palette, lens or medium, and a negative prompt. Then name two models "
                "on build.nvidia.com that would suit it."),
        temperature=0.8,
        kind="image",
    ),
    Tentacle(
        id="analyst",
        name="Analyst",
        color="#9fe86a",
        blurb="Reasons over data, compares options, finds the flaw",
        keywords=["analy", "compare", "evaluate", "why", "risk", "trade-off", "tradeoff", "data",
                  "metric", "root cause", "assess", "review", "pros and cons", "decide",
                  "estimate", "forecast", "investigate"],
        candidates=["nemotron-3-ultra", "deepseek-r1", "qwq-32b", "nemotron-ultra",
                    "llama-3.3-nemotron-super-49b", "gpt-oss-120b", "nemotron-3-super",
                    "nemotron-70b", "magistral", "llama-3.3-70b",
                    "gemini-flash-latest", "gemini-pro-latest", "gemini-2.5-flash",
                    # Local: the 7B first — analysis is where the 3B's ceiling shows, and
                    # aya ahead of it for anything not in English.
                    "aya-expanse", "qwen2.5", "gemma2", "llama3.2", "mistral"],
        fallback_families=["reasoning", "chat"],
        system=("You are an analyst. Lead with the conclusion, then the reasoning that supports it. "
                "Quantify where the input allows and say plainly where it does not. Name the "
                "strongest objection to your own conclusion before you finish."),
        temperature=0.3,
        max_tokens=1800,
    ),
    Tentacle(
        id="researcher",
        name="Researcher",
        color="#6ec8ff",
        blurb="Looks things up on the live web and answers with sources",
        keywords=["latest", "current", "today", "recent", "news", "look up", "find out",
                  "search", "who is", "what happened", "price", "pricing", "release",
                  "changelog", "documentation", "docs", "version", "compare", "up to date",
                  "this year", "right now", "trend", "announcement"],
        candidates=["nemotron-3-super", "gpt-oss-120b", "llama-3.3-70b", "mistral-large",
                    "nemotron-3-ultra", "llama-3.1-70b", "llama-3.1-8b",
                    "gemini-flash-latest", "gemini-pro-latest", "gemini-2.5-flash",
                    # Local: this role reads several thousand characters of fetched page
                    # before it answers, so the 7B rather than the 3B — and aya first,
                    # since a fetched page is as likely to be Hindi as English.
                    "aya-expanse", "qwen2.5", "gemma2", "mistral", "llama3.2"],
        fallback_families=["chat", "reasoning"],
        system=("You are a research specialist. You are given excerpts from web pages that "
                "were fetched moments ago. Answer strictly from them: quote figures as they "
                "appear, attribute every claim to the source URL it came from, and say "
                "plainly when the sources do not answer the question rather than filling "
                "the gap from memory. Note disagreement between sources rather than "
                "silently picking one. End with a 'Sources' list of the URLs you used."),
        temperature=0.2,
        max_tokens=1800,
    ),
]

BY_ID = {t.id: t for t in TENTACLES}

# Every model name any tentacle asks for, in order. Used to rank generic fallbacks: a
# curated name beats whatever happens to sort first alphabetically.
CURATED: list[str] = [c for t in TENTACLES for c in t.candidates]


def curation_rank(model_id: str) -> int:
    lowered = model_id.lower()
    return next((i for i, c in enumerate(CURATED) if c in lowered), len(CURATED))


def shortlist(t: Tentacle, model_ids: list[str], limit: int = 8) -> list[tuple[str, str]]:
    """Every model this tentacle would accept, best first, as (model_id, how) pairs.

    Preferred candidates come first, then whatever the fallback families offer. Callers
    walk this in order and take the first model that proves it is actually serving.
    """
    grouped = classify(model_ids)
    seen: set[str] = set()
    out: list[tuple[str, str]] = []

    def add(mid: str, how: str) -> None:
        if mid and mid not in seen:
            seen.add(mid)
            out.append((mid, how))

    for candidate in t.candidates:
        # Shortest match wins: 'llama-3.1-8b' should prefer the plain instruct model over
        # any longer derivative that happens to embed the same substring.
        for mid in sorted((m for m in model_ids if candidate in m.lower()), key=len):
            add(mid, "preferred")

    def ranked(family: str) -> list[str]:
        # Curated names first, then alphabetical. Family lists arrive alphabetical, and
        # since the shortlist is capped, an unranked tail would spend the whole budget on
        # whatever sorts early rather than on models we actually want.
        return sorted((m for m in grouped.get(family, []) if not is_utility(m)),
                      key=lambda m: (curation_rank(m), m))

    for fam in t.fallback_families:
        for mid in ranked(fam):
            add(mid, f"fallback from {fam}")

    # An imager with no image model still works — it writes prompts instead.
    if t.kind == "image":
        for mid in ranked("chat"):
            add(mid, "no image model entitled — writing prompts instead")

    return out[:limit]


def bind(model_ids: list[str], live: set[str] | None = None,
         provider: str | None = None) -> list[dict]:
    """Attach a model to every tentacle. Returns a report of what got bound and how.

    `live` is the set of models confirmed to actually answer. Pass it whenever you can:
    the catalog happily lists models that 404, and worse, models that accept a request
    and never reply. Without it this falls back to trusting the catalog.

    `provider` narrows the pool to one place before any of that happens. The router
    decides where an agent's work should go, and this decides which model there does it —
    keeping the two separate is what stops a 'route this locally' decision from quietly
    binding to a hosted model because the hosted one ranked higher.
    """
    report = []
    model_ids = for_provider(model_ids, provider)
    grouped = classify(model_ids)
    for t in TENTACLES:
        chosen, how = None, "nothing entitled"
        for mid, why in shortlist(t, model_ids):
            if live is None or mid in live:
                chosen, how = mid, why
                break
        if chosen and live is not None and how == "preferred":
            how = "preferred, verified"

        # The shortlist is capped, and its fallback tail is alphabetical rather than
        # ranked — so a tentacle can run out of options while perfectly good models sit
        # in the verified set. Draw on that directly before giving up.
        if not chosen and live:
            families = t.fallback_families + (["chat", "reasoning"] if t.kind == "image" else [])
            for fam in families:
                pool = sorted((m for m in grouped.get(fam, []) if m in live and not is_utility(m)),
                              key=lambda m: (curation_rank(m), m))
                if pool:
                    chosen, how = pool[0], f"any working {fam} model"
                    break
        report.append({
            "id": t.id, "name": t.name, "color": t.color, "blurb": t.blurb,
            "model": chosen, "bound_via": how, "kind": t.kind,
            "provider": providers.provider_of(chosen) if chosen else provider,
            "ready": chosen is not None,
            "generates_images": t.kind == "image" and chosen is not None
                                and family_of(chosen) == "image",
        })
    return report


# --- planning ---------------------------------------------------------------

ROLE_MENU = ("writer (prose), coder (code), scheduler (dated plans), "
             "imager (visuals), analyst (reasoning and analysis), "
             "researcher (looks it up on the live web and cites sources)")

SCHEMA = ('{"agents": [{"role": str, "label": str, "subtask": str}], "why": str}')

ROLE_RULE = ('"role" must be exactly one of these six strings: "writer", "coder", '
             '"scheduler", "imager", "analyst", "researcher". Never invent another role '
             'name; pick the closest of the six. Anything reasoned out or worked out from '
             'what you already know is "analyst"; anything that needs a current fact — '
             'a price, a version, a release date, what happened recently, what a specific '
             'page says — is "researcher", because only that role can read the internet.')


def planner_system(budget: int) -> str:
    """Decompose a task into as many parallel agents as it genuinely warrants."""
    return (
        f"You decompose a task into independent work items for specialist agents that all "
        f"run at the same time. Roles: {ROLE_MENU}. "
        f"Respond with ONLY JSON, no prose and no Markdown fences, shaped: {SCHEMA}. "
        f"{ROLE_RULE} "
        f"Create one agent per genuinely separable piece of work — as many as the task "
        f"needs, up to {budget}. Reuse a role as often as you like: three coders on three "
        f"different modules is correct and expected. "
        f"Size the pool by counting the distinct deliverables the task actually asks for, "
        f"and create exactly that many agents. 'What is 12 percent of 340?' asks for one "
        f"answer — that is one agent, not three. 'Assess the risks, write the migration "
        f"code, plan the cutover and draft the announcement' asks for four things — that "
        f"is four agents. Never split one deliverable into think/do/present stages: a "
        f"single agent produces its whole piece by itself. Always return at least one. "
        f"'label' is two to four words naming that agent's slice, "
        f"unique across agents. Each 'subtask' must stand alone and carry its own context — "
        f"an agent sees only its own subtask, never the task or the other agents. Never "
        f"create an agent whose work needs another agent's output first; they are "
        f"concurrent, not sequential."
    )


def supervisor_system(budget: int) -> str:
    """Decide, having seen the first results, whether the task needs more agents."""
    return (
        f"You supervise a pool of parallel agents working one task. You are given the task "
        f"and a digest of what has been produced so far. Decide whether more agents are "
        f"needed: gaps left open, pieces the finished work revealed, depth the task asks "
        f"for and has not got. Roles: {ROLE_MENU}. "
        f"Respond with ONLY JSON, no prose and no Markdown fences, shaped: {SCHEMA}. "
        f"{ROLE_RULE} "
        f"Digest entries are excerpts: one ending in '[EXCERPT ENDS]' was cut for length "
        f"and its agent finished in full. Never treat a truncated entry as unfinished work. "
        f"Only an entry explicitly marked FAILED did not complete. "
        f"Apply this test first: if every entry in the digest has real content and none is "
        f'marked FAILED, the task is covered — answer {{"agents": [], "why": "covered"}} '
        f"and nothing else. Add an agent only for a distinct part of the original task that "
        f"no entry addresses at all, or to redo an entry marked FAILED. Never add an agent "
        f"to verify, check, recompute, review, polish or summarise another agent's answer, "
        f"and never add one merely because budget remains. Up to {budget} agents. "
        f"Each subtask must stand alone; the new agents run concurrently and cannot see "
        f"each other."
    )


# Planning is the one call whose quality sets the shape of everything after it, so it gets
# the strongest model available rather than the cheapest. Small models answer fast and size
# the pool badly — they split a two-part task into eight. Ordered best first; anything here
# must emit strict JSON, which rules out models that narrate their reasoning first.
PLANNER_PREFERENCE = [
    "nemotron-3-super", "gpt-oss-120b", "llama-3.3-70b", "mistral-large",
    "nemotron-3.5-lightning", "step-3.7-flash", "gpt-oss-20b", "nemotron-3-ultra",
    "llama-3.1-70b", "llama-3.1-8b",
    "gemini-flash-latest", "gemini-pro-latest", "gemini-2.5-flash",
    # Local, for when nothing hosted is reachable. The 7B first: the 3B plans a
    # two-part task into six agents, and every one of those costs a completion.
    "qwen2.5", "mistral", "llama3.2",
]


def pick_planner(live: set[str], provider: str | None = None) -> str | None:
    """Best available model for planning and supervision, or None if nothing is usable."""
    pool = set(for_provider(sorted(live), provider))
    for pref in PLANNER_PREFERENCE:
        matches = sorted((m for m in pool if pref in m.lower()), key=len)
        if matches:
            return matches[0]
    return next((m for m in sorted(pool) if not is_utility(m)), None)


def role_for_text(text: str) -> str:
    """Best-matching role for free text.

    Planners cheerfully invent roles that are not on the menu ('calculator', 'researcher').
    Dropping those agents loses real work, so an invented role gets mapped onto the closest
    real one by keyword instead. Analyst is the default: it is the most general of the five.
    """
    lowered = text.lower()
    best_score, best_id = 0, "analyst"
    for t in TENTACLES:
        score = sum(1 for k in t.keywords if k in lowered)
        if score > best_score:
            best_score, best_id = score, t.id
    return best_id


def keyword_plan(task: str, budget: int) -> list[dict]:
    """Deterministic fallback when the planner model is unavailable or returns junk.

    Scales with the task the only way keywords can: every role the text actually implicates
    gets an agent, rather than a fixed top-two.
    """
    lowered = task.lower()
    scored = [(sum(1 for k in t.keywords if k in lowered), t) for t in TENTACLES]
    picked = [t for score, t in sorted(scored, key=lambda s: -s[0]) if score > 0][:budget]
    return [{"role": t.id, "label": t.name.lower(), "subtask": task}
            for t in (picked or [BY_ID["writer"]])]
