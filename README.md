# Octopus

A local multi-agent console. Describe a task; it works out what kind of work it is,
splits it into as many agents as the task warrants, runs them in parallel, and decides
for each one whether it should run **on your machine** or **on a hosted free tier**.

Drive it from a browser, from **Discord**, or from **WhatsApp** — by typing or by voice.

**[Live demo →](https://bablu-singh.github.io/octopus-nim/)** — a recorded dispatch,
replayed. GitHub Pages cannot run the Python backend, so the demo replays a real run
captured from the live API. Everything else on that page is the app itself.

**[Architecture →](docs/ARCHITECTURE.md)** — module map, routing logic, dispatch lifecycle.

## What it does

You never pick an agent type. A planner reads the task and sizes the pool to it:

| Task | Result |
|---|---|
| "What is 12 percent of 340?" | 1 agent, 1 wave |
| "Write a function that validates IBANs, with unit tests." | 2 coders |
| A migration naming 7 deliverables | 7 agents across 4 roles |
| "What does Ollama's changelog say about structured outputs?" | 1 researcher, reading the live web |

There are six **roles** — writer, coder, scheduler, imager, analyst, researcher — but
they are templates, not a roster. A role is instantiated as often as the work divides, so
three coders on three modules is normal. Nothing caps the pool at six.

Each wave runs concurrently. When it finishes, a supervisor reads what came back and adds
agents for anything still uncovered, repeating until it says stop or a budget runs out —
so the pool grows with the task rather than being fixed when you press Dispatch.

## Local and hosted, decided per agent

The providers fail in opposite directions. Local is free, private, offline and never
rate-limited, but it is CPU inference — a 7B produces maybe ten tokens a second. The
hosted free tiers (NIM, Gemini) are fast and far larger, but they are shared, quota'd and
a network round trip away. Neither kind is the right default for everything.

So every agent's subtask is weighed before a model is bound to it. Light work — a short
email, a definition, a quick read on some numbers — stays on your machine. Heavy work —
architecture, migrations, anything asking for depth or length — goes to a hosted model.
One dispatch routinely splits across both:

```
WAVE 1
  writer-1    local    qwen2.5:7b                       light (0.18): 1 brevity cue
  analyst-1   nvidia   llama-3.3-nemotron-super-49b-v1  heavy (0.80): 5 depth cues
```

Scoring is deterministic keyword-and-length work, not a model call: asking a model which
model should answer would add a round trip to every agent, which on a one-line question
costs more than simply answering it.

**Availability always beats preference.** Stop Ollama and everything goes to a hosted
provider. Delete every key and everything runs locally, offline, with no account at all.
Neither case has a code path of its own — they fall out of which providers are reachable.

Pin a whole run with `OCTOPUS_ROUTE=local`, `=nvidia` or `=gemini`, or dry-run the
decision for any subtask with `POST /api/route` to see the score before you touch the
threshold.

## Providers

| Provider | Cost | Needs | Status |
|---|---|---|---|
| **local** (Ollama) | free, offline | nothing | on |
| **nvidia** (NIM) | free tier | `NVIDIA_API_KEY` | on when the key is present |
| **gemini** | free tier | `GEMINI_API_KEY` | on when the key is present |
| openai | paid | key + `ENABLE_OPENAI=1` | off |
| anthropic | paid | key + an adapter | off |

All three live providers are free, which is the constraint this project is built to.
Drop a Gemini key from [aistudio.google.com](https://aistudio.google.com/apikey) into
`.env` and it joins the rotation on the next restart — no code change, because Google
publishes an OpenAI-compatible surface and the router derives its ordering from the
registry rather than from hardcoded names.

OpenAI and Anthropic are registered and **deliberately disabled** — they are paid.
Enabling OpenAI needs only a key and a flag. Anthropic's `/messages` differs in request
shape, streaming events and auth header, so it also needs an adapter at the documented
seam in `providers.py`.

A provider is a base URL, an optional key, and a wire format. Models are addressed as
`provider:model` everywhere above `providers.py`, so adding one is a row in a registry —
not a refactor of the octopus, the tentacles, or the UI.

## It can read the internet

A model's knowledge stops at its training cut-off. The **researcher** role closes that
gap — it searches the live web, reads the top results and answers with citations — and any
agent gets a page fetched for it when its subtask contains a URL.

Keyless, like the rest of the stack: DuckDuckGo's lite endpoint for search, direct HTTP
for pages, a small built-in HTML-to-text extractor. Throttled, size-capped and cached.

Two things matter more than the plumbing:

**Fetched content is untrusted.** A page can contain text aimed at the model reading it.
Every excerpt is fenced between `<<<UNTRUSTED_WEB_CONTENT>>>` markers, and the agent's
system prompt says that nothing inside them is an instruction — it is evidence to cite,
and anything directive-shaped should be reported as suspicious rather than obeyed. That
does not make injection impossible; it removes the easy version and keeps the boundary
visible.

**It will not read your machine.** Fetches are checked against the *resolved* address, so
loopback, private ranges and the cloud metadata endpoint are refused even behind a public
hostname.

## Voice

Speak a task, hear the answer, **in English or Hindi**. Kokoro for speech (Apache-2.0, 54
voices, sounds like a person) and faster-whisper for listening — both local, both on CPU,
no key, no API, nothing leaves the machine. Devanagari picks the Hindi voice on its own. Send a WhatsApp voice note and get one back; same on Discord; in the browser a 🎤
records and every answer has a 🔊.

`/voice auto` (the default) speaks only when you spoke. Audio always accompanies the text
rather than replacing it, since speech cannot be skimmed or copied.

## Drive it from your phone

Two chat front doors, both reaching the same agent pool. **Discord is the one to reach
for**: its gateway is an outbound WebSocket, so there is no public URL, no tunnel, no
webhook signature and no inbound port — the app dials out. Setup is a bot token and an
invite link, and it works from the Discord mobile app.

WhatsApp is supported two ways, neither as clean:

- **Cloud API** — official, no account risk, but needs a Meta app, a `cloudflared` tunnel,
  a signed webhook and a 24-hour messaging window.
- **QR scan** — no Meta account at all, links like WhatsApp Web. But it drives an
  unofficial client, which **violates WhatsApp's terms and can get the phone number
  banned**. Off by default behind two separate switches. Use a spare number.

```
you  ▸ draft a release note for v2 and list the migration risks
bot  ◂ Wave 1 — 2 agent(s) in parallel
       • ReleaseNote — local / qwen2.5:7b
       • MigrationRisks — nvidia / nemotron-3-super-120b
     ◂ ✅ ReleaseNote …
     ◂ 🐙 Done — 2 agents, 1 wave, 2 ok · ~3.1k tokens · free
```

`/status` `/models` `/cost` `/mode` `/web` `/queue` `/history` `/stop` `/new` — anything
else is a task.

Tasks **queue** rather than being refused while one runs, and each one **remembers the
turns before it**, so "now make that shorter" resolves against what was actually produced.
Both last until `/new`.

Both doors are peers of the browser UI: all three call `octopus.dispatch()` directly, and
one shared bridge turns dispatch events into messages, so a platform differs only in its
message limit, how bold is spelled, and how a message is sent. Neither can drift from the
other.

Neither answers anyone by default. An empty allowlist means the door connects and ignores
everyone — deliberately useless rather than deliberately open.

Setup for both is in [nim-test-layer/README.md](nim-test-layer/README.md).

## Cost, quota and switching a provider off

Everything reachable today is free, so "cost" here means quota and wall-clock rather
than money — and the cheapest call is the one that never happens:

- **Trivial tasks skip the planner.** A one-line question does not need a planning
  completion to establish that it is one agent's work. Below a weight of 0.20 the first
  wave is planned by keyword, saving a whole call on exactly the runs where that call was
  the largest share of the cost.
- **Work already done is never redone**, a failed slice gets exactly one retry, and a
  light task never grows past one wave.
- **Token caps per provider**, and an optional hard ceiling for a whole dispatch
  (`OCTOPUS_TOKEN_BUDGET`).
- **Every run reports what it spent** — tokens per provider, and money only if something
  billable was actually involved. Counts are estimates: agents stream, and streaming
  responses carry no usage block.
- **Free before paid** in the routing order, so enabling a paid provider later adds a
  fallback rather than quietly becoming the default.

**Rate limits are respected rather than discovered.** Each provider is gated to a
requests-per-minute figure set below its published ceiling, spaced evenly rather than
burst-and-wait — a burst of eight agents is the shape that trips a per-minute quota. If a
`429` arrives anyway, the provider is stood down for as long as it asked for
(`Retry-After`, or the delay in the error body), and the router simply stops offering it
until it recovers. A `429` never marks a model dead: it means we asked too often, not that
the model is broken.

**If a free tier stops being free**, set `ENABLE_NVIDIA=0` (or `ENABLE_GEMINI=0`,
`ENABLE_LOCAL=0`). That provider leaves the pool and everything routes to what is left —
no code change, nothing to uninstall, and one restart to undo. Availability was always the
only thing the router trusted, so this needed no new machinery:

```
ENABLE_NVIDIA=0
  nvidia   disabled       Switched off — set ENABLE_NVIDIA=1 to use it.
  heavy work  -> gemini
  light work  -> local
```

## The part that makes it work

`GET /v1/models` advertises far more than a key can run. On the key this was built
against, **102 models were listed and 6 actually served completions** — most return 404,
and several accept a request and then never reply, which surfaces as a read timeout
minutes later.

So nothing is bound on the strength of the catalog. Every NIM candidate is probed with a
tiny completion first, and only models that return real content are used. A model that
fails mid-run is dropped and its role re-bound to a different one.

Local is the opposite case and is treated as such: Ollama lists only what is already on
disk, so probing it would add minutes to a catalog read to confirm something already
known. Those models are trusted on sight and dropped the normal way if they ever fail.

## Run it

**Local models** (optional but recommended — this is what makes it work with no key):

```bash
winget install Ollama.Ollama
```

```bash
ollama pull llama3.2:3b && ollama pull qwen2.5:7b
```

**The app** — macOS / Linux:

```bash
cd nim-test-layer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app:app --reload --port 8000
```

Windows (PowerShell):

```bash
cd nim-test-layer
python -m venv .venv; .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
python -m uvicorn app:app --reload --port 8000
```

Open <http://127.0.0.1:8000>. Full documentation, API reference and tuning knobs are in
[nim-test-layer/README.md](nim-test-layer/README.md).

A NIM key is now **optional**: with Ollama running the app works with no key and no
network. Add one to `.env` to unlock the big models for heavy work. Your key stays in
`.env` on your machine — it is gitignored, never sent to the browser, and only ever
fingerprinted in logs.

## Rebuilding the demo

```bash
python3 docs/record.py     # capture a fresh dispatch (needs a working key)
python3 docs/build.py      # regenerate docs/index.html from the app's own UI
```

`docs/index.html` is generated, not hand-maintained: it is `static/octopus.html` plus a
banner and a shim that replays `docs/demo.json` in place of the API.
