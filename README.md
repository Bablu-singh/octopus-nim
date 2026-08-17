# Octopus

A local multi-agent console. Describe a task; it works out what kind of work it is,
splits it into as many agents as the task warrants, runs them in parallel, and decides
for each one whether it should run **on your machine** or **on NVIDIA NIM**.

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

There are five **roles** — writer, coder, scheduler, imager, analyst — but they are
templates, not a roster. A role is instantiated as often as the work divides, so three
coders on three modules is normal. Nothing caps the pool at five.

Each wave runs concurrently. When it finishes, a supervisor reads what came back and adds
agents for anything still uncovered, repeating until it says stop or a budget runs out —
so the pool grows with the task rather than being fixed when you press Dispatch.

## Local and hosted, decided per agent

The two providers fail in opposite directions. Local is free, private, offline and never
rate-limited, but it is CPU inference — a 7B produces maybe ten tokens a second. NIM is
fast and far larger, but it is a shared free tier with quotas and a network round trip.
Neither is the right default for everything.

So every agent's subtask is weighed before a model is bound to it. Light work — a short
email, a definition, a quick read on some numbers — stays on your machine. Heavy work —
architecture, migrations, anything asking for depth or length — goes to NIM. One dispatch
routinely splits across both:

```
WAVE 1
  writer-1    local    qwen2.5:7b                       light (0.18): 1 brevity cue
  analyst-1   nvidia   llama-3.3-nemotron-super-49b-v1  heavy (0.80): 5 depth cues
```

Scoring is deterministic keyword-and-length work, not a model call: asking a model which
model should answer would add a round trip to every agent, which on a one-line question
costs more than simply answering it.

**Availability always beats preference.** Stop Ollama and everything goes to NIM. Delete
the key and everything runs locally, offline, with no account at all. Neither case has a
code path of its own — they fall out of which providers are reachable.

Pin a whole run with `OCTOPUS_ROUTE=local` or `=nvidia`, or dry-run the decision for any
subtask with `POST /api/route` to see the score before you touch the threshold.

## Extending it

A provider is a base URL, an optional key, and a wire format. Models are addressed as
`provider:model` everywhere above `providers.py`, so adding one is a row in a registry —
not a refactor of the octopus, the tentacles, or the UI.

OpenAI and Anthropic are **registered and deliberately disabled**. Octopus is
free-resources-only for now; those rows exist so that enabling one later is a config
change. (OpenAI would need only a key. Anthropic's `/messages` differs in request shape,
streaming events and auth header, so it also needs an adapter at the documented seam.)

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
