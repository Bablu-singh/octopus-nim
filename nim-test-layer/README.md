# Octopus — test layer and agent console

A local wrapper around several model providers — an **Ollama server on this machine**,
the **NVIDIA NIM API** (`build.nvidia.com`) and **Google Gemini**, all of which have a
free tier — so you can verify a key, list reachable models,
and fire prompts without pasting the key into a browser page or hitting CORS. The key
stays in `.env` on your machine; the browser only ever talks to `127.0.0.1`.

On top of that sits **Octopus**: describe a task and it decides what kind of work it is,
splits it into as many agents as the task warrants, runs them in parallel, and routes each
one to whichever provider suits it. See [../docs/ARCHITECTURE.md](../docs/ARCHITECTURE.md).

## Setup

macOS / Linux:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Windows (PowerShell):

```bash
python -m venv .venv; .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```

A NIM key in `.env` is optional. For local models, install Ollama and pull at least one:

```bash
ollama pull llama3.2:3b
```

## Run

```bash
python -m uvicorn app:app --reload --port 8000
```

Open http://127.0.0.1:8000 for Octopus, or `/console` for the key self-test.

Terminal only:

```bash
python cli_test.py                      # exits 0 on pass, 1 on failure
python cli_test.py "summarise CQRS"     # custom prompt
```

## What the self-test checks

| Check | Catches |
|---|---|
| Key loaded | Missing `.env`, blank value, wrong variable name |
| Key authenticates | Revoked, expired, or wrong-account key (401/403) |
| Models actually serving | Listed-but-dead models — see below |
| Completion round trip | Model name typos, quota exhaustion (429), empty responses |
| Streaming | Proxy or firewall breaking SSE, first-token latency |

## How the agent pool is sized

Open `/` and describe a task. You never pick an agent type — the planner reads the task,
works out what kind of work it is, and sizes the pool to it:

| Task | Agents |
|---|---|
| "What is 12 percent of 340?" | 1 analyst |
| "Write a function that validates IBANs, with unit tests." | 2 coders |
| A migration naming 7 deliverables | 7 agents across 4 roles |

There are five **roles** — writer, coder, scheduler, imager, analyst — but they are
templates, not a roster. A role is instantiated as often as the work divides, so three
coders on three modules is normal. Nothing caps the pool at five.

Everything in a wave runs concurrently, with `OCTOPUS_MAX_PARALLEL` (default 6) talking
upstream at once. When the wave finishes, a supervisor sees what came back and adds agents
for anything still uncovered; that repeats until it says stop or a budget runs out. So the
pool grows with the task rather than being fixed when you press Dispatch.

Budgets, all `.env`-tunable: `OCTOPUS_MAX_AGENTS` (24 total), `OCTOPUS_FIRST_WAVE` (8),
`OCTOPUS_MAX_WAVES` (4), `OCTOPUS_MAX_PARALLEL` (6).

Two rules keep it from spinning: a deliverable that already succeeded is never rerun, and
one that fails gets exactly one retry — on a different model, because a failure that looks
like the model's fault drops that model and re-binds the role mid-run.

## Where each agent runs

Every subtask is weighed before a model is bound to it — role baseline, plus depth cues
("comprehensive", "migration", "production-ready"), minus brevity cues ("short", "quick",
"typo"), adjusted for request length and listed requirements. Score `≥ 0.55` is heavy.

| | local | nvidia | gemini |
|---|---|---|---|
| Key | none | `NVIDIA_API_KEY` | `GEMINI_API_KEY` |
| Catalog | trusted — lists only what is on disk | probed model by model | trusted — lists what it serves |
| Read timeout | 300 s — a cold 7B loads from disk | 45 s — fail fast | 60 s |
| Token cap | 700 (`LOCAL_MAX_TOKENS`) | role default, 1400–2000 | role default |
| Parallelism | 2 (`OCTOPUS_LOCAL_PARALLEL`) | 6 (`OCTOPUS_MAX_PARALLEL`) | 6 |
| Gets | light work | heavy work, image generation, planning | heavy work, planning |

Order within a branch is not hardcoded: each provider declares `prefers` (`small`/`large`)
and `priority` in `providers.py`, and the router reads that — so adding a key adds a
destination without touching `routing.py`.

Planning and supervision are never weighed: they always take the strongest provider that
is up, because a small model sizes a pool badly and every extra agent it invents costs a
completion.

Availability beats preference — with a provider down, everything routes to what is left.

```bash
# dry-run the decision without dispatching anything
curl -s localhost:8000/api/route -H 'Content-Type: application/json' \
  -d '{"role":"writer","subtask":"Write a short thank-you note."}' | jq
```

Tune with `OCTOPUS_ROUTE` (`auto`, or any provider name to pin a run) and
`OCTOPUS_ROUTE_THRESHOLD`.

### Adding Gemini

Get a free key at [aistudio.google.com/apikey](https://aistudio.google.com/apikey), put
`GEMINI_API_KEY=...` in `.env`, restart. It appears in `/api/providers` and joins the
routing rotation — no code change, because Google publishes an OpenAI-compatible surface
at `/v1beta/openai`. Two quirks are handled in `providers.py`: it answers a rejected key
with `400` rather than `401`, and it prefixes listed ids with `models/`.

## Cost, rate limits, and turning a provider off

Everything reachable is free, so the budget is quota and wall-clock. Savings, largest
first:

| Knob | Default | Effect |
|---|---|---|
| `OCTOPUS_PLANNER_SKIP_BELOW` | `0.20` | Below this task weight, wave 1 is planned by keyword — one fewer completion on every trivial run |
| `OCTOPUS_TOKEN_BUDGET` | `0` (off) | Hard ceiling on estimated tokens per dispatch; running agents still finish |
| `LOCAL_MAX_TOKENS` | `700` | Caps local generation, which is what keeps small work faster than the network |
| `<PROVIDER>_RPM` | `36` NIM, `12` Gemini, `0` local | Requests per minute, spaced evenly rather than burst-and-wait |

Every dispatch reports its spend in the `complete` event — tokens per provider, and money
only when something billable was involved. Counts are estimates: agents stream, and
streaming responses carry no usage block.

If a provider answers `429` anyway, it is stood down for as long as it asked
(`Retry-After`, or the `retryDelay` in the body) and the router stops offering it until it
recovers — visible as `cooling down` in `/api/providers`. A `429` never marks a model
dead; it means we asked too often, not that the model broke.

### Turning a provider off

```bash
ENABLE_NVIDIA=0   # or ENABLE_GEMINI=0, ENABLE_LOCAL=0
```

That provider leaves the pool and everything routes to what is left. This is the answer to
"what if a free tier stops being free" — no code change, one restart to undo. It needed no
new machinery because availability was always the only thing the router trusted.

## Listed is not the same as served

`GET /v1/models` returns everything the catalog knows about, not what your key can
actually run. On a typical key most of those models return 404, and a handful accept the
request and then never respond at all — which surfaces as a read timeout minutes later.

So nothing is bound to a model on the strength of the catalog alone. `/api/catalog`
probes each candidate with a tiny completion and binds only models that return actual
content — a 200 with an empty message does not count;
the result is cached for 15 minutes per key, and a model that fails during a real
dispatch is dropped from the cache so the next run re-binds instead of failing again.

That probe is why the first `/api/catalog` call takes ~15 s and later ones are instant.
Force a re-probe with `/api/catalog?force=true`.

Timeouts are tunable in `.env`: `NIM_CONNECT_TIMEOUT`, `NIM_READ_TIMEOUT` (per read, so
for streaming it is the gap between chunks), and `NIM_PROBE_TIMEOUT`.

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | What is usable right now (key fingerprint only, never the key) |
| GET | `/api/providers` | Every registered provider, its state, and the routing config |
| POST | `/api/route` | Dry-run: where would this subtask go, and why? Dispatches nothing |
| GET | `/api/models` | Every model ID one provider lists (`?provider=local`) |
| GET | `/api/catalog` | Which of those actually answer, grouped, with per-provider bindings |
| POST | `/api/dispatch` | Plan and run an agent pool over one task, streamed as SSE |
| POST | `/api/chat` | Single completion, returns text + token usage |
| POST | `/api/chat/stream` | SSE passthrough |
| POST | `/api/selftest` | All five checks as JSON |

Send `X-NIM-Key` on any of these to override **NIM's** key for one call — handy for
comparing two keys without a restart. It applies to NIM alone; every other provider keeps
reading its own environment variable, so the header cannot hand Google an `nvapi-` string.

Model ids are qualified as `provider:model`. A bare id means NIM, which is what every id
meant before providers existed, so old scripts keep working.

```bash
curl -s localhost:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"prompt":"ping","model":"local:qwen2.5:7b"}' | jq

curl -s localhost:8000/api/chat -H 'Content-Type: application/json' \
  -d '{"prompt":"ping","model":"nvidia:meta/llama-3.1-8b-instruct"}' | jq
```

`POST /api/dispatch` also takes `{"mode": "local"|"nvidia"}` to pin one run.

## Reading failures

- **401** — key is wrong or revoked. Regenerate at build.nvidia.com/settings/api-keys.
- **403** — key is valid but not entitled to that model. Use **Load models** to see what it can reach.
- **404** — model ID typo, or a catalogued model your key cannot actually run. NIM IDs
  are namespaced, e.g. `meta/llama-3.1-8b-instruct`.
- **429** — free-tier credits exhausted or rate limited.
- **503** — a provider is not answering at all. For local that almost always means Ollama
  is not running: start it with `ollama serve`, or check `ollama list` shows a model.
- **504** — raised locally, not by NVIDIA: the model took the request and sent nothing
  back. Almost always a listed-but-not-served model rather than a problem with your key.
- **`bad key` at startup** — the key is present but was rejected. NIM serves its model
  list to anyone, so this is caught by a separate one-token auth probe rather than by the
  catalog read appearing to succeed. Providers disagree on the status code for this —
  Gemini says `400`, NIM says `403` — so the body is inspected too.

Never commit `.env` — `.gitignore` already covers it.
