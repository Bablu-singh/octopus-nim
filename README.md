# Octopus

A local multi-agent console for the NVIDIA NIM API (`build.nvidia.com`). Describe a task;
it works out what kind of work it is, splits it into as many agents as the task warrants,
and runs them in parallel against whichever models your key can actually reach.

**[Live demo →](https://bablu-singh.github.io/octopus-nim/)** — a recorded dispatch,
replayed. GitHub Pages cannot run the Python backend, so the demo replays a real run
captured from the live API. Everything else on that page is the app itself.

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

## The part that makes it work

`GET /v1/models` advertises far more than a key can run. On the key this was built
against, **102 models were listed and 6 actually served completions** — most return 404,
and several accept a request and then never reply, which surfaces as a read timeout
minutes later.

So nothing is bound on the strength of the catalog. Every candidate is probed with a tiny
completion first, and only models that return real content are used. A model that fails
mid-run is dropped and its role re-bound to a different one, so the retry lands somewhere
that works.

## Run it

```bash
cd nim-test-layer
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # paste your nvapi- key into .env
uvicorn app:app --reload --port 8000
```

Open <http://127.0.0.1:8000>. Full documentation, API reference and tuning knobs are in
[nim-test-layer/README.md](nim-test-layer/README.md).

Your key stays in `.env` on your machine — it is gitignored, never sent to the browser,
and only ever fingerprinted in logs.

## Rebuilding the demo

```bash
python3 docs/record.py     # capture a fresh dispatch (needs a working key)
python3 docs/build.py      # regenerate docs/index.html from the app's own UI
```

`docs/index.html` is generated, not hand-maintained: it is `static/octopus.html` plus a
banner and a shim that replays `docs/demo.json` in place of the API.
