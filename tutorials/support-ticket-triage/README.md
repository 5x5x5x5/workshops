# Support Ticket Triage — Intro to Modal

A pure-Python workflow that triages customer support tickets in parallel on [Modal](https://modal.com) using `Function.map` for fan-out.

**What it shows:**
- Workflows as plain Python functions (`@app.function`, functions calling functions)
- Parallel fan-out with `classify_ticket.map` — every ticket scored simultaneously
- The same code runs locally-driven or fully in the cloud, no config changes

## Setup

```bash
cd tutorials/support-ticket-triage
uv venv .venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Modal account (one-time)

```bash
uv run modal setup
```

This opens a browser to authenticate. Don't have an account? Sign up at [modal.com](https://modal.com).

## OpenAI secret

`classify_ticket` reads `OPENAI_API_KEY` from a Modal secret named `openai-secret`. Create it once:

```bash
uv run modal secret create openai-secret OPENAI_API_KEY=sk-...
```

(Or create it in the Modal dashboard under **Secrets** using the OpenAI template.)

## Run

```bash
uv run modal run workflow.py
```

The classification functions execute in the cloud; a ranked triage report is written to `support_ticket_triage_report.html` in your working directory.

## How it works

1. **`main`** — `local_entrypoint` that holds a batch of sample tickets, fans out classification, and writes the HTML report
2. **`classify_ticket`** — map function, scores each ticket's category, urgency & sentiment with GPT-4o-mini and computes a priority score
3. **`build_report`** — ranks tickets by combined priority score, prints a ranked report, and returns the HTML

The fan-out happens in one line: `classify_ticket.map(tickets)` — Modal runs all 10 classification calls in parallel across containers.
