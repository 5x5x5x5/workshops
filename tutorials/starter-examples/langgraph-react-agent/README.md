# LangGraph ReAct Agent

A ReAct agent using LangGraph's prebuilt `create_react_agent` with math tools, running on [Modal](https://modal.com).

## What it does

- Creates a ReAct (Reason + Act) agent with OpenAI and LangGraph
- Defines simple math tools (`add`, `multiply`)
- The agent reasons about which tool to call, observes results, and loops until it has an answer
- Runs inside a Modal container with the `OPENAI_API_KEY` injected from a Modal secret

## Setup

```bash
cd tutorials/starter-examples/langgraph-react-agent

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

The agent reads `OPENAI_API_KEY` from a Modal secret named `openai-secret`. Create it once:

```bash
uv run modal secret create openai-secret OPENAI_API_KEY=sk-...
```

(Or create it in the Modal dashboard under **Secrets** using the OpenAI template.)

## Run

```bash
uv run modal run langgraph_react_agent.py --request "What is 12 * 7 plus 3?"
```

Logs and traces for each run are available in the [Modal dashboard](https://modal.com/apps).

## Notes

- Dependencies are declared inline in the `modal.Image`, so the only local requirement is `modal`
- `utils/file_viewer.py` is a notebook display helper used by the companion notebook
- The companion notebook `tutorial_langgraph_react_agent.ipynb` still uses the Flyte SDK and has not been ported
