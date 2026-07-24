# DuckDB Data Pipeline

Extract CSV data, transform with DuckDB SQL, and render the results as an HTML report on [Modal](https://modal.com).

## What it does

- **`extract`** — Loads the Titanic CSV from a public URL using DuckDB's `read_csv_auto`
- **`transform`** — Aggregates survival statistics by passenger class using SQL
- **`pipeline`** — Orchestrates extract -> transform, returns the results as an HTML table
- **`main`** — A `local_entrypoint` that runs the pipeline and writes `duckdb_etl_report.html`

Each function runs in a container built from the `modal.Image` defined at the top of the script — no cluster to provision.

## Setup

```bash
cd tutorials/starter-examples/duckdb-etl

uv venv .venv --python 3.11
source .venv/bin/activate

uv pip install -r requirements.txt
```

## Modal account (one-time)

```bash
uv run modal setup
```

This opens a browser to authenticate. Don't have an account? Sign up at [modal.com](https://modal.com).

## Run

```bash
uv run modal run duckdb_etl.py
```

The functions execute in the cloud; the report is written to `duckdb_etl_report.html` in your working directory.

## Notes

- Fully self-contained — no external services or accounts beyond Modal needed
- DuckDB can query pandas DataFrames directly with SQL
- Dependencies are declared inline in the `modal.Image`, so the only local requirement is `modal`
