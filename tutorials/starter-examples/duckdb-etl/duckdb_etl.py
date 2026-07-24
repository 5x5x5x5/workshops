import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "duckdb", "pandas", "pyarrow"
)

app = modal.App("duckdb-etl", image=image)

SAMPLE_CSV = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


@app.function(cpu=1, memory=1024)
def extract():
    """Load raw data from a CSV source."""
    import duckdb

    df = duckdb.sql(f"SELECT * FROM read_csv_auto('{SAMPLE_CSV}')").df()
    print(f"Extracted {len(df)} rows")
    return df


@app.function(cpu=1, memory=1024)
def transform(raw):
    """Aggregate survival stats by passenger class using DuckDB SQL."""
    import duckdb

    summary = duckdb.sql("""
        SELECT
            Pclass AS passenger_class,
            COUNT(*) AS total,
            SUM(Survived) AS survived,
            ROUND(AVG(Survived) * 100, 1) AS survival_rate,
            ROUND(AVG(Fare), 2) AS avg_fare
        FROM raw
        GROUP BY Pclass
        ORDER BY Pclass
    """).df()
    print(summary.to_string(index=False))
    return summary


@app.function(cpu=1, memory=1024)
def pipeline() -> str:
    """Extract → Transform pipeline. Returns an HTML report."""
    raw = extract.remote()
    summary = transform.remote(raw)

    return (
        f"<h2>DuckDB ETL Results</h2>"
        f"<p>Processed {len(raw)} rows into {len(summary)} groups</p>"
        f"<h3>Survival by Passenger Class</h3>"
        f"{summary.to_html(index=False)}"
    )


@app.local_entrypoint()
def main():
    html = pipeline.remote()
    out = "duckdb_etl_report.html"
    with open(out, "w") as f:
        f.write(html)
    print(f"Wrote report to {out}")


# uv run modal run duckdb_etl.py
