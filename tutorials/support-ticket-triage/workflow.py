"""Support Ticket Triage: LLM-powered classification with parallel fan-out.

Scores a batch of support tickets using GPT-4o-mini in parallel on Modal,
then ranks them and generates a visual triage report written to disk.
"""

import modal

from report_helpers import build_html

# Environment configuration -------------------
image = modal.Image.debian_slim(python_version="3.11").pip_install("openai")

app = modal.App("ticket-triage", image=image)

# Default batch of sample support tickets
TICKETS = [
    "URGENT: Production database is down, customers cannot log in",
    "The export button gives an error when I click it",
    "Love the new dashboard update, works great!",
    "App has been slow and unresponsive for 2 days, very frustrated",
    "Security breach detected — need immediate investigation",
    "How do I reset my password?",
    "Billing is wrong, I was charged twice. Want a refund",
    "Great customer support, issue resolved quickly. Thanks!",
    "Critical outage on the payments service, blocked on all orders",
    "Minor UI bug: the footer overlaps on mobile",
]


# Functions -----------------------------------
@app.function(
    cpu=1,
    memory=256,
    secrets=[modal.Secret.from_name("openai-secret")],
)
def classify_ticket(ticket: str) -> dict:
    """Use an LLM to classify a single support ticket."""
    import json
    import os

    from openai import OpenAI

    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a support ticket classifier. Analyze the ticket and return JSON with:\n"
                    '- "category": one of [billing, bug, outage, security, feature_request, general]\n'
                    '- "urgency": float 0-1 (1 = most urgent)\n'
                    '- "sentiment": float -1 to 1 (1 = positive)\n'
                    '- "summary": one-sentence summary of the issue\n'
                    '- "suggested_action": brief recommended next step'
                ),
            },
            {"role": "user", "content": ticket},
        ],
    )

    result = json.loads(response.choices[0].message.content)
    result["ticket"] = ticket[:120]
    result["priority"] = round(
        result["urgency"] * 0.6 + max(-result["sentiment"], 0) * 0.4, 2
    )
    return result


@app.function(cpu=1, memory=256)
def build_report(scored: list[dict]) -> dict:
    """Rank tickets by priority and render a triage report."""
    ranked = sorted(scored, key=lambda t: t["priority"], reverse=True)

    html = build_html(ranked)

    # Also print for terminal visibility
    print("\n=== Ticket Priority Report ===\n")
    for i, t in enumerate(ranked, 1):
        print(f"  {i}. [{t['priority']:.2f}] [{t.get('category','')}] {t['ticket']}")
        print(f"     → {t.get('suggested_action','')}\n")

    return {"ranked": ranked, "html": html}


@app.local_entrypoint()
def main():
    """Fan out LLM classification across all tickets, then build a report."""
    # Fan out: classify every ticket in parallel on Modal
    scored = list(classify_ticket.map(TICKETS))

    # Aggregate: rank and report
    result = build_report.remote(scored)

    # Write the HTML report locally
    out = "support_ticket_triage_report.html"
    with open(out, "w") as f:
        f.write(result["html"])
    print(f"Wrote report to {out}")


# uv run modal run workflow.py
