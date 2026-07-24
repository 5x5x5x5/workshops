"""
Emotion Classifier — Gradio frontend on Modal.

Type text and see emotion predictions with confidence scores and an attention
heatmap showing which words the model focuses on. The UI loads the fine-tuned
model (produced by workflow.py) from the shared model volume and runs inference
in-process, so it works on its own after training. serve.py is the standalone
REST API if you prefer the Gradio-as-thin-HTTP-client split.

Usage:
    # Dev server with hot-reload (ephemeral URL)
    uv run modal serve app_gradio.py

    # Persistent deployment
    uv run modal deploy app_gradio.py
"""

import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install(
        "transformers",
        "accelerate",
        "gradio>=5.0.0",
        "fastapi[standard]",
    )
)

app = modal.App("emotion-classifier-ui", image=image)

# The fine-tuned model is written to this volume by workflow.py.
model_volume = modal.Volume.from_name("bert-emotion-model", create_if_missing=True)
MODEL_PATH = "/models"
FINETUNED_DIR = f"{MODEL_PATH}/finetuned_model"
EMOTION_LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]


# ------------------------------------------------------------------
# Example texts
# ------------------------------------------------------------------

EXAMPLES = [
    "I am so happy right now, everything is going great!",
    "I feel really sad and lonely tonight",
    "I can't believe they surprised me with a puppy!",
    "This makes me so angry, I can't stand it",
    "I'm terrified of what might happen next",
    "I love you more than anything in the world",
    "I feel so proud of what we accomplished together",
    "The news was shocking, I never expected that",
    "I'm worried about the upcoming exam results",
    "She gave me the sweetest compliment and I'm blushing",
]

EMOTION_COLORS = {
    "sadness": "#4a6fa5",
    "joy": "#f4a261",
    "love": "#e76f51",
    "anger": "#e63946",
    "fear": "#6c567b",
    "surprise": "#2a9d8f",
}

EMOTION_EMOJI = {
    "sadness": "😢",
    "joy": "😊",
    "love": "❤️",
    "anger": "😠",
    "fear": "😨",
    "surprise": "😲",
}


# ------------------------------------------------------------------
# Gradio app
# ------------------------------------------------------------------

@app.function(gpu="T4", memory=8192, volumes={MODEL_PATH: model_volume})
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    from pathlib import Path

    import gradio as gr
    import torch
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    # -- Load the fine-tuned model once per container --
    model_path = Path(FINETUNED_DIR)
    model = None
    tokenizer = None
    if model_path.exists():
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path),
            output_attentions=True,
            attn_implementation="eager",
        )
        model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)

    def run_inference(text: str):
        """Run the model and return the same shape serve.py's /predict returns."""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        probs = torch.softmax(outputs.logits[0], dim=-1).cpu().tolist()
        pred_idx = int(torch.argmax(outputs.logits[0]).item())

        last_attention = outputs.attentions[-1][0]
        cls_attention = last_attention.mean(dim=0)[0].cpu().tolist()

        token_ids = inputs["input_ids"][0]
        raw_tokens = tokenizer.convert_ids_to_tokens(token_ids)

        clean_tokens = []
        clean_attention = []
        for i, tok in enumerate(raw_tokens):
            if tok in ("[CLS]", "[SEP]", "<s>", "</s>", "[PAD]", "<pad>"):
                continue
            if tok == tokenizer.pad_token:
                continue
            display_tok = tok.replace("##", "").replace("Ġ", "").replace("▁", "")
            if not display_tok.strip():
                continue
            clean_tokens.append(display_tok)
            clean_attention.append(cls_attention[i])

        if clean_attention:
            max_att = max(clean_attention)
            min_att = min(clean_attention)
            att_range = max_att - min_att or 1
            clean_attention = [(a - min_att) / att_range for a in clean_attention]

        return {
            "predicted_emotion": EMOTION_LABELS[pred_idx],
            "confidence": round(probs[pred_idx], 4),
            "scores": [
                {"label": EMOTION_LABELS[i], "score": round(probs[i], 4)}
                for i in range(len(EMOTION_LABELS))
            ],
            "tokens": clean_tokens,
            "attention_weights": clean_attention,
        }

    def predict(text: str):
        if not text.strip():
            return "", "", ""

        if model is None:
            return (
                "Model not found in the volume. Run the pipeline first: "
                "`uv run modal run workflow.py`",
                "",
                "",
            )

        try:
            result = run_inference(text)
        except Exception as e:
            return f"Error: {e}", "", ""

        # Build prediction summary
        emotion = result["predicted_emotion"]
        confidence = result["confidence"]
        emoji = EMOTION_EMOJI.get(emotion, "")

        summary = f"## {emoji} {emotion.title()} ({confidence:.1%})\n\n"

        # Confidence bars for all emotions
        summary += "### Confidence Scores\n\n"
        for score in sorted(result["scores"], key=lambda s: s["score"], reverse=True):
            pct = score["score"] * 100
            bar_emoji = EMOTION_EMOJI.get(score["label"], "")
            filled = int(pct / 5)
            bar = "█" * filled + "░" * (20 - filled)
            summary += f"`{bar}` **{score['label']}** {bar_emoji} {pct:.1f}%\n\n"

        # Attention heatmap as HTML
        tokens = result.get("tokens", [])
        weights = result.get("attention_weights", [])

        attention_html = ""
        if tokens and weights:
            attention_html = (
                '<div style="font-family:system-ui;padding:16px;">'
                '<h3 style="color:#16213e;margin-bottom:8px;">Attention Heatmap</h3>'
                '<p style="color:#6c757d;font-size:0.85em;margin-bottom:12px;">'
                'Which words the model focuses on for its prediction. Darker = more attention.</p>'
                '<div style="line-height:2.4;font-size:1.1em;">'
            )
            for tok, w in zip(tokens, weights):
                # Gradient: light blue (#dbe9f7) → mid blue (#5a9bd5) → deep navy (#0f3460)
                if w < 0.5:
                    # Low attention: light background, dark text
                    t = w / 0.5
                    r = int(219 + (90 - 219) * t)
                    g = int(233 + (155 - 233) * t)
                    b = int(247 + (213 - 247) * t)
                    text_color = "#1a1a2e"
                else:
                    # High attention: dark background, white text
                    t = (w - 0.5) / 0.5
                    r = int(90 + (15 - 90) * t)
                    g = int(155 + (52 - 155) * t)
                    b = int(213 + (96 - 213) * t)
                    text_color = "#ffffff"
                attention_html += (
                    f'<span style="background:rgb({r},{g},{b});'
                    f'color:{text_color};padding:4px 6px;border-radius:4px;'
                    f'margin:2px;display:inline-block;">{tok}</span>'
                )
            attention_html += '</div></div>'

        # Top attended words
        top_words_md = ""
        if tokens and weights:
            pairs = sorted(zip(weights, tokens), reverse=True)
            top_words_md = "### Most Attended Words\n\n"
            for w, tok in pairs[:5]:
                bar_len = int(w * 20)
                bar = "█" * bar_len + "░" * (20 - bar_len)
                top_words_md += f"`{bar}` **{tok}** ({w:.2f})\n\n"

        return summary + top_words_md, attention_html, ""

    # Build the Gradio UI
    with gr.Blocks(
        title="Emotion Classifier",
        theme=gr.themes.Soft(),
        css="""
        .main-header { text-align: center; margin-bottom: 20px; }
        .emotion-card { border-radius: 12px; padding: 20px; }
        """,
    ) as demo:
        gr.Markdown(
            "# Emotion Classifier\n"
            "Enter text and see what emotion the model detects, "
            "with confidence scores and an attention heatmap showing "
            "which words influenced the prediction.\n\n"
            "Powered by a fine-tuned [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base) "
            "model trained on the [emotion](https://huggingface.co/datasets/dair-ai/emotion) dataset "
            "(sadness, joy, love, anger, fear, surprise)."
        )

        with gr.Row():
            with gr.Column(scale=1):
                text_input = gr.Textbox(
                    label="Enter text to classify",
                    placeholder="Type something emotional...",
                    lines=3,
                )
                classify_btn = gr.Button(
                    "Classify Emotion", variant="primary", size="lg",
                )
                gr.Examples(
                    examples=[[ex] for ex in EXAMPLES],
                    inputs=[text_input],
                    label="Try these examples",
                )

            with gr.Column(scale=1):
                prediction_output = gr.Markdown(label="Prediction")
                attention_output = gr.HTML(label="Attention")
                error_output = gr.Textbox(label="", visible=False)

        classify_btn.click(
            predict,
            inputs=[text_input],
            outputs=[prediction_output, attention_output, error_output],
        )

        text_input.submit(
            predict,
            inputs=[text_input],
            outputs=[prediction_output, attention_output, error_output],
        )

    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")
