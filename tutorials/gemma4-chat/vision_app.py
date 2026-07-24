"""Gemma 4 vision app, served on Modal.

Upload an image; two tabs:
- **Ask**: free-form Q&A streaming, with optional thinking panel and budget cap
- **Detect**: ask for bounding-box JSON in the Gemini `box_2d` format and draw
  them on the image (emergent VLM capability, not a trained detector head)

Same vLLM backend (`gemma4-26b-a4b-it-vllm`) as `chat_app.py` and
`live_camera_app.py`. Modal serves the UI over HTTPS, so image uploads from a
remote browser work with no extra tunnel.

Dev (hot-reload, ephemeral URL):
    modal serve vision_app.py

Deploy (persistent URL — run after vllm_server.py is deployed):
    modal deploy vision_app.py
"""

from __future__ import annotations

import modal


VISION_APP_NAME = "gemma4-vision"

vision_image = (
    modal.Image.debian_slim(python_version="3.11")
    # fonts-dejavu-core supplies DejaVuSans-Bold.ttf for the box labels.
    .apt_install("fonts-dejavu-core")
    .pip_install(
        "gradio==5.42.0",
        "openai>=1.50.0",
        "pillow>=10.0.0",
        "fastapi[standard]",
    )
)

app = modal.App(VISION_APP_NAME, image=vision_image)


# Same parser as chat_app.py — Gemma 4 IT wraps the thought channel in
# `<|channel>thought\n...<channel|>` special tokens (kept visible in output
# via skip_special_tokens=False).
def _split_thinking(text: str) -> tuple[str, str]:
    OPEN, OPEN_TAIL = "<|channel>", "thought\n"
    CLOSE = "<channel|>"
    j = text.find(OPEN)
    if j == -1:
        return "", text.strip()
    pre = text[:j]
    rest = text[j + len(OPEN):]
    if rest.startswith(OPEN_TAIL):
        rest = rest[len(OPEN_TAIL):]
    k = rest.find(CLOSE)
    if k == -1:
        thinking, answer = rest, pre
    else:
        thinking = rest[:k]
        answer = (pre + rest[k + len(CLOSE):])
    return thinking.strip(), answer.strip()


@app.function(cpu=1, memory=2048, scaledown_window=900)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    """Build and serve the Gradio vision UI."""
    import base64
    import io
    import json
    import re

    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    from openai import OpenAI
    from PIL import Image, ImageDraw, ImageFont, ImageOps

    from config import MODEL, resolve_vllm_url

    model_id = MODEL.model_id
    base_url = resolve_vllm_url() + "/v1"
    print(f"[vision] gradio={gr.__version__}  vllm={base_url}  model={model_id}", flush=True)
    client = OpenAI(base_url=base_url, api_key="not-used")

    CHARS_PER_TOKEN = 3.5

    PRESET_PROMPTS = [
        "Describe this image in detail.",
        "List every object you can see, roughly where it is in the frame.",
        "Read any text visible in the image.",
        "What's unusual or unsafe about this scene?",
        "Count the people / animals / vehicles visible.",
    ]

    BBOX_COLORS = [
        (31, 119, 180), (255, 127, 14), (44, 160, 44), (214, 39, 40),
        (148, 103, 189), (140, 86, 75), (227, 119, 194), (127, 127, 127),
        (188, 189, 34), (23, 190, 207),
    ]
    BBOX_FILL_ALPHA = 90
    BBOX_OUTLINE_ALPHA = 255

    def _load_and_orient(image_path: str) -> Image.Image:
        """Apply EXIF orientation so the model and the overlay agree on
        which way is up. Phones embed rotation in EXIF; without this the
        bounding boxes land on the wrong axis."""
        return ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")

    def _encode_to_data_url(img: Image.Image) -> str:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"

    # ---------------- Ask tab -----------------------------------------------

    def ask(image_path, question, temperature, think_budget):
        if not image_path:
            yield "", "Upload an image first."
            return
        if not question or not question.strip():
            yield "", "Ask a question about the image."
            return

        img = _load_and_orient(image_path)
        data_url = _encode_to_data_url(img)
        budget_chars = int(think_budget * CHARS_PER_TOKEN) if think_budget else 0

        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": question.strip()},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }]

        yield "", "Thinking..."
        stream = client.chat.completions.create(
            model=model_id,
            messages=messages,
            stream=True,
            temperature=float(temperature),
            max_tokens=2048,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": True},
                "skip_special_tokens": False,
            },
        )

        buf = ""
        capped = False
        try:
            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue
                buf += delta
                thinking, answer = _split_thinking(buf)
                yield thinking, answer or "Thinking..."

                if (budget_chars and not answer
                        and len(thinking) >= budget_chars):
                    capped = True
                    break
        finally:
            stream.close()

        if capped:
            thinking, _ = _split_thinking(buf)
            thinking += f"\n\n_[capped at ~{think_budget} tokens]_"
            yield thinking, "Generating answer..."

            followup = messages + [
                {"role": "assistant", "content": thinking},
                {"role": "user", "content": "Stop thinking. Give your final answer now, concisely."},
            ]
            answer_stream = client.chat.completions.create(
                model=model_id,
                messages=followup,
                stream=True,
                temperature=float(temperature),
                max_tokens=2048,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": False},
                    "skip_special_tokens": False,
                },
            )
            buf2 = ""
            try:
                for chunk in answer_stream:
                    delta = chunk.choices[0].delta.content or ""
                    if not delta:
                        continue
                    buf2 += delta
                    _, ans = _split_thinking(buf2)
                    yield thinking, ans or "Generating answer..."
            finally:
                answer_stream.close()

    # ---------------- Detect tab --------------------------------------------

    DETECT_PROMPT = (
        "Detect {target} in this image. Respond with ONLY a JSON array (no "
        "prose, no markdown fence). Each detection is "
        '{{"label": "<short name>", "box_2d": [ymin, xmin, ymax, xmax]}} '
        "where coords are normalized 0–1000 (y axis down). Skip duplicates."
    )

    def _extract_json_array(text: str) -> list[dict]:
        fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL)
        candidate = fenced.group(1) if fenced else text
        start = candidate.find("[")
        end = candidate.rfind("]")
        if start == -1 or end == -1 or end <= start:
            return []
        try:
            data = json.loads(candidate[start:end + 1])
        except json.JSONDecodeError:
            return []
        if not isinstance(data, list):
            return []
        out = []
        for item in data:
            if not isinstance(item, dict):
                continue
            box = item.get("box_2d") or item.get("bbox") or item.get("box")
            label = item.get("label") or item.get("name") or ""
            if isinstance(box, list) and len(box) == 4:
                out.append({"label": str(label), "box_2d": [float(v) for v in box]})
        return out

    def _draw_boxes(base: Image.Image, detections: list[dict]) -> Image.Image:
        img = base.convert("RGBA")
        w, h = img.size
        overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        fill_draw = ImageDraw.Draw(overlay)
        try:
            font = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                size=max(14, int(min(w, h) * 0.025)),
            )
        except OSError:
            font = ImageFont.load_default()
        line_w = max(2, int(min(w, h) * 0.004))

        for i, det in enumerate(detections):
            ymin, xmin, ymax, xmax = det["box_2d"]
            x1, y1 = int(xmin / 1000 * w), int(ymin / 1000 * h)
            x2, y2 = int(xmax / 1000 * w), int(ymax / 1000 * h)
            r, g, b = BBOX_COLORS[i % len(BBOX_COLORS)]
            fill_draw.rectangle([x1, y1, x2, y2], fill=(r, g, b, BBOX_FILL_ALPHA))

        img = Image.alpha_composite(img, overlay)
        draw = ImageDraw.Draw(img)
        for i, det in enumerate(detections):
            ymin, xmin, ymax, xmax = det["box_2d"]
            x1, y1 = int(xmin / 1000 * w), int(ymin / 1000 * h)
            x2, y2 = int(xmax / 1000 * w), int(ymax / 1000 * h)
            r, g, b = BBOX_COLORS[i % len(BBOX_COLORS)]
            color = (r, g, b, BBOX_OUTLINE_ALPHA)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)
            label = det["label"]
            if label:
                tb = draw.textbbox((x1, y1), label, font=font)
                pad = 3
                draw.rectangle(
                    [tb[0] - pad, tb[1] - pad, tb[2] + pad, tb[3] + pad],
                    fill=color,
                )
                draw.text((x1, y1), label, fill=(255, 255, 255, 255), font=font)
        return img.convert("RGB")

    def detect(image_path, target, temperature):
        if not image_path:
            return None, "Upload an image first.", ""
        target = (target or "").strip() or "the main objects"
        base = _load_and_orient(image_path)
        data_url = _encode_to_data_url(base)

        resp = client.chat.completions.create(
            model=model_id,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": DETECT_PROMPT.format(target=target)},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }],
            temperature=float(temperature),
            max_tokens=2048,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False},
                "skip_special_tokens": True,
            },
        )
        raw = resp.choices[0].message.content or ""
        detections = _extract_json_array(raw)
        print(f"[detect] target={target!r} parsed={len(detections)}", flush=True)

        if not detections:
            return base, "No detections parsed.", raw
        annotated = _draw_boxes(base, detections)
        return annotated, json.dumps(detections, indent=2), raw

    # ---------------- UI -----------------------------------------------------

    with gr.Blocks(title="Gemma 4 Vision") as demo:
        gr.Markdown(
            "# Gemma 4 Vision\n"
            "Upload an image. **Ask** mode: free-form Q&A. **Detect** mode: "
            "Gemma 4 returns normalized bounding boxes that we draw on the image. "
            "Detection is emergent (not a trained head) — quality varies. "
            f"Endpoint: `{base_url}` · model: `{model_id}`"
        )

        with gr.Row():
            temperature = gr.Slider(0.0, 1.5, value=0.2, step=0.05, label="Temperature")
            think_budget = gr.Slider(
                0, 4000, value=0, step=100,
                label="Thinking budget (tokens, 0 = unlimited)",
                info="Caps thinking on the Ask tab. Detect ignores this.",
            )

        with gr.Tabs():
            with gr.Tab("Ask"):
                with gr.Row():
                    with gr.Column():
                        image = gr.Image(type="filepath", label="Image", height=400)
                        question = gr.Textbox(
                            label="Question", lines=2,
                            placeholder="e.g. What's happening in this photo?",
                        )
                        preset = gr.Radio(PRESET_PROMPTS, label="Preset prompts", value=None)
                        submit = gr.Button("Ask", variant="primary")
                    with gr.Column():
                        with gr.Accordion("🧠 Thinking", open=False):
                            thinking = gr.Textbox(
                                show_label=False, lines=10,
                                placeholder="Thinking tokens stream here...",
                            )
                        answer = gr.Textbox(label="Answer", lines=15)

                ask_inputs = [image, question, temperature, think_budget]
                ask_outputs = [thinking, answer]
                preset.change(lambda p: p or "", inputs=preset, outputs=question)
                submit.click(ask, inputs=ask_inputs, outputs=ask_outputs)
                question.submit(ask, inputs=ask_inputs, outputs=ask_outputs)

            with gr.Tab("Detect (bounding boxes)"):
                with gr.Row():
                    with gr.Column():
                        det_image = gr.Image(type="filepath", label="Image", height=400)
                        target = gr.Textbox(
                            label="What to detect",
                            value="the main objects",
                            placeholder="e.g. 'all people and cars', 'dogs', 'faces'",
                        )
                        det_submit = gr.Button("Detect", variant="primary")
                    with gr.Column():
                        annotated = gr.Image(label="Annotated", height=400)
                        detections_json = gr.Code(
                            language="json", label="Detections", lines=12,
                        )
                        raw_output = gr.Textbox(label="Raw model output", lines=4)

                det_submit.click(
                    detect,
                    inputs=[det_image, target, temperature],
                    outputs=[annotated, detections_json, raw_output],
                )

    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")
