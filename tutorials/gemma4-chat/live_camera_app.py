"""Gemma 4 live-camera describer, served on Modal.

Webcam → Gemma 4 IT vision → streaming caption every few seconds. Reuses the
same `gemma4-26b-a4b-it-vllm` server as `chat_app.py`; this is a separate
Gradio frontend pointed at the same OpenAI-compatible endpoint.

Two modes:
- independent: each caption is a fresh description of the current frame
- narrative:   the prompt includes recent captions so the model focuses on
               what changed — feels like live commentary

Webcam access (`getUserMedia`) requires a secure context. Modal serves this UI
over HTTPS, so the webcam works from any browser with no extra tunnel.

Dev (hot-reload, ephemeral URL):
    modal serve live_camera_app.py

Deploy (persistent URL — run after vllm_server.py is deployed):
    modal deploy live_camera_app.py

Tunables (set as env vars before `modal serve`/`modal deploy`):
    CAMERA_CADENCE  seconds between captions (default 3)
    MAX_SIDE        downsample long side in px (default 384)
    MAX_OUT_TOKENS  cap reply length (default 60)
"""

from __future__ import annotations

import os

import modal


LIVE_CAM_APP_NAME = "gemma4-live-camera"

# Propagate the caption tunables from the deploy shell into the container.
_propagated_envs = {
    k: os.environ[k]
    for k in ("CAMERA_CADENCE", "MAX_SIDE", "MAX_OUT_TOKENS")
    if k in os.environ
}

cam_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "gradio==5.42.0",
        "openai>=1.50.0",
        "pillow>=10.0.0",
        "fastapi[standard]",
    )
    .env(_propagated_envs)
)

app = modal.App(LIVE_CAM_APP_NAME, image=cam_image)


@app.function(cpu=1, memory=2048, scaledown_window=900)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    """Build and serve the Gradio live-camera UI."""
    import base64
    import io
    import os
    import threading

    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    from openai import OpenAI
    from PIL import Image

    from config import MODEL, resolve_vllm_url

    model_id = MODEL.model_id
    base_url = resolve_vllm_url() + "/v1"
    print(f"[live_camera] gradio={gr.__version__}  vllm={base_url}  model={model_id}", flush=True)
    client = OpenAI(base_url=base_url, api_key="not-used")

    # --- tunables ---
    CADENCE_S = float(os.environ.get("CAMERA_CADENCE", "3"))
    HISTORY_LEN = 4                                                # narrative context window
    MAX_SIDE = int(os.environ.get("MAX_SIDE", "384"))              # downsample long side
    MAX_OUT_TOKENS = int(os.environ.get("MAX_OUT_TOKENS", "60"))   # cap reply length

    # Explicit backpressure: drop frames that arrive while a caption is in flight.
    inflight = threading.Lock()

    INDEPENDENT_PROMPT = (
        "Describe this image in one short sentence. Direct, specific, no preamble."
    )
    NARRATIVE_PROMPT = (
        "You are narrating a live camera feed. Recent captions:\n{history}\n\n"
        "Describe the new frame in one short sentence. Focus on what changed. "
        "Present tense, no preamble."
    )

    def encode_frame(frame) -> str:
        """numpy RGB → base64 JPEG. Downsampled so the vision encoder is quick."""
        img = Image.fromarray(frame)
        img.thumbnail((MAX_SIDE, MAX_SIDE))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode()

    def format_history(history: list[str]) -> str:
        return "\n".join(f"- {h}" for h in history[-HISTORY_LEN:])

    def stash_frame(frame):
        """Fast no-op for gr.Image.stream — just records the latest frame."""
        return frame

    def caption_tick(frame, mode: str, history: list[str], running: bool):
        """gr.Timer handler: caption the latest stashed frame on cadence."""
        if frame is None or not running:
            return

        if not inflight.acquire(blocking=False):
            return

        try:
            img_b64 = encode_frame(frame)
            if mode == "narrative" and history:
                prompt = NARRATIVE_PROMPT.format(history=format_history(history))
            else:
                prompt = INDEPENDENT_PROMPT

            yield f"_querying **{model_id}**…_", history, format_history(history)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                        },
                    ],
                }
            ]

            try:
                stream = client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    stream=True,
                    temperature=0.4,
                    max_tokens=MAX_OUT_TOKENS,
                    extra_body={
                        # Caption use-case: skip thinking entirely.
                        "chat_template_kwargs": {"enable_thinking": False},
                        "skip_special_tokens": True,  # we don't parse <|channel> here
                    },
                )
            except Exception as e:
                yield f"**Error**: {e}", history, format_history(history)
                return

            reply = ""
            try:
                for chunk in stream:
                    delta = chunk.choices[0].delta.content or ""
                    if not delta:
                        continue
                    reply += delta
                    yield reply, history, format_history(history)
            except Exception as e:
                yield (f"**Error during streaming**: {e}\n\nPartial: {reply}",
                       history, format_history(history))
                return
            finally:
                stream.close()

            reply = reply.strip()
            if not reply:
                yield (
                    "_(empty response — model may have stopped on a stop token or "
                    "refused the frame)_",
                    history,
                    format_history(history),
                )
                return
            new_history = (history + [reply])[-HISTORY_LEN:]
            yield reply, new_history, format_history(new_history)
        finally:
            inflight.release()

    def start():
        return gr.Timer(value=CADENCE_S, active=True), True, [], "_Starting…_", ""

    def stop():
        return gr.Timer(active=False), False

    with gr.Blocks(title="Gemma 4 Live Camera") as demo:
        gr.Markdown(
            "# Gemma 4 Live Camera\n"
            "Webcam → Gemma 4 IT vision → streaming caption. "
            f"Captioning every **{CADENCE_S:g}s** "
            "(set `CAMERA_CADENCE` env var on the deploy to change). "
            f"Endpoint: `{base_url}` · model: `{model_id}`"
        )

        with gr.Row():
            mode = gr.Radio(
                ["narrative", "independent"], value="narrative", label="Mode",
                info="narrative = use recent captions as context (commentary feel)",
            )

        with gr.Row():
            with gr.Column(scale=1):
                cam = gr.Image(
                    sources=["webcam"], streaming=True, type="numpy",
                    label="Camera", height=360,
                )
                with gr.Row():
                    start_btn = gr.Button("Start", variant="primary")
                    stop_btn = gr.Button("Stop")
            with gr.Column(scale=1):
                caption = gr.Markdown(
                    "_Press **Start** to begin captioning…_",
                    label="Latest caption",
                )
                with gr.Accordion("Recent captions (narrative context)", open=False):
                    history_md = gr.Markdown()

        running = gr.State(False)
        history = gr.State([])
        latest_frame = gr.State(None)

        cam.stream(
            stash_frame,
            inputs=cam,
            outputs=latest_frame,
            stream_every=0.2,
            concurrency_limit=None,
        )

        caption_timer = gr.Timer(value=CADENCE_S, active=False)
        caption_timer.tick(
            caption_tick,
            inputs=[latest_frame, mode, history, running],
            outputs=[caption, history, history_md],
        )

        start_btn.click(
            start,
            outputs=[caption_timer, running, history, caption, history_md],
        )
        stop_btn.click(stop, outputs=[caption_timer, running])

    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")
