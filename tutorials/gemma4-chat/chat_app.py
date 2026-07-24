"""Gradio chat UI for Gemma 4, fronting the vLLM server, served on Modal.

The UI is a Gradio app mounted on FastAPI and served as a Modal ASGI app. The
vLLM server's URL is resolved at container startup via `resolve_vllm_url()`
(looks up the deployed vLLM web endpoint by name, or honors `VLLM_URL`).

Dev (hot-reload, ephemeral URL):
    modal serve chat_app.py

Deploy (persistent URL — run after vllm_server.py is deployed):
    modal deploy chat_app.py
"""

from __future__ import annotations

import modal

from config import CHAT_APP_NAME, MODEL


# Frontend image. We don't need vllm here — just an OpenAI client + Gradio.
# Gradio 5.x is required for `gr.Chatbot(type="messages")` (the metadata-titled
# 🧠 Thinking panel relies on the messages format). Gradio 6.x dropped that
# kwarg and 4.x doesn't have messages format at all.
chat_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("gradio==5.42.0", "openai>=1.50.0", "fastapi[standard]")
)

app = modal.App(CHAT_APP_NAME, image=chat_image)


def _split_thinking(text: str) -> tuple[str, str]:
    """Split a Gemma-4 chat response into (thinking, answer).

    The IT model emits its chain-of-thought wrapped between two special
    tokens, which vLLM renders as text in the streamed completion:
      <|channel>thought
      ...reasoning...
      <channel|>
      ...final answer...

    These are present whenever the model is capable of thinking, even when
    thinking is disabled (in which case the thought block is empty). We
    treat content between the markers as thinking, and content after the
    closing marker as the answer.

    Robust to partial markers so it can be called incrementally on a
    growing streaming buffer.
    """
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


@app.function(cpu=1, memory=2048, scaledown_window=1800)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def ui():
    """Build and serve the Gradio chat UI."""
    import gradio as gr
    from fastapi import FastAPI
    from gradio.routes import mount_gradio_app
    from openai import OpenAI

    from config import MODEL, resolve_vllm_url

    model_id = MODEL.model_id
    base_url = resolve_vllm_url() + "/v1"
    print(f"[chat] gradio={gr.__version__}  vllm={base_url}  model={model_id}", flush=True)
    client = OpenAI(base_url=base_url, api_key="not-used")

    DEFAULT_SYSTEM = "You are a helpful assistant."

    # Rough chars-per-token heuristic for converting the user-facing thinking-
    # budget slider (in tokens) to a character cap on the streamed buffer.
    CHARS_PER_TOKEN = 3.5

    # Internal hard cap on total tokens (thinking + answer). The user-facing
    # control is the "Thinking budget" slider; this is just a safety ceiling
    # so the model can't ramble past it. Well below the model's max_model_len.
    MAX_TOTAL_TOKENS = 4096

    def chat(message, history, system_prompt, enable_thinking, think_budget,
             temperature, top_p):
        if not message or not message.strip():
            yield "", history
            return

        history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": "", "metadata": {"title": "🧠 Thinking"}},
            {"role": "assistant", "content": ""},
        ]
        yield "", history

        # Build the messages list for /v1/chat/completions. The -it model's
        # chat_template.jinja handles `<|turn>...<turn|>` formatting and the
        # `<|think|>` insertion when we set `chat_template_kwargs.enable_thinking`.
        sys_text = system_prompt.strip() or "You are a helpful assistant."
        msgs = [{"role": "system", "content": sys_text}]
        for t in history[:-2]:
            if "metadata" in t:
                continue   # skip the prior thinking-panel placeholders
            msgs.append({"role": t["role"], "content": t["content"]})

        budget_chars = int(think_budget * CHARS_PER_TOKEN) if think_budget else 0

        stream = client.chat.completions.create(
            model=model_id,
            messages=msgs,
            stream=True,
            temperature=float(temperature),
            top_p=float(top_p),
            max_tokens=MAX_TOTAL_TOKENS,
            extra_body={
                # Forwarded to the chat template as `enable_thinking=...`.
                # vLLM passes this through to the jinja template kwargs.
                "chat_template_kwargs": {"enable_thinking": bool(enable_thinking)},
                # Keep special tokens like <|channel> / <channel|> in the
                # streamed output so _split_thinking can find the boundary
                # between the thought block and the answer. Default is True,
                # which strips them and leaves us with just `thought\n...`.
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
                history[-2]["content"] = thinking
                history[-1]["content"] = answer
                yield "", history

                # Cap thinking length: if we've exceeded the budget AND the
                # model hasn't started the answer yet (no `<channel|>` seen
                # → answer is still empty), abort and do a second pass.
                if (budget_chars and not answer and len(thinking) >= budget_chars):
                    capped = True
                    break
        finally:
            stream.close()

        if capped:
            history[-2]["content"] += f"\n\n_[capped at ~{think_budget} tokens]_"
            yield "", history

            # Second pass: thinking disabled, force a direct answer using the
            # truncated thought as priming context. The original Ollama
            # version did exactly this; OpenAI-compat API maps cleanly.
            followup = msgs + [
                {"role": "assistant", "content": history[-2]["content"]},
                {"role": "user", "content": "Stop thinking. Give your final answer now, concisely."},
            ]
            answer_stream = client.chat.completions.create(
                model=model_id,
                messages=followup,
                stream=True,
                temperature=float(temperature),
                top_p=float(top_p),
                max_tokens=MAX_TOTAL_TOKENS,
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
                    # With thinking disabled the model still emits an empty
                    # `<|channel>thought\n<channel|>` envelope before the
                    # answer — strip it the same way.
                    _, ans = _split_thinking(buf2)
                    history[-1]["content"] = ans
                    yield "", history
            finally:
                answer_stream.close()

        # If the model never wrote a `<think>...</think>` block, drop the
        # empty thinking placeholder so the UI doesn't show a blank panel.
        if not history[-2]["content"]:
            history.pop(-2)
            yield "", history

    with gr.Blocks(title=f"Gemma 4 Chat ({model_id})") as demo:
        gr.Markdown(
            f"# Gemma 4 Chat\n"
            f"Served by vLLM on Modal. Model: `{model_id}` · Endpoint: `{base_url}`"
        )
        with gr.Row():
            temperature = gr.Slider(0.0, 1.5, value=1.0, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.95, step=0.05, label="Top-p")
            think_budget = gr.Slider(
                0, 4000, value=0, step=100,
                label="Thinking budget (tokens, 0 = unlimited)",
                info="Caps the chain-of-thought. When hit, we stop reasoning and re-prompt for a direct answer.",
            )
        with gr.Row():
            system_prompt = gr.Textbox(
                value=DEFAULT_SYSTEM, label="System prompt", lines=2, scale=4,
            )
            enable_thinking = gr.Checkbox(
                value=True, label="Enable thinking",
                info="Adds <|think|> to the system prompt — model reasons step-by-step before answering.",
                scale=1,
            )
        chatbot = gr.Chatbot(type="messages", label="Conversation", height=500)
        msg = gr.Textbox(label="Your message", placeholder="Type and press Enter")
        with gr.Row():
            send = gr.Button("Send", variant="primary")
            clear = gr.Button("Clear")

        inputs = [msg, chatbot, system_prompt, enable_thinking, think_budget, temperature, top_p]
        outputs = [msg, chatbot]
        msg.submit(chat, inputs=inputs, outputs=outputs)
        send.click(chat, inputs=inputs, outputs=outputs)
        clear.click(lambda: [], outputs=chatbot)

    return mount_gradio_app(app=FastAPI(), blocks=demo, path="/")
