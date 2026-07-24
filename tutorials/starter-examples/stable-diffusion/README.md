# Stable Diffusion Image Generation

Generate images from text prompts using SDXL Turbo on a GPU with [Modal](https://modal.com).

## What it does

- Loads the SDXL Turbo model from HuggingFace (no auth required)
- Generates an image from a text prompt using the diffusers pipeline
- Runs on an A10G GPU, caching model weights in a `modal.Volume` so re-runs skip the download
- Returns the PNG bytes; the `local_entrypoint` writes them to `output.png`

## Setup

```bash
cd tutorials/starter-examples/stable-diffusion

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
uv run modal run stable_diffusion.py --prompt "a cat astronaut floating in space, digital art"
```

The image is written to `output.png` in your working directory.

## Notes

- The GPU (`gpu="A10G"`) and CUDA 12.4 PyTorch are configured in the `modal.Image` / `@app.function` — no cluster to set up
- SDXL Turbo is optimized for few steps — try `--steps 4` for faster results
- First run downloads the model into the `hf-cache` volume; later runs reuse it
- To use a cheaper GPU, change `gpu="A10G"` to `gpu="T4"` in `stable_diffusion.py`
