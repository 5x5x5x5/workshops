import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "torchvision",
        extra_index_url="https://download.pytorch.org/whl/cu124",
    )
    .pip_install("diffusers==0.31.0", "transformers==4.44.2", "accelerate")
)

app = modal.App("stable-diffusion", image=image)

# Cache HuggingFace model weights across runs so the model is only downloaded once.
hf_cache = modal.Volume.from_name("hf-cache", create_if_missing=True)
HF_CACHE_PATH = "/root/.cache/huggingface"


@app.function(gpu="A10G", timeout=600, volumes={HF_CACHE_PATH: hf_cache})
def generate(prompt: str, steps: int = 30) -> bytes:
    """Generate an image from a text prompt using Stable Diffusion."""
    import torch
    from diffusers import AutoPipelineForText2Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = AutoPipelineForText2Image.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        variant="fp16" if device == "cuda" else None,
    ).to(device)

    image = pipe(prompt, num_inference_steps=steps, guidance_scale=0.0).images[0]

    path = "/tmp/output.png"
    image.save(path)
    with open(path, "rb") as f:
        return f.read()


@app.local_entrypoint()
def main(prompt: str = "a cat astronaut floating in space, digital art", steps: int = 30):
    png = generate.remote(prompt, steps)

    with open("output.png", "wb") as f:
        f.write(png)
    print(f"Wrote output.png ({len(png)} bytes) for prompt: {prompt!r}")


# uv run modal run stable_diffusion.py --prompt "a cat astronaut floating in space, digital art"
