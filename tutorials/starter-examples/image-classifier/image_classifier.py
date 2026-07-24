"""
Image Classification Training on Modal

Fine-tune a pretrained ResNet on the Beans dataset from HuggingFace.

Usage:
    uv run modal run image_classifier.py --num-epochs 3
"""

import modal

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "torchvision", "datasets", "Pillow"
)

app = modal.App("image-classifier", image=image)

# Shared volume so `load_data` can hand the prepared tensors to `train`.
data_volume = modal.Volume.from_name("image-classifier-data", create_if_missing=True)
DATA_PATH = "/data"


@app.function(cpu=2, memory=8192, timeout=1800, volumes={DATA_PATH: data_volume})
def load_data() -> str:
    """Download the Beans dataset and save as tensors to the shared volume."""
    import torch
    from datasets import load_dataset
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    ds = load_dataset("beans", split="train")
    images, labels = [], []
    for sample in ds:
        images.append(transform(sample["image"].convert("RGB")))
        labels.append(sample["labels"])

    path = f"{DATA_PATH}/beans_data.pt"
    torch.save({"images": torch.stack(images), "labels": torch.tensor(labels)}, path)
    data_volume.commit()
    print(f"Saved {len(images)} samples to {path}")
    return path


@app.function(gpu="T4", memory=8192, timeout=1800, volumes={DATA_PATH: data_volume})
def train(data_path: str, num_epochs: int = 3, lr: float = 0.001) -> bytes:
    """Fine-tune ResNet18 on the Beans dataset."""
    import io

    import torch
    import torch.nn as nn
    from torchvision import models

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = torch.load(data_path, weights_only=False)
    images, labels = dataset["images"].to(device), dataset["labels"].to(device)
    print(f"Training on {len(images)} images, {labels.unique().numel()} classes")

    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, labels.unique().numel())
    model.to(device).train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    batch_size = 32

    for epoch in range(num_epochs):
        epoch_loss, correct = 0.0, 0
        for i in range(0, len(images), batch_size):
            batch_x = images[i:i + batch_size]
            batch_y = labels[i:i + batch_size]

            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(batch_x)
            correct += (preds.argmax(1) == batch_y).sum().item()

        acc = correct / len(images) * 100
        print(f"Epoch {epoch + 1}/{num_epochs} — loss: {epoch_loss / len(images):.4f}, acc: {acc:.1f}%")

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


@app.function()
def pipeline(num_epochs: int = 3) -> bytes:
    """Load data → Train model."""
    data_path = load_data.remote()
    return train.remote(data_path, num_epochs=num_epochs)


@app.local_entrypoint()
def main(num_epochs: int = 3):
    model_bytes = pipeline.remote(num_epochs)
    out = "resnet_beans.pt"
    with open(out, "wb") as f:
        f.write(model_bytes)
    print(f"Wrote trained model to {out} ({len(model_bytes)} bytes)")


# uv run modal run image_classifier.py --num-epochs 3
