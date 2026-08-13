"""FastAPI inference service: GET /health and POST /predict."""
import io
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dataset import CIFAR10_CLASSES, get_transforms  # noqa: E402
from model import get_model  # noqa: E402

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt")

STATE: dict = {"model": None, "classes": CIFAR10_CLASSES, "loaded": False}
_transform = get_transforms(train=False)


def _load_model() -> None:
    path = Path(CHECKPOINT_PATH)
    if not path.exists():
        return
    checkpoint = torch.load(path, map_location="cpu")
    architecture = checkpoint.get("architecture", "resnet18")
    num_classes = checkpoint.get("num_classes", len(CIFAR10_CLASSES))
    model = get_model(architecture=architecture, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    STATE["model"] = model
    STATE["loaded"] = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield
    STATE["model"] = None
    STATE["loaded"] = False


app = FastAPI(title="CIFAR-10 Classifier API", lifespan=lifespan)


@app.get("/health")
def health():
    if not STATE["loaded"]:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ok", "checkpoint": CHECKPOINT_PATH, "classes": len(STATE["classes"])}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if not STATE["loaded"]:
        raise HTTPException(status_code=503, detail="model not loaded")

    raw = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"invalid image: {exc}") from exc

    tensor = _transform(pil_image).unsqueeze(0)
    with torch.no_grad():
        logits = STATE["model"](tensor)
        probabilities = F.softmax(logits, dim=1).squeeze(0).tolist()

    scored = sorted(
        (
            {"class": name, "probability": round(prob, 6)}
            for name, prob in zip(STATE["classes"], probabilities)
        ),
        key=lambda item: item["probability"],
        reverse=True,
    )
    return {"filename": image.filename, "top": scored[0]["class"], "predictions": scored}
