import os
from datetime import datetime, timezone

import numpy as np
import torch
from PIL import Image
from sqlalchemy import text
from transformers import AutoImageProcessor, AutoModel

MODEL_ID = os.getenv("MODEL_ID", "facebook/dinov2-base")
MODEL_REVISION = os.getenv("MODEL_REVISION", "main")

MODEL_STATE = {
    "model_id": None,
    "model_revision": None,
    "model": None,
    "processor": None,
    "loaded_at": None,
}


def _fetch_registry(db):
    row = db.execute(
        text("SELECT model_id, model_revision FROM model_registry WHERE id = 1")
    ).mappings().first()
    if not row:
        return {"model_id": MODEL_ID, "model_revision": MODEL_REVISION}
    return {"model_id": row["model_id"], "model_revision": row["model_revision"]}


def _load_model(model_id: str, model_revision: str, logger=None):
    processor = AutoImageProcessor.from_pretrained(model_id, revision=model_revision)
    model = AutoModel.from_pretrained(model_id, revision=model_revision)
    model.eval()
    return model, processor


def ensure_model(db, logger):
    target = _fetch_registry(db)
    if (
        MODEL_STATE["model"] is None
        or MODEL_STATE["model_id"] != target["model_id"]
        or MODEL_STATE["model_revision"] != target["model_revision"]
    ):
        model, processor = _load_model(
            target["model_id"], target["model_revision"], logger
        )
        MODEL_STATE.update(
            {
                "model_id": target["model_id"],
                "model_revision": target["model_revision"],
                "model": model,
                "processor": processor,
                "loaded_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        logger.info(
            "model_loaded",
            extra={
                "model_id": target["model_id"],
                "model_revision": target["model_revision"],
                "loaded_at": MODEL_STATE["loaded_at"],
            },
        )
    return MODEL_STATE


def embed_image(pil_image: Image.Image, model, processor) -> np.ndarray:
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0]
    cls_embedding = torch.nn.functional.normalize(cls_embedding, p=2, dim=1)
    return cls_embedding.squeeze().numpy()
