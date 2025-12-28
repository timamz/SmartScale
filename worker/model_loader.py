import json
import os
from datetime import datetime, timezone

import tensorflow as tf
from huggingface_hub import snapshot_download
from sqlalchemy import text

MODEL_ID = os.getenv("MODEL_ID", "Adriana213/vgg16-fruit-classifier")
MODEL_REVISION = os.getenv("MODEL_REVISION", "main")

MODEL_STATE = {
    "model_id": None,
    "model_revision": None,
    "model": None,
    "labels": None,
    "model_path": None,
    "loaded_at": None,
}


def _fetch_registry(db):
    row = db.execute(
        text("SELECT model_id, model_revision FROM model_registry WHERE id = 1")
    ).mappings().first()
    if not row:
        return {"model_id": MODEL_ID, "model_revision": MODEL_REVISION}
    return {"model_id": row["model_id"], "model_revision": row["model_revision"]}


def _load_labels(labels_path: str, logger=None) -> list[str] | None:
    if not os.path.exists(labels_path):
        if logger:
            logger.warning("labels_missing", extra={"labels_path": labels_path})
        return None
    with open(labels_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    items = sorted(data.items(), key=lambda item: int(item[0]))
    return [label for _, label in items]


def _load_model(model_id: str, model_revision: str, logger=None):
    model_path = snapshot_download(repo_id=model_id, revision=model_revision)
    labels = _load_labels(os.path.join(model_path, "class_labels.json"), logger=logger)
    model = tf.keras.models.load_model(model_path, compile=False)
    return model, labels, model_path


def ensure_model(db, logger):
    target = _fetch_registry(db)
    if (
        MODEL_STATE["model"] is None
        or MODEL_STATE["model_id"] != target["model_id"]
        or MODEL_STATE["model_revision"] != target["model_revision"]
    ):
        model, labels, model_path = _load_model(
            target["model_id"], target["model_revision"], logger
        )
        MODEL_STATE.update(
            {
                "model_id": target["model_id"],
                "model_revision": target["model_revision"],
                "model": model,
                "labels": labels,
                "model_path": model_path,
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
