import json
import os
import time

import numpy as np
from sqlalchemy import text
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image as keras_image

from db import get_db_session
from logging_utils import setup_logging
from model_loader import ensure_model
from worker_app import celery_app

DEFAULT_PRICE_PER_KG = float(os.getenv("DEFAULT_PRICE_PER_KG", "2.99"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
LOG_PATH = os.getenv("LOG_PATH")
TARGET_SIZE = (100, 100)

logger = setup_logging("smartscale.worker", LOG_PATH)


@celery_app.task(name="worker_tasks.classify")
def classify(job_id: str, top_k: int = 3) -> None:
    start_time = time.time()
    db = get_db_session()
    try:
        row = db.execute(
            text("SELECT image_path, weight_kg FROM inference_requests WHERE id = :id"),
            {"id": job_id},
        ).mappings().first()
        if not row:
            logger.error("job_missing", extra={"job_id": job_id})
            return

        db.execute(
            text("UPDATE inference_requests SET status = 'running' WHERE id = :id"),
            {"id": job_id},
        )
        db.commit()

        state = ensure_model(db, logger)
        model = state["model"]
        labels = state["labels"] or []

        img = keras_image.load_img(row["image_path"], target_size=TARGET_SIZE)
        x = keras_image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        predictions = model.predict(x, verbose=0)
        probs = np.squeeze(predictions)

        k = max(1, min(int(top_k), 5, len(probs)))
        indices = np.argsort(probs)[::-1][:k]
        top_k_list = []
        for idx in indices.tolist():
            label = labels[int(idx)] if int(idx) < len(labels) else str(idx)
            top_k_list.append({"label": label, "confidence": float(probs[int(idx)])})

        predicted_label = top_k_list[0]["label"] if top_k_list else None
        confidence = top_k_list[0]["confidence"] if top_k_list else None

        price_per_kg = None
        total_price = None
        if predicted_label and row["weight_kg"] is not None:
            price_row = db.execute(
                text("SELECT price_per_kg FROM product_prices WHERE label = :label"),
                {"label": predicted_label},
            ).mappings().first()
            price_per_kg = price_row["price_per_kg"] if price_row else DEFAULT_PRICE_PER_KG
            total_price = price_per_kg * row["weight_kg"]

        latency_ms = int((time.time() - start_time) * 1000)
        db.execute(
            text(
                """
                UPDATE inference_requests
                SET status = 'done',
                    predicted_label = :predicted_label,
                    confidence = :confidence,
                    top_k = CAST(:top_k AS jsonb),
                    price_per_kg = :price_per_kg,
                    total_price = :total_price,
                    latency_ms = :latency_ms,
                    model_id = :model_id,
                    model_revision = :model_revision
                WHERE id = :id
                """
            ),
            {
                "id": job_id,
                "predicted_label": predicted_label,
                "confidence": confidence,
                "top_k": json.dumps(top_k_list),
                "price_per_kg": price_per_kg,
                "total_price": total_price,
                "latency_ms": latency_ms,
                "model_id": state["model_id"],
                "model_revision": state["model_revision"],
            },
        )
        db.commit()

        logger.info(
            "job_completed",
            extra={
                "job_id": job_id,
                "status": "done",
                "predicted_label": predicted_label,
                "confidence": confidence,
                "latency_ms": latency_ms,
                "low_confidence": confidence is not None and confidence < CONFIDENCE_THRESHOLD,
            },
        )
    except Exception as exc:
        db.rollback()
        db.execute(
            text(
                "UPDATE inference_requests SET status = 'error', error = :error WHERE id = :id"
            ),
            {"id": job_id, "error": str(exc)},
        )
        db.commit()
        logger.error("job_error", extra={"job_id": job_id, "error": str(exc)})
    finally:
        db.close()
