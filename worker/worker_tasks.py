import json
import os
import time

import numpy as np
from PIL import Image
from sqlalchemy import text

from db import get_db_session
from logging_utils import setup_logging
from model_loader import embed_image, ensure_model
from worker_app import celery_app

DEFAULT_PRICE_PER_KG = float(os.getenv("DEFAULT_PRICE_PER_KG", "2.99"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))
LOG_PATH = os.getenv("LOG_PATH")

logger = setup_logging("smartscale.worker", LOG_PATH)


def _embedding_to_pgvector(embedding: np.ndarray) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in embedding.tolist()) + "]"


@celery_app.task(name="worker_tasks.classify")
def classify(job_id: str, top_k: int = 10) -> None:
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
        processor = state["processor"]

        img = Image.open(row["image_path"]).convert("RGB")
        query_embedding = embed_image(img, model, processor)
        query_vec = _embedding_to_pgvector(query_embedding)

        k = max(1, min(int(top_k), 10))
        matches = db.execute(
            text(
                """
                SELECT id, caption, label, image_path,
                       1 - (embedding <=> CAST(:query_vec AS vector)) AS similarity
                FROM reference_images
                WHERE embedding IS NOT NULL
                ORDER BY embedding <=> CAST(:query_vec AS vector)
                LIMIT :k
                """
            ),
            {"query_vec": query_vec, "k": k},
        ).mappings().all()

        top_k_list = []
        for m in matches:
            top_k_list.append({
                "ref_id": m["id"],
                "caption": m["caption"],
                "label": m["label"],
                "similarity": round(float(m["similarity"]), 4),
                "image_path": m["image_path"],
            })

        predicted_label = top_k_list[0]["label"] if top_k_list else None
        confidence = top_k_list[0]["similarity"] if top_k_list else None

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


@celery_app.task(name="worker_tasks.embed_reference")
def embed_reference(ref_id: int) -> None:
    db = get_db_session()
    try:
        row = db.execute(
            text("SELECT image_path FROM reference_images WHERE id = :id"),
            {"id": ref_id},
        ).mappings().first()
        if not row:
            logger.error("ref_missing", extra={"ref_id": ref_id})
            return

        state = ensure_model(db, logger)
        model = state["model"]
        processor = state["processor"]

        ref_storage = os.getenv("REFERENCE_STORAGE_PATH", "/data/reference")
        full_path = os.path.join(ref_storage, row["image_path"])
        img = Image.open(full_path).convert("RGB")
        embedding = embed_image(img, model, processor)
        vec_str = _embedding_to_pgvector(embedding)

        db.execute(
            text(
                "UPDATE reference_images SET embedding = CAST(:vec AS vector) WHERE id = :id"
            ),
            {"vec": vec_str, "id": ref_id},
        )
        db.commit()
        logger.info("ref_embedded", extra={"ref_id": ref_id})
    except Exception as exc:
        db.rollback()
        logger.error("ref_embed_error", extra={"ref_id": ref_id, "error": str(exc)})
    finally:
        db.close()
