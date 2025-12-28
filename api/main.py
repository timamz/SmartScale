import hashlib
import os
import time
import uuid
from datetime import datetime
from typing import Any

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from celery_app import celery_app
from db import get_db
from logging_utils import setup_logging

IMAGE_STORAGE_PATH = os.getenv("IMAGE_STORAGE_PATH", "/data/images")
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "changeme")
MODEL_ID = os.getenv("MODEL_ID", "Adriana213/vgg16-fruit-classifier")
MODEL_REVISION = os.getenv("MODEL_REVISION", "main")
LOG_PATH = os.getenv("LOG_PATH")

logger = setup_logging("smartscale.api", LOG_PATH)

app = FastAPI(title="SmartScale API", version="1.0.0")


class PredictResponse(BaseModel):
    job_id: str
    status: str


class PredictionResult(BaseModel):
    predicted_label: str | None
    confidence: float | None
    top_k: list[dict[str, Any]] | None
    price_per_kg: float | None
    total_price: float | None
    confirmed_label: str | None


class ResultResponse(BaseModel):
    status: str
    prediction: PredictionResult | None = None
    error: str | None = None


class ReloadRequest(BaseModel):
    model_id: str | None = None
    model_revision: str | None = None


class ConfirmRequest(BaseModel):
    confirmed_label: str


class ModelInfo(BaseModel):
    model_id: str
    model_revision: str
    updated_at: datetime


def _ensure_storage_dir() -> None:
    os.makedirs(IMAGE_STORAGE_PATH, exist_ok=True)


def _fetch_model_registry(db: Session) -> dict[str, str]:
    row = db.execute(
        text("SELECT model_id, model_revision FROM model_registry WHERE id = 1")
    ).mappings().first()
    if not row:
        return {"model_id": MODEL_ID, "model_revision": MODEL_REVISION}
    return {"model_id": row["model_id"], "model_revision": row["model_revision"]}


@app.post("/v1/predict", response_model=PredictResponse)
async def predict(
    image: UploadFile = File(...),
    weight_kg: float | None = Form(None),
    top_k: int = Form(3),
    db: Session = Depends(get_db),
) -> PredictResponse:
    if top_k < 1:
        raise HTTPException(status_code=400, detail="top_k must be >= 1")
    if weight_kg is not None and weight_kg <= 0:
        raise HTTPException(status_code=400, detail="weight_kg must be > 0")

    _ensure_storage_dir()
    job_id = str(uuid.uuid4())
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="empty image payload")

    image_sha = hashlib.sha256(image_bytes).hexdigest()
    ext = os.path.splitext(image.filename or "")[1] or ".jpg"
    image_path = os.path.join(IMAGE_STORAGE_PATH, f"{job_id}{ext}")
    with open(image_path, "wb") as f:
        f.write(image_bytes)

    model_info = _fetch_model_registry(db)
    db.execute(
        text(
            """
            INSERT INTO inference_requests (
                id, status, image_path, image_sha256, weight_kg, model_id, model_revision
            ) VALUES (
                :id, :status, :image_path, :image_sha256, :weight_kg, :model_id, :model_revision
            )
            """
        ),
        {
            "id": job_id,
            "status": "queued",
            "image_path": image_path,
            "image_sha256": image_sha,
            "weight_kg": weight_kg,
            "model_id": model_info["model_id"],
            "model_revision": model_info["model_revision"],
        },
    )
    db.commit()

    celery_app.send_task("worker_tasks.classify", args=[job_id, top_k])

    logger.info(
        "job_queued",
        extra={
            "job_id": job_id,
            "status": "queued",
            "image_sha256": image_sha,
            "weight_kg": weight_kg,
        },
    )

    return PredictResponse(job_id=job_id, status="queued")


@app.get("/v1/result/{job_id}", response_model=ResultResponse)
def result(job_id: str, db: Session = Depends(get_db)) -> ResultResponse:
    row = db.execute(
        text(
            """
            SELECT status, predicted_label, confidence, top_k, price_per_kg,
                   total_price, error, confirmed_label
            FROM inference_requests WHERE id = :id
            """
        ),
        {"id": job_id},
    ).mappings().first()

    if not row:
        raise HTTPException(status_code=404, detail="job_id not found")

    prediction = None
    if row["status"] == "done":
        prediction = PredictionResult(
            predicted_label=row["predicted_label"],
            confidence=row["confidence"],
            top_k=row["top_k"],
            price_per_kg=row["price_per_kg"],
            total_price=row["total_price"],
            confirmed_label=row["confirmed_label"],
        )

    return ResultResponse(status=row["status"], prediction=prediction, error=row["error"])


@app.get("/v1/history")
def history(
    limit: int = 50,
    offset: int = 0,
    label: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    min_confidence: float | None = None,
    db: Session = Depends(get_db),
):
    if limit < 1 or limit > 500:
        raise HTTPException(status_code=400, detail="limit must be 1..500")

    clauses = ["1=1"]
    params: dict[str, Any] = {"limit": limit, "offset": offset}

    if label:
        clauses.append("predicted_label = :label")
        params["label"] = label
    if date_from:
        clauses.append("created_at >= :date_from")
        params["date_from"] = date_from
    if date_to:
        clauses.append("created_at <= :date_to")
        params["date_to"] = date_to
    if min_confidence is not None:
        clauses.append("confidence >= :min_confidence")
        params["min_confidence"] = min_confidence

    query = text(
        f"""
        SELECT id, created_at, status, predicted_label, confidence, top_k,
               weight_kg, price_per_kg, total_price, confirmed_label, error
        FROM inference_requests
        WHERE {' AND '.join(clauses)}
        ORDER BY created_at DESC
        LIMIT :limit OFFSET :offset
        """
    )
    rows = db.execute(query, params).mappings().all()
    return {"items": rows, "limit": limit, "offset": offset}


@app.get("/v1/health")
def health():
    return {"status": "ok", "time": time.time()}


@app.post("/v1/confirm/{job_id}")
def confirm_label(job_id: str, payload: ConfirmRequest, db: Session = Depends(get_db)):
    result = db.execute(
        text(
            """
            UPDATE inference_requests
            SET confirmed_label = :confirmed_label
            WHERE id = :id
            RETURNING id
            """
        ),
        {"id": job_id, "confirmed_label": payload.confirmed_label},
    ).mappings().first()
    if not result:
        raise HTTPException(status_code=404, detail="job_id not found")
    db.commit()

    logger.info(
        "label_confirmed",
        extra={"job_id": job_id, "confirmed_label": payload.confirmed_label},
    )
    return {"status": "ok", "job_id": job_id, "confirmed_label": payload.confirmed_label}


@app.post("/v1/admin/reload-model", response_model=ModelInfo)
def reload_model(
    payload: ReloadRequest,
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    db: Session = Depends(get_db),
) -> ModelInfo:
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="invalid admin token")

    current = _fetch_model_registry(db)
    model_id = payload.model_id or current["model_id"]
    model_revision = payload.model_revision or current["model_revision"]

    row = db.execute(
        text(
            """
            UPDATE model_registry
            SET model_id = :model_id,
                model_revision = :model_revision,
                updated_at = now()
            WHERE id = 1
            RETURNING model_id, model_revision, updated_at
            """
        ),
        {"model_id": model_id, "model_revision": model_revision},
    ).mappings().first()
    db.commit()

    logger.info(
        "model_reload_requested",
        extra={
            "model_id": row["model_id"],
            "model_revision": row["model_revision"],
        },
    )

    return ModelInfo(**row)


@app.get("/v1/admin/model", response_model=ModelInfo)
def model_info(
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
    db: Session = Depends(get_db),
) -> ModelInfo:
    if x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="invalid admin token")

    row = db.execute(
        text("SELECT model_id, model_revision, updated_at FROM model_registry WHERE id = 1")
    ).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="model registry not initialized")

    return ModelInfo(**row)
