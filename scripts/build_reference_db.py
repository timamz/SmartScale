#!/usr/bin/env python3
"""Embed reference images with DINOv2 and populate the reference_images table.

Reads images from a directory structure: {root}/{label}/{variety}.jpg
For each image, computes a DINOv2 embedding and inserts into PostgreSQL (pgvector).

Usage (standalone):
    python scripts/build_reference_db.py \
        --source-dir ./reference_data \
        --db-url postgresql://smartscale:smartscale@localhost:5432/smartscale

Usage (inside Docker):
    docker-compose exec worker python /app/scripts/build_reference_db.py \
        --source-dir /data/reference \
        --db-url postgresql://smartscale:smartscale@db:5432/smartscale
"""

import argparse
import os

import numpy as np
import torch
from PIL import Image
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from transformers import AutoImageProcessor, AutoModel


def embed_image(pil_image: Image.Image, model, processor) -> np.ndarray:
    inputs = processor(images=pil_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0]
    cls_embedding = torch.nn.functional.normalize(cls_embedding, p=2, dim=1)
    return cls_embedding.squeeze().numpy()


def embedding_to_pgvector(embedding: np.ndarray) -> str:
    return "[" + ",".join(f"{v:.8f}" for v in embedding.tolist()) + "]"


def main():
    parser = argparse.ArgumentParser(description="Build reference image DB")
    parser.add_argument("--source-dir", required=True, help="Root of reference images")
    parser.add_argument(
        "--db-url",
        default="postgresql+psycopg2://smartscale:smartscale@localhost:5432/smartscale",
    )
    parser.add_argument("--model-id", default="facebook/dinov2-base")
    args = parser.parse_args()

    print(f"Loading model {args.model_id}...")
    processor = AutoImageProcessor.from_pretrained(args.model_id)
    model = AutoModel.from_pretrained(args.model_id)
    model.eval()
    print("Model loaded.")

    db_url = args.db_url
    if not db_url.startswith("postgresql+"):
        db_url = db_url.replace("postgresql://", "postgresql+psycopg2://", 1)
    engine = create_engine(db_url)
    Session = sessionmaker(bind=engine)
    db = Session()

    existing = set()
    rows = db.execute(text("SELECT image_path FROM reference_images")).fetchall()
    for r in rows:
        existing.add(r[0])

    source_dir = args.source_dir

    # Count how many images are on disk
    total_on_disk = 0
    for ld in os.listdir(source_dir):
        lp = os.path.join(source_dir, ld)
        if os.path.isdir(lp):
            for f in os.listdir(lp):
                if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                    total_on_disk += 1

    if total_on_disk > 0 and len(existing) >= total_on_disk:
        print(f"Reference DB is already populated ({len(existing)} images). Nothing to do.")
        print("To re-populate, clear the table first:")
        print("  docker compose exec db psql -U smartscale -c 'DELETE FROM reference_images;'")
        db.close()
        return

    if existing:
        print(f"Found {len(existing)} existing entries, will skip those.")

    inserted = 0
    skipped = 0
    errors = 0

    for label_dir in sorted(os.listdir(source_dir)):
        label_path = os.path.join(source_dir, label_dir)
        if not os.path.isdir(label_path):
            continue

        label = label_dir.replace("_", " ").title()

        for img_file in sorted(os.listdir(label_path)):
            if not img_file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                continue

            rel_path = f"{label_dir}/{img_file}"
            if rel_path in existing:
                skipped += 1
                continue

            full_path = os.path.join(label_path, img_file)
            variety = os.path.splitext(img_file)[0].replace("_", " ").title()
            caption = f"{label} {variety}" if variety != label else label

            try:
                img = Image.open(full_path).convert("RGB")
                embedding = embed_image(img, model, processor)
                vec_str = embedding_to_pgvector(embedding)

                db.execute(
                    text(
                        """
                        INSERT INTO reference_images (caption, label, image_path, embedding)
                        VALUES (:caption, :label, :image_path, CAST(:embedding AS vector))
                        """
                    ),
                    {
                        "caption": caption,
                        "label": label,
                        "image_path": rel_path,
                        "embedding": vec_str,
                    },
                )
                db.commit()
                inserted += 1
                print(f"  [{inserted}] {label} / {caption}")
            except Exception as e:
                db.rollback()
                errors += 1
                print(f"  ERROR: {rel_path}: {e}")

    db.close()
    print(f"\nDone: {inserted} inserted, {skipped} skipped, {errors} errors")


if __name__ == "__main__":
    main()
