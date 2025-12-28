CREATE EXTENSION IF NOT EXISTS "pgcrypto";

CREATE TABLE IF NOT EXISTS inference_requests (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  created_at timestamptz NOT NULL DEFAULT now(),
  status text NOT NULL,
  image_path text NOT NULL,
  image_sha256 text NOT NULL,
  weight_kg float8 NULL,
  predicted_label text NULL,
  confidence float8 NULL,
  top_k jsonb NULL,
  price_per_kg float8 NULL,
  total_price float8 NULL,
  model_id text NOT NULL,
  model_revision text NOT NULL,
  latency_ms int NULL,
  error text NULL,
  confirmed_label text NULL
);

CREATE TABLE IF NOT EXISTS product_prices (
  label text PRIMARY KEY,
  price_per_kg float8 NOT NULL
);

CREATE TABLE IF NOT EXISTS model_registry (
  id int PRIMARY KEY DEFAULT 1,
  model_id text NOT NULL,
  model_revision text NOT NULL,
  updated_at timestamptz NOT NULL DEFAULT now()
);

INSERT INTO model_registry (id, model_id, model_revision)
VALUES (1, 'Adriana213/vgg16-fruit-classifier', 'main')
ON CONFLICT (id) DO UPDATE
SET model_id = EXCLUDED.model_id,
    model_revision = EXCLUDED.model_revision,
    updated_at = now();

INSERT INTO product_prices (label, price_per_kg) VALUES
  ('Apple Braeburn', 2.49),
  ('Banana', 1.19),
  ('Kiwi', 3.49),
  ('Lemon', 2.10),
  ('Orange', 1.89),
  ('Pear', 2.29),
  ('Pineapple', 2.99),
  ('Strawberry', 4.99),
  ('Tomato 1', 1.99),
  ('Avocado', 3.99)
ON CONFLICT (label) DO NOTHING;

CREATE OR REPLACE VIEW metrics_daily AS
WITH daily AS (
  SELECT
    date_trunc('day', created_at)::date AS date,
    count(*) AS total_requests,
    avg(latency_ms) AS avg_latency_ms,
    percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95_latency_ms,
    avg(CASE WHEN confidence IS NOT NULL AND confidence < 0.55 THEN 1 ELSE 0 END) AS low_conf_rate,
    sum(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS error_count
  FROM inference_requests
  GROUP BY date_trunc('day', created_at)::date
),
top_labels AS (
  SELECT
    date_trunc('day', created_at)::date AS date,
    predicted_label,
    count(*) AS cnt,
    row_number() OVER (
      PARTITION BY date_trunc('day', created_at)::date
      ORDER BY count(*) DESC
    ) AS rn
  FROM inference_requests
  WHERE predicted_label IS NOT NULL
  GROUP BY date_trunc('day', created_at)::date, predicted_label
)
SELECT
  d.date,
  d.total_requests,
  d.avg_latency_ms,
  d.p95_latency_ms,
  d.low_conf_rate,
  t.predicted_label AS top_label,
  d.error_count
FROM daily d
LEFT JOIN top_labels t
  ON d.date = t.date AND t.rn = 1;
