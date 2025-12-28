import json
import os
import time
from typing import Any

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000/v1")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.55"))


def _post_predict(image_file, weight_kg: float | None, top_k: int) -> dict[str, Any]:
    files = {
        "image": (
            image_file.name,
            image_file.getvalue(),
            image_file.type or "application/octet-stream",
        )
    }
    data: dict[str, Any] = {"top_k": top_k}
    if weight_kg is not None:
        data["weight_kg"] = weight_kg
    response = requests.post(f"{API_BASE_URL}/predict", files=files, data=data, timeout=30)
    response.raise_for_status()
    return response.json()


def _get_result(job_id: str) -> dict[str, Any]:
    response = requests.get(f"{API_BASE_URL}/result/{job_id}", timeout=15)
    response.raise_for_status()
    return response.json()


def _post_confirm(job_id: str, confirmed_label: str) -> None:
    response = requests.post(
        f"{API_BASE_URL}/confirm/{job_id}",
        json={"confirmed_label": confirmed_label},
        timeout=10,
    )
    response.raise_for_status()


def _fetch_history(params: dict[str, Any]) -> dict[str, Any]:
    response = requests.get(f"{API_BASE_URL}/history", params=params, timeout=15)
    response.raise_for_status()
    return response.json()


st.set_page_config(page_title="SmartScale", layout="wide")

st.title("SmartScale")

weigh_tab, analytics_tab = st.tabs(["Weigh & classify", "Analytics"])

with weigh_tab:
    st.subheader("Weigh & classify")
    if "last_job_id" not in st.session_state:
        st.session_state["last_job_id"] = None
    if "last_result" not in st.session_state:
        st.session_state["last_result"] = None

    image_file = st.file_uploader("Upload produce image", type=["jpg", "jpeg", "png"])
    weight_kg = st.number_input("Weight (kg)", min_value=0.0, value=0.0, step=0.05)
    top_k = st.slider("Top-K alternatives", min_value=1, max_value=5, value=3)

    if st.button("Classify", type="primary", disabled=image_file is None):
        try:
            payload = _post_predict(image_file, weight_kg if weight_kg > 0 else None, top_k)
        except requests.RequestException as exc:
            st.error(f"Failed to submit job: {exc}")
            st.stop()

        job_id = payload["job_id"]
        st.session_state["last_job_id"] = job_id
        st.session_state["last_result"] = None
        st.info(f"Job submitted: {job_id}")

        with st.spinner("Running inference..."):
            result = {"status": "queued"}
            for _ in range(40):
                try:
                    result = _get_result(job_id)
                except requests.RequestException:
                    time.sleep(0.5)
                    continue
                if result["status"] in {"done", "error"}:
                    break
                time.sleep(0.5)

        st.session_state["last_result"] = result

    if st.session_state["last_job_id"]:
        if st.session_state["last_result"] is None:
            st.warning("No result yet. Click refresh to check status.")
        result = st.session_state["last_result"]
        job_id = st.session_state["last_job_id"]

        if st.button("Refresh status") and job_id:
            try:
                st.session_state["last_result"] = _get_result(job_id)
                result = st.session_state["last_result"]
            except requests.RequestException as exc:
                st.error(f"Failed to fetch result: {exc}")

        if result and result["status"] == "done":
            prediction = result.get("prediction") or {}
            st.success("Prediction ready")
            st.metric("Predicted label", prediction.get("predicted_label") or "-")
            st.metric("Confidence", f"{prediction.get('confidence', 0):.3f}")

            if prediction.get("price_per_kg") is not None:
                st.write(
                    f"Price per kg: ${prediction['price_per_kg']:.2f} | "
                    f"Total: ${prediction['total_price']:.2f}"
                )

            top_k_list = prediction.get("top_k") or []
            if top_k_list:
                st.write("Top alternatives")
                st.table(top_k_list)

            if prediction.get("confidence") is not None and prediction.get("confidence") < CONFIDENCE_THRESHOLD:
                st.warning("Low confidence. Please confirm the correct label.")
                options = [item["label"] for item in top_k_list] or []
                if options:
                    selected = st.selectbox("Select correct label", options)
                    if st.button("Confirm selection"):
                        try:
                            _post_confirm(job_id, selected)
                            st.success(f"Confirmed label saved: {selected}")
                        except requests.RequestException as exc:
                            st.error(f"Failed to confirm label: {exc}")
        elif result and result["status"] == "error":
            st.error(result.get("error") or "Inference failed")
        elif result:
            st.warning("Inference still running. Try refresh in a moment.")

with analytics_tab:
    st.subheader("Analytics")
    st.caption("Filters apply to history fetched from the API.")

    col1, col2, col3 = st.columns(3)
    with col1:
        label_filter = st.text_input("Label filter")
    with col2:
        min_conf = st.slider("Minimum confidence", min_value=0.0, max_value=1.0, value=0.0)
    with col3:
        limit = st.number_input("Rows", min_value=10, max_value=200, value=50, step=10)

    date_range = st.date_input("Date range", value=())
    params: dict[str, Any] = {"limit": int(limit), "offset": 0}
    if label_filter:
        params["label"] = label_filter
    if min_conf > 0:
        params["min_confidence"] = min_conf
    if len(date_range) == 2:
        params["date_from"] = date_range[0].isoformat()
        params["date_to"] = date_range[1].isoformat()

    try:
        data = _fetch_history(params)
        items = data.get("items", [])
    except requests.RequestException as exc:
        st.error(f"Failed to fetch history: {exc}")
        items = []

    if items:
        df = pd.DataFrame(items)
        if "top_k" in df.columns:
            df["top_k"] = df["top_k"].apply(
                lambda value: json.dumps(value, ensure_ascii=True)
                if isinstance(value, (list, dict))
                else value
            )
        st.dataframe(df, use_container_width=True)

        if "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"])
            requests_over_time = df.set_index("created_at").resample("D").size()
            if not requests_over_time.empty:
                st.line_chart(requests_over_time, height=200)

        if "predicted_label" in df.columns:
            top_labels = df["predicted_label"].value_counts().head(10)
            if not top_labels.empty:
                st.bar_chart(top_labels, height=200)

        if "confidence" in df.columns:
            conf_series = df["confidence"].dropna()
            if not conf_series.empty:
                hist = pd.cut(conf_series, bins=10).value_counts().sort_index()
                if not hist.empty:
                    hist_df = hist.reset_index()
                    hist_df.columns = ["bin", "count"]
                    hist_df["bin"] = hist_df["bin"].astype(str)
                    st.bar_chart(hist_df, x="bin", y="count", height=200)
    else:
        st.info("No history yet.")

    st.markdown("[OpenAPI docs](http://localhost:8000/docs)")
