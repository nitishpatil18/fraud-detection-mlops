"""streamlit frontend for the fraud detection api.

three tabs:
  - predict: single transaction form -> probability + decision
  - batch: upload a csv, get predictions back
  - monitoring: charts from the postgres prediction log

deploy via streamlit community cloud: just point it at this file on github.
"""
from __future__ import annotations

import os
from io import StringIO

import httpx
import pandas as pd
import streamlit as st

API_URL = os.environ.get("API_URL", "https://fraud-api-5dxn.onrender.com")
DATABASE_URL = os.environ.get("DATABASE_URL", "")

st.set_page_config(
    page_title="fraud detection demo",
    page_icon="🛡️",
    layout="wide",
)


# -----------------------------------------------------------------------------
# api helpers
# -----------------------------------------------------------------------------

@st.cache_resource
def http_client() -> httpx.Client:
    return httpx.Client(timeout=30.0)


def call_predict(payload: dict) -> dict:
    r = http_client().post(f"{API_URL}/predict", json=payload)
    r.raise_for_status()
    return r.json()


def call_info() -> dict:
    r = http_client().get(f"{API_URL}/info")
    r.raise_for_status()
    return r.json()


def call_health() -> dict:
    try:
        r = http_client().get(f"{API_URL}/health", timeout=60.0)
        return r.json()
    except Exception as e:
        return {"status": "unreachable", "error": str(e)}


# -----------------------------------------------------------------------------
# ui
# -----------------------------------------------------------------------------

st.title("fraud detection")
st.caption(f"api: {API_URL}")

with st.sidebar:
    st.header("service status")
    health = call_health()
    if health.get("status") == "ok":
        st.success(f"api: ok (model loaded: {health.get('model_loaded')})")
    else:
        st.error(f"api: {health.get('status')}")

    try:
        info = call_info()
        st.metric("features", info["n_features"])
        st.metric("threshold", info["threshold"])
        st.caption(f"model run: `{info['model_run_id'][:12]}...`")
    except Exception as e:
        st.warning(f"info endpoint failed: {e}")

tab_predict, tab_batch, tab_monitor = st.tabs(["predict", "batch", "monitoring"])


# -----------------------------------------------------------------------------
# tab: single prediction
# -----------------------------------------------------------------------------

with tab_predict:
    st.subheader("score a single transaction")

    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input(
            "transaction amount (usd)", min_value=0.0, value=75.5, step=10.0,
        )
        product_cd = st.selectbox(
            "product code", ["W", "C", "H", "R", "S"], index=0,
        )
        card4 = st.selectbox(
            "card network",
            ["visa", "mastercard", "american express", "discover"],
            index=0,
        )
        card6 = st.selectbox("card type", ["debit", "credit"], index=0)
    with col2:
        email_domain = st.selectbox(
            "payer email domain",
            ["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "other"],
            index=0,
        )
        device_type = st.selectbox(
            "device type", ["desktop", "mobile"], index=0,
        )
        card5 = st.number_input("card issuer bin", value=100.0)
        addr1 = st.number_input("billing addr bucket", value=315.0)

    if st.button("predict", type="primary"):
        payload = {
            "TransactionAmt": amount,
            "ProductCD": product_cd,
            "card4": card4,
            "card6": card6,
            "card5": card5,
            "addr1": addr1,
            "P_emaildomain": email_domain,
            "DeviceType": device_type,
        }
        with st.spinner("calling model..."):
            try:
                resp = call_predict(payload)
            except Exception as e:
                st.error(f"prediction failed: {e}")
                st.stop()

        proba = resp["fraud_probability"]
        is_fraud = resp["is_fraud"]

        # color-coded decision
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("fraud probability", f"{proba:.2%}")
        with col_b:
            st.metric("threshold", f"{resp['threshold']:.2%}")
        with col_c:
            if is_fraud:
                st.metric("decision", "⚠️ flag as fraud")
            else:
                st.metric("decision", "✅ approve")

        # risk band
        if proba < 0.05:
            st.success("low risk")
        elif proba < 0.5:
            st.warning("medium risk")
        else:
            st.error("high risk")

        st.caption(f"served by model run: `{resp['model_run_id']}`")
        with st.expander("request payload"):
            st.json(payload)
        with st.expander("full response"):
            st.json(resp)


# -----------------------------------------------------------------------------
# tab: batch
# -----------------------------------------------------------------------------

with tab_batch:
    st.subheader("batch scoring")
    st.caption(
        "upload a csv with transaction columns. the api accepts any subset "
        "of fields; missing ones are filled with nan by the model."
    )

    uploaded = st.file_uploader("csv file", type=["csv"])
    max_rows = st.slider("max rows to score", 10, 500, 100, step=10)

    if uploaded is not None:
        try:
            df = pd.read_csv(uploaded, nrows=max_rows)
        except Exception as e:
            st.error(f"could not read csv: {e}")
            st.stop()

        st.write(f"loaded {len(df)} rows, {df.shape[1]} columns")
        st.dataframe(df.head(10))

        if st.button("score all", type="primary"):
            results = []
            progress = st.progress(0, text="scoring...")
            for i, (_, row) in enumerate(df.iterrows(), start=1):
                payload = {
                    k: (v if pd.notna(v) else None)
                    for k, v in row.to_dict().items()
                }
                # json doesn't like numpy types - coerce
                for k, v in payload.items():
                    if hasattr(v, "item"):
                        payload[k] = v.item()
                try:
                    resp = call_predict(payload)
                    results.append({
                        **row.to_dict(),
                        "fraud_probability": resp["fraud_probability"],
                        "is_fraud": resp["is_fraud"],
                    })
                except Exception as e:
                    results.append({
                        **row.to_dict(),
                        "fraud_probability": None,
                        "is_fraud": None,
                        "error": str(e),
                    })
                progress.progress(i / len(df), text=f"scored {i}/{len(df)}")
            progress.empty()

            out = pd.DataFrame(results)
            st.success(
                f"done. flagged {int(out['is_fraud'].sum())} of {len(out)} "
                f"({100 * out['is_fraud'].mean():.1f}%)"
            )
            st.dataframe(out[["fraud_probability", "is_fraud"]].head(20))

            # download link for the full scored csv
            csv_buf = StringIO()
            out.to_csv(csv_buf, index=False)
            st.download_button(
                "download results csv",
                csv_buf.getvalue(),
                file_name="scored.csv",
                mime="text/csv",
            )


# -----------------------------------------------------------------------------
# tab: monitoring
# -----------------------------------------------------------------------------

with tab_monitor:
    st.subheader("prediction log")
    if not DATABASE_URL:
        st.info(
            "set the DATABASE_URL environment variable to read prediction logs. "
            "grafana dashboard at http://localhost:3000 for the full view."
        )
    else:
        try:
            from sqlalchemy import create_engine
            engine = create_engine(DATABASE_URL, pool_pre_ping=True)
            df = pd.read_sql(
                "select ts, fraud_probability, is_fraud, latency_ms, model_run_id "
                "from prediction_logs order by ts desc limit 1000",
                engine,
            )
        except Exception as e:
            st.error(f"could not query db: {e}")
            st.stop()

        if df.empty:
            st.info("no predictions logged yet. send some via the predict tab.")
        else:
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("total predictions", f"{len(df):,}")
            col2.metric("flagged rate", f"{100 * df['is_fraud'].mean():.1f}%")
            col3.metric("avg latency", f"{df['latency_ms'].mean():.0f} ms")
            col4.metric("p95 latency", f"{df['latency_ms'].quantile(0.95):.0f} ms")

            df_bucket = (
                df.set_index("ts")
                .groupby(pd.Grouper(freq="5min"))
                .agg(requests=("is_fraud", "count"),
                     avg_proba=("fraud_probability", "mean"))
                .reset_index()
            )
            c1, c2 = st.columns(2)
            with c1:
                st.caption("requests per 5 min")
                st.line_chart(df_bucket.set_index("ts")["requests"])
            with c2:
                st.caption("avg fraud probability per 5 min")
                st.line_chart(df_bucket.set_index("ts")["avg_proba"])

            st.caption("last 20 predictions")
            st.dataframe(df.head(20))