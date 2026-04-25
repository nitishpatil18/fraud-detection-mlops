"""generate prediction traffic using real test data.

usage:
  python scripts/generate_traffic.py                  # 500 predictions
  python scripts/generate_traffic.py --n 1000         # 1000 predictions
  python scripts/generate_traffic.py --n 500 --drift  # inject synthetic drift
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import time

import httpx
import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000/predict")

# a reasonable subset of fields to send - realistic production callers
# rarely have all 431 features
FIELDS_TO_SEND = [
    "TransactionAmt",
    "ProductCD",
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
    "DeviceType",
    "DeviceInfo",
    "C1",
    "C2",
    "C13",
    "C14",
    "D1",
    "D2",
    "D15",
    "V95",
    "V96",
    "V97",
]


def load_sample_transactions(n: int) -> pd.DataFrame:
    """pull a random sample of RAW transactions (pre-encoding).

    the api's pydantic schema expects string categoricals (like ProductCD='W')
    and float numerics, which is what real callers send. the processed parquet
    has everything encoded to ints, so we load raw here.
    """
    from src.config import RAW_TXN_PATH

    df = pd.read_csv(RAW_TXN_PATH, nrows=50000)
    return df.sample(n=n, random_state=random.randint(0, 10000))


def apply_synthetic_drift(df: pd.DataFrame) -> pd.DataFrame:
    """simulate production drift: 3x transaction amounts, force ProductCD='C'."""
    df = df.copy()
    df["TransactionAmt"] = df["TransactionAmt"].astype(float) * 3.0
    if "ProductCD" in df.columns:
        df["ProductCD"] = "C"
    return df


def row_to_payload(row: pd.Series) -> dict:
    """convert a row to a request json, keeping only common fields."""
    payload = {}
    for field in FIELDS_TO_SEND:
        if field in row and pd.notna(row[field]):
            val = row[field]
            # json doesn't like numpy types
            if isinstance(val, np.integer | np.int32 | np.int64):
                val = int(val)
            elif isinstance(val, np.floating | np.float32 | np.float64):
                val = float(val)
            payload[field] = val
    return payload


def send_prediction(client: httpx.Client, payload: dict) -> dict:
    r = client.post(API_URL, json=payload, timeout=10.0)
    r.raise_for_status()
    return r.json()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument(
        "--drift",
        action="store_true",
        help="inject synthetic drift",
    )
    args = parser.parse_args()

    log.info("loading sample transactions")
    sample = load_sample_transactions(args.n)

    if args.drift:
        log.info("applying synthetic drift")
        sample = apply_synthetic_drift(sample)

    log.info("sending %d requests", args.n)
    start = time.perf_counter()
    fraud_count = 0
    with httpx.Client() as client:
        for i, (_, row) in enumerate(sample.iterrows(), start=1):
            payload = row_to_payload(row)
            try:
                resp = send_prediction(client, payload)
                if resp["is_fraud"]:
                    fraud_count += 1
            except Exception:
                log.exception("request %d failed", i)
            if i % 100 == 0:
                log.info("sent %d/%d", i, args.n)

    elapsed = time.perf_counter() - start
    log.info(
        "done. %d requests in %.1fs (%.1f rps), %d flagged as fraud",
        args.n,
        elapsed,
        args.n / elapsed,
        fraud_count,
    )


if __name__ == "__main__":
    main()
