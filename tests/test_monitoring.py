"""tests for prediction logging.

uses an in-memory sqlite db so no real postgres is needed.
"""

from __future__ import annotations

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from src.monitoring.schema import Base, PredictionLog


def test_schema_round_trip() -> None:
    """insert a row, read it back, confirm all fields match."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    with session_factory() as s:
        s.add(
            PredictionLog(
                model_run_id="abc123",
                features={"TransactionAmt": 50.0, "ProductCD": "W"},
                fraud_probability=0.23,
                is_fraud=False,
                threshold=0.5,
                latency_ms=4.2,
            )
        )
        s.commit()

    with session_factory() as s:
        rows = s.execute(select(PredictionLog)).scalars().all()
        assert len(rows) == 1
        row = rows[0]
        assert row.model_run_id == "abc123"
        assert row.features["TransactionAmt"] == 50.0
        assert row.fraud_probability == 0.23
        assert row.is_fraud is False
        assert row.ts is not None  # default set by sqlalchemy
