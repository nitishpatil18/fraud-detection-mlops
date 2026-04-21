"""database engine and helpers for writing prediction logs."""
from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.monitoring.schema import Base, PredictionLog

log = logging.getLogger(__name__)

_engine: Engine | None = None
_SessionLocal: sessionmaker[Session] | None = None


def get_database_url() -> str:
    """read postgres url from env, fall back to local docker default."""
    return os.environ.get(
        "DATABASE_URL",
        "postgresql+psycopg://fraud:fraud@localhost:5432/fraud",
    )


def init_db() -> Engine:
    """create engine + session factory, create tables if missing.

    idempotent. safe to call multiple times.
    """
    global _engine, _SessionLocal
    if _engine is not None:
        return _engine

    url = get_database_url()
    log.info("connecting to db: %s", url.split("@")[-1])  # don't log password
    _engine = create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=5)
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)

    Base.metadata.create_all(_engine)
    log.info("db ready, tables ensured")
    return _engine


@contextmanager
def session_scope() -> Iterator[Session]:
    """context manager for a db session with auto commit/rollback."""
    if _SessionLocal is None:
        init_db()
    assert _SessionLocal is not None
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def write_prediction_log(
    model_run_id: str,
    features: dict,
    fraud_probability: float,
    is_fraud: bool,
    threshold: float,
    latency_ms: float,
) -> None:
    """insert one prediction log row. swallows errors so serving never breaks."""
    try:
        with session_scope() as s:
            s.add(PredictionLog(
                model_run_id=model_run_id,
                features=features,
                fraud_probability=fraud_probability,
                is_fraud=is_fraud,
                threshold=threshold,
                latency_ms=latency_ms,
            ))
    except Exception:
        log.exception("failed to write prediction log")