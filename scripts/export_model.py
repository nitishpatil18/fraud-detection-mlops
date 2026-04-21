"""export a model + mappings from mlflow to a local folder.

used by the dockerfile to bake a specific model run into the image so
the deployed service doesn't need mlflow at runtime.

usage:
  MODEL_RUN_ID=<run_id> python -m scripts.export_model
  MODEL_RUN_ID=<run_id> EXPORT_DIR=build/model python -m scripts.export_model
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

import mlflow

log = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    run_id = os.environ.get("MODEL_RUN_ID")
    if not run_id:
        raise RuntimeError("MODEL_RUN_ID env var required")

    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    export_dir = Path(os.environ.get("EXPORT_DIR", "build/model"))

    mlflow.set_tracking_uri(tracking_uri)
    log.info("exporting run %s to %s", run_id, export_dir)

    if export_dir.exists():
        log.info("cleaning existing export dir")
        shutil.rmtree(export_dir)
    export_dir.mkdir(parents=True)

    # pull the xgboost model artifact folder
    log.info("downloading model artifact")
    mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="model",
        dst_path=str(export_dir),
    )

    # pull the category mappings file
    log.info("downloading category_mappings.json")
    mlflow.artifacts.download_artifacts(
        run_id=run_id,
        artifact_path="category_mappings.json",
        dst_path=str(export_dir),
    )

    # write the run id into a small metadata file so the api knows what it loaded
    (export_dir / "run_id.txt").write_text(run_id)

    log.info("export complete")
    for p in sorted(export_dir.rglob("*")):
        rel = p.relative_to(export_dir)
        if p.is_file():
            log.info("  %s (%.1fKB)", rel, p.stat().st_size / 1024)


if __name__ == "__main__":
    main()
