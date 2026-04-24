# fraud detection mlops

[![ci](https://github.com/nitishpatil18/fraud-detection-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/nitishpatil18/fraud-detection-mlops/actions/workflows/ci.yml)
[![docker](https://github.com/nitishpatil18/fraud-detection-mlops/actions/workflows/docker.yml/badge.svg)](https://github.com/nitishpatil18/fraud-detection-mlops/actions/workflows/docker.yml)

an end-to-end machine learning system that scores credit card transactions for fraud risk in real time. built from scratch as a final-year project to demonstrate production-style mlops practices: experiment tracking, model serving, monitoring, ci/cd, and cloud deployment.

## live demo

- **frontend**: https://nitish-fraud-detection.streamlit.app
- **api**: https://fraud-api-5dxn.onrender.com
- **api docs**: https://fraud-api-5dxn.onrender.com/docs
- **health**: https://fraud-api-5dxn.onrender.com/health

(the api runs on render free tier, which sleeps after 15 min idle. first request after sleep takes ~30s to cold-start.)

## results

| metric | value |
|---|---|
| dataset | ieee-cis, 590k transactions, 3.5% fraud rate |
| model | xgboost (300 trees, depth 6) |
| test pr-auc | 0.51 |
| test roc-auc | 0.89 |
| test recall at precision=0.5 | 0.45 |
| test recall at precision=0.9 | 0.27 |
| production inference latency | ~1s p50 (render free tier, 0.1 cpu) |

## what's inside

- **training pipeline**: hydra config, mlflow tracking, time-based train/val/test split
- **feature pipeline**: parquet storage, category-mapping fit-on-train-only, saved as artifact
- **serving api**: fastapi with pydantic validation, model baked into docker image
- **prediction logging**: every request persisted to postgres asynchronously
- **drift monitoring**: evidently ai, daily cron via github actions, configurable threshold
- **dashboard**: grafana with 8 panels reading from postgres
- **frontend**: streamlit app with predict / batch / monitoring tabs
- **ci/cd**: github actions runs lint, format, type check, tests, and docker build on every push

## architecture

see [docs/architecture.md](docs/architecture.md) for the component diagram and data-flow description.

## design decisions

see [docs/decisions.md](docs/decisions.md) for the ~10 key choices made while building this (why pr-auc, why time-based split, why bake the model into docker, etc.).

## stack

- **model**: xgboost 2.1
- **training/tracking**: python 3.11, hydra, mlflow
- **api**: fastapi, pydantic, uvicorn
- **database**: postgres 16 (render managed)
- **monitoring**: evidently ai, grafana
- **frontend**: streamlit
- **container**: docker
- **ci**: github actions
- **deploy**: render (api + db), streamlit community cloud (frontend)

## local development

```bash
# clone and set up env
git clone https://github.com/nitishpatil18/fraud-detection-mlops.git
cd fraud-detection-mlops
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

download the dataset from kaggle:

```bash
kaggle competitions download -c ieee-fraud-detection -p data/raw
cd data/raw && unzip ieee-fraud-detection.zip && rm ieee-fraud-detection.zip && cd ../..
```

start services (postgres + grafana):

```bash
docker compose up -d
```

start mlflow server in a separate terminal:

```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 --port 5000
```

run the feature pipeline and train:

```bash
python -m src.prepare_data
python -m src.train
```

override any hyperparameter from the cli:

```bash
python -m src.train run_name=exp1 model.max_depth=8 model.learning_rate=0.1
```

start the api pointing at an mlflow run:

```bash
export MODEL_RUN_ID=<your_run_id>
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
export DATABASE_URL=postgresql+psycopg://fraud:fraud@localhost:5432/fraud
uvicorn src.api.main:app --reload
```

or build and run the production image:

```bash
MODEL_RUN_ID=<your_run_id> python -m scripts.export_model
docker build -f Dockerfile.prod -t fraud-api:local .
docker run --rm -p 8000:8000 -e DATABASE_URL=... fraud-api:local
```

start the frontend:

```bash
streamlit run frontend/app.py
```

## testing

```bash
pytest tests/ -v
ruff check src/ tests/ scripts/
ruff format --check src/ tests/ scripts/
```

## project structure
configs/         hydra yaml configs
data/
raw/           raw csvs from kaggle (gitignored)
processed/     parquet features + category mappings
build/           exported model artifacts for docker baking
src/
config.py      paths and constants
data.py        raw csv loading, time-based split
features.py    categorical encoding, fit/apply pattern
prepare_data.py feature pipeline script
evaluate.py    metrics (pr-auc, recall at precision)
train.py       training with mlflow + hydra
api/           fastapi app (main, model_loader, schemas)
monitoring/    postgres logging + evidently drift detection
scripts/
export_model.py  download model + mappings from mlflow
generate_traffic.py  send synthetic predictions for testing
check_drift.py   one-shot drift check with exit codes
frontend/
app.py         streamlit ui
grafana/dashboards/ provisioned dashboard json
tests/           pytest unit tests (features, api, monitoring)
docs/
architecture.md  system diagram
decisions.md     key design choices
problem.md       problem statement
report.md        academic project report
.github/workflows/
ci.yml         lint + test on every push
docker.yml     image build on every push
drift-check.yml daily drift cron
Dockerfile.ci    image for ci (no baked model)
Dockerfile.prod  image for production (model baked in)
docker-compose.yml postgres + grafana for local dev
render.yaml      render deploy config
pyproject.toml   ruff + pytest config

## dataset

ieee-cis fraud detection, originally from the 2019 kaggle competition. 590,540 labeled transactions with 394 features (transaction + device identity). fraud rate 3.5%. data covers 182 consecutive days.

## license

mit

## author

nitish patil. built as a final-year project at ramaiah institute of technology.