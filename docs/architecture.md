# architecture

## overview

a production-style ml system that scores credit card transactions for fraud risk in real time. the system is structured as separate services (data prep, training, serving, monitoring) that communicate via shared artifacts in mlflow and a shared database.

## component diagram
┌─────────────────┐
      │  ieee-cis kaggle│
      │  dataset (csv)  │
      └────────┬────────┘
               │
      ┌────────▼────────┐
      │ prepare_data.py │  time-based split, categorical encoding
      └────────┬────────┘
               │
      ┌────────▼────────┐
      │ train.py        │  xgboost + scale_pos_weight, hydra config
      │ (offline)       │  logs to mlflow (params, metrics, model)
      └────────┬────────┘
               │
      ┌────────▼────────┐
      │ export_model.py │  downloads mlflow artifacts to build/model/
      └────────┬────────┘
               │
    ┌──────────▼──────────┐
    │ docker build        │  bake model into image
    │ (Dockerfile.prod)   │
    └──────────┬──────────┘
               │
  ┌────────────▼────────────┐       ┌───────────────┐
  │  render.com (singapore) │       │ render postgres│
  │  fastapi + baked model  │──────►│ prediction_logs│
  │  /health, /info, /predict│       │               │
  └────────────┬────────────┘       └───────┬───────┘
               │                             │
        http json                     sql queries
               │                             │
  ┌────────────▼────────────┐       ┌───────▼───────┐
  │ streamlit cloud ui      │       │ grafana (local)│
  │ predict / batch /       │       │ 8-panel dash   │
  │ monitoring tabs         │       └────────────────┘
  └─────────────────────────┘

  ## design principles

**separation of concerns.** data prep, training, serving, and monitoring are distinct scripts. each has one responsibility and communicates with others through artifacts (parquet, mlflow runs, database rows). this mirrors how production ml systems are built.

**training-serving skew prevention.** the same `apply_category_mappings` function is used at training time and at inference time. mappings are fit on the training split only, saved as json, and bundled with the model. the api never re-fits anything.

**model as build artifact.** the docker image bakes in a specific model run via `scripts/export_model.py`. the production container has no runtime dependency on mlflow. retraining means rebuilding the image.

**graceful degradation on logging failure.** the api logs every prediction to postgres in a background task. if the database is unreachable, the api still returns predictions. prediction logging is observability, not critical path.

**observability as a first-class concern.** every prediction row includes `model_run_id`, `latency_ms`, and the full input features (as json). this lets us query "what did model X predict between time A and B" without an application code change.

## data flow for a single prediction

1. client sends `POST /predict` with a transaction json.
2. pydantic validates the payload. malformed requests return 422 before the model sees them.
3. the api builds a 431-column dataframe, filling unknown columns with nan, and applies the saved category mappings.
4. xgboost returns a fraud probability in [0, 1].
5. the api returns `{fraud_probability, is_fraud, threshold, model_run_id}` synchronously.
6. in a background task, the request payload, response, model run id, and latency are written to postgres.

## technology choices

| layer | chosen | alternative | why |
|---|---|---|---|
| model | xgboost | random forest / deep nn | best tabular-data trade-off, handles missing data natively, fast training and inference |
| imbalance handling | `scale_pos_weight` | smote / undersampling | no data leakage, no synthetic samples, simpler to reason about |
| experiment tracking | mlflow | wandb, neptune | open source, self-hostable, the de-facto standard |
| config | hydra + yaml | argparse, click | composable configs, command-line overrides, standard in research codebases |
| api framework | fastapi | flask, django | async support, auto-generated swagger docs, pydantic validation |
| prediction log | postgres | redis, mongodb | structured queries for monitoring, jsonb for feature flexibility |
| drift detection | evidently ai | alibi-detect, whylogs | has presets for tabular drift, good html reports |
| dashboard | grafana | streamlit, bespoke | proper dashboard tool, industry standard |
| orchestration | docker compose (local), render (prod) | k8s, ecs | simplest way to get a real deploy on a student budget |
| ci | github actions | circleci, gitlab ci | free for public repos, integrated with github |

## known limitations

1. **inference latency is ~1s on render free tier** (0.1 cpu, 512mb ram). the per-request pandas overhead in `_prepare_row` dominates. converting to numpy arrays would cut this to <100ms.
2. **reference data is in a 50mb parquet**. the drift-check github actions job cannot run end-to-end because that file is not in git. a production fix is to precompute reference feature statistics into a small json.
3. **no shadow or canary deployment**. a new model replaces the old one immediately. in production i would run both in parallel, compare, then promote.
4. **single model, single threshold**. a real fraud system uses multiple model thresholds per merchant category, transaction amount, and risk tolerance.