# design decisions

this document captures the key choices made while building this system. for each, the alternatives and the reasoning.

## 1. time-based train/val/test split, not random

**question**: how to split 590k transactions across 182 days into train/val/test?

**chosen**: sort by `TransactionDT`, take the last 15% as test, the prior 15% as val, the rest as train.

**why**: this is time-series data. a random split lets the model "see the future", which inflates validation metrics and breaks in production. the time-based split mirrors how the model would see data when deployed. the ~0.02 pr-auc gap between val and test confirms real temporal drift in this dataset.

## 2. pr-auc as the primary metric

**question**: what's the right metric for a 3.5% positive class?

**chosen**: pr-auc (average precision). report roc-auc, recall at p=0.5, and recall at p=0.9 as secondary.

**why**: accuracy is useless at 3.5% imbalance (97% baseline from always predicting the majority class). roc-auc looks artificially high on imbalanced data because the true-negative pile is so big. pr-auc directly measures the precision-recall trade-off that matters for a business threshold.

## 3. `scale_pos_weight` instead of resampling

**question**: how to handle the 27:1 class imbalance?

**chosen**: set `scale_pos_weight = neg / pos` in xgboost, no oversampling or undersampling.

**why**: keeps the full dataset, no synthetic samples, no information loss. smote on 431 columns with heavy missingness is unreliable. `scale_pos_weight` is a clean one-liner that adjusts the loss function directly.

## 4. label encoding for 31 categoricals, not one-hot

**question**: how to encode product codes, card networks, emails, etc.?

**chosen**: integer label encoding, with mappings fit on train only.

**why**: xgboost (tree-based) handles label-encoded categoricals as well as one-hot, without the feature explosion. one-hot would push the feature count past 5,000 and slow training significantly. fitting on train only prevents leakage and mimics the inference-time behavior for unseen categories (mapped to -1).

## 5. separate `prepare_data.py` and `train.py`

**question**: should feature engineering happen inside the training script?

**chosen**: two scripts. `prepare_data.py` writes parquet files once. `train.py` only reads parquet.

**why**: parquet is 10x faster to read than csv, which makes experiment iteration faster. more importantly, it enforces that training cannot secretly recompute features differently from serving. feature code lives in `features.py` and is imported by both the prep script and the api, so both share the same transformation logic.

## 6. fastapi + pydantic for the serving layer

**question**: flask, fastapi, or a grpc service?

**chosen**: fastapi with pydantic schemas.

**why**: input validation comes for free via pydantic. malformed requests get 422 with a clear error before the model sees them. `/docs` gives us a swagger ui for zero extra code, which is valuable for demos and debugging. async support is built in. the ergonomics are better than flask for modern python.

## 7. background task for prediction logging

**question**: should the api write to postgres synchronously on each request?

**chosen**: enqueue writes in a fastapi `BackgroundTasks`, let the response return first.

**why**: logging is observability, not a user-facing feature. adding db latency to every response hurts users for no benefit. additionally, `write_prediction_log` catches all exceptions, so if the db is unreachable the api still works. this is the standard "degrade gracefully" pattern.

## 8. bake the model into the docker image for production

**question**: should production load the model from mlflow, or embed it?

**chosen**: `scripts/export_model.py` downloads the model to `build/model/`, the dockerfile copies it in.

**why**: production should not depend on the experiment tracker at runtime. mlflow is a training-time tool. baking the model in means deploys are deterministic, deploys are self-contained, and there's no network dependency on mlflow at request time. the model-loader module supports both paths: `MODEL_LOCAL_DIR` (prod) or `MODEL_RUN_ID` (dev).

## 9. postgres jsonb for the feature payload

**question**: how to store the full input payload per prediction?

**chosen**: single `jsonb` column in the `prediction_logs` table, not a wide schema.

**why**: the model has 431 features, most often null per request. a wide schema would be brittle across model versions. jsonb supports indexing and direct queries, is fast for our workload, and makes schema migration trivial when the feature set changes.

## 10. separate `Dockerfile.ci` and `Dockerfile.prod`

**question**: ci doesn't have the baked model; how do we build the image there?

**chosen**: two dockerfiles. ci builds `Dockerfile.ci` (no model copy, just import smoke test). prod uses `Dockerfile.prod` (with model copy).

**why**: ci should verify the python environment and imports without depending on training artifacts. mixing the two into one dockerfile with conditional logic is fragile. two small, purpose-specific files are clearer than one complex file.

## things i would change given more time

- **convert single-row inference from pandas to numpy.** the pandas overhead is the dominant cost at inference; this would drop production latency from ~1s to <100ms.
- **precompute reference feature statistics** into a small json so the drift-check github action can run end-to-end in ci.
- **add a proper model registry stage** (`staging` -> `production`) in mlflow, with a cli to promote runs and trigger a rebuild.
- **shadow deployment** for new models: run the new and old models side by side, log both, compare.
- **per-feature drift thresholds** rather than a single share-of-drifted-columns threshold. some features drifting is less risky than others.
- **typing coverage**: run mypy in strict mode and get to zero errors.