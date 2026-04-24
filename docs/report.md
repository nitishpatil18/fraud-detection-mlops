# fraud detection mlops: a production-style end-to-end system

**author**: nitish patil  
**institute**: ramaiah institute of technology  
**department**: computer science and engineering (ai & ml)  
**project**: final-year major project  

## abstract

this project presents an end-to-end machine learning system for credit card fraud detection. unlike typical student projects that stop at model training, this work implements the complete lifecycle: data preparation, experiment tracking, model serving via a rest api, prediction logging, drift monitoring, a web-based frontend, automated ci/cd, and public cloud deployment. the system is built around the ieee-cis fraud detection dataset (590,540 transactions, 3.5% fraud rate, 182 days) and achieves a test-set pr-auc of 0.51 using gradient boosted trees. the emphasis of the work is not on model accuracy but on the engineering patterns required to deploy and operate a machine learning model in production. the final system exposes a public rest api (https://fraud-api-5dxn.onrender.com) and a streamlit frontend (https://nitish-fraud-detection.streamlit.app), with monitoring infrastructure that captures every prediction for subsequent drift analysis.

## 1. introduction

credit card fraud detection is one of the most widely studied applications of machine learning, and it is also one where the gap between research models and production systems is largest. a notebook that trains a high-accuracy model is not deployable software. this project was motivated by the observation that recruiters hiring for machine learning engineering roles look for evidence that candidates can ship a model into production, monitor it, and handle failure modes such as data drift. existing final-year projects in the department predominantly focus on model training in jupyter notebooks; this project instead focuses on the engineering around the model.

the work is structured as a multi-service system with clear boundaries: offline data preparation and training, an online serving api, a monitoring pipeline, a dashboard, and a frontend. each component is implemented using industry-standard open source tooling (mlflow, fastapi, postgres, grafana, evidently, docker, github actions) and deployed to free-tier cloud services (render, streamlit community cloud). the result is a system that can be demonstrated as a live application to reviewers and interviewers, rather than solely as a code submission.

## 2. problem formulation

given a credit card transaction characterized by 394 features (transaction amount, product code, card information, device identity, and hundreds of anonymized engineered features), predict the probability that the transaction is fraudulent. the prediction is then compared against a configurable threshold to produce a binary decision.

the target metric for model selection is precision-recall area under curve (pr-auc), which is appropriate for the heavy class imbalance (3.5% positive rate) that characterizes this problem. accuracy is not a useful metric because a model that always predicts the majority class achieves 96.5% accuracy while catching zero fraud. additional secondary metrics are recall at fixed precision thresholds, which map more directly to business trade-offs between false positives (blocking legitimate customers) and false negatives (letting fraud pass through).

the business constraint is that transactions must be scored in real time, so the system must support low-latency single-transaction inference. the engineering constraint is that the model must be monitorable in production: it must be possible to observe prediction volume, prediction distribution, latency, and data drift over time, without instrumenting each caller.

## 3. dataset

the ieee-cis fraud detection dataset, published as a kaggle competition in 2019, comprises 590,540 labeled transactions spanning 182 consecutive days. each transaction carries 394 columns: identifiers, timing, amount, product code, card network, billing address, email domains, device information, and 339 anonymized feature columns (v1 through v339) engineered by the dataset provider. the dataset is distributed across two joinable tables, `train_transaction` and `train_identity`, connected by a shared transaction id. only approximately 24% of transactions have associated identity data; the rest arrive with all identity columns null, which is informative in itself.

exploratory data analysis revealed three properties that shaped subsequent design. first, the fraud rate varies non-monotonically with transaction amount, suggesting that linear models would underperform tree-based alternatives on this problem. second, product code is a strong univariate predictor, with product `c` exhibiting a 11.7% fraud rate versus 2.0% for product `w`. third, 174 of 394 columns are more than 50% null, indicating that the pattern of missingness itself is likely informative, which rules out aggressive imputation strategies that would discard this signal.

## 4. system architecture

the system is organized as four logical layers: data preparation, training, serving, and monitoring. each layer is a distinct python module with a narrow contract, and communication between layers happens through persistent artifacts rather than direct function calls. this separation mirrors how production machine learning systems are typically structured, where training is a batch process that produces an artifact, and serving consumes that artifact.

the data preparation layer is a single script (`prepare_data.py`) that loads the raw csv files, performs a time-based split into train, validation, and test sets, fits categorical mappings on the training set only, applies those mappings to all three splits, and writes the output to parquet files in `data/processed/`. the parquet format is both smaller (approximately 76 mb versus 1.3 gb raw) and faster to read, which enables rapid iteration during training experiments. the category mappings are written as a json file alongside the parquet, so that the inference path can apply identical transformations without reloading training data.

the training layer (`train.py`) reads the parquet files and trains an xgboost classifier, using `scale_pos_weight` to account for class imbalance. all hyperparameters are exposed through a hydra yaml config, enabling command-line overrides for experiment sweeps. every run logs its parameters, metrics, and model artifact to an mlflow tracking server, providing a persistent record of all experiments and enabling direct comparison between runs in the mlflow ui. the category mappings file is also logged as an artifact, so that a specific model run is always paired with the specific mappings that were used to train it.

the serving layer is a fastapi application that loads a model and its mappings at startup. the loader supports two modes: loading from an mlflow run (used in development, where the tracking server is reachable) and loading from a local folder (used in production, where the model is baked into the docker image). the api exposes three endpoints: `/health` for liveness checks, `/info` for model metadata, and `/predict` for inference. input is validated via pydantic, which rejects malformed requests before they reach the model. the `/predict` handler applies the category mappings to the incoming payload, constructs a one-row dataframe padded with nulls for features the caller did not provide, and returns the model's fraud probability along with the discrete decision and the model's run id.

the monitoring layer consists of two complementary components. first, the api logs every prediction to a postgres table through a fastapi background task, so that logging does not add latency to responses. each row captures the timestamp, model run id, input features (stored as jsonb for schema flexibility), prediction probability, decision, and inference latency in milliseconds. second, a scheduled drift-check script compares the distribution of recent live predictions against the training reference distribution using evidently ai, and exits with a non-zero code when the share of drifted features exceeds a configurable threshold. a github actions cron is configured to run this check daily, and the alert-on-failure step is where a real deployment would integrate slack, pagerduty, or an automated retraining trigger.

production deployment uses docker. the training artifacts are exported from mlflow into a local folder, then copied into the image at build time; the resulting container has no runtime dependency on mlflow. the api is deployed to render's singapore region on the free tier, and the postgres database is a managed render instance in the same region. the frontend is a streamlit application deployed to streamlit community cloud, which calls the render api over https and reads the prediction log directly from postgres. a grafana dashboard, run locally via docker compose, provides 8 panels of real-time monitoring data sourced from the same postgres database.

## 5. implementation notes

the project is written in python 3.11 and consists of approximately 1,800 lines of application code plus 400 lines of tests. dependencies are pinned in `requirements.txt`. continuous integration runs on every push via github actions: the pipeline installs dependencies, runs ruff for linting and format checking, mypy for type checking, and pytest for the test suite. a separate github actions workflow builds the docker image and verifies that the api module imports correctly inside the container. tests use `fastapi.testclient` and `pytest.monkeypatch` to test the api without a live model or database, by injecting a mocked model-loader function.

several subtle correctness issues were encountered and resolved during development. the most instructive was training-serving skew arising from dtype mismatches: the postgres prediction log stored categorical features as their raw string representations (such as `"W"`), while the reference parquet stored them as label-encoded integers. a naive drift comparison between the two produced nonsensical results because every categorical feature appeared as entirely null after dtype coercion. the fix was to apply the saved category mappings to the current data before comparison, ensuring that reference and current frames share the same encoded representation.

another issue was that the drift-check script, when given small current samples and a full training reference, reported high drift rates even when the current data was drawn from the test split. this is a known artifact of the kolmogorov-smirnov test used by evidently, which is sensitive to sample size asymmetry and to the subset of features actually populated by the traffic generator. in a production system, the appropriate fix is to precompute reference statistics over a fixed window and to drift-check only the features that are expected to be populated.

## 6. results and evaluation

the baseline model achieves a test-set pr-auc of 0.510, with roc-auc of 0.89, recall of 0.45 at precision 0.5, and recall of 0.27 at precision 0.9. the val-to-test degradation of roughly 0.03 pr-auc is consistent with the temporal drift observed in the dataset: the fraud rate rises gradually across the 182-day window, so the final 15% of days differ somewhat from the middle 15% used for validation. the model used was xgboost with 300 trees, max depth 6, learning rate 0.05, and `scale_pos_weight` set to the negative-to-positive ratio in the training split.

the model was compared against a shallower configuration (max depth 4) and a deeper training run (500 trees). the shallower configuration underperformed on every metric, confirming that the dataset supports deeper trees without overfitting within the number of estimators tested. the 500-tree configuration improved pr-auc by less than 0.01 while increasing training time by 40%, so 300 trees was chosen for the deployed baseline.

production inference latency measured from the render singapore deployment averaged approximately 1 second per request at the 50th percentile and 1.7 seconds at the 95th percentile. this is an order of magnitude slower than the same model running on local hardware (roughly 100 milliseconds). profiling revealed that the majority of the overhead is pandas dataframe construction in the `_prepare_row` function, which runs per request. converting this path to operate on numpy arrays directly is a documented future improvement.

after deployment, a traffic generator was used to send a mixture of real test transactions and synthetically drifted transactions (with amounts multiplied by three and product codes replaced with the highest-fraud category) to the production api. the drift-check script correctly identified the synthetic drift, exiting with a non-zero code; the grafana dashboard and streamlit monitoring tab both displayed the resulting latency and volume curves live from the postgres log.

## 7. conclusion and future work

this project demonstrates that a final-year student can build a machine learning system that resembles production software, using only free or open source tooling and free-tier cloud infrastructure. the concrete deliverables are a public api, a public frontend, a monitored deployment, and a ci/cd pipeline, in addition to the model itself. the engineering patterns implemented, including training-serving feature consistency via shared mapping files, baking models into serving containers, asynchronous prediction logging with graceful degradation, and scheduled drift detection with exit-code-based alerting, are the patterns used by production teams at technology companies.

future work falls into three categories. first, performance: converting inference from pandas to numpy should reduce production latency by an order of magnitude; introducing request batching would enable the system to handle higher throughput. second, model lifecycle: integrating an mlflow model registry with staged promotion (staging, production) and implementing shadow deployments would enable safe rollout of retrained models. third, monitoring maturity: replacing the coarse share-of-drifted-features threshold with per-feature policies, and computing reference statistics as a small precomputed artifact instead of relying on the full training parquet, would make the drift detection production-ready.

## 8. references

[1] kaggle, "ieee-cis fraud detection," competition dataset, vesta corporation, 2019.

[2] t. chen and c. guestrin, "xgboost: a scalable tree boosting system," in proceedings of the 22nd acm sigkdd international conference on knowledge discovery and data mining, 2016.

[3] d. sculley et al., "hidden technical debt in machine learning systems," in advances in neural information processing systems (nips), 2015.

[4] mlflow documentation, "mlflow tracking," https://mlflow.org/docs/latest/tracking.html

[5] evidently ai documentation, "data drift," https://docs.evidentlyai.com/

[6] fastapi documentation, "tutorial," https://fastapi.tiangolo.com/tutorial/

[7] d. j. hand and r. j. till, "a simple generalisation of the area under the roc curve for multiple class classification problems," machine learning, 2001.