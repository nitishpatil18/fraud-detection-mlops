# demo script (4 minutes)

use this script for the project review / viva / any live demo. each step has exact timing and exact actions. practice once end-to-end before the real thing.

## prereqs (do 30 min before)

1. wake the render api: open https://fraud-api-5dxn.onrender.com/health in a browser. wait for `{"status":"ok"}`.
2. confirm streamlit is alive: open https://nitish-fraud-detection.streamlit.app, let it load.
3. optional local services (if presenting on your laptop):
   - `docker compose up -d` for grafana
   - `mlflow server ...` in a terminal

keep 5 tabs open, in this order (makes clicking painless):
- tab 1: github repo
- tab 2: api docs (swagger)
- tab 3: streamlit frontend
- tab 4: grafana dashboard (localhost)
- tab 5: mlflow ui (localhost)

## script

### 0:00 - 0:30 | the hook

"this is a production-style credit card fraud detection system. the model itself is a small part of what i built. the larger part is the engineering around it: how i track experiments, deploy to the cloud, log every prediction, monitor for drift, and automatically test every change i make."

### 0:30 - 1:00 | the repo

open tab 1 (github). point out:

- the badges: "ci and docker builds pass on every push."
- the live demo links at the top of the readme: "both deployed and public."
- the project structure: "clear separation between data, training, api, monitoring, and frontend."

### 1:00 - 1:45 | the live api

open tab 2 (swagger at /docs). point out:

- three endpoints: `/health`, `/info`, `/predict`.
- click `/info`, try it out, execute: "431 features, threshold 0.5, baseline_fast model loaded."
- click `/predict`, try it out, submit `{"TransactionAmt": 75, "ProductCD": "W"}`, execute: "returns 1% fraud probability, the model says approve, and tells me which model version served the request."

### 1:45 - 2:30 | the frontend

open tab 3 (streamlit). point out:

- predict tab: fill a transaction, click predict, show the colored result.
- batch tab: "i can upload a csv and score hundreds at once, download results."
- monitoring tab: "this reads live from the same postgres database the api writes to. i can see every prediction."

### 2:30 - 3:00 | the dashboard

open tab 4 (grafana at localhost:3000). point out:

- "grafana is open source, used by google, stripe, and most tech companies for dashboards."
- 8 panels: total predictions, fraud rate, latency percentiles, traffic over time, fraud probability over time.
- "this is exactly how an on-call engineer would watch model performance in production."

### 3:00 - 3:30 | the experiments

open tab 5 (mlflow at localhost:5000). click the `fraud-detection-baseline` experiment. point out:

- 5 runs: baseline, depth4, smoke_test, baseline_v2, baseline_fast.
- the metrics columns: test_pr_auc values.
- "every experiment is reproducible. i can see exactly what config gave me this number."

### 3:30 - 4:00 | the close

back to tab 1 (github). click actions tab. point out:

- "every push runs lint, format check, type check, tests, and docker build."
- "there's a daily cron that runs drift detection against the live database. if drift exceeds 30%, the job fails and in a real setup it would page an on-call engineer."

"so, to summarize: one model, one api, one frontend, one dashboard, and a full ci/cd pipeline with monitoring. this is the system a small team would build at a startup."

## likely questions

**q: what's the biggest weakness?**  
a: production inference latency. about 1 second per request on render's free tier. profiling shows pandas dataframe construction dominates; converting to numpy arrays would bring it under 100ms.

**q: why this model, not a neural network?**  
a: tabular data, ~600k rows, many categorical features, lots of missingness. xgboost is the standard strong baseline and usually beats neural networks on tabular without extensive tuning. also much cheaper to train and serve.

**q: how would you retrain?**  
a: re-run `prepare_data.py` to refresh features, run `train.py` with a new run name, use the mlflow ui to compare the new run's test pr-auc against the current production run, run `export_model.py` with the new run id, commit, push. render auto-redeploys. shadow deployment would be a future improvement.

**q: how does drift detection work?**  
a: i pull the last 24 hours of predictions from postgres, encode them with the same category mappings the model uses, and pass them to evidently ai along with my training reference. evidently runs ks tests on numeric features and chi-square tests on categoricals. if more than 30% of features show statistically significant drift, the github actions job fails, which triggers the alert step.

**q: what would change for 1000 requests per second instead of 1?**  
a: three things. first, numpy inference path instead of pandas. second, request batching at the api layer. third, replace sqlite-backed mlflow with postgres-backed mlflow, and the local docker-compose postgres with a managed cloud postgres with read replicas for the analytics workload.

**q: do you own this repo or did you use a template?**  
a: built from scratch, commit by commit, over 12 weekly sessions. the commit history and branch activity on github shows the work.