# fraud detection mlops

end-to-end machine learning system for credit card fraud detection on the ieee-cis dataset.

## stack

- python 3.11, xgboost, pandas
- mlflow for experiment tracking
- hydra for config management
- pytest for testing
- parquet for feature storage

## setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

download the dataset:
```bash
kaggle competitions download -c ieee-fraud-detection -p data/raw
cd data/raw && unzip ieee-fraud-detection.zip && rm ieee-fraud-detection.zip && cd ../..
```

## usage

start mlflow server (separate terminal):
```bash
mlflow server \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns \
  --host 127.0.0.1 --port 5000
```

prepare features (run once after dataset changes):
```bash
python -m src.prepare_data
```

train a model:
```bash
python -m src.train
```

override any config from the command line:
```bash
python -m src.train run_name=deep model.max_depth=8 model.learning_rate=0.1
```

run tests:
```bash
pytest tests/ -v
```

## structure
configs/          hydra yaml configs
data/
raw/            raw csvs from kaggle (gitignored)
processed/      parquet features + category mappings
src/
config.py       paths and constants
data.py         raw csv loading and time-based split
features.py     categorical encoding, fit/apply pattern
prepare_data.py feature pipeline script
evaluate.py     metrics (pr-auc, recall at precision)
train.py        training script with mlflow logging
tests/            pytest unit tests
docs/             problem doc, architecture

## baseline results

- dataset: 590k transactions, 3.5% fraud rate, 182 days
- split: time-based (not random) train/val/test
- model: xgboost with `scale_pos_weight` for imbalance
- test pr-auc: 0.514