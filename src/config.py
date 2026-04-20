"""central config: paths, constants."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_TXN_PATH = RAW_DIR / "train_transaction.csv"
RAW_ID_PATH = RAW_DIR / "train_identity.csv"

TARGET_COL = "isFraud"
ID_COL = "TransactionID"
TIME_COL = "TransactionDT"

RANDOM_SEED = 42