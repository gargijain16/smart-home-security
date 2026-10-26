from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DB_PATH = DATA_DIR / "smart_home_security.db"
DATASET_PATH = DATA_DIR / "synthetic_iot_traffic.csv"

DEFAULT_ROWS = 6000
DEFAULT_SEED = 42

# Risk levels: Low < 30 <= Medium < 55 <= High < 75 <= Critical
RISK_BINS = [0, 30, 55, 75, 100.01]
RISK_LABELS = ["Low", "Medium", "High", "Critical"]
EVENT_THRESHOLD = 55  # flows scoring High/Critical become security events

SYNTHETIC_NOTICE = (
    "All traffic in this project is SYNTHETIC (simulated by a Python script). "
    "Metrics describe performance on simulated data only and are not evidence of real-world detection ability."
)
