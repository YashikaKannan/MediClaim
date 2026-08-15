"""
config.py — central paths & constants for Member 5 (LOF provider anomaly detection).

NEW DATA VERSION: this zip already ships PROVIDER_ML_READY.csv (1 row = 1 provider,
all 6 features present), so we read it directly — no aggregation step needed.
"""
from pathlib import Path

DATA_DIR   = Path(__file__).resolve().parents[1] / "data"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Source files (only these two are needed for Member 5)
PROVIDER_ML_READY_CSV = DATA_DIR / "PROVIDER_ML_READY.csv"            # ready-made, 1 row = 1 provider
LEIE_CSV              = DATA_DIR / "leie_clean_specialty_filled.csv"  # exclusion list, for validation

# Result file
RESULTS_CSV = OUTPUT_DIR / "member5_lof_results.csv"

# ---- model constants ---------------------------------------------------------
MIN_PEERS     = 20
N_NEIGHBORS   = 20
CONTAMINATION = 0.03
SCORE_CAP_Q   = 0.995

FEATURES = [
    "Tot_Srvcs",
    "Tot_Benes",
    "Tot_Mdcr_Pymt_Amt",
    "Services_Per_Beneficiary",
    "Charge_Per_Service",
    "Payment_to_Allowed_Ratio",
]
LOG_COLS = ["Tot_Srvcs", "Tot_Benes", "Tot_Mdcr_Pymt_Amt", "Charge_Per_Service"]
RISK_BANDS = [(30, "Low"), (60, "Medium"), (80, "High"), (100, "Critical")]
