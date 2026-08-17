"""
Medicare Provider Anomaly Risk Engine - Configuration
Centralized configuration for all parameters
"""

import os
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent

# Data paths
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_PROVIDER_FILE = PROJECT_ROOT / "Train-1542865627584.csv"
TRAIN_BENEFICIARY_FILE = PROJECT_ROOT / "Train_Beneficiarydata-1542865627584.csv"
TRAIN_INPATIENT_FILE = PROJECT_ROOT / "Train_Inpatientdata-1542865627584.csv"
TRAIN_OUTPATIENT_FILE = PROJECT_ROOT / "Train_Outpatientdata-1542865627584.csv"

TEST_PROVIDER_FILE = PROJECT_ROOT / "Test-1542969243754.csv"
TEST_BENEFICIARY_FILE = PROJECT_ROOT / "Test_Beneficiarydata-1542969243754.csv"
TEST_INPATIENT_FILE = PROJECT_ROOT / "Test_Inpatientdata-1542969243754.csv"
TEST_OUTPATIENT_FILE = PROJECT_ROOT / "Test_Outpatientdata-1542969243754.csv"

# Model paths
MODELS_DIR = PROJECT_ROOT / "models"
ISOLATION_FOREST_MODEL = MODELS_DIR / "isolation_forest_model.pkl"
PREPROCESSING_PIPELINE = MODELS_DIR / "preprocessing_pipeline.pkl"
FEATURE_COLUMNS = MODELS_DIR / "feature_columns.pkl"
MODEL_METADATA = MODELS_DIR / "model_metadata.pkl"

# Output paths
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
PROVIDER_PREDICTIONS = OUTPUTS_DIR / "provider_anomaly_predictions.csv"
TOP_SUSPICIOUS = OUTPUTS_DIR / "top_suspicious_providers.csv"
EVALUATION_METRICS = OUTPUTS_DIR / "evaluation_metrics.json"
EVALUATION_REPORT = OUTPUTS_DIR / "evaluation_report.txt"
DATA_QUALITY_REPORT = OUTPUTS_DIR / "data_quality_report.csv"

# Create directories if they don't exist
for directory in [MODELS_DIR, OUTPUTS_DIR, PLOTS_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Isolation Forest Configuration
ISOLATION_FOREST_CONFIG = {
    "n_estimators": 500,
    "max_samples": "auto",
    "max_features": 1.0,
    "contamination": "auto",
    "bootstrap": False,
    "random_state": 42,
    "n_jobs": -1,
}

# Multiple random seeds for ensemble stability
RANDOM_SEEDS = [42, 52, 62, 72, 82]

# Contamination options (unsupervised selection)
CONTAMINATION_OPTIONS = ["auto", 0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.18, 0.20]
DEFAULT_CONTAMINATION = 0.18

# Risk level percentile thresholds (configurable)
RISK_LEVEL_THRESHOLDS = {
    "LOW": (0, 90),
    "MEDIUM": (90, 95),
    "HIGH": (95, 99),
    "CRITICAL": (99, 100),
}

# Data types for efficient memory usage
PANDAS_DTYPES = {
    "Provider": "object",
    "BeneID": "object",
    "ClaimID": "object",
    "Gender": "object",
    "Race": "object",
    "State": "object",
    "County": "object",
    "AttendingPhysician": "object",
    "OperatingPhysician": "object",
    "OtherPhysician": "object",
}

# Chronic condition columns
CHRONIC_CONDITIONS = [
    "ChronicCond_Alzheimer",
    "ChronicCond_Heartfailure",
    "ChronicCond_KidneyDisease",
    "ChronicCond_Cancer",
    "ChronicCond_ObstrPulmonary",
    "ChronicCond_Depression",
    "ChronicCond_Diabetes",
    "ChronicCond_IschemicHeart",
    "ChronicCond_Osteoporasis",
    "ChronicCond_rheumatoidarthritis",
    "ChronicCond_stroke",
]

# Diagnosis code columns
DIAGNOSIS_COLUMNS = [
    "ClmDiagnosisCode_1",
    "ClmDiagnosisCode_2",
    "ClmDiagnosisCode_3",
    "ClmDiagnosisCode_4",
    "ClmDiagnosisCode_5",
    "ClmDiagnosisCode_6",
    "ClmDiagnosisCode_7",
    "ClmDiagnosisCode_8",
    "ClmDiagnosisCode_9",
    "ClmDiagnosisCode_10",
]

# Procedure code columns
PROCEDURE_COLUMNS = [
    "ClmProcedureCode_1",
    "ClmProcedureCode_2",
    "ClmProcedureCode_3",
    "ClmProcedureCode_4",
    "ClmProcedureCode_5",
    "ClmProcedureCode_6",
]

# Feature engineering thresholds
HIGH_COST_PERCENTILE = 0.90  # Claims above 90th percentile are high-cost
LONG_DURATION_DAYS = 30  # Claims longer than 30 days are long-duration
HIGH_DIAGNOSIS_THRESHOLD = 5  # Claims with more than 5 diagnoses
HIGH_PROCEDURE_THRESHOLD = 3  # Claims with more than 3 procedures

# Reimbursement log transformation threshold (for skewness)
LOG_TRANSFORM_THRESHOLD = 100  # Values > $100 may be log-transformed if skewed

# Random state for reproducibility
RANDOM_STATE = 42

# Logging
LOGGING_LEVEL = "INFO"
