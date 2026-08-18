"""
Configuration for One-Class SVM Anomaly Detection Pipeline
"""

from pathlib import Path
import logging


# ============================================================================
# PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent

DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DATA_DIR = DATA_DIR / "train"
TEST_DATA_DIR = DATA_DIR / "test"

# IMPORTANT:
# Keep this name exactly as it exists in your current project.
RAW_DATA_DIR = PROJECT_ROOT / "sythetic data"

ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
REPORTS_DIR = PROJECT_ROOT / "reports"
TESTS_DIR = PROJECT_ROOT / "tests"


ARTIFACTS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)


# ============================================================================
# DATA FILE NAMES
# ============================================================================

TRAIN_PROVIDER_FILE = "Train-1542865627584.csv"
TRAIN_BENEFICIARY_FILE = "Train_Beneficiarydata-1542865627584.csv"
TRAIN_INPATIENT_FILE = "Train_Inpatientdata-1542865627584.csv"
TRAIN_OUTPATIENT_FILE = "Train_Outpatientdata-1542865627584.csv"

TEST_PROVIDER_FILE = "Test-1542969243754.csv"
TEST_BENEFICIARY_FILE = "Test_Beneficiarydata-1542969243754.csv"
TEST_INPATIENT_FILE = "Test_Inpatientdata-1542969243754.csv"
TEST_OUTPATIENT_FILE = "Test_Outpatientdata-1542969243754.csv"


# ============================================================================
# COLUMN NAMES
# ============================================================================

PROVIDER_ID_COL = "Provider"
FRAUD_LABEL_COL = "PotentialFraud"

BENE_ID_COL = "BeneID"
CLAIM_ID_COL = "ClaimID"


# ============================================================================
# DATE COLUMNS
# ============================================================================

CLAIM_START_DT_COL = "ClaimStartDt"
CLAIM_END_DT_COL = "ClaimEndDt"
ADMISSION_DT_COL = "AdmissionDt"
DISCHARGE_DT_COL = "DischargeDt"
DOB_COL = "DOB"


# ============================================================================
# FINANCIAL COLUMNS
# ============================================================================

REIMBURSEMENT_COL = "InscClaimAmtReimbursed"
DEDUCTIBLE_COL = "DeductibleAmtPaid"


# ============================================================================
# BENEFICIARY COLUMNS
# ============================================================================

GENDER_COL = "Gender"
RACE_COL = "Race"


# ============================================================================
# PHYSICIAN COLUMNS
# ============================================================================

ATTENDING_PHYSICIAN_COL = "AttendingPhysician"
OPERATING_PHYSICIAN_COL = "OperatingPhysician"
OTHER_PHYSICIAN_COL = "OtherPhysician"


# ============================================================================
# DIAGNOSIS / PROCEDURE COLUMNS
# ============================================================================

DIAGNOSIS_COLS_PREFIX = "ClmDiagnosisCode_"
PROCEDURE_COLS_PREFIX = "ClmProcedureCode_"
ADMIT_DIAGNOSIS_COL = "ClmAdmitDiagnosisCode"


# ============================================================================
# ONE-CLASS SVM
# ============================================================================

RANDOM_STATE = 42


# --------------------------------------------------------------------------
# Expanded hyperparameter search
# --------------------------------------------------------------------------
#
# nu:
# Controls the expected proportion of outliers and the lower bound on
# support vectors.
#
# gamma:
# Controls the influence/radius of individual training points.
#
# kernel:
# Different kernels are evaluated rather than forcing only RBF.
#
# NOTE:
# These are candidate values. The validation results determine which
# combination performs best.
# --------------------------------------------------------------------------

NU_VALUES = [
    0.005,
    0.01,
    0.02,
    0.03,
    0.05,
    0.08,
    0.10,
    0.15,
    0.20
]

GAMMA_VALUES = [
    "scale",
    "auto",
    0.0001,
    0.0005,
    0.001,
    0.005,
    0.01,
    0.05,
    0.1,
    0.5
]

KERNEL_VALUES = [
    "rbf",
    "sigmoid",
    "linear"
]

# Default kernel used when a specific kernel is not supplied.
KERNEL = "rbf"


# ============================================================================
# MODEL SELECTION
# ============================================================================

# Minimum recall that an acceptable model should maintain.
#
# Your current recall is approximately 67.33%, so 0.65 prevents the tuning
# process from selecting a model that improves precision/F1 by destroying
# recall.
MIN_RECALL = 0.65

# Primary metric for model selection.
PRIMARY_SELECTION_METRIC = "f1_score"

# Secondary metric used when models have similar F1.
SECONDARY_SELECTION_METRIC = "precision"


# ============================================================================
# TRAIN / VALIDATION SPLIT
# ============================================================================

TRAIN_SPLIT_RATIO = 0.8
VALIDATION_SPLIT_RATIO = 0.2

STRATIFIED_SPLIT = True


# ============================================================================
# RISK SCORE THRESHOLDS
# ============================================================================

RISK_LEVEL_THRESHOLDS = {
    "Low": (0, 25),
    "Medium": (26, 50),
    "High": (51, 75),
    "Critical": (76, 100)
}


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

USE_LOG_TRANSFORM = True

LOG_FEATURES = [
    "OP_Claim_Count",
    "OP_Total_Reimbursement",
    "IP_Claim_Count",
    "IP_Total_Reimbursement",
    "Total_Claims",
    "Total_Reimbursement",
    "Total_Deductible"
]


# ============================================================================
# SCALING
# ============================================================================

USE_ROBUST_SCALER = True
USE_STANDARD_SCALER = False


# ============================================================================
# PREPROCESSING
# ============================================================================

IMPUTATION_STRATEGY = "median"

MISSING_CATEGORICAL_VALUE = "Unknown"


# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = logging.INFO

LOG_FORMAT = (
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def get_logger(name):
    """Get a configured logger instance."""

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)

        logger.addHandler(handler)
        logger.setLevel(LOG_LEVEL)

    return logger


# ============================================================================
# VALIDATION RULES
# ============================================================================

REQUIRED_PROVIDER_COLS = [
    PROVIDER_ID_COL,
    FRAUD_LABEL_COL
]

REQUIRED_INPATIENT_COLS = [
    BENE_ID_COL,
    CLAIM_ID_COL,
    PROVIDER_ID_COL,
    CLAIM_START_DT_COL,
    CLAIM_END_DT_COL,
    REIMBURSEMENT_COL
]

REQUIRED_OUTPATIENT_COLS = [
    BENE_ID_COL,
    CLAIM_ID_COL,
    PROVIDER_ID_COL,
    CLAIM_START_DT_COL,
    CLAIM_END_DT_COL,
    REIMBURSEMENT_COL
]

REQUIRED_BENEFICIARY_COLS = [
    BENE_ID_COL
]


# ============================================================================
# MODEL METADATA
# ============================================================================

MODEL_NAME = "One-Class SVM Provider Anomaly Detector"
MODEL_ALGORITHM = "One-Class SVM"
MODEL_KERNEL = KERNEL

SKLEARN_VERSION = None
PYTHON_VERSION = None