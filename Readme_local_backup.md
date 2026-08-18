# Medicare Provider Anomaly Risk Engine

A production-ready unsupervised anomaly detection system for identifying fraudulent or anomalous Medicare provider behavior using **Isolation Forest**.

## Overview

This system analyzes CMS Medicare claims data at the provider level to identify providers exhibiting unusual behavior patterns that may indicate fraud or compliance issues. The model is trained completely unsupervised without using fraud labels, ensuring no data leakage.

## Algorithm

**Isolation Forest** - A scalable, stochastic ensemble method specifically designed for anomaly detection:
- Isolates anomalies using random partitioning
- No distance calculations required (efficient for high dimensions)
- Unsupervised learning (no labels required)
- Multiple ensemble members for stability

## Key Features

### Data Processing
- **Automated column detection** - Discovers actual columns dynamically
- **Comprehensive data quality audit** - Reports missing values, duplicates, statistics
- **Provider-level aggregation** - Groups claims by provider to create behavioral features
- **No fraud label leakage** - PotentialFraud label used only for post-hoc evaluation

### Feature Engineering
50+ behavioral features across multiple categories:

#### Claim Volume Features
- Total claims, inpatient/outpatient claims
- Unique beneficiaries, physicians
- Claims per beneficiary/physician
- Active months and temporal statistics

#### Financial Features
- Total/average/median reimbursement
- High-cost claim detection (>90th, >95th, >99th percentiles)
- Reimbursement concentration metrics
- Per-claim and per-beneficiary economics

#### Beneficiary Features
- Demographics (age, gender distribution)
- Chronic condition prevalence
- High-risk beneficiary percentages
- Condition diversity metrics

#### Diagnosis & Procedure Features
- Diagnosis/procedure counts per claim
- Unique diagnosis/procedure diversity
- High-complexity claim detection

#### Temporal & Peer Features
- Monthly utilization patterns
- Provider percentile rankings vs peers
- Deviation from peer medians

### Anomaly Scoring & Risk Levels

**Risk Score**: 0-100 (normalized, higher = more suspicious)

**Risk Levels** (percentile-based):
- **LOW**: 0-90th percentile
- **MEDIUM**: 90-95th percentile
- **HIGH**: 95-99th percentile
- **CRITICAL**: 99-100th percentile

### Evaluation

Post-training evaluation against PotentialFraud label:
- Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Confusion matrix
- Fraud detection rate
- False positive rate

### Explanations

For each flagged provider:
- Top 5 most unusual features
- Percentile ranking within peers
- Deviation metrics from normal
- Risk score and level rationale

## Project Structure

```
Medicare_Provider_Anomaly_Risk_Engine/
├── data/                          # Raw data directory
├── src/
│   ├── __init__.py
│   ├── utils.py                   # Utility functions
│   ├── data_loader.py             # Data loading
│   ├── data_audit.py              # Data quality audit
│   ├── data_cleaning.py           # Data cleaning/preprocessing
│   ├── feature_engineering.py     # Feature creation
│   ├── preprocessing.py           # Model preprocessing pipeline
│   ├── isolation_forest_engine.py # IF model implementations
│   ├── anomaly_scoring.py         # Score normalization
│   ├── evaluation.py              # Model evaluation
│   └── anomaly_explanation.py     # Explanation generation
├── models/
│   ├── isolation_forest_model.pkl     # Trained IF model
│   ├── preprocessing_pipeline.pkl     # Fitted preprocessing
│   ├── feature_columns.pkl            # Feature names/order
│   └── model_metadata.pkl             # Training metadata
├── outputs/
│   ├── provider_anomaly_predictions.csv   # All predictions
│   ├── top_suspicious_providers.csv       # Top 50 anomalies
│   ├── evaluation_metrics.json            # Metrics report
│   ├── evaluation_report.txt              # Text report
│   ├── data_quality_report.csv            # Data audit
│   └── plots/
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       ├── precision_recall_curve.png
│       └── anomaly_distribution.png
├── train.py                       # Training pipeline
├── predict.py                     # Inference script
├── config.py                      # Configuration
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

```bash
# Clone or navigate to project directory
cd Medicare_Provider_Anomaly_Risk_Engine

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

Run the complete training pipeline:

```bash
python train.py
```

This will:
1. Load and audit all training datasets
2. Clean and prepare data
3. Engineer 50+ features
4. Train ensemble Isolation Forest models
5. Generate predictions and risk scores
6. Evaluate against PotentialFraud label
7. Save all models and outputs

**Time**: ~5-10 minutes depending on data size and hardware

### Prediction

Generate predictions on new test data:

```bash
python predict.py
```

This will:
1. Load test datasets
2. Engineer features using exact same pipeline
3. Apply preprocessing
4. Generate anomaly predictions
5. Save results to `outputs/test_provider_anomaly_predictions.csv`

## Output Files

### Model Files (in `models/`)
- **isolation_forest_model.pkl** - Trained Isolation Forest (single or ensemble)
- **preprocessing_pipeline.pkl** - Fitted imputer and RobustScaler
- **feature_columns.pkl** - Feature names in training order
- **model_metadata.pkl** - Algorithm, parameters, features, metrics

### Prediction Files (in `outputs/`)
- **provider_anomaly_predictions.csv**
  - Provider ID
  - anomaly_prediction (0=normal, 1=anomaly)
  - anomaly_score_raw (model scores)
  - risk_score (0-100)
  - risk_level (LOW/MEDIUM/HIGH/CRITICAL)
  - Important behavioral features

- **top_suspicious_providers.csv**
  - Top 50 most anomalous providers
  - Risk scores and levels
  - Key behavioral metrics

### Evaluation Files (in `outputs/`)
- **evaluation_metrics.json** - Precision, Recall, F1, ROC-AUC, Confusion Matrix
- **evaluation_report.txt** - Human-readable summary
- **data_quality_report.csv** - Dataset statistics and missing values

### Visualizations (in `outputs/plots/`)
- **confusion_matrix.png** - True/False positives vs negatives
- **roc_curve.png** - ROC curve with AUC
- **precision_recall_curve.png** - PR curve with AUC
- **anomaly_distribution.png** - Anomaly score distribution by risk level

## Configuration

Edit `config.py` to customize:

```python
# Isolation Forest parameters
ISOLATION_FOREST_CONFIG = {
    "n_estimators": 500,          # Number of trees
    "contamination": "auto",      # Expected anomaly rate
    "max_samples": "auto",        # Samples per tree
}

# Ensemble seeds for stability
RANDOM_SEEDS = [42, 52, 62, 72, 82]

# Risk level percentiles
RISK_LEVEL_THRESHOLDS = {
    "LOW": (0, 90),
    "MEDIUM": (90, 95),
    "HIGH": (95, 99),
    "CRITICAL": (99, 100),
}

# Feature engineering thresholds
HIGH_COST_PERCENTILE = 0.90
LONG_DURATION_DAYS = 30
HIGH_DIAGNOSIS_THRESHOLD = 5
HIGH_PROCEDURE_THRESHOLD = 3
```

## Model Architecture

### Training Pipeline

```
Raw CMS Data
    ↓
[Data Loading & Audit]
    ↓
[Data Cleaning]
    ↓
[Provider-Level Aggregation]
    ↓
[Feature Engineering] ← 50+ behavioral features
    ↓
[Preprocessing] ← Median imputation + RobustScaler
    ↓
[Ensemble Isolation Forest] ← 5 models with different seeds
    ↓
[Anomaly Scores]
    ↓
[Anomaly Detection] ← 0/1 predictions
    ↓
[Risk Scoring] ← 0-100 normalized
    ↓
[Risk Levels] ← LOW/MEDIUM/HIGH/CRITICAL
    ↓
[Evaluation] ← Precision/Recall vs PotentialFraud
    ↓
[Model Persistence] ← .pkl files
```

### Prediction Pipeline

```
Test Data
    ↓
[Feature Engineering] ← Same as training
    ↓
[Load Preprocessing] ← Fitted pipeline
    ↓
[Transform Features] ← Same scaling
    ↓
[Load Trained Model] ← IF ensemble
    ↓
[Predict] ← Ensemble voting
    ↓
[Score] ← Median aggregation
    ↓
[Risk Scores & Levels]
    ↓
[CSV Output]
```

## Important Notes

### No Label Leakage
✓ PotentialFraud label NOT used during:
- Feature engineering
- Preprocessing
- Model training
- Contamination selection
- Hyperparameter optimization

✗ PotentialFraud used ONLY for:
- Post-training evaluation
- Performance metrics calculation
- Confusion matrix generation

### Unsupervised Training
- Model learns provider behavior WITHOUT knowing labels
- No assumption about fraud distribution
- Real anomalies detected based on behavior patterns
- Transparent, explainable results

### Ensemble Stability
- 5 Isolation Forest models trained with different random seeds
- Predictions aggregated by majority voting
- Scores aggregated by median
- Ensures stable, reproducible results

### Feature Quality
- 50+ engineered features
- Zero-variance and constant features removed
- Infinite values handled
- Missing values imputed with median
- Features scaled with RobustScaler (resistant to outliers)

## Evaluation Metrics

When evaluated against PotentialFraud label:

- **Precision**: % of flagged providers actually fraudulent
- **Recall**: % of actual frauds detected
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under receiver operating characteristic
- **PR-AUC**: Area under precision-recall curve
- **Fraud Detection Rate**: % of labeled frauds caught
- **False Positive Rate**: % of normal providers incorrectly flagged

**Note**: These are evaluation-only metrics. The model makes predictions unsupervised.

## Example Output

```
========================================================
MEDICARE PROVIDER ANOMALY RISK ENGINE
========================================================

Algorithm: Isolation Forest
Providers: 5200
Features: 47

Anomalies: 260
Normal: 4940
Anomaly Percentage: 5.00%

Precision: 0.7245
Recall: 0.6892
F1: 0.7065
ROC-AUC: 0.8123
PR-AUC: 0.7654

========================================================
RISK DISTRIBUTION
========================================================

CRITICAL:...... 52 (1.0%)
HIGH:.......... 260 (5.0%)
MEDIUM:........ 520 (10.0%)
LOW:........... 4368 (84.0%)
========================================================

Top Suspicious Provider: PR123456
Risk Score: 94.5/100
Risk Level: CRITICAL

Reasons:
1. Total reimbursement at 99th percentile
2. Average claim amount 250% above peer median
3. High-cost claim percentage at 98th percentile
4. Claims per beneficiary 3x normal
```

## Troubleshooting

### Missing Columns
- The system auto-detects columns from CSV headers
- If a column is truly missing, the feature is skipped with warning
- Check data_quality_report.csv for available columns

### Insufficient Memory
- Use selected columns only
- Process claims in chunks
- Reduce ensemble size in config.py

### Prediction Errors
- Ensure test data has same feature format as training
- Missing features are filled with 0
- Check that models exist in `models/` directory

### Low Evaluation Metrics
- This is unsupervised learning; fraud is rare
- Anomaly detection != fraud detection
- Tune `contamination` parameter in config.py
- Review explanations for detected anomalies

## Citation

**Algorithm**: Liu, F. T., Ting, K. M., & Zhou, Z.-H. (2008). Isolation Forest. *ICDM*, 413-422.

**Implementation**: scikit-learn

## License

Internal use only. CMS data is confidential.

## Author

Healthcare ML Engineering Team

---

**Last Updated**: 2024
**Version**: 1.0
