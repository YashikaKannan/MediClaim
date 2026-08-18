# One-Class SVM Healthcare Provider Anomaly Detection Pipeline

## Overview

This is a **production-ready, end-to-end machine learning pipeline** for detecting anomalous healthcare providers using One-Class SVM on CMS Synthetic Medicare data.

The pipeline:
- Loads and validates CMS healthcare provider claim data
- Performs comprehensive data cleaning and preprocessing
- Aggregates claim-level data to provider-level features
- Implements advanced feature engineering
- Creates stratified train/validation splits at provider level
- Performs hyperparameter tuning with multiple metrics
- Trains a One-Class SVM model on normal provider behavior
- Evaluates model performance with multiple metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC)
- Generates predictions on test data
- Produces risk scores and risk levels

## Key Features

### Data Handling
- **Provider-level aggregation**: Converts claim-level data to provider-level features
- **Multiple data sources**: Integrates inpatient claims, outpatient claims, and beneficiary data
- **Comprehensive validation**: Checks for missing values, duplicates, infinite values, and data consistency

### Feature Engineering
- **Log transformation**: Applied to highly skewed financial features
- **Robust scaling**: Uses RobustScaler to handle outliers in healthcare data
- **Multiple aggregate metrics**: 
  - Claim counts and reimbursement aggregates
  - Beneficiary and physician uniqueness metrics
  - Diagnosis and procedure code diversity
  - Length of stay metrics for inpatient claims
  - Derived ratios (reimbursement per claim, claims per beneficiary, etc.)

### Machine Learning
- **One-Class SVM**: Learns normal provider behavior patterns
- **Stratified splits**: Ensures balanced fraud representation in train/validation
- **Only trains on normal providers**: Fraud providers excluded from model fitting
- **Comprehensive hyperparameter tuning**: Tests multiple nu and gamma values
- **Multi-metric evaluation**: Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC

### Risk Scoring
- **Continuous risk scores**: Normalized to 0-100 range
- **Risk level assignments**: Low, Medium, High, Critical
- **Decision scores**: Raw anomaly scores from One-Class SVM decision function

## Project Structure

```
one_class_svm_project/
├── data/
│   ├── train/              # Training data
│   └── test/               # Test data
│
├── src/
│   ├── __init__.py         # Package init
│   ├── config.py           # Configuration and constants
│   ├── utils.py            # Utility functions
│   ├── data_loader.py      # Data loading
│   ├── preprocessing.py    # Data preprocessing
│   ├── provider_aggregation.py  # Claim to provider aggregation
│   ├── feature_engineering.py   # Feature creation and transformation
│   ├── train.py            # Model training and hyperparameter tuning
│   ├── evaluate.py         # Model evaluation and metrics
│   └── predict.py          # Prediction and artifact loading
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loading.py
│   ├── test_preprocessing.py
│   ├── test_feature_engineering.py
│   ├── test_provider_aggregation.py
│   ├── test_model_training.py
│   ├── test_prediction.py
│   ├── test_no_data_leakage.py
│   └── test_artifacts.py
│
├── artifacts/
│   ├── one_class_svm.pkl          # Trained One-Class SVM model
│   ├── scaler.pkl                 # Fitted feature scaler
│   ├── feature_columns.pkl        # Feature column names and order
│   └── model_metadata.pkl         # Model metadata and hyperparameters
│
├── outputs/
│   ├── validation_predictions.csv  # Validation set predictions
│   ├── test_predictions.csv        # Test set predictions
│   ├── validation_metrics.json     # Validation metrics
│   ├── confusion_matrix.csv        # Confusion matrix
│   ├── feature_summary.csv         # Feature statistics
│   └── model_comparison.csv        # Hyperparameter tuning results
│
├── reports/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   ├── precision_recall_curve.png
│   ├── risk_distribution.png
│   └── decision_score_distribution.png
│
├── requirements.txt        # Python dependencies
├── README.md              # This file
├── run_pipeline.py        # Main orchestration script
└── inspect_dataset.py     # Data inspection utility
```

## Installation

### Prerequisites
- Python 3.8+
- pip

### Setup

1. **Clone or extract the project:**
   ```bash
   cd one_class_svm_project
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The pipeline expects the CMS Synthetic Medicare dataset with the following files:

### Training Data
- `Train-1542865627584.csv` - Provider labels (Provider, PotentialFraud)
- `Train_Beneficiarydata-1542865627584.csv` - Beneficiary demographics
- `Train_Inpatientdata-1542865627584.csv` - Inpatient claims
- `Train_Outpatientdata-1542865627584.csv` - Outpatient claims

### Test Data
- `Test-1542969243754.csv` - Provider identifiers
- `Test_Beneficiarydata-1542969243754.csv` - Beneficiary demographics
- `Test_Inpatientdata-1542969243754.csv` - Inpatient claims
- `Test_Outpatientdata-1542969243754.csv` - Outpatient claims

These should be placed in the `sythetic data/` directory.

## Running the Pipeline

### Complete Pipeline Execution

Run the entire pipeline from data loading to test predictions:

```bash
python run_pipeline.py
```

This will:
1. Load all datasets
2. Perform data preprocessing
3. Aggregate claims to provider level
4. Engineer features
5. Create train/validation split
6. Tune hyperparameters
7. Evaluate on validation set
8. Train final model
9. Generate test predictions
10. Save all artifacts and outputs

**Expected runtime**: ~2-5 minutes depending on data size and system

### Individual Components

You can also run individual pipeline components:

```bash
# Just load and inspect data
python -c "from src.data_loader import load_training_data; data = load_training_data()"

# Run tests
python -m pytest tests/

# Create visualizations (if implemented)
# python src/visualize.py
```

## Output Files

### Predictions
- **`outputs/validation_predictions.csv`**
  - Validation set predictions with actual labels
  - Columns: Provider, ActualFraud, PredictedAnomaly, RiskScore, RiskLevel
  
- **`outputs/test_predictions.csv`**
  - Test set predictions (no actual labels)
  - Columns: Provider, PredictedAnomaly, DecisionScore, RiskScore, RiskLevel

### Metrics
- **`outputs/validation_metrics.json`**
  - All calculated metrics: accuracy, precision, recall, f1, ROC-AUC, PR-AUC, etc.

- **`outputs/confusion_matrix.csv`**
  - Confusion matrix for validation set

- **`outputs/model_comparison.csv`**
  - Hyperparameter tuning results for all tested combinations

### Artifacts
- **`artifacts/one_class_svm.pkl`**
  - Trained One-Class SVM model

- **`artifacts/scaler.pkl`**
  - Fitted feature scaler (RobustScaler)

- **`artifacts/feature_columns.pkl`**
  - Feature column names in correct order

- **`artifacts/model_metadata.pkl`**
  - Model parameters, training info, Python/sklearn versions

## Key Metrics Explained

### Classification Metrics
- **Accuracy**: Overall fraction of correct predictions
- **Precision**: Among providers predicted anomalous, how many were actually fraudulent
- **Recall**: Among actually fraudulent providers, how many were detected
- **F1-Score**: Harmonic mean of precision and recall

### AUC Metrics
- **ROC-AUC**: Area under the Receiver Operating Characteristic curve (robustness to threshold changes)
- **PR-AUC**: Area under Precision-Recall curve (especially useful for imbalanced data)

### Confusion Matrix
```
                Predicted Normal    Predicted Anomaly
Actual Normal       TN                  FP
Actual Anomaly      FN                  TP
```

### Derived Metrics
- **Sensitivity**: Recall = TP / (TP + FN)
- **Specificity**: TN / (TN + FP)
- **False Positive Rate**: FP / (FP + TN)
- **False Negative Rate**: FN / (FN + TP)

## Risk Scoring

Risk scores are normalized to 0-100:
- **0-25**: Low risk (normal provider behavior)
- **26-50**: Medium risk (some unusual patterns)
- **51-75**: High risk (significant anomalies)
- **76-100**: Critical risk (extreme anomalies)

Higher risk scores indicate more anomalous provider behavior patterns.

## Data Leakage Prevention

The pipeline strictly prevents data leakage:

1. **PotentialFraud is NEVER used as a feature**
   - Only used for: creating training split, validation labels, evaluation
   - NOT used in: preprocessing, scaling, feature engineering, model training

2. **Preprocessing fitted only on training data**
   - Scaler is fitted on training set only
   - Imputer is fitted on training set only
   - Feature transformations use training statistics

3. **Provider-level split**
   - No provider appears in both training and validation sets
   - Prevents information leakage from same provider

4. **Normal-only training**
   - One-Class SVM is trained ONLY on normal providers
   - Fraud providers are excluded from model fitting
   - This ensures model learns true "normal" behavior

## Important Notes

### One-Class SVM Interpretation
- One-Class SVM predicts: +1 (normal) or -1 (anomaly)
- Decision function output is NOT a probability
- It's a continuous anomaly score (lower = more anomalous)
- Risk score conversion uses min-max normalization

### Class Imbalance
- Training data typically has ~10% fraud (506/5410 providers)
- Validation set is stratified to maintain this ratio
- F1-score is primary metric (balances precision/recall)
- PR-AUC is also important for imbalanced evaluation

### Hyperparameter Tuning
Tested parameters:
- **nu**: [0.01, 0.02, 0.03, 0.05, 0.10] (expected anomaly fraction)
- **gamma**: ["scale", 0.001, 0.01, 0.1] (RBF kernel parameter)
- **kernel**: "rbf" (Radial Basis Function)

Best parameters are selected based on F1-score, with consideration for precision and recall balance.

## Troubleshooting

### FileNotFoundError: Dataset not found
- Ensure CMS data files are in `sythetic data/` directory
- Check file names match exactly (including case and special characters)

### Memory Issues with Large Datasets
- The pipeline uses efficient pandas groupby operations
- Large datasets may require increased RAM
- Consider processing in batches if necessary

### Poor Model Performance
- Check feature distributions (use `log_feature_statistics`)
- Verify data quality and preprocessing
- Inspect hyperparameter tuning results
- Consider adjusting nu parameter (controls contamination)

## Advanced Usage

### Custom Feature Engineering

Edit `src/feature_engineering.py` to:
- Add new provider-level features
- Modify log transformation strategy
- Implement custom feature selection

### Hyperparameter Modification

Edit `src/config.py`:
```python
NU_VALUES = [0.01, 0.02, 0.03, 0.05, 0.10]
GAMMA_VALUES = ["scale", 0.001, 0.01, 0.1]
```

### Risk Score Customization

Edit `src/config.py` or `src/train.py`:
```python
RISK_LEVEL_THRESHOLDS = {
    "Low": (0, 25),
    "Medium": (26, 50),
    "High": (51, 75),
    "Critical": (76, 100)
}
```

## Model Reloading and Prediction

```python
from src.predict import load_artifacts, predict_on_test_data
import pandas as pd

# Load artifacts
artifacts = load_artifacts("artifacts/")

# Load new provider features
test_features = pd.read_csv("test_provider_features.csv")

# Make predictions
predictions = predict_on_test_data(test_features, artifacts)

# Save results
predictions.to_csv("new_predictions.csv", index=False)
```

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

Tests verify:
- Data loading correctness
- Preprocessing pipeline
- Feature engineering logic
- Provider aggregation
- No data leakage
- Model training and prediction
- Artifact persistence
- Feature column order consistency

## Performance Expectations

### Training Time
- Data loading: ~5-10 seconds
- Preprocessing: ~10-15 seconds
- Feature aggregation: ~30-45 seconds
- Hyperparameter tuning (5×4 = 20 combinations): ~3-5 minutes
- Model training: ~1-2 minutes
- Total: ~5-10 minutes

### Memory Requirements
- Dataset size: ~200-300 MB
- Working memory during processing: ~500 MB - 1 GB
- Fitted model size: ~10-50 MB

### Prediction Time
- Test data (1000s of providers): <1 second

## References

### One-Class SVM
- Schölkopf et al., "Support Vector Method for Novelty Detection"
- Scikit-learn OneClassSVM: https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html

### Healthcare Fraud Detection
- CMS Fraud & Abuse Prevention: https://www.cms.gov/
- Synthetic Medicare Data: Public domain research dataset

## License

This project is provided for educational and research purposes.

## Contact & Support

For issues or questions:
1. Check this README
2. Review inline code comments
3. Inspect log output for detailed error messages
4. Check test files for usage examples

---

**Version**: 1.0.0  
**Last Updated**: 2026-08-17  
**Status**: Production-Ready
