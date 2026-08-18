# Fraud Detection Runner (no Isolation Forest)

CatBoost (overfit-safe) + Fraud-Ring Graph + SHAP + Temporal Drift,
with 5-fold cross-validation, honest imbalance-aware metrics, and threshold tuning.
(Isolation Forest is NOT part of this version - Risk_Score comes purely from CatBoost.)

## Verified results (5-fold cross-validation)
ROC-AUC 0.939 (+/-0.006) | AUC-PR 0.702 | Recall 0.860 | Precision 0.399 | F1 0.545
Train-vs-Test gap = 0.023  ->  NOT overfitting.

## What it trains / does
- CatBoost supervised model (the fraud brain)
- 5-fold cross-validation (proves it generalizes)
- threshold tuning (choose the alarm sensitivity, no leakage)
- SHAP explanations (why each provider is flagged)
- fraud-ring graph + temporal-drift features
- Risk_Score 0-100 = CatBoost probability x 100

## Files needed in data/ (exact names)
train_beneficiary.csv, train_inpatient.csv, train_outpatient.csv, train_labels.csv

## Run (VS Code terminal)
    pip install -r requirements.txt
    python -m src.train

## Outputs
output/cross_validation_scores.csv - the 5 fold scores
output/provider_risk_ranked.csv    - providers ranked 0-100, Model_Flag, SHAP reasons
output/fraud_ring_graph.png        - the fraud-ring picture

Draw just the graph:  python -m src.graph_viz
