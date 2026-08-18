# AI-Powered Healthcare Claims Fraud Intelligence & Investigation Platform

## 1. Project Overview

### Project Objective

Build an AI-powered healthcare claims fraud, waste, and abuse risk detection platform called "MediClaim" using the given (dataset) healthcare provider fraud dataset.

The platform will:

1. Process healthcare provider/claim data.
2. Engineer provider-level behavioral features.
3. Detect anomalous behavior using multiple unsupervised ML techniques.
4. Benchmark providers against comparable peers.
5. Generate statistically explainable evidence.
6. Combine multiple signals into an overall risk score.
7. Prioritize suspicious providers for investigation.
8. Present the results through a polished React dashboard.
9. Expose the analytical results through a FastAPI backend.
10. Provide charts, diagrams, explanations, and investigation workflows rather than displaying raw model outputs only.

### Core Principle

The project should not present the system as a black-box "fraud predictor."

The intended workflow is:

**Data → Feature Engineering → Anomaly Detection → Statistical Analysis → Peer Benchmarking → Risk Scoring → Explainability → Investigation Queue → Human Investigator**

The ML models identify unusual behavior. Statistical and peer-analysis components explain the behavior. The final system helps investigators decide which providers deserve attention.

---

# 2. Problem Statement

Healthcare insurers process very large numbers of claims. Fraudulent, wasteful, or abusive behavior can represent only a small fraction of the overall data, making manual identification difficult.

A simple fraud label is not sufficient for an investigator. The investigator needs:

- A prioritized list of suspicious providers.
- A numerical risk score.
- Evidence explaining why the provider was flagged.
- Comparison with similar providers.
- Identification of unusual billing/service patterns.
- Multiple independent signals rather than dependence on a single model.

The platform therefore focuses on **risk detection and investigation support**, not automatic declaration of guilt.

---

# 3. Dataset Strategy

## 3.1 Primary Dataset

Use the given (dataset) healthcare provider fraud detection dataset.
Test-1542969243754.csv
Test_Beneficiarydata-1542969243754.csv
Test_Inpatientdata-1542969243754.csv
Test_Outpatientdata-1542969243754.csv
Train-1542865627584.csv
Train_Beneficiarydata-1542865627584.csv
Train_Inpatientdata-1542865627584.csv
Train_Outpatientdata-1542865627584.csv


Before modeling, document:

- Dataset source.
- Dataset license/usage conditions.
- Number of records.
- Number of providers.
- Number of claims/services.
- Available provider attributes.
- Available beneficiary attributes.
- Payment/reimbursement fields.
- Diagnosis/procedure information.
- Existing fraud labels, if present.
- Missing-value patterns.
- Duplicate records.
- Time coverage.
- Target/label distribution.

## 3.2 Important Dataset Principle

Do not immediately train the six models.

First understand the data dictionary and determine what each field actually represents.

The team should create a dataset documentation sheet containing:

| Field | Meaning | Type | Missing % | Used For |
|---|---|---|---:|---|
| Provider ID | Provider identifier | ID | - | Grouping |
| Claim fields | Claim information | Numeric/Categorical | TBD | Feature engineering |
| Payment fields | Reimbursement information | Numeric | TBD | Financial features |
| Beneficiary fields | Beneficiary information | Numeric | TBD | Utilization features |
| Specialty | Provider specialty | Categorical | TBD | Peer grouping |
| Fraud label | Known label, if available | Binary | TBD | Evaluation |

Exact fields must be confirmed from the selected given (dataset) dataset rather than assumed.

---

# 4. Target Unit of Analysis

The recommended primary unit is the **provider** and "Claims".

Raw claim/service records should be transformed into provider-level and claim-level behavioral features.

Example:

```text
Provider
 ├── Total claims
 ├── Total services
 ├── Total beneficiaries
 ├── Total reimbursement
 ├── Average reimbursement per claim
 ├── Average reimbursement per beneficiary
 ├── Services per beneficiary
 ├── Claims per beneficiary
 ├── Payment per service
 ├── Diagnosis diversity
 ├── Procedure diversity
 └── Other utilization/payment features
```

This makes the system suitable for provider-level and claim - level investigation.

If the dataset supports sufficient granularity, a secondary claim-level analysis can be added later.

---

# 5. Overall System Architecture

```text
                         given (dataset) HEALTHCARE DATA
                                   |
                                   v
                         DATA INGESTION / CLEANING
                                   |
                                   v
                         FEATURE ENGINEERING
                                   |
                    +--------------+--------------+
                    |                             |
                    v                             v
             ML ANOMALY LAYER                  EXPLAINABILITY LAYER
                    |                             |
          +---------+---------+          +--------+---------+
          |         |         |          |        |         |
          v         v         v          v        v         v
Autoencoder  Isolation   LOF and catboost     Z-Score  Percentile  Peer
                 Forest                                  Ratios
          |         |         |
          +---------+---------+
                    |
                    v
             ONE-CLASS SVM
                    |
                    +--------------------+
                                         |
                                         v
                                RISK SCORING ENGINE
                                         |
                                         v
                              EXPLAINABILITY ENGINE
                                         |
                              +----------+----------+
                              |                     |
                              v                     v
                       RISK SCORE / LEVEL     EVIDENCE / REASONS
                              |                     |
                              +----------+----------+
                                         |
                                         v
                              INVESTIGATION QUEUE
                                         |
                                         v
                              FASTAPI REST API
                                         |
                                         v
                               REACT FRONTEND
                                         |
                         +---------------+---------------+
                         |               |               |
                         v               v               v
                    DASHBOARD      ANALYTICS       INVESTIGATION
                                                     WORKSPACE
```

---

# 6. ML and Analytics Components

## 6.1 Autoencoder

### Purpose

Learn the structure of normal provider behavior and identify providers that have high reconstruction error.

### Concept

Input features are passed through:

```text
Input Features
      |
      v
Encoder
      |
Latent Representation
      |
      v
Decoder
      |
      v
Reconstructed Features
```

Calculate reconstruction error:

```text
Reconstruction Error =
Difference between original and reconstructed feature vector
```

High error indicates that the provider's behavior is difficult for the model to reconstruct and may therefore be anomalous.

### Advantages

- Captures nonlinear relationships.
- Can detect complex behavioral patterns.
- Useful when anomalies are not obvious from a single feature.

### Limitations

- Requires careful normalization.
- Threshold selection is important.
- Harder to explain than statistical methods.
- Anomaly does not automatically mean fraud.

### Output

```text
autoencoder_score
reconstruction_error
autoencoder_flag
```

---

# 7. Isolation Forest

## Purpose

Detect providers that are easily isolated from the majority of providers.

### Concept

Isolation Forest recursively partitions the feature space.

Unusual observations tend to be isolated using fewer splits.

### Why use it

- Efficient.
- Suitable for large datasets.
- Strong baseline for unsupervised anomaly detection.
- Relatively easy to operationalize.

### Output

```text
isolation_score
isolation_flag
```

### Recommended Role

Use Isolation Forest as one of the primary anomaly detectors.

---

# 8. One-Class SVM

## Purpose

Learn a boundary around normal provider behavior and identify providers outside that boundary.

### Workflow

```text
Normal provider features
          |
          v
     One-Class SVM
          |
          v
Normal region / Outlier
```

### Important Consideration

One-Class SVM can become computationally expensive and is sensitive to feature scaling and hyperparameters.

Therefore:

- Use standardized/scaled features.
- Tune `nu`.
- Tune kernel parameters where appropriate.
- Evaluate performance carefully.
- Treat it as a comparison/ensemble signal rather than automatically making it the primary model.

### Output

```text
ocsvm_score
ocsvm_flag
```

---

# 9. CatBoost
Purpose

Predict the probability that a provider exhibits fraudulent, wasteful, or abusive behavior based on historical patterns and engineered provider-level features.

Why it matters

Unlike anomaly detection models that identify unusual behavior, CatBoost learns from known fraud labels and discovers patterns associated with previously identified fraudulent providers.

Example:

Provider Features
      |
      +-- Total Claims
      +-- Total Services
      +-- Total Reimbursement
      +-- Beneficiary Count
      +-- Claims per Beneficiary
      +-- Payment per Service
      |
      v
    CatBoost
      |
      v
 Fraud Probability

A provider may not appear as an extreme outlier but can still match patterns commonly observed in known fraudulent providers.

Output
catboost_score
fraud_probability
catboost_prediction
Advantages
Handles categorical and numerical features effectively.
Requires minimal feature preprocessing.
Captures complex nonlinear relationships.
Strong performance on tabular healthcare data.
Provides feature importance for explainability.
Limitations
Requires labeled fraud data.
Performance depends on label quality.
May learn historical biases present in the training data.
Not suitable when only unlabeled data is available.
---

# 10. Peer Benchmarking Engine

This is one of the most important explainability components.

## Purpose

Compare a provider with comparable providers instead of comparing it blindly against the entire population.

Potential peer-group dimensions:

- Specialty.
- Provider type.
- Geography, if available and meaningful.
- Service characteristics.
- Other clinically/business-relevant attributes.

Avoid overly small peer groups.

## Metrics

Examples:

```text
Provider service volume
Peer median service volume
Provider payment
Peer median payment
Provider beneficiary count
Peer median beneficiary count
Provider percentile
Provider / peer median ratio
```

Example explanation:

> Provider performs 4.2× the median service volume of comparable providers in the same specialty.

This is significantly more understandable to an investigator than a raw anomaly score.

---

# 11. Robust Z-Score

Use robust statistics to reduce the influence of extreme values.

A median/MAD-based approach is preferred when the distribution contains significant outliers.

Conceptually:

```text
How far is the provider from the typical population?
```

Possible features:

- Payment.
- Service volume.
- Beneficiary count.
- Services per beneficiary.
- Payment per service.

Output:

```text
robust_z_payment
robust_z_services
robust_z_beneficiaries
```

---

# 12. Percentile Analysis

Percentiles provide intuitive explanations.

Examples:

> Provider is in the 98.7th percentile for service volume.

> Provider is in the 99.2nd percentile for reimbursement.

Percentile information should be surfaced directly in the UI.

---

# 13. Peer Ratio Analysis

Example:

```text
Peer Ratio =
Provider Metric / Peer Median Metric
```

Examples:

```text
Service volume ratio = 4.2x
Payment ratio        = 3.1x
Beneficiary ratio    = 1.7x
```

Peer ratios are especially useful for investigator-facing explanations.

---

# 14. Feature Engineering Plan

Feature engineering should be treated as a major project phase.

## Provider-Level Features

### Claim Features

- Total claims.
- Claims per beneficiary.
- Claims per service.
- Claim frequency.
- Average claim amount.

### Service Features

- Total services.
- Services per beneficiary.
- Services per claim.
- Service frequency.
- Specialty-specific service metrics.

### Financial Features

- Total reimbursement.
- Average reimbursement per claim.
- Average reimbursement per service.
- Reimbursement per beneficiary.
- Payment-to-service ratios.

### Beneficiary Features

- Total beneficiaries.
- Beneficiaries per claim.
- Beneficiaries per service.
- Beneficiary concentration.

### Diversity Features

Where available:

- Number of unique diagnosis codes.
- Number of unique procedure codes.
- Number of service categories.

### Distribution Features

Where useful:

- Median payment.
- Standard deviation.
- IQR.
- MAD.
- Maximum payment.
- Percentiles.

Exact features should be finalized after inspecting the given (dataset) dataset.

---

# 15. Data Preprocessing

Pipeline:

```text
Raw Data
   |
   v
Schema Validation
   |
   v
Duplicate Handling
   |
   v
Missing Value Treatment
   |
   v
Data Type Conversion
   |
   v
Categorical Encoding
   |
   v
Numerical Scaling
   |
   v
Outlier-Aware Feature Processing
   |
   v
Model-Ready Dataset
```

Important:

Do not blindly remove outliers because the objective is anomaly detection. Some extreme values may be precisely the behavior the system needs to detect.

---

# 16. Handling Fraud Labels

If the given (dataset) dataset contains a known fraud label:

### Do not automatically use the fraud label as an input feature to the unsupervised models.

Instead:

```text
Features
   |
   +--> Unsupervised models 
   |
   +--> Statistical analysis
   |
   +--> Peer analysis
   |
   v
Anomaly/Risk score

Known fraud label
   |
   v
Evaluation only
```

This allows the team to demonstrate whether unsupervised anomaly signals correspond to known fraudulent providers.

---

# 17. Model Evaluation

Because fraud/anomaly datasets can be highly imbalanced, accuracy should not be the primary metric.

Use:

- Precision.
- Recall.
- F1-score.
- ROC-AUC.
- PR-AUC.
- Precision@K.
- Recall@K.

## Precision@K

If investigators can investigate only the top 100 providers:

```text
Top 100 highest-risk providers
             |
             v
How many are actually known fraudulent?
```

This is highly relevant to the real-world workflow.

## Recommended comparison

Create a model comparison table:

| Model | Precision | Recall | F1 | PR-AUC | Precision@100 |
|---|---:|---:|---:|---:|---:|
| Isolation Forest | TBD | TBD | TBD | TBD | TBD |
| Autoencoder | TBD | TBD | TBD | TBD | TBD |
| LOF | TBD | TBD | TBD | TBD | TBD |
| Catboost | TBD | TBD | TBD | TBD | TBD |
| One-Class SVM | TBD | TBD | TBD | TBD | TBD |
| Combined Risk Engine | TBD | TBD | TBD | TBD | TBD |

Do not fill these with invented values. Populate them after experiments.

---

# 18. Risk Scoring Engine

The system should combine multiple signals.

A possible starting framework:

```text
ML anomaly signals
        +
Statistical deviation
        +
Peer deviation
        +
Behavioral evidence
        |
        v
Final Risk Score: 0–100
```

Example conceptual weighting:

```text
Isolation Forest       15%
Autoencoder            10%
LOF                    10%
One-Class SVM          5%
Statistical Risk       20%
Peer Benchmark Risk    20%
catboost               20%
```

These weights are initial design values only.

They must be validated experimentally.

## Risk Levels

Initial UI classification:

```text
0–30     Low
31–60    Medium
61–80    High
81–100   Critical
```

Thresholds should be adjustable after validation.

---

# 19. Explainability Engine

For every high-risk provider, generate structured reasons.

Example:

```text
Provider Risk Score: 91

Reasons:
1. Service volume is 4.2× the peer median.
2. Provider is in the 98.7th percentile for reimbursement.
3. LOF identifies the provider as a local outlier.
4. Autoencoder reconstruction error is above the anomaly threshold.
5. Payment per beneficiary is significantly above peer norms.
```

The system should distinguish:

- Model signal.
- Statistical evidence.
- Peer evidence.

Never claim:

> "This provider committed fraud."

Prefer:

> "This provider exhibits multiple patterns associated with elevated fraud risk and should be reviewed."

---

# 20. React Frontend Plan

## Technology

Recommended:

- React.
- TypeScript.
- Vite or the team's existing React setup.
- React Router.
- Recharts or another mature charting library.
- Tailwind CSS or an established UI component system.
- Lucide icons or equivalent.
- API client using `fetch` or Axios.

The exact frontend libraries can be finalized based on the team's existing project setup.

---

# 21. Frontend Pages

## Page 0: Data Ingestion & Analysis

Purpose:

Allow analysts to upload healthcare claims and provider datasets and trigger the fraud risk analysis pipeline.

Components:

- Upload Claims Dataset
- Upload Provider Dataset
- Dataset Validation Status
- Record Count
- Processing Status
- Run Analysis Button
- Last Analysis Timestamp

Workflow:

Upload Dataset
      ↓
Data Validation
      ↓
Feature Engineering
      ↓
Model Execution
      ↓
Risk Score Generation
      ↓
Dashboard Refresh

## Page 1: Executive Dashboard

Purpose:

Give an immediate overview.

Components:

- Total providers.
- High-risk providers.
- Critical providers.
- Investigation queue count.
- Total claims.
- Risk distribution.
- Risk by specialty.
- Top suspicious providers.
- Recent investigation activity.

Recommended charts:

- Risk distribution bar/pie chart.
- Risk by specialty bar chart.
- Provider risk ranking.
- Payment/service distribution.

---

# 22. Page 2: Investigation Queue

Purpose:

Prioritize work.

Columns:

- Provider ID.
- Risk score.
- Risk level.
- Peer deviation.
- Main reason.
- Investigation status.

Filters:

- Risk level.
- Provider type.
- Risk score.
- Investigation status.

Actions:

- View provider.
- Mark as under review.
- Mark reviewed.
- Add investigator notes.

For the hackathon MVP, status and notes can be implemented simply without a complex case-management system.

---

# 23. Page 3: Provider Investigation

This should be the most polished page.

## Header

```text
Provider ID
Specialty
Risk score
Risk level
Investigation status
```

## Risk Explanation

Display the top evidence.

## Peer Comparison

Charts comparing:

- Service volume.
- Payment.
- Beneficiaries.
- Claims.
Graph:

Provider
vs
Peer Median

Metrics:

Claims
Payments
Services
Beneficiaries

## Model Signals

Show:

- Isolation Forest.
- Autoencoder.
- LOF.
- One-Class SVM.

## Statistical Signals

Show:

- Percentile.
- Robust Z-score.
- Peer ratios.

## Behavioral Trends

If temporal data supports it:

- Monthly claim volume.
- Monthly payment.
- Monthly service volume.

---

# 24. Page 4: Analytics

Recommended charts:

### Risk Distribution

Low / Medium / High / Critical.

### Risk by Specialty

Rank specialties by suspicious-provider count or aggregate risk.

### Payment vs Service Volume

Scatter plot.

### Provider Risk Ranking

Top-N providers.

### Anomaly Model Comparison

Compare the distribution of anomaly scores.

### Statistical Deviations

Show the distribution of key metrics.

---

# 25. Page 5: How It Works

Create a visual architecture/ML pipeline diagram.

```text
Data
 ↓
Preprocessing
 ↓
Feature Engineering
 ↓
 ┌───────────────┬──────────────────┐
 ↓               ↓                  ↓
Autoencoder   Isolation Forest     LOF
 ↓               ↓                  ↓
 └───────────────┼──────────────────┘
                 ↓
        Statistical Analysis
                 ↓
         Peer Benchmarking
                 ↓
          Risk Score Engine
                 ↓
          Explainable Result
                 ↓
        Investigation Queue
```

This page is useful for judges, mentors, and technical reviewers.

---

# 26. FastAPI Backend Plan

## Suggested API Structure

```text
/api/v1/dashboard
/api/v1/providers
/api/v1/providers/{provider_id}
/api/v1/providers/{provider_id}/risk
/api/v1/providers/{provider_id}/peers
/api/v1/providers/{provider_id}/signals
/api/v1/analytics/risk-distribution
/api/v1/analytics/specialties
/api/v1/investigations
/api/v1/model-performance
```

## Example Responses

### Provider Risk

```json
{
  "provider_id": "12345",
  "risk_score": 91,
  "risk_level": "critical",
  "signals": {
    "isolation_forest": 0.91,
    "autoencoder": 0.87,
    "lof": 0.94,
    "catboost": 0.88,
    "one_class_svm": 0.82
  },
  "peer_metrics": {
    "service_ratio": 4.2,
    "payment_ratio": 3.1,
    "service_percentile": 98.7
  },
  "reasons": [
    "Service volume is 4.2x the peer median",
    "Service volume is in the 98.7th percentile",
    "High local anomaly score"
  ]
}
```

Actual schema should be finalized after implementation.

---

# 27. Backend Internal Structure

Recommended conceptual structure:

```text
backend/
├── app/
│   ├── main.py
│   ├── api/
│   │   ├── dashboard.py
│   │   ├── providers.py
│   │   ├── investigations.py
│   │   └── analytics.py
│   ├── services/
│   │   ├── risk_engine.py
│   │   ├── peer_engine.py
│   │   ├── explanation_engine.py
│   │   └── analytics_service.py
│   ├── models/
│   │   ├── autoencoder.py
│   │   ├── isolation_forest.py
│   │   ├── lof.py
│   │   └── one_class_svm.py
│   ├── schemas/
│   └── db/
└── tests/
```

---

# 28. ML Training Architecture

Training should be separate from request-time API processing.

Recommended:

```text
ml/
├── data/
├── preprocessing/
├── features/
├── models/
│   ├── autoencoder/
│   ├── isolation_forest/
│   ├── lof/
│   └── one_class_svm/
├── evaluation/
├── risk_engine/
└── notebooks/
```

Training workflow:

```text
Dataset
   ↓
Preprocessing
   ↓
Feature Engineering
   ↓
Train Models
   ↓
Generate Scores
   ↓
Evaluate
   ↓
Tune
   ↓
Freeze Selected Models
   ↓
Generate Provider Risk Results
   ↓
Store Results
```

---

<!-- # 29. Database Strategy(no need as of now)

For the MVP, store:

## Providers

```text
provider_id
specialty
provider_type
other metadata
```

## Provider Features

```text
provider_id
total_claims
total_services
total_beneficiaries
total_payment
derived metrics
```

## Model Scores

```text
provider_id
autoencoder_score
isolation_score
lof_score
ocsvm_score
```

## Statistical Scores

```text
provider_id
robust_z_scores
percentiles
peer_ratios
```

## Risk Results

```text
provider_id
risk_score
risk_level
primary_reasons
created_at
```

## Investigations

```text
provider_id
status
notes
created_at
updated_at
```

A relational database such as PostgreSQL is appropriate if the team already uses it. SQLite can be acceptable for a simple local prototype. -->

---

# 30. Important ML Design Decision

Do not train all models on raw columns.

The models should consume engineered numerical features.

For example:

```text
Raw claims
     ↓
Provider aggregation
     ↓
Behavioral features
     ↓
Scaling / normalization
     ↓
ML models
```

This is essential.

---

# 31. Data Leakage Prevention

If fraud labels exist, do not accidentally use them in features. Use only when needed

Also inspect whether any field indirectly reveals the target.

The evaluation pipeline should separate:

```text
X = model features
y = known fraud label
```

The unsupervised models operate on `X`.

`y` is used for evaluation.

---

# 32. Threshold Selection

Do not arbitrarily decide that:

```text
anomaly score > 0.8 = fraud
```

Instead:

1. Generate anomaly scores.
2. Inspect score distributions.
3. Compare against known labels.
4. Evaluate precision/recall.
5. Select thresholds.
6. Validate on held-out data.

The final UI should expose:

```text
Risk score
Risk level
Evidence
```

rather than the raw threshold logic.

---

# 33. Recommended Development Phases

## Phase 1 — Dataset Understanding

Deliverables:

- Dataset downloaded.
- Data dictionary.
- Schema report.
- Missing-value report.
- Duplicate report.
- Label distribution.
- Provider/claim relationships understood.

---

## Phase 2 — Data Pipeline

Deliverables:

- Cleaning pipeline.
- Provider aggregation.
- Feature engineering pipeline.
- Reproducible preprocessing.

---

## Phase 3 — Baseline Model

Start with:

**Isolation Forest**

Deliver:

- Anomaly scores.
- Initial ranking.
- Initial evaluation.

This gives the team a working baseline early.

---

## Phase 4 — Additional Models

Add:

1. Autoencoder.
2. LOF.
3. One-Class SVM.

Compare them systematically.

---

## Phase 5 — Explainability

Implement:

- Robust Z-score.
- Percentile.
- Peer ratios.
- Peer benchmarking.
- Human-readable reason generation.

---

## Phase 6 — Risk Engine

Combine:

```text
ML + Statistical + Peer signals
```

into a unified risk score.

---

## Phase 7 — FastAPI

Expose:

- Dashboard data.
- Provider risk.
- Peer comparisons.
- Analytics.
- Investigation queue.
- Model performance.

---

## Phase 8 — React UI

Build:

1. Dashboard.
2. Investigation Queue.
3. Provider Investigation.
4. Claim Investigation.
5. Analytics. Reports.
6. How It Works.
7. AI Invstigation Assistant(RAG integrated)
8. Setting for the end users

---

## Phase 9 — Visualization Polish

Add:

- Charts.
- KPI cards.
- Risk indicators.
- Peer comparison visuals.
- Model score visualizations.
- Scatter plots.
- Architecture diagrams.
- Tooltips.
- Filters.
- Responsive layout.

---

## Phase 10 — Evaluation and Demo

Validate:

- Model metrics.
- Top-K performance.
- Explanation correctness.
- API performance.
- UI responsiveness.
- End-to-end workflow.

---

Final Risk Score =
40% ML Signals
30% Statistical Signals
30% Peer Benchmark Signals

# 34. MVP Priority

If time becomes limited, prioritize in this order.

### Must Have

- given (dataset) dataset.
- Provider-level feature engineering.
- Isolation Forest.
- Peer benchmarking.
- Statistical analysis.
- Risk scoring.
- Explainable reasons.
- FastAPI.
- React dashboard.
- Provider investigation page.
- Investigation queue.

### Should Have

- Autoencoder.
- catboost.
- Analytics page.
- Model comparison.
- Architecture page.

### Nice to Have

- One-Class SVM.
- Advanced temporal analysis.
- Investigation notes.
- Exportable reports.
- Advanced filtering.
- Authentication.
- RAG Assistant
- Role-based access.

Do not sacrifice the core workflow just to include more algorithms.

---

# 35. What Makes This Project Strong

The project should demonstrate five things.

## 1. Detection

The system finds anomalous providers.

## 2. Comparison

The system knows whether a provider is unusual relative to comparable providers.

## 3. Explanation

The system can explain why the provider is suspicious.

## 4. Prioritization

The system ranks cases so investigators can focus on the highest-risk providers.

## 5. Visualization

The system communicates complex model results through a professional UI.

---

# 36. Final Demo Story

The final presentation should follow this sequence:

### Step 1

Open the dashboard.

> "This is the overall healthcare fraud-risk landscape."

### Step 2

Show the high-risk provider count.

> "The system has identified these providers as requiring further review."

### Step 3

Open a provider.

> "Let's investigate this provider."

### Step 4

Show the risk score.

> "This provider has a risk score of 91/100."

### Step 5

Show evidence.

> "The provider performs 4.2× the peer median service volume and is in the 98.7th percentile for reimbursement."

### Step 6

Show model agreement.

> "Multiple independent anomaly detectors also identify this provider as unusual."

### Step 7

Show the peer comparison chart.

> "This is the provider compared with similar providers."

### Step 8

Show investigation queue.

> "The system automatically prioritizes the highest-risk cases."

### Step 9

Explain architecture.

> "The platform combines unsupervised ML, statistical analysis, peer benchmarking, and explainability."

This tells a complete story from **raw data → AI → evidence → investigation**.

---

# 37. Final Recommended Technology Stack

## Frontend

- React
- TypeScript
- Tailwind CSS / component library
- Recharts or equivalent
- React Router
- Axios/fetch

## Backend

- Python
- FastAPI
- Pydantic
- Pandas
- NumPy
- Scikit-learn

## ML

- Scikit-learn
- PyTorch or TensorFlow for Autoencoder
- Isolation Forest
- catboost
- autoencoder
- One-Class SVM

## Data

- given (dataset) healthcare provider fraud dataset
- PostgreSQL for application results, if required

## Visualization

- Bar charts.
- Line charts.
- Scatter plots.
- Risk distribution.
- Peer comparison charts.
- Model signal charts.
- Architecture/flow diagrams.

---

# 38. Final Architecture in One View

```text
                         ┌──────────────────────┐
                         │   given (dataset) DATASET     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ DATA PREPROCESSING   │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ FEATURE ENGINEERING  │
                         └──────────┬───────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
          ┌───────────────────┐          ┌────────────────────┐
          │ ML ANOMALY LAYER  │          │ ANALYTICS LAYER    │
          ├───────────────────┤          ├────────────────────┤
          │ Autoencoder       │          │ Robust Z-score     │
          │ Isolation Forest   │          │ Percentile         │
          │         LOF          │          │ Peer Ratio         │
          │One-Class SVM  
          SUPERVISED: CatBOOST   │          │ Peer Benchmarking  │
          └─────────┬─────────┘          └──────────┬─────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                         ┌──────────────────────┐
                         │   RISK SCORE ENGINE  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ EXPLAINABILITY       │
                         │ ENGINE                │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ INVESTIGATION QUEUE  │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │      FASTAPI         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       REACT          │
                         │  INVESTIGATION UI    │
                         ├──────────────────────┤
                         │ Dashboard            │
                         │ Analytics, Reports   │
                         │ Provider Investigation│
                         │ Investigation Queue  │
                         │ How It Works      
                         │Claim Investigation
                          AI Invstigation Assistant(RAG integrated)
                           Setting for the end users
                         └──────────────────────┘
```

---

FINAL FLOW
Upload
↓
Feature Engineering
↓
Model Execution
↓
Risk Fusion
↓
Explainability
↓
Dashboard
↓
Investigation Queue
↓
Provider Investigation

# 39. Core Project Statement

The final project should be described as:

> **An AI-powered healthcare claims fraud, waste, and abuse risk intelligence platform named as "MediClaim" that combines unsupervised anomaly detection, supervised catboost, statistical deviation analysis, peer benchmarking, and explainable risk scoring to identify and prioritize suspicious healthcare providers for human investigation.**

The strongest part of the project is **not the number of algorithms**.

The strongest part is the complete chain:

**Detect → Compare → Explain → Score → Prioritize → Investigate**

That should guide both the ML implementation and the React UI design.
