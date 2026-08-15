# Peer Comparison Module

## Files

- `peer-to-peerComp.py` → Main peer comparison and fraud risk scoring script
- `PEER_ANALYSIS_RESULTS.csv` → Output results
- `leie_clean_specialty_filled.csv` → OIG/LEIE exclusion reference dataset
- `screenshots/` → Dashboard screenshots
- `README.md` → Documentation

## Objective

Identify potentially suspicious Medicare providers by comparing each provider against peers within the same specialty and geographic region.

## Methods Used

### 1. Robust Z-Score Analysis
Measures how far a provider's payments, services, and beneficiaries deviate from the median of their peer group using Median Absolute Deviation (MAD). More resistant to extreme outliers than traditional Z-score.

### 2. Percentile Analysis
Ranks providers within their specialty peer group to identify top-performing and unusually high-billing providers.

### 3. Peer Ratio Analysis
Compares provider metrics against peer-group medians:

- Payment Ratio
- Service Ratio
- Beneficiary Ratio

Higher ratios indicate unusual behavior relative to peers.

### 4. Geographic Peer Comparison
Compares provider payments against providers from the same specialty and state.

### 5. OIG / LEIE Validation
Matches providers against the OIG List of Excluded Individuals and Entities (LEIE) to identify known excluded providers.

### 6. Financial Leakage Estimation
Estimates potential excess Medicare spending by comparing provider payments against peer median payments.

## Output Columns

- Peer_Risk_Score
- Risk_Level
- Fraud_Flag
- Explainability
- Payment_Ratio
- Service_Ratio
- Beneficiary_Ratio
- Geo_Payment_Ratio
- Estimated_Leakage
- OIG_Flag

## Output

Generated file:

`PEER_ANALYSIS_RESULTS.csv`

