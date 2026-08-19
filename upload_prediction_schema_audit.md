# Upload Prediction Pipeline Schema Audit

## Scope

This audit follows `backend/app/main.py` upload validation and `backend/app/pipeline.py` through claim scoring, provider aggregation, model scoring, peer benchmarking, robust z-scores, and database output.

The schema compatibility score means: **the percentage of the model's runtime input contract that is available from the minimum source CSVs after the pipeline's documented derivations and defaults**. A score of 100% means the model can receive every expected feature; it does not mean that an arbitrary dataset is statistically calibrated to the training population.

## Minimum source schemas

### Claims CSV

Mandatory columns:

`ClaimID`, `BeneID`, `Provider`, `InscClaimAmtReimbursed`, `DeductibleAmtPaid`, `ClaimStartDt`, `ClaimEndDt`, `AttendingPhysician`, `OperatingPhysician`, `OtherPhysician`

Optional columns:

`is_inpatient`, `AdmissionDt`, `DischargeDt`, `ClmDiagnosisCode_1` through `ClmDiagnosisCode_10`, and `ClmProcedureCode_1` through `ClmProcedureCode_6`.

Defaults and derivations:

- Missing `is_inpatient` is inferred from admission/discharge dates; with both absent it becomes outpatient (`0`).
- Missing admission/discharge dates produce `AdmissionDuration = 0`.
- Missing diagnosis/procedure code columns produce counts of `0`.
- `ClaimDuration` is computed from claim dates and defaults to `1` when unusable.
- Claim numeric nulls and infinities are filled/clipped to `0` before claim model scoring.

The physician columns are mandatory even though the upload endpoint currently validates only four claim columns. Provider aggregation indexes all three physician columns directly.

### Beneficiary CSV

Mandatory columns:

`BeneID`, `DOB`, `DOD`, `Gender`, `Race`, `State`, `County`, `RenalDiseaseIndicator`, `IPAnnualReimbursementAmt`, `IPAnnualDeductibleAmt`, `OPAnnualReimbursementAmt`, `OPAnnualDeductibleAmt`

Optional columns:

`ChronicCond_Alzheimer`, `ChronicCond_Heartfailure`, `ChronicCond_KidneyDisease`, `ChronicCond_Cancer`, `ChronicCond_ObstrPulmonary`, `ChronicCond_Depression`, `ChronicCond_Diabetes`, `ChronicCond_IschemicHeart`, `ChronicCond_Osteoporasis`, `ChronicCond_rheumatoidarthritis`, `ChronicCond_stroke`.

Defaults and derivations:

- Missing chronic-condition columns result in `ChronicCondCount = 0`.
- Chronic values `1` are true and `2` are false; missing values become `0`.
- Missing or invalid DOB values result in age `65`.
- Missing/invalid renal indicators become `0`.
- `DOD` is used only as a null/non-null deceased flag, but the column itself must exist because the preprocessing indexes it.

The six columns currently checked by the beneficiary upload endpoint are insufficient for a full pipeline run.

### Provider CSV

Mandatory column:

`Provider`

Optional columns:

`PotentialFraud`, plus any NPI-compatible column (`NPI`, `Provider_NPI`, `ProviderNPI`, or `NPI_ID`) for LEIE screening.

Defaults:

- Missing `PotentialFraud` becomes `0` (`No`).
- Missing NPI means LEIE score `0`; it is not required for the model pipeline.

## Model matrix

| Model | Required input features at prediction | Mandatory source columns | Optional source columns | Defaults if optional data is absent | Runs with optional columns absent? | Compatibility |
|---|---|---|---|---|---|---|
| Autoencoder | Claims: 12 engineered features; provider: 27 engineered provider features | Claims and beneficiary mandatory schemas; provider ID joins | All diagnosis/procedure/chronic/admission fields; provider labels/NPI | Counts/durations/chronic count as described above; provider label `0` | Yes, with the mandatory schemas | 100% |
| Isolation Forest | Claims: 12 engineered features; provider: 27 engineered provider features | Claims and beneficiary mandatory schemas; provider ID joins | Same optional columns | Same defaults | Yes | 100% |
| One-Class SVM | Claims: 12 engineered features and current-batch fit; provider: 27 engineered provider features | Claims and beneficiary mandatory schemas; provider ID joins | Same optional columns | Same defaults | Yes | 100% |
| LOF | 27 engineered provider features | Claims and beneficiary mandatory schemas; provider ID joins | Same optional columns | Same defaults | Yes | 100% |
| CatBoost | 27 engineered provider features | Claims and beneficiary mandatory schemas; provider ID joins | Same optional columns | Same defaults | Yes | 100% |
| Peer Benchmarking | Provider totals, beneficiary totals, reimbursement, claims, and state-derived peer group | Claims: `Provider`, `BeneID`, reimbursement; beneficiary: `State`; all mandatory join columns | `is_inpatient`, chronic, diagnosis/procedure, provider labels/NPI | Outpatient default, missing peer medians fall back to batch median | Yes, but scores are weak for an unseen state/peer group | 100% |
| Robust Z-Score | `total_reimbursement`, `total_claims`, `claims_per_beneficiary` | Claims: `Provider`, `BeneID`, reimbursement; beneficiary join | All other optional columns | No special raw input beyond the mandatory schema | Yes | 100% |

`LOF` and provider-level `Autoencoder`, `Isolation Forest`, and `One-Class SVM` do not consume raw CSV columns directly. They consume the same 27-column provider feature vector generated from the claims and beneficiary tables.

## Upload API versus successful inference

The API accepts a claims file with only `ClaimID`, `BeneID`, `Provider`, and `InscClaimAmtReimbursed`; a beneficiary file with six demographic columns; and a provider file with `Provider`. Those checks are not the successful-inference contract. A file can pass upload validation and still fail in `run_risk_pipeline()` because the pipeline directly indexes additional columns.

## Can a different Medicare dataset be scored without retraining?

**Yes, conditionally.** A mentor can obtain scores without retraining if the alternate dataset is converted to the minimum schemas above, preserves the same meanings/units/date conventions, and contains compatible provider and beneficiary identifiers. Provider models and their scaler, peer medians, and robust-z parameters are reused. Claim autoencoder and claim One-Class SVM are actually fitted dynamically on the uploaded claim batch.

This is not a guarantee of valid calibration for a materially different population, coding system, reimbursement year, or field definition. A dataset with different semantics, missing mandatory fields, or a shifted distribution requires mapping and validation, and usually model recalibration or retraining before the scores should be treated as operationally reliable.