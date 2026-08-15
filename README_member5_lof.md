# Member 5 — LOF Provider Anomaly Detection (v2 — ready-made data)

Detects providers whose billing behaviour differs from peers **within the same
specialty**, using Local Outlier Factor. Unsupervised — no fraud labels, so we
do **not** report accuracy/precision/recall.

## What changed in this data version

This zip already ships **`PROVIDER_ML_READY.csv`** (80,000 providers, 1 row each,
all 6 features present). So there is **no aggregation step** — the model reads
that file directly. It also ships a richer LEIE file
(`leie_clean_specialty_filled.csv`) with `IS_CURRENTLY_EXCLUDED` and
`IS_FRAUD_RELATED_EXCLUSION` flags, which we use for stronger validation.

Only **two** files feed Member 5:
- `PROVIDER_ML_READY.csv`            → the model input
- `leie_clean_specialty_filled.csv` → validation (LEIE overlap)

The other files (`carrier_clean_final_perfect.csv`, `outpatient_features.csv`,
`inpatient_cleaned.csv`, `beneficiary_dataset.xlsx`, the FFS zip) belong to other
members. You consume their outputs at combine time, not their raw rows.

## Project layout
```
member5_lof/
├── data/                               # put the two source files here
│   ├── PROVIDER_ML_READY.csv
│   └── leie_clean_specialty_filled.csv
├── output/
│   └── member5_lof_results.csv         # generated
├── src/
│   ├── config.py
│   └── lof_provider_anomaly.py
├── requirements.txt
└── README.md
```

## Run (two lines)
```bash
pip install -r requirements.txt
python -m src.lof_provider_anomaly
```

## Verified result on the new data
80,000 analysed · 79,868 scored · 132 insufficient-peer · Low 76,959 / Medium
2,084 / High 300 / Critical 525 · LEIE overlap 8 (2 fraud-related land in the
top-3% risk tier).

## Combining with the team later
Join on `Rndrng_NPI`; use the 0–100 `Risk_Score` (already comparable across
members); namespace columns as `m5_*` when merged; keep `INSUFFICIENT_PEERS`
rows with null scores; don't re-scale after merging.

## Output columns
`Rndrng_NPI, Rndrng_Prvdr_Type, LOF_Score, Risk_Score, Risk_Level,
Payment_Ratio, Service_Ratio, Beneficiary_Ratio, LEIE_Match,
Currently_Excluded, Fraud_Related, Explanation`
