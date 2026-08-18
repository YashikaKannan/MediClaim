# Peer Benchmarking Audit Report

## Executive Summary

The current peer-benchmarking module is not producing a valid peer comparison because the expected peer grouping columns are missing from the processed provider dataset. In the actual data, the expected columns do not exist:

- `Rndrng_Prvdr_Type` → missing
- `Provider_Type` → missing
- `Specialty` → missing

Because of that, the current logic falls back to grouping on `Provider`, which creates one provider per group. This makes each provider a singleton peer set, and the current score logic ranks the sole value as 100th percentile, yielding `Peer_Risk_Score = 100` for every provider.

This is not an isolated bug in the final risk fusion logic; it begins in the peer-benchmarking module itself and propagates into the fusion layer.

---

## 1) Peer Group Audit

### Result: FAIL

The required grouping fields were checked directly in the processed dataset at [processed/PROVIDER_ML_READY_KAGGLE.csv](processed/PROVIDER_ML_READY_KAGGLE.csv).

Evidence:
- `Rndrng_Prvdr_Type` present: False
- `Provider_Type` present: False
- `Specialty` present: False

Because none of the intended peer group columns exist, the module falls back to `Provider`.

### Group Summary

The current peer-benchmark output shows the true peer group behavior:
- Total peer groups: 5410
- Minimum providers per group: 1
- Maximum providers per group: 1

This means every provider is being compared only to itself.

### Peer Group Summary Output

See [peer-benchmarking/output/peer_group_summary.csv](peer-benchmarking/output/peer_group_summary.csv).

---

## 2) Why Peer Score = 100

### Root cause

The current peer-benchmarking pipeline generates a per-group percentile rank after computing the feature-level risk scores. For a singleton group:

- group size = 1
- percentile rank of the single value is 1.0
- multiplied by 100 = 100

In other words, a provider with no peer comparison is scored as the 100th percentile by definition.

### Verified output stats

From the current [peer-benchmarking/output/peer_benchmark_scores.csv](peer-benchmarking/output/peer_benchmark_scores.csv):

- Min: 100.0
- Q1: 100.0
- Median: 100.0
- Q3: 100.0
- 95th percentile: 100.0
- 99th percentile: 100.0
- Max: 100.0

Counts:
- `Peer_Risk_Score > 90`: 5410
- `Peer_Risk_Score > 95`: 5410
- `Peer_Risk_Score = 100`: 5410

### Debug artifact

See [peer-benchmarking/output/top_peer_outliers_debug.csv](peer-benchmarking/output/top_peer_outliers_debug.csv).

### Conclusion

This is not a valid peer anomaly signal. It is a degenerate grouping outcome.

---

## 3) Robust Z-Score Validation

The intended robust z-score formula is:

$$
Robust\_Z = \frac{value - median}{1.4826 \times MAD}
$$

This formula is conceptually correct.

### What happens in practice

When peer groups contain a single provider:
- `MAD = 0`
- the code enters the `MAD=0` guard and effectively returns 0 for robust z

That means the robust z contribution is not extreme, but the downstream scoring is still forced to 100 because it ranks a singleton value as top percentile.

### Extreme robust z check

With singleton groups, the robust z-score distribution is effectively flat and non-informative, because there is no peer variance.

Result:
- `Robust_Z > 3`: 0
- `Robust_Z > 5`: 0
- `Robust_Z > 10`: 0

This is expected under the current one-provider grouping condition.

---

## 4) Feature Contribution Audit

The peer benchmarking design intends to rank a provider against peers on multiple provider-level financial and utilization features.

However, there are no true peer groups, so feature contribution is not being computed relative to a meaningful peer baseline. This makes the feature contribution table non-actionable until the group construction issue is fixed.

See [peer-benchmarking/output/peer_feature_contributions.csv](peer-benchmarking/output/peer_feature_contributions.csv).

---

## 5) Explainability and RAG Readiness

The module is currently generating score files, but it does not have a reliable explanation layer because the peer scores are produced from singleton groups and are therefore not meaningful.

The explanation file exists in principle, but it is only a wrapper around a bad score. The actual root cause remains the missing peer-group columns and singleton fallback.

See:
- [peer-benchmarking/output/peer_explanations.csv](peer-benchmarking/output/peer_explanations.csv)
- [peer-benchmarking/output/peer_knowledge_base.csv](peer-benchmarking/output/peer_knowledge_base.csv)

These are audit artifacts, not valid production explanations yet.

---

## 6) Fusion Validation

The fusion layer is successfully loading the peer benchmark output and registering it in the model registry.

Validated in the current code state:
- Isolation Forest: loaded
- OCSVM: loaded
- Autoencoder: loaded
- Claim Autoencoder: loaded
- CatBoost: loaded
- Peer Benchmarking: loaded

The fusion weights currently in [risk_fusion/fusion_engine.py](risk_fusion/fusion_engine.py) are:

- Isolation Forest: 0.22
- OCSVM: 0.18
- CatBoost: 0.22
- Autoencoder: 0.18
- Claim Autoencoder: 0.10
- Peer Benchmarking: 0.10

Total = 1.00

This means the peer score contributes exactly 10% to the final fusion, which is correct as configured.

### Important caveat

The fusion is mathematically correct with respect to the configured weights, but it consumes a bad peer signal because the peer module is invalid before the fusion stage.

---

## 7) Root Cause Classification

### Best fit: D + E + C

- D) Provider compared against entire dataset: Not exactly the entire dataset, but effectively against itself via singleton grouping.
- E) Risk scaling bug exists: Yes, because percentile rank is being applied inside a one-provider peer set.
- C) Peer groups are too small: Yes, effectively size 1.

### Not the primary issue

- A) Percentile normalization is wrong: The percentile calculation is not inherently wrong, but it is being applied to invalid peer groups.
- B) MAD calculation is wrong: The MAD formula is valid. The actual issue is the lack of meaningful peer variance in singleton groups.

---

## 8) Recommended Fixes

1. Restore the correct peer grouping source
   - Use `Rndrng_Prvdr_Type` as the primary grouping key.
   - Fallback to `Provider_Type`.
   - Fallback to `Specialty`.
   - If none are present, do not compare against provider-level singleton groups.

2. Add a minimum peer group threshold
   - Example: if a group has fewer than 5 providers, either combine with a broader peer bucket or mark as `Not_Enough_Peer_Data`.

3. Guard the percentile assignment
   - Do not compute `rank(pct)` when peer group size is 1.
   - Set peer risk to 0 or `NaN` and mark as insufficient data.

4. Apply robust z-score only for valid groups
   - When `MAD == 0`, skip the anomaly score and flag the provider as having no peer variance.

5. Keep the fusion weight intact after the peer score is fixed
   - The 10% peer weight is valid once valid peer groups exist.

---

## Final Conclusion

The peer benchmarking module is currently invalid because it is operating on a dataset without the expected peer-group columns and therefore collapses to singleton-provider groups. The resulting 100% peer score is an artifact of that grouping issue, not a real anomaly signal.

At this point, the correct action is to repair the grouping logic and peer-score calculation before re-running the fusion layer.

The code has not been changed in this audit phase; this report captures the verified root cause and the required repair path.
