import pandas as pd
from pathlib import Path
from sklearn.ensemble import IsolationForest
import joblib

# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

INPUT_FILE = Path("processed/model_features_final.csv")
MODEL_FILE = Path("processed/isolation_forest.joblib")
SCORES_FILE = Path("processed/anomaly_scores.csv")

# --------------------------------------------------
# 2. Load final features
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

X = df.drop(columns=["Provider"])

print("=" * 70)
print("ISOLATION FOREST TRAINING")
print("=" * 70)

print("\nInput shape:", X.shape)
print("Providers:", len(df))
print("Features:", X.shape[1])

# --------------------------------------------------
# 3. Create Isolation Forest
# --------------------------------------------------

model = IsolationForest(
    n_estimators=300,
    max_samples="auto",
    contamination="auto",
    random_state=42,
    n_jobs=-1
)

# --------------------------------------------------
# 4. Train
# --------------------------------------------------

print("\nTraining Isolation Forest...")

model.fit(X)

print("Training completed.")

# --------------------------------------------------
# 5. Generate anomaly scores
# --------------------------------------------------

# sklearn's score_samples:
# higher = more normal
# lower  = more anomalous

raw_score = model.score_samples(X)

# Convert so:
# higher = more anomalous

anomaly_score = -raw_score

# Model prediction:
#  1  = normal
# -1  = anomaly

prediction = model.predict(X)

# Convert to easier interpretation:
# 0 = normal
# 1 = anomaly

anomaly_flag = (prediction == -1).astype(int)

# --------------------------------------------------
# 6. Create result dataframe
# --------------------------------------------------

results = pd.DataFrame({
    "Provider": df["Provider"],
    "Anomaly_Score": anomaly_score,
    "Anomaly_Flag": anomaly_flag
})

# Highest anomaly first
results = results.sort_values(
    "Anomaly_Score",
    ascending=False
)

# --------------------------------------------------
# 7. Save results
# --------------------------------------------------

results.to_csv(
    SCORES_FILE,
    index=False
)

joblib.dump(
    model,
    MODEL_FILE
)

# --------------------------------------------------
# 8. Summary
# --------------------------------------------------

print("\n" + "=" * 70)
print("ISOLATION FOREST TRAINING COMPLETE")
print("=" * 70)

print("\nModel:")
print(model)

print("\nNormal providers:",
      (anomaly_flag == 0).sum())

print("Anomalous providers:",
      (anomaly_flag == 1).sum())

print(
    "Anomaly percentage:",
    round(anomaly_flag.mean() * 100, 2),
    "%"
)

print("\nAnomaly score statistics:")
print(results["Anomaly_Score"].describe())

print("\nTop 20 most anomalous providers:")
print(results.head(20))

print("\nSaved model:")
print(MODEL_FILE)

print("\nSaved scores:")
print(SCORES_FILE)