import pandas as pd
import numpy as np

FILE = "processed/model_features.csv"

df = pd.read_csv(FILE)

X = df.drop(columns=["Provider"])

print("=" * 70)
print("MODEL FEATURE VALIDATION")
print("=" * 70)

print("\nShape:")
print(df.shape)

print("\nSkewness after log transformation:")
print(
    X.skew()
    .sort_values(ascending=False)
)

print("\nHighly correlated features (> 0.90):")

corr = X.corr()

found = False

for i in range(len(corr.columns)):
    for j in range(i + 1, len(corr.columns)):
        value = corr.iloc[i, j]

        if abs(value) > 0.90:
            print(
                f"{corr.columns[i]} <-> "
                f"{corr.columns[j]} : {value:.3f}"
            )
            found = True

if not found:
    print("None")

print("\nSummary statistics:")
print(X.describe().T)

print("\nMissing values:")
print(X.isnull().sum().sum())

print("\nInfinite values:")
print(np.isinf(X).sum().sum())

print("\nModel feature validation completed.")