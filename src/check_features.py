import pandas as pd
from pathlib import Path

DATA_FILE = Path("processed/provider_features.csv")

df = pd.read_csv(DATA_FILE)

print("=" * 70)
print("PROVIDER FEATURE QUALITY CHECK")
print("=" * 70)

# --------------------------------------------------
# 1. Basic information
# --------------------------------------------------

print("\nShape:")
print(df.shape)

print("\nDuplicate provider IDs:")
print(df["Provider"].duplicated().sum())

# --------------------------------------------------
# 2. Data types
# --------------------------------------------------

print("\nData types:")
print(df.dtypes)

# --------------------------------------------------
# 3. Constant / zero-variance features
# --------------------------------------------------

numeric_df = df.drop(columns=["Provider"])

constant_features = [
    col for col in numeric_df.columns
    if numeric_df[col].nunique() <= 1
]

print("\nConstant features:")
print(constant_features)

# --------------------------------------------------
# 4. Summary statistics
# --------------------------------------------------

print("\nSummary statistics:")
print(numeric_df.describe().T)

# --------------------------------------------------
# 5. Skewness
# --------------------------------------------------

print("\nSkewness:")
print(
    numeric_df.skew()
    .sort_values(ascending=False)
)

# --------------------------------------------------
# 6. Correlation
# --------------------------------------------------

print("\nHighly correlated feature pairs (> 0.90):")

corr = numeric_df.corr()

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

# --------------------------------------------------
# 7. Negative values
# --------------------------------------------------

negative_counts = (numeric_df < 0).sum()

print("\nFeatures containing negative values:")

for col, count in negative_counts.items():
    if count > 0:
        print(col, ":", count)

# --------------------------------------------------
# 8. Infinite values
# --------------------------------------------------

print("\nInfinite values:")

print(
    numeric_df.isin([float("inf"), float("-inf")])
    .sum()
    .sum()
)

print("\nFeature quality check completed.")