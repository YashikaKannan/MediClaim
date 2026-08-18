import pandas as pd

df = pd.read_csv("processed/PROVIDER_ML_READY_KAGGLE.csv")

print(df.shape)
print(df.describe())

corr = df.corr(numeric_only=True)