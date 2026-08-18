import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

df = pd.read_csv("processed/validation_results.csv")

y_true = df["Actual"]
y_pred = df["Predicted"]

print("Precision:", precision_score(y_true, y_pred))
print("Recall:", recall_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred))