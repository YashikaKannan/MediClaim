import pandas as pd

f1 = pd.read_csv("outputs/suspicious_claims_1pct.csv")
f3 = pd.read_csv("outputs/suspicious_claims_3pct.csv")
f5 = pd.read_csv("outputs/suspicious_claims_5pct.csv")

print(len(f1))
print(len(f3))
print(len(f5))