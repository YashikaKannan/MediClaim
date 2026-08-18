import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw")

files = [
    "Train-1542865627584.csv",
    "Train_Beneficiarydata-1542865627584.csv",
    "Train_Inpatientdata-1542865627584.csv",
    "Train_Outpatientdata-1542865627584.csv",
    "Test-1542969243754.csv",
    "Test_Beneficiarydata-1542969243754.csv",
    "Test_Inpatientdata-1542969243754.csv",
    "Test_Outpatientdata-1542969243754.csv",
]

for file in files:
    path = DATA_DIR / file

    print("\n" + "=" * 80)
    print(file)
    print("=" * 80)

    df = pd.read_csv(path)

    print("Rows    :", df.shape[0])
    print("Columns :", df.shape[1])

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nMissing values:")
    print(df.isnull().sum()[df.isnull().sum() > 0])

    print("\nFirst 3 rows:")
    print(df.head(3))