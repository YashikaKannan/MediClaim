import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import RobustScaler

from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping


# -----------------------------------
# Dataset
# -----------------------------------

DATA_PATH = "CLAIM_LEVEL_AUTOENCODER_DATA.csv"


# -----------------------------------
# Claim-level features
# -----------------------------------

features = [
    "CLM_PMT_AMT",
    "NCH_CARR_CLM_SBMTD_CHRG_AMT",
    "NCH_CARR_CLM_ALOWD_AMT",
    "LINE_SBMTD_CHRG_AMT",
    "LINE_ALOWD_CHRG_AMT",
    "LINE_PRVDR_PMT_AMT",
    "LINE_SRVC_CNT",
    "DIAGNOSIS_COUNT"
]


print("Loading claim-level dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Unique claims:", df["CLM_ID"].nunique())


# -----------------------------------
# Select features
# -----------------------------------

print("\nPreparing features...")

X_df = df[features].copy()

X_df = X_df.apply(
    pd.to_numeric,
    errors="coerce"
)

X_df = X_df.replace(
    [np.inf, -np.inf],
    np.nan
)

if X_df.isna().any().any():

    print("Missing values detected.")

    X_df = X_df.fillna(
        X_df.median()
    )

else:
    print("No missing values.")


# -----------------------------------
# Safety check
# -----------------------------------

negative_count = (
    X_df < 0
).sum().sum()

print(
    "Negative values:",
    negative_count
)

if negative_count > 0:
    raise ValueError(
        "Negative values detected. "
        "log1p cannot safely continue."
    )


# -----------------------------------
# Log transformation
# -----------------------------------

print("Applying log1p transformation...")

X_log = np.log1p(X_df)


# -----------------------------------
# Robust scaling
# -----------------------------------

print("Applying RobustScaler...")

scaler = RobustScaler()

X = scaler.fit_transform(
    X_log
)

print("Final model input shape:", X.shape)

print(
    "NaN values:",
    np.isnan(X).sum()
)

print(
    "Infinite values:",
    np.isinf(X).sum()
)


# -----------------------------------
# Save final scaler
# -----------------------------------

joblib.dump(
    scaler,
    "claim_level_scaler.pkl"
)

print(
    "Scaler saved: claim_level_scaler.pkl"
)


# -----------------------------------
# Autoencoder
# -----------------------------------

print("\nBuilding Claim-Level Autoencoder...")

input_dim = X.shape[1]

inputs = Input(
    shape=(input_dim,)
)

encoded = Dense(
    16,
    activation="relu"
)(inputs)

bottleneck = Dense(
    4,
    activation="relu"
)(encoded)

decoded = Dense(
    16,
    activation="relu"
)(bottleneck)

outputs = Dense(
    input_dim,
    activation="linear"
)(decoded)

autoencoder = Model(
    inputs=inputs,
    outputs=outputs
)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

autoencoder.summary()


# -----------------------------------
# Early stopping
# -----------------------------------

early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# -----------------------------------
# Training
# -----------------------------------

print("\nTraining Claim-Level Autoencoder...")

history = autoencoder.fit(
    X,
    X,
    epochs=50,
    batch_size=256,
    validation_split=0.1,
    callbacks=[early_stop],
    verbose=1
)


# -----------------------------------
# Save model
# -----------------------------------

autoencoder.save(
    "claim_level_autoencoder.keras"
)

print(
    "\nModel saved: "
    "claim_level_autoencoder.keras"
)

print("\n================================")
print("CLAIM-LEVEL TRAINING COMPLETED!")
print("================================")