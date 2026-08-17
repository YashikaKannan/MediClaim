import os
import json
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "processed",
    "autoencoder_claim_dataset.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "InscClaimAmtReimbursed",
    "DeductibleAmtPaid",
    "ClaimDuration",
    "DiagnosisCount",
    "ProcedureCount",
    "PhysicianCount",
    "HospitalStayDays",
    "IPAnnualReimbursementAmt",
    "IPAnnualDeductibleAmt",
    "OPAnnualReimbursementAmt",
    "OPAnnualDeductibleAmt",
    "ChronicConditionCount"
]


print("=" * 70)
print("AUTOENCODER MODEL TRAINING")
print("=" * 70)


# ============================================================
# 1. LOAD PROCESSED DATA
# ============================================================

print("\n[1] Loading processed claim dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("Total claims:", len(df))


# ============================================================
# 2. PREPARE FEATURES
# ============================================================

print("\n[2] Preparing Autoencoder features...")

X = df[FEATURES].copy()

X = X.replace([np.inf, -np.inf], 0)
X = X.fillna(0)

print("Number of features:", len(FEATURES))
print("Missing values:", int(X.isnull().sum().sum()))


# ============================================================
# 3. LOG TRANSFORMATION
# ============================================================

print("\n[3] Applying log transformation...")

X_log = np.log1p(X)

print("[OK] Log transformation completed")


# ============================================================
# 4. TRAIN / VALIDATION SPLIT
# ============================================================

print("\n[4] Creating train/validation split...")

X_train, X_val = train_test_split(
    X_log,
    test_size=0.10,
    random_state=RANDOM_STATE
)

print("Training claims:", len(X_train))
print("Validation claims:", len(X_val))


# ============================================================
# 5. ROBUST SCALING
# ============================================================

print("\n[5] Scaling features...")

scaler = RobustScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

joblib.dump(
    scaler,
    os.path.join(MODEL_DIR, "robust_scaler.pkl")
)

print("[OK] RobustScaler saved")


# ============================================================
# 6. BUILD AUTOENCODER
# ============================================================

print("\n[6] Building Autoencoder...")

input_dim = X_train_scaled.shape[1]

input_layer = Input(
    shape=(input_dim,),
    name="claim_features"
)

encoded = Dense(
    16,
    activation="relu"
)(input_layer)

encoded = Dense(
    8,
    activation="relu"
)(encoded)

bottleneck = Dense(
    4,
    activation="relu",
    name="bottleneck"
)(encoded)

decoded = Dense(
    8,
    activation="relu"
)(bottleneck)

decoded = Dense(
    16,
    activation="relu"
)(decoded)

output_layer = Dense(
    input_dim,
    activation="linear"
)(decoded)

autoencoder = Model(
    inputs=input_layer,
    outputs=output_layer
)

autoencoder.compile(
    optimizer="adam",
    loss="mse"
)

autoencoder.summary()


# ============================================================
# 7. EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# ============================================================
# 8. TRAIN MODEL
# ============================================================

print("\n[7] Training Autoencoder...")

history = autoencoder.fit(
    X_train_scaled,
    X_train_scaled,

    validation_data=(
        X_val_scaled,
        X_val_scaled
    ),

    epochs=50,
    batch_size=512,
    shuffle=True,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


# ============================================================
# 9. SAVE MODEL
# ============================================================

model_path = os.path.join(
    MODEL_DIR,
    "claim_autoencoder.keras"
)

autoencoder.save(model_path)


# ============================================================
# 10. SAVE FEATURE INFORMATION
# ============================================================

feature_path = os.path.join(
    MODEL_DIR,
    "features.json"
)

with open(feature_path, "w") as f:
    json.dump(FEATURES, f, indent=4)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("[SUCCESS] AUTOENCODER TRAINING COMPLETE")
print("=" * 70)

print("\nModel saved:")
print(model_path)

print("\nScaler saved:")
print(
    os.path.join(
        MODEL_DIR,
        "robust_scaler.pkl"
    )
)

print("\nTraining epochs completed:")
print(len(history.history["loss"]))

print("\nFinal training loss:")
print(history.history["loss"][-1])

print("\nFinal validation loss:")
print(history.history["val_loss"][-1])