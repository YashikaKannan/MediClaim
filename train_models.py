import pandas as pd
import numpy as np
import os
import sqlite3
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import precision_score, recall_score, f1_score, precision_recall_curve, auc
import joblib
from catboost import CatBoostClassifier

# Make directories
os.makedirs("e:/CTS - MediClaim/backend/app/ml_models", exist_ok=True)
os.makedirs("e:/CTS - MediClaim/backend/app/db", exist_ok=True)

# 1. PyTorch Autoencoder definition
class AutoencoderModel(nn.Module):
    def __init__(self, input_dim):
        super(AutoencoderModel, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, input_dim)
        )
        
    def forward(self, x):
        return self.decoder(self.encoder(x))

def train_autoencoder(X_scaled, epochs=30, batch_size=64):
    input_dim = X_scaled.shape[1]
    model = AutoencoderModel(input_dim)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # Convert data to tensor
    dataset = torch.tensor(X_scaled, dtype=torch.float32)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = criterion(outputs, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(X_scaled):.6f}")
        
    return model

def main():
    print("Loading engineered features...")
    df = pd.read_csv("e:/CTS - MediClaim/datas/engineered_train_features.csv")
    
    providers = df["Provider"].values
    y = df["PotentialFraud"].values
    
    # Select features (drop index columns and target)
    feature_cols = [c for c in df.columns if c not in ["Provider", "PotentialFraud", "primary_state"]]
    X = df[feature_cols].copy()
    
    # Impute any missing values with median
    for col in X.columns:
        if X[col].isnull().any():
            X[col] = X[col].fillna(X[col].median())
            
    print(f"Features count: {X.shape[1]}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save the scaler
    joblib.dump(scaler, "e:/CTS - MediClaim/backend/app/ml_models/scaler.joblib")
    
    # ----------------------------------------------------
    # Model 1: Isolation Forest
    # ----------------------------------------------------
    print("Training Isolation Forest...")
    iforest = IsolationForest(n_estimators=150, contamination=0.1, random_state=42)
    iforest.fit(X_scaled)
    joblib.dump(iforest, "e:/CTS - MediClaim/backend/app/ml_models/iforest.joblib")
    
    # Isolation Forest score: higher decision function means less anomalous.
    # We want a 0-100 score where higher means MORE anomalous.
    if_decision = iforest.decision_function(X_scaled)
    # decision_function ranges from ~ -0.5 to 0.5
    if_scores = np.clip((0.2 - if_decision) / 0.5 * 100, 0, 100)
    
    # ----------------------------------------------------
    # Model 2: Local Outlier Factor (LOF)
    # ----------------------------------------------------
    print("Training Local Outlier Factor...")
    # Using novelty=True so we can run predict/score on new samples
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, novelty=True)
    lof.fit(X_scaled)
    joblib.dump(lof, "e:/CTS - MediClaim/backend/app/ml_models/lof.joblib")
    
    lof_decision = lof.score_samples(X_scaled)  # Negative outlier factor
    lof_scores = np.clip((-lof_decision - 1.0) * 100, 0, 100)
    
    # ----------------------------------------------------
    # Model 3: One-Class SVM
    # ----------------------------------------------------
    print("Training One-Class SVM...")
    ocsvm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
    ocsvm.fit(X_scaled)
    joblib.dump(ocsvm, "e:/CTS - MediClaim/backend/app/ml_models/ocsvm.joblib")
    
    ocsvm_decision = ocsvm.decision_function(X_scaled)
    ocsvm_scores = np.clip((-ocsvm_decision) * 100, 0, 100)
    
    # ----------------------------------------------------
    # Model 4: PyTorch Autoencoder
    # ----------------------------------------------------
    print("Training PyTorch Autoencoder...")
    # Train only on non-fraud cases for pure reconstruction logic
    X_normal = X_scaled[y == 0]
    ae_model = train_autoencoder(X_normal, epochs=40, batch_size=64)
    torch.save(ae_model.state_dict(), "e:/CTS - MediClaim/backend/app/ml_models/autoencoder.pth")
    
    # Compute reconstruction error
    ae_model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        X_reconstructed = ae_model(X_tensor)
        ae_errors = torch.mean((X_tensor - X_reconstructed)**2, dim=1).numpy()
    
    # Scale reconstruction error: Map 99th percentile of error to 100
    ae_threshold = np.percentile(ae_errors, 99.5)
    ae_scores = np.clip((ae_errors / ae_threshold) * 100, 0, 100)
    
    # Save the reconstruction threshold for inference
    joblib.dump(ae_threshold, "e:/CTS - MediClaim/backend/app/ml_models/ae_threshold.joblib")
    
    # ----------------------------------------------------
    # Model 5: CatBoost (Supervised)
    # ----------------------------------------------------
    print("Training CatBoostClassifier...")
    cat_model = CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        eval_metric='Logloss',
        random_seed=42,
        verbose=0
    )
    cat_model.fit(X_scaled, y)
    cat_model.save_model("e:/CTS - MediClaim/backend/app/ml_models/catboost.cbm")
    
    cat_probs = cat_model.predict_proba(X_scaled)[:, 1]
    cat_scores = cat_probs * 100
    
    # ----------------------------------------------------
    # Peer Benchmarking Engine
    # ----------------------------------------------------
    print("Computing Peer Benchmarking...")
    # Let's group providers by type: Inpatient-heavy (inpatient_ratio >= 0.2) or Outpatient-heavy (inpatient_ratio < 0.2)
    # and State
    df["provider_type"] = df["inpatient_ratio"].apply(lambda r: "Inpatient-heavy" if r >= 0.2 else "Outpatient-heavy")
    df["peer_group"] = df["provider_type"] + "_" + df["primary_state"].astype(str)
    
    # Calculate peer group medians for: total_reimbursement, total_claims, total_beneficiaries
    peer_medians = df.groupby("peer_group")[["total_reimbursement", "total_claims", "total_beneficiaries"]].median()
    peer_medians.columns = ["peer_median_reimbursement", "peer_median_claims", "peer_median_beneficiaries"]
    
    # Save peer medians for inference
    peer_medians.to_csv("e:/CTS - MediClaim/backend/app/ml_models/peer_medians.csv")
    
    df = df.join(peer_medians, on="peer_group")
    
    # Compute Ratios
    # Handle division by zero
    df["reimbursement_ratio"] = df["total_reimbursement"] / df["peer_median_reimbursement"].replace(0, 1)
    df["claims_ratio"] = df["total_claims"] / df["peer_median_claims"].replace(0, 1)
    df["beneficiary_ratio"] = df["total_beneficiaries"] / df["peer_median_beneficiaries"].replace(0, 1)
    
    # Scale peer ratios: A ratio of 1.0 is normal, 4.0+ is highly abnormal
    # Scale so that 1.0 is 0 risk, 4.0 or above is 100 risk
    df["reimbursement_ratio_score"] = np.clip((df["reimbursement_ratio"] - 1.0) / 3.0 * 100, 0, 100)
    df["claims_ratio_score"] = np.clip((df["claims_ratio"] - 1.0) / 3.0 * 100, 0, 100)
    df["beneficiary_ratio_score"] = np.clip((df["beneficiary_ratio"] - 1.0) / 3.0 * 100, 0, 100)
    
    peer_benchmark_score = (df["reimbursement_ratio_score"] + df["claims_ratio_score"] + df["beneficiary_ratio_score"]) / 3.0
    
    # ----------------------------------------------------
    # Robust Z-Score Engine (Median & MAD)
    # ----------------------------------------------------
    print("Computing Robust Z-Scores...")
    # Calculate global medians and MADs
    z_cols = ["total_reimbursement", "total_claims", "claims_per_beneficiary"]
    medians = {}
    mads = {}
    
    for col in z_cols:
        med = df[col].median()
        mad = np.median(np.abs(df[col] - med)) * 1.4826
        if mad == 0:
            mad = 1e-5
        medians[col] = med
        mads[col] = mad
        
        z_col_name = f"robust_z_{col}"
        df[z_col_name] = (df[col] - med) / mad
        
    # Save medians and MADs
    joblib.dump({"medians": medians, "mads": mads}, "e:/CTS - MediClaim/backend/app/ml_models/robust_z_params.joblib")
    
    # Calculate a composite robust Z-score scaled to 0-100
    # Z-scores > 3 are unusual. Scale Z-scores between 0 and 5 to 0-100.
    df["robust_z_reimbursement_score"] = np.clip(df["robust_z_total_reimbursement"] / 5.0 * 100, 0, 100)
    df["robust_z_claims_score"] = np.clip(df["robust_z_total_claims"] / 5.0 * 100, 0, 100)
    df["robust_z_cpb_score"] = np.clip(df["robust_z_claims_per_beneficiary"] / 5.0 * 100, 0, 100)
    
    statistical_score = (df["robust_z_reimbursement_score"] + df["robust_z_claims_score"] + df["robust_z_cpb_score"]) / 3.0
    
    # ----------------------------------------------------
    # Percentile Analysis
    # ----------------------------------------------------
    print("Computing Percentiles...")
    df["reimbursement_percentile"] = df["total_reimbursement"].rank(pct=True) * 100
    df["claims_percentile"] = df["total_claims"].rank(pct=True) * 100
    df["beneficiary_percentile"] = df["total_beneficiaries"].rank(pct=True) * 100
    
    # ----------------------------------------------------
    # Risk Score Fusion Engine
    # ----------------------------------------------------
    print("Combining signals into Final Risk Score...")
    # Final Risk Score = 40% ML + 30% Statistical + 30% Peer
    # ML Signals composite (weighted blend)
    ml_score = (0.25 * if_scores + 0.20 * ae_scores + 0.15 * lof_scores + 0.10 * ocsvm_scores + 0.30 * cat_scores)
    
    final_risk_score = 0.40 * ml_score + 0.30 * statistical_score + 0.30 * peer_benchmark_score
    
    df["ml_score"] = ml_score
    df["statistical_score"] = statistical_score
    df["peer_score"] = peer_benchmark_score
    df["risk_score"] = final_risk_score
    
    # Risk Level classification
    # 0-30 Low, 31-60 Medium, 61-80 High, 81-100 Critical
    def classify_risk(score):
        if score <= 30:
            return "Low"
        elif score <= 60:
            return "Medium"
        elif score <= 80:
            return "High"
        else:
            return "Critical"
            
    df["risk_level"] = df["risk_score"].apply(classify_risk)
    
    # ----------------------------------------------------
    # Evaluate Models against labels
    # ----------------------------------------------------
    print("\n==============================================")
    print("Model Evaluation:")
    print("==============================================")
    
    # Define a threshold to evaluate unsupervised layers (e.g. top 10% highest scores as positive)
    def evaluate_model(scores, y_true, name):
        thresh = np.percentile(scores, 90.65) # Top 9.35% (contamination matching target)
        preds = (scores >= thresh).astype(int)
        prec = precision_score(y_true, preds)
        rec = recall_score(y_true, preds)
        f1 = f1_score(y_true, preds)
        
        # PR AUC
        precisions, recalls, _ = precision_recall_curve(y_true, scores / 100.0)
        pr_auc = auc(recalls, precisions)
        
        # Precision@100
        top_100_indices = np.argsort(scores)[::-1][:100]
        prec_at_100 = y_true[top_100_indices].mean()
        
        print(f"| {name:<22} | {prec:.4f} | {rec:.4f} | {f1:.4f} | {pr_auc:.4f} | {prec_at_100:.2f} |")
        return prec, rec, f1, pr_auc, prec_at_100
        
    print("| Model                  | Precision | Recall | F1-Score | PR-AUC | Precision@100 |")
    print("|------------------------|-----------|--------|----------|--------|---------------|")
    evaluate_model(if_scores, y, "Isolation Forest")
    evaluate_model(ae_scores, y, "Autoencoder")
    evaluate_model(lof_scores, y, "LOF")
    evaluate_model(ocsvm_scores, y, "One-Class SVM")
    evaluate_model(cat_scores, y, "CatBoost")
    evaluate_model(final_risk_score, y, "Combined Risk Engine")
    
    # Save the dataframe of results
    df.to_csv("e:/CTS - MediClaim/datas/evaluated_train_providers.csv", index=False)
    print("\nSaved evaluated results to CSV.")
    
    # ----------------------------------------------------
    # Populate SQLite database
    # ----------------------------------------------------
    print("Populating SQLite database...")
    db_path = "e:/CTS - MediClaim/backend/app/db/mediclaim.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Drop existing tables
    cursor.execute("DROP TABLE IF EXISTS providers")
    cursor.execute("DROP TABLE IF EXISTS model_scores")
    cursor.execute("DROP TABLE IF EXISTS peer_benchmarks")
    cursor.execute("DROP TABLE IF EXISTS explanations")
    cursor.execute("DROP TABLE IF EXISTS investigations")
    
    # Create tables
    cursor.execute("""
    CREATE TABLE providers (
        provider_id TEXT PRIMARY KEY,
        total_claims INTEGER,
        inpatient_claims INTEGER,
        outpatient_claims INTEGER,
        inpatient_ratio REAL,
        total_beneficiaries INTEGER,
        total_reimbursement REAL,
        mean_reimbursement REAL,
        risk_score REAL,
        risk_level TEXT,
        primary_state INTEGER,
        provider_type TEXT,
        PotentialFraud INTEGER
    )
    """)
    
    cursor.execute("""
    CREATE TABLE model_scores (
        provider_id TEXT PRIMARY KEY,
        isolation_score REAL,
        autoencoder_score REAL,
        lof_score REAL,
        ocsvm_score REAL,
        catboost_score REAL,
        ml_score REAL,
        statistical_score REAL,
        peer_score REAL,
        FOREIGN KEY(provider_id) REFERENCES providers(provider_id)
    )
    """)
    
    cursor.execute("""
    CREATE TABLE peer_benchmarks (
        provider_id TEXT PRIMARY KEY,
        reimbursement_ratio REAL,
        claims_ratio REAL,
        beneficiary_ratio REAL,
        reimbursement_percentile REAL,
        claims_percentile REAL,
        beneficiary_percentile REAL,
        FOREIGN KEY(provider_id) REFERENCES providers(provider_id)
    )
    """)
    
    cursor.execute("""
    CREATE TABLE explanations (
        provider_id TEXT,
        reason TEXT,
        FOREIGN KEY(provider_id) REFERENCES providers(provider_id)
    )
    """)
    
    cursor.execute("""
    CREATE TABLE investigations (
        provider_id TEXT PRIMARY KEY,
        status TEXT,
        notes TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(provider_id) REFERENCES providers(provider_id)
    )
    """)
    
    # Insert data
    print("Inserting data into database tables...")
    for idx, row in df.iterrows():
        p_id = row["Provider"]
        
        # 1. providers table
        cursor.execute("""
        INSERT INTO providers (
            provider_id, total_claims, inpatient_claims, outpatient_claims, inpatient_ratio,
            total_beneficiaries, total_reimbursement, mean_reimbursement, risk_score,
            risk_level, primary_state, provider_type, PotentialFraud
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            p_id, int(row["total_claims"]), int(row["inpatient_claims"]), int(row["outpatient_claims"]),
            float(row["inpatient_ratio"]), int(row["total_beneficiaries"]), float(row["total_reimbursement"]),
            float(row["mean_reimbursement"]), float(row["risk_score"]), row["risk_level"],
            int(row["primary_state"]), row["provider_type"], int(row["PotentialFraud"])
        ))
        
        # 2. model_scores table
        cursor.execute("""
        INSERT INTO model_scores (
            provider_id, isolation_score, autoencoder_score, lof_score, ocsvm_score,
            catboost_score, ml_score, statistical_score, peer_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            p_id, float(if_scores[idx]), float(ae_scores[idx]), float(lof_scores[idx]),
            float(ocsvm_scores[idx]), float(cat_scores[idx]), float(row["ml_score"]),
            float(row["statistical_score"]), float(row["peer_score"])
        ))
        
        # 3. peer_benchmarks table
        cursor.execute("""
        INSERT INTO peer_benchmarks (
            provider_id, reimbursement_ratio, claims_ratio, beneficiary_ratio,
            reimbursement_percentile, claims_percentile, beneficiary_percentile
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            p_id, float(row["reimbursement_ratio"]), float(row["claims_ratio"]), float(row["beneficiary_ratio"]),
            float(row["reimbursement_percentile"]), float(row["claims_percentile"]), float(row["beneficiary_percentile"])
        ))
        
        # 4. Generate & Insert Explanations
        reasons = []
        # Statistical Reasons
        if row["reimbursement_ratio"] > 3.0:
            reasons.append(f"Reimbursement volume is {row['reimbursement_ratio']:.1f}x the peer median for their specialty type in state {int(row['primary_state'])}.")
        if row["claims_ratio"] > 3.0:
            reasons.append(f"Claims volume is {row['claims_ratio']:.1f}x the peer median for their specialty type in state {int(row['primary_state'])}.")
        if row["reimbursement_percentile"] > 95:
            reasons.append(f"Provider's total reimbursement is in the {row['reimbursement_percentile']:.1f}th percentile nationally.")
        if row["claims_percentile"] > 95:
            reasons.append(f"Provider's total claims are in the {row['claims_percentile']:.1f}th percentile nationally.")
        if row["patient_death_rate"] > 0.05:
            reasons.append(f"Billed claims for deceased beneficiaries make up {row['patient_death_rate']*100:.1f}% of total claims (extremely high risk).")
        if row["mean_reimbursement"] > df["mean_reimbursement"].median() * 3:
            reasons.append(f"Average reimbursement per claim (${row['mean_reimbursement']:.2f}) is over 3x the national median (${df['mean_reimbursement'].median():.2f}).")
            
        # Model Agreement
        model_flags = 0
        if if_scores[idx] > 70: model_flags += 1
        if ae_scores[idx] > 70: model_flags += 1
        if lof_scores[idx] > 70: model_flags += 1
        if ocsvm_scores[idx] > 70: model_flags += 1
        
        if model_flags >= 3:
            reasons.append(f"High agreement across multiple unsupervised anomaly detection models ({model_flags} models flagging provider behavior).")
        if cat_scores[idx] > 80:
            reasons.append(f"Supervised ML classifier flags behavior patterns aligning with known past fraudulent providers (Confidence: {cat_scores[idx]:.1f}%).")
            
        if len(reasons) == 0:
            reasons.append("Overall metrics align with normal provider benchmarks. Minor deviations observed.")
            
        for reason in reasons:
            cursor.execute("INSERT INTO explanations (provider_id, reason) VALUES (?, ?)", (p_id, reason))
            
        # 5. Default investigations table (unassigned)
        cursor.execute("INSERT INTO investigations (provider_id, status, notes) VALUES (?, ?, ?)", (p_id, "New", ""))
        
    conn.commit()
    conn.close()
    print("Database populate finished successfully.")

if __name__ == "__main__":
    main()
