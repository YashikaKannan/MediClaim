import sqlite3
import os

DB_PATH = "e:/CTS - MediClaim/backend/app/db/mediclaim.db"

def init_database():
    print(f"Initializing database at: {DB_PATH}")
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 1. providers table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS providers (
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

    # 2. model_scores table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS model_scores (
        provider_id TEXT PRIMARY KEY,
        isolation_score REAL,
        autoencoder_score REAL,
        lof_score REAL,
        ocsvm_score REAL,
        catboost_score REAL,
        ml_score REAL,
        statistical_score REAL,
        peer_score REAL,
        leie_score REAL,
        FOREIGN KEY(provider_id) REFERENCES providers(provider_id)
    )
    """)

    # 3. peer_benchmarks table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS peer_benchmarks (
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

    # 4. explanations table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS explanations (
        provider_id TEXT,
        reason TEXT,
        FOREIGN KEY(provider_id) REFERENCES providers(provider_id)
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS provider_explanations (
        provider_id TEXT PRIMARY KEY,
        explanation_json TEXT NOT NULL,
        generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(provider_id) REFERENCES providers(provider_id)
    )
    """)

    # 5. investigations table (with assigned_investigator)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS investigations (
        provider_id TEXT PRIMARY KEY,
        status TEXT,
        notes TEXT,
        assigned_investigator TEXT DEFAULT 'Unassigned',
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(provider_id) REFERENCES providers(provider_id)
    )
    """)

    # 6. claims table (new schema)
    try:
        cursor.execute("SELECT business_interpretation FROM claims LIMIT 1")
    except sqlite3.OperationalError:
        print("Dropping old claims table for schema upgrade...")
        cursor.execute("DROP TABLE IF EXISTS claims")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS claims (
        claim_id TEXT PRIMARY KEY,
        provider_id TEXT,
        bene_id TEXT,
        risk_score REAL,
        risk_category TEXT,
        is_anomaly INTEGER,
        explanation_1 TEXT,
        explanation_2 TEXT,
        explanation_3 TEXT,
        business_interpretation TEXT,
        explanation_json TEXT,
        claim_amount REAL,
        claim_type TEXT,
        claim_date TEXT,
        FOREIGN KEY(provider_id) REFERENCES providers(provider_id)
    )
    """)

    try:
        cursor.execute("SELECT explanation_json FROM claims LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE claims ADD COLUMN explanation_json TEXT")

    # 7. provider_drift table (new)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS provider_drift (
        provider_id TEXT PRIMARY KEY,
        drift_score REAL,
        drift_level TEXT,
        claims_spike_ratio REAL,
        reimbursement_spike_ratio REAL,
        coding_shift_index REAL,
        historical_monthly_data TEXT,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY(provider_id) REFERENCES providers(provider_id)
    )
    """)

    # 8. uploads table (new)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS uploads (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        file_type TEXT,
        row_count INTEGER,
        status TEXT,
        error_message TEXT,
        uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Drop old settings if they don't have catboost_weight
    try:
        cursor.execute("SELECT catboost_weight FROM settings LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("DROP TABLE IF EXISTS settings")

    # 8. settings table (new)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        id INTEGER PRIMARY KEY,
        api_url TEXT,
        catboost_weight REAL,
        iforest_weight REAL,
        lof_weight REAL,
        robust_z_weight REAL,
        peer_benchmark_weight REAL,
        leie_weight REAL,
        z_cutoff REAL,
        high_risk_limit REAL,
        crit_risk_limit REAL
    )
    """)

    # Insert default settings if not exists
    cursor.execute("SELECT COUNT(*) FROM settings")
    if cursor.fetchone()[0] == 0:
        cursor.execute("""
        INSERT INTO settings (id, api_url, catboost_weight, iforest_weight, lof_weight, robust_z_weight, peer_benchmark_weight, leie_weight, z_cutoff, high_risk_limit, crit_risk_limit)
        VALUES (1, 'http://localhost:8000', 35.0, 20.0, 15.0, 10.0, 15.0, 5.0, 3.0, 65.0, 85.0)
        """)
        print("Default settings seeded.")

    # Check if investigations needs migrations (for existing databases)
    try:
        cursor.execute("SELECT assigned_investigator FROM investigations LIMIT 1")
    except sqlite3.OperationalError:
        print("Migrating investigations table to add assigned_investigator column...")
        cursor.execute("ALTER TABLE investigations ADD COLUMN assigned_investigator TEXT DEFAULT 'Unassigned'")

    # Check if model_scores needs migrations (for existing databases)
    try:
        cursor.execute("SELECT leie_score FROM model_scores LIMIT 1")
    except sqlite3.OperationalError:
        print("Migrating model_scores table to add leie_score column...")
        cursor.execute("ALTER TABLE model_scores ADD COLUMN leie_score REAL DEFAULT 0.0")

    conn.commit()
    conn.close()
    print("Database initialization/upgrade complete.")

if __name__ == "__main__":
    init_database()
