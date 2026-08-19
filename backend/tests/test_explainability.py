import sqlite3

import app.main as main_module


def build_mock_db():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE providers (
            provider_id TEXT,
            provider_type TEXT,
            primary_state INTEGER,
            total_claims REAL,
            inpatient_claims REAL,
            outpatient_claims REAL,
            inpatient_ratio REAL,
            total_reimbursement REAL,
            total_beneficiaries REAL,
            mean_reimbursement REAL,
            risk_score REAL,
            risk_level TEXT,
            investigation_status TEXT,
            investigation_notes TEXT,
            assigned_investigator TEXT,
            PotentialFraud INTEGER
        );
        CREATE TABLE model_scores (
            provider_id TEXT,
            catboost_score REAL,
            isolation_score REAL,
            autoencoder_score REAL,
            lof_score REAL,
            ocsvm_score REAL,
            ml_score REAL,
            statistical_score REAL,
            peer_score REAL,
            leie_score REAL
        );
        CREATE TABLE peer_benchmarks (
            provider_id TEXT,
            reimbursement_ratio REAL,
            claims_ratio REAL,
            beneficiary_ratio REAL,
            reimbursement_percentile REAL,
            claims_percentile REAL,
            beneficiary_percentile REAL
        );
        CREATE TABLE explanations (
            provider_id TEXT,
            reason TEXT
        );
        CREATE TABLE provider_explanations (
            provider_id TEXT,
            explanation_json TEXT
        );
        CREATE TABLE provider_drift (
            provider_id TEXT,
            drift_score REAL,
            drift_level TEXT,
            claims_spike_ratio REAL,
            reimbursement_spike_ratio REAL,
            coding_shift_index REAL,
            historical_monthly_data TEXT
        );
        CREATE TABLE investigations (
            provider_id TEXT,
            status TEXT,
            notes TEXT,
            assigned_investigator TEXT,
            updated_at TEXT
        );
        """
    )
    conn.execute(
        """
        INSERT INTO providers (
            provider_id, provider_type, primary_state, total_claims, inpatient_claims,
            outpatient_claims, inpatient_ratio, total_reimbursement, total_beneficiaries,
            mean_reimbursement, risk_score, risk_level, investigation_status,
            investigation_notes, assigned_investigator, PotentialFraud
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "PRV-101",
            "Cardiology",
            12,
            240,
            96,
            144,
            0.4,
            820000,
            180,
            3416.67,
            88.0,
            "Critical",
            "New",
            "Escalated review",
            "Audit team",
            1,
        ),
    )
    conn.execute(
        """
        INSERT INTO model_scores (
            provider_id, catboost_score, isolation_score, autoencoder_score,
            lof_score, ocsvm_score, ml_score, statistical_score, peer_score, leie_score
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "PRV-101",
            92.0,
            81.0,
            74.0,
            79.0,
            76.0,
            80.0,
            85.0,
            88.0,
            0.0,
        ),
    )
    conn.execute(
        """
        INSERT INTO peer_benchmarks (
            provider_id, reimbursement_ratio, claims_ratio, beneficiary_ratio,
            reimbursement_percentile, claims_percentile, beneficiary_percentile
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "PRV-101",
            4.3,
            2.8,
            2.1,
            98.5,
            96.2,
            92.0,
        ),
    )
    conn.execute(
        "INSERT INTO explanations (provider_id, reason) VALUES (?, ?)",
        ("PRV-101", "Reimbursement is 4.3x the median for comparable providers."),
    )
    conn.execute(
        "INSERT INTO provider_drift (provider_id, drift_score, drift_level, claims_spike_ratio, reimbursement_spike_ratio, coding_shift_index, historical_monthly_data) VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("PRV-101", 76.0, "High", 1.9, 2.6, 1.4, "[]"),
    )
    return conn


def test_grounded_copilot_rewrites_technical_jargon(monkeypatch):
    conn = build_mock_db()
    monkeypatch.setattr(main_module, "get_db_connection", lambda: conn)

    result = main_module.query_ai_assistant(main_module.AIQueryRequest(provider_id="PRV-101", query="Why was this provider flagged?"))

    response = result["response"]
    assert "Isolation Forest" not in response
    assert "LOF" not in response
    assert "Autoencoder" not in response
    assert "feature vector" not in response.lower()
    assert "model threshold" not in response.lower()
    assert "Reimbursement" in response or "billing" in response.lower()
    assert "Why" in response or "flagged" in response.lower()


def test_service_explanations_are_business_friendly():
    explanation = main_module.build_provider_explanation(
        provider={
            "provider_id": "PRV-101",
            "risk_score": 88,
            "risk_level": "Critical",
            "total_claims": 240,
            "total_beneficiaries": 180,
            "total_reimbursement": 820000,
            "mean_reimbursement": 3416.67,
        },
        scores={
            "isolation_score": 81,
            "lof_score": 79,
            "ocsvm_score": 76,
            "autoencoder_score": 74,
            "catboost_score": 92,
            "statistical_score": 85,
            "peer_score": 88,
            "leie_score": 0,
        },
        benchmarks={
            "reimbursement_ratio": 4.3,
            "claims_ratio": 2.8,
            "reimbursement_percentile": 98.5,
            "claims_percentile": 96.2,
        },
        drift={
            "drift_score": 76,
            "drift_level": "High",
            "claims_spike_ratio": 1.9,
            "reimbursement_spike_ratio": 2.6,
        },
        reasons=["Reimbursement is 4.3x the median for comparable providers.", "Claim volume is 2.8x the median for comparable providers."],
    )

    joined = "\n".join(explanation["why_flagged"] + explanation["peer_comparison"] + explanation["financial_impact"] + [explanation["recommended_action"]])
    assert "Critical" in explanation["risk_category"] or explanation["risk_category"] == "Critical"
    assert explanation["priority"] in {"P1", "P2", "P3", "P4"}
    assert "Isolation Forest" not in joined
    assert "LOF" not in joined
    assert "Autoencoder" not in joined
    assert "4.3x" in joined or "2.8x" in joined
