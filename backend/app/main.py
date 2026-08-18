from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import os
from typing import List, Optional

app = FastAPI(title="MediClaim Risk Intelligence Backend API", version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "e:/CTS - MediClaim/backend/app/db/mediclaim.db"

def get_db_connection():
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Database file not found. Run training first.")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

# Pydantic models
class StatusUpdateRequest(BaseModel):
    status: str

class NotesUpdateRequest(BaseModel):
    notes: str

class AIQueryRequest(BaseModel):
    provider_id: str
    query: str

@app.get("/")
def read_root():
    return {"message": "Welcome to MediClaim API. Use /docs for documentation."}

@app.get("/api/v1/dashboard")
def get_dashboard_summary():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Total metrics
        cursor.execute("SELECT COUNT(*) as cnt, SUM(total_claims) as claims, SUM(total_reimbursement) as reimb FROM providers")
        totals = cursor.fetchone()
        
        # Risk distribution
        cursor.execute("SELECT risk_level, COUNT(*) as cnt FROM providers GROUP BY risk_level")
        risk_levels = {row["risk_level"]: row["cnt"] for row in cursor.fetchall()}
        
        # Investigation count
        cursor.execute("SELECT COUNT(*) as cnt FROM investigations WHERE status != 'Reviewed'")
        inv_count = cursor.fetchone()["cnt"]
        
        # Top 10 suspicious providers
        cursor.execute("""
            SELECT p.provider_id, p.risk_score, p.risk_level, p.total_reimbursement, p.total_claims, p.provider_type
            FROM providers p
            ORDER BY p.risk_score DESC
            LIMIT 10
        """)
        top_suspicious = [dict(row) for row in cursor.fetchall()]
        
        # Risk by specialty/type
        cursor.execute("SELECT provider_type, AVG(risk_score) as avg_risk, COUNT(*) as cnt FROM providers GROUP BY provider_type")
        risk_by_type = [dict(row) for row in cursor.fetchall()]
        
        # Risk by State (top 10 states by average risk score)
        cursor.execute("""
            SELECT primary_state as state, AVG(risk_score) as avg_risk, COUNT(*) as cnt 
            FROM providers 
            GROUP BY primary_state 
            ORDER BY avg_risk DESC 
            LIMIT 10
        """)
        risk_by_state = [dict(row) for row in cursor.fetchall()]
        
        # Recent activity
        cursor.execute("""
            SELECT i.provider_id, i.status, i.updated_at, p.risk_score, p.risk_level
            FROM investigations i
            JOIN providers p ON i.provider_id = p.provider_id
            WHERE i.status != 'New'
            ORDER BY i.updated_at DESC
            LIMIT 5
        """)
        recent_activity = [dict(row) for row in cursor.fetchall()]
        
        return {
            "total_providers": totals["cnt"],
            "total_claims": totals["claims"],
            "total_reimbursement": totals["reimb"],
            "investigation_queue_count": inv_count,
            "risk_distribution": {
                "Low": risk_levels.get("Low", 0),
                "Medium": risk_levels.get("Medium", 0),
                "High": risk_levels.get("High", 0),
                "Critical": risk_levels.get("Critical", 0),
            },
            "top_suspicious": top_suspicious,
            "risk_by_type": risk_by_type,
            "risk_by_state": risk_by_state,
            "recent_activity": recent_activity
        }
    finally:
        conn.close()

@app.get("/api/v1/providers")
def list_providers(
    page: int = 1,
    page_size: int = 20,
    search: Optional[str] = None,
    risk_level: Optional[str] = None,
    provider_type: Optional[str] = None,
    state: Optional[int] = None,
    sort_by: str = "risk_score",
    sort_order: str = "desc"
):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        conditions = []
        params = []
        
        if search:
            conditions.append("p.provider_id LIKE ?")
            params.append(f"%{search}%")
        if risk_level:
            conditions.append("p.risk_level = ?")
            params.append(risk_level)
        if provider_type:
            conditions.append("p.provider_type = ?")
            params.append(provider_type)
        if state is not None:
            conditions.append("p.primary_state = ?")
            params.append(state)
            
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        
        # Sort validation
        allowed_sorts = ["risk_score", "total_claims", "total_reimbursement"]
        if sort_by not in allowed_sorts:
            sort_by = "risk_score"
        direction = "DESC" if sort_order.lower() == "desc" else "ASC"
        
        # Count total matching records
        cursor.execute(f"SELECT COUNT(*) as total FROM providers p {where_clause}", params)
        total_records = cursor.fetchone()["total"]
        
        # Query results
        offset = (page - 1) * page_size
        query = f"""
            SELECT p.*, i.status as investigation_status
            FROM providers p
            LEFT JOIN investigations i ON p.provider_id = i.provider_id
            {where_clause}
            ORDER BY p.{sort_by} {direction}
            LIMIT ? OFFSET ?
        """
        cursor.execute(query, params + [page_size, offset])
        rows = cursor.fetchall()
        
        return {
            "page": page,
            "page_size": page_size,
            "total_records": total_records,
            "total_pages": (total_records + page_size - 1) // page_size,
            "data": [dict(row) for row in rows]
        }
    finally:
        conn.close()

@app.get("/api/v1/providers/{provider_id}")
def get_provider_details(provider_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Provider profile
        cursor.execute("""
            SELECT p.*, i.status as investigation_status, i.notes as investigation_notes, i.updated_at as status_updated_at
            FROM providers p
            LEFT JOIN investigations i ON p.provider_id = i.provider_id
            WHERE p.provider_id = ?
        """, (provider_id,))
        profile_row = cursor.fetchone()
        
        if not profile_row:
            raise HTTPException(status_code=404, detail="Provider not found")
            
        profile = dict(profile_row)
        
        # Model Scores
        cursor.execute("SELECT * FROM model_scores WHERE provider_id = ?", (provider_id,))
        scores_row = cursor.fetchone()
        scores = dict(scores_row) if scores_row else {}
        
        # Peer Benchmarks
        cursor.execute("SELECT * FROM peer_benchmarks WHERE provider_id = ?", (provider_id,))
        peer_row = cursor.fetchone()
        peer = dict(peer_row) if peer_row else {}
        
        # Explanations
        cursor.execute("SELECT reason FROM explanations WHERE provider_id = ?", (provider_id,))
        reasons = [row["reason"] for row in cursor.fetchall()]
        
        return {
            "profile": profile,
            "model_scores": scores,
            "peer_benchmarks": peer,
            "reasons": reasons
        }
    finally:
        conn.close()

@app.post("/api/v1/investigations/{provider_id}/status")
def update_investigation_status(provider_id: str, payload: StatusUpdateRequest):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT provider_id FROM providers WHERE provider_id = ?", (provider_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Provider not found")
            
        cursor.execute("""
            INSERT INTO investigations (provider_id, status, notes, updated_at)
            VALUES (?, ?, '', CURRENT_TIMESTAMP)
            ON CONFLICT(provider_id) DO UPDATE SET
                status = excluded.status,
                updated_at = CURRENT_TIMESTAMP
        """, (provider_id, payload.status))
        conn.commit()
        return {"success": True, "message": f"Status updated to {payload.status}"}
    finally:
        conn.close()

@app.post("/api/v1/investigations/{provider_id}/notes")
def update_investigation_notes(provider_id: str, payload: NotesUpdateRequest):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT provider_id FROM providers WHERE provider_id = ?", (provider_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Provider not found")
            
        cursor.execute("""
            INSERT INTO investigations (provider_id, status, notes, updated_at)
            VALUES (?, 'Under Review', ?, CURRENT_TIMESTAMP)
            ON CONFLICT(provider_id) DO UPDATE SET
                notes = excluded.notes,
                updated_at = CURRENT_TIMESTAMP
        """, (provider_id, payload.notes))
        conn.commit()
        return {"success": True, "message": "Investigation notes updated."}
    finally:
        conn.close()

@app.get("/api/v1/model-performance")
def get_model_performance():
    # Hardcoded from training script outputs for consistency and speed
    return [
        {"model": "CatBoost (Supervised)", "precision": 0.9209, "recall": 0.9209, "f1": 0.9209, "pr_auc": 0.9682, "precision_at_100": 1.00, "type": "Supervised"},
        {"model": "Combined Risk Engine", "precision": 0.5573, "recall": 0.5573, "f1": 0.5573, "pr_auc": 0.6484, "precision_at_100": 1.00, "type": "Hybrid"},
        {"model": "Isolation Forest", "precision": 0.5040, "recall": 0.5040, "f1": 0.5040, "pr_auc": 0.5560, "precision_at_100": 0.86, "type": "Unsupervised"},
        {"model": "Autoencoder", "precision": 0.4565, "recall": 0.4565, "f1": 0.4565, "pr_auc": 0.4776, "precision_at_100": 0.80, "type": "Unsupervised"},
        {"model": "One-Class SVM", "precision": 0.3300, "recall": 0.3300, "f1": 0.3300, "pr_auc": 0.3566, "precision_at_100": 0.36, "type": "Unsupervised"},
        {"model": "LOF", "precision": 0.1087, "recall": 0.1087, "f1": 0.1087, "pr_auc": 0.1097, "precision_at_100": 0.10, "type": "Unsupervised"},
    ]

@app.post("/api/v1/ai-assistant/query")
def query_ai_assistant(payload: AIQueryRequest):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        p_id = payload.provider_id
        user_query = payload.query.lower()
        
        # Fetch provider data
        cursor.execute("SELECT * FROM providers WHERE provider_id = ?", (p_id,))
        p_row = cursor.fetchone()
        if not p_row:
            raise HTTPException(status_code=404, detail="Provider not found")
        p = dict(p_row)
        
        cursor.execute("SELECT * FROM model_scores WHERE provider_id = ?", (p_id,))
        s = dict(cursor.fetchone())
        
        cursor.execute("SELECT * FROM peer_benchmarks WHERE provider_id = ?", (p_id,))
        b = dict(cursor.fetchone())
        
        cursor.execute("SELECT reason FROM explanations WHERE provider_id = ?", (p_id,))
        reasons = [row["reason"] for row in cursor.fetchall()]
        
        # Craft a highly detailed response based on the query type
        response = ""
        
        if "why" in user_query or "reason" in user_query or "flag" in user_query or "suspicious" in user_query:
            reasons_bullet = "\n".join([f"- {r}" for r in reasons])
            response = (
                f"**Clinical Audit Assistant Analysis for Provider {p_id}**\n\n"
                f"This provider is classified as **{p['risk_level']} Risk** with a composite score of **{p['risk_score']:.1f}/100**.\n\n"
                f"### Core Anomalies Detected:\n"
                f"{reasons_bullet}\n\n"
                f"### Layered Risk Score Breakdown:\n"
                f"- **Supervised Pattern Matching (CatBoost):** {s['catboost_score']:.1f}/100 - indicating behavior strongly matching known historical fraud cases.\n"
                f"- **Unsupervised Anomaly Score (Ensemble):** {s['ml_score']:.1f}/100 (Isolation Forest: {s['isolation_score']:.1f}, Autoencoder: {s['autoencoder_score']:.1f}).\n"
                f"- **Statistical Outlier Score:** {s['statistical_score']:.1f}/100.\n"
                f"- **Peer Group Deviation:** {s['peer_score']:.1f}/100."
            )
        elif "peer" in user_query or "benchmark" in user_query or "compare" in user_query:
            response = (
                f"**Peer Benchmarking Report for Provider {p_id}**\n\n"
                f"The provider operates primarily in **State {p['primary_state']}** and falls into the **{p['provider_type']}** peer group.\n\n"
                f"### Peer Ratios (Provider Metric / Peer Group Median):\n"
                f"- **Reimbursement Ratio:** **{b['reimbursement_ratio']:.2f}x** (Provider billed a total of **${p['total_reimbursement']:,.2f}** compared to the peer median).\n"
                f"- **Claims Ratio:** **{b['claims_ratio']:.2f}x** (Provider submitted **{p['total_claims']}** claims).\n"
                f"- **Beneficiary Service Ratio:** **{b['beneficiary_ratio']:.2f}x** (Provider served **{p['total_beneficiaries']}** unique patients).\n\n"
                f"### National Percentiles:\n"
                f"- **Reimbursement Percentile:** {b['reimbursement_percentile']:.1f}th percentile.\n"
                f"- **Claims Volume Percentile:** {b['claims_percentile']:.1f}th percentile.\n"
                f"- **Patient Volume Percentile:** {b['beneficiary_percentile']:.1f}th percentile.\n\n"
                f"**Summary:** A reimbursement ratio of **{b['reimbursement_ratio']:.2f}x** indicates that this provider receives significantly higher insurance payouts than peers with a similar share of inpatient/outpatient claims in the same state."
            )
        elif "model" in user_query or "algorithm" in user_query or "ml" in user_query:
            response = (
                f"**ML Detection Signals for Provider {p_id}**\n\n"
                f"The hybrid engine evaluates five distinct models to avoid black-box limitations:\n\n"
                f"| Algorithm | Anomaly Score (0-100) | Flagged Status |\n"
                f"| :--- | :---: | :---: |\n"
                f"| **Isolation Forest** | {s['isolation_score']:.1f} | {'Flagged' if s['isolation_score'] > 70 else 'Normal'} |\n"
                f"| **PyTorch Autoencoder** | {s['autoencoder_score']:.1f} | {'Flagged' if s['autoencoder_score'] > 70 else 'Normal'} |\n"
                f"| **Local Outlier Factor (LOF)** | {s['lof_score']:.1f} | {'Flagged' if s['lof_score'] > 70 else 'Normal'} |\n"
                f"| **One-Class SVM** | {s['ocsvm_score']:.1f} | {'Flagged' if s['ocsvm_score'] > 70 else 'Normal'} |\n"
                f"| **CatBoost Classifier** | {s['catboost_score']:.1f} | {'Flagged' if s['catboost_score'] > 80 else 'Normal'} |\n\n"
                f"- **Unsupervised Agreement:** Multiple algorithms identify this provider as a structural outlier in the high-dimensional feature space.\n"
                f"- **Supervised Correlation:** The CatBoost model indicates a probability of **{s['catboost_score']:.1f}%** that the provider shares billing features with past established fraud cases."
            )
        else:
            response = (
                f"Hello! I am your AI clinical audit assistant. I have loaded all records for **Provider {p_id}**.\n\n"
                f"You can ask me questions such as:\n"
                f"1. *Why was this provider flagged?*\n"
                f"2. *How does this provider compare to peers?*\n"
                f"3. *What models or algorithms flagged this provider?*\n\n"
                f"Currently, Provider {p_id} has a composite risk score of **{p['risk_score']:.1f}/100** (**{p['risk_level']}** risk category) and is marked as **{p['PotentialFraud'] == 1 and 'Ground Truth Fraud' or 'Ground Truth Normal'}** in historical databases."
            )
            
        return {"response": response}
    finally:
        conn.close()
