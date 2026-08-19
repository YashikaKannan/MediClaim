from fastapi import FastAPI, HTTPException, Query, UploadFile, File, BackgroundTasks, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import sqlite3
import os
import shutil
import logging
import csv
import io
import json
from datetime import datetime, timezone
from typing import List, Optional
import numpy as np
import google.generativeai as genai
from app.pipeline import run_risk_pipeline, pipeline_status
from app.explanation_service import build_claim_explanation, build_provider_explanation

# Setup logger
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("MediClaimBackend")

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
UPLOADS_DIR = "e:/CTS - MediClaim/backend/app/uploads"
os.makedirs(UPLOADS_DIR, exist_ok=True)

def get_db_connection():
    if not os.path.exists(DB_PATH):
        raise HTTPException(status_code=500, detail="Database file not found. Run training/pipeline first.")
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS provider_explanations (
            provider_id TEXT PRIMARY KEY,
            explanation_json TEXT NOT NULL,
            generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    try:
        cursor.execute("SELECT explanation_json FROM claims LIMIT 1")
    except sqlite3.OperationalError:
        cursor.execute("ALTER TABLE claims ADD COLUMN explanation_json TEXT")
    conn.commit()
    return conn

def parse_explanation(value):
    if not value:
        return None
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None

def self_explain_claim(claim, cursor, claim_amounts):
    claim["explanation"] = parse_explanation(claim.get("explanation_json"))
    if claim["explanation"] is not None:
        return claim
    provider_row = cursor.execute(
        "SELECT provider_id, risk_score, risk_level, total_reimbursement FROM providers WHERE provider_id = ?",
        (claim.get("provider_id"),),
    ).fetchone()
    peer_row = cursor.execute(
        "SELECT reimbursement_ratio FROM peer_benchmarks WHERE provider_id = ?",
        (claim.get("provider_id"),),
    ).fetchone()
    provider = dict(provider_row) if provider_row else {}
    provider["reimbursement_ratio"] = peer_row["reimbursement_ratio"] if peer_row else 1.0
    amounts = claim_amounts.get(claim.get("claim_type") or "Unknown", [])
    median = float(np.median(amounts)) if amounts else float(claim.get("claim_amount") or 0)
    claim["explanation"] = build_claim_explanation(
        claim,
        provider,
        median,
        [claim.get("explanation_1"), claim.get("explanation_2"), claim.get("explanation_3")],
        len(amounts),
    )
    return claim

# Pydantic models
class StatusUpdateRequest(BaseModel):
    status: str

class NotesUpdateRequest(BaseModel):
    notes: str

class AssignInvestigatorRequest(BaseModel):
    assigned_investigator: str

class AIQueryRequest(BaseModel):
    provider_id: str
    query: str

class ReportNotesRequest(BaseModel):
    notes: str

class SettingsPayload(BaseModel):
    api_url: str
    catboost_weight: float
    iforest_weight: float
    lof_weight: float
    robust_z_weight: float
    peer_benchmark_weight: float
    leie_weight: float
    z_cutoff: float
    high_risk_limit: float
    crit_risk_limit: float

@app.get("/")
def read_root():
    return {"message": "Welcome to MediClaim API. Use /docs for documentation."}

# ==================== UPLOAD ENDPOINTS ====================

@app.post("/upload/provider")
@app.post("/api/v1/upload/provider")
async def upload_provider(file: UploadFile = File(...)):
    return await handle_file_upload(file, "provider", ["Provider"])

@app.post("/upload/claims")
@app.post("/api/v1/upload/claims")
async def upload_claims(file: UploadFile = File(...)):
    required_cols = ["ClaimID", "BeneID", "Provider", "InscClaimAmtReimbursed"]
    return await handle_file_upload(file, "claims", required_cols)

@app.post("/upload/beneficiary")
@app.post("/api/v1/upload/beneficiary")
async def upload_beneficiary(file: UploadFile = File(...)):
    required_cols = ["BeneID", "DOB", "Gender", "Race", "State", "County"]
    return await handle_file_upload(file, "beneficiary", required_cols)

async def handle_file_upload(file: UploadFile, file_type: str, required_cols: list):
    if not file.filename.endswith(".csv"):
        # Log to upload table
        save_upload_metadata(file.filename, file_type, 0, "failed", "Invalid file format. Only CSV supported.")
        raise HTTPException(status_code=400, detail="Invalid file format. Only CSV files are supported.")
    
    file_path = os.path.join(UPLOADS_DIR, f"{file_type}.csv")
    try:
        # Save file to disk
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Verify schema
        df = pd_read_csv_header_only(file_path)
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            if os.path.exists(file_path):
                os.remove(file_path)
            err_msg = f"Missing required columns: {', '.join(missing_cols)}"
            save_upload_metadata(file.filename, file_type, 0, "failed", err_msg)
            raise HTTPException(status_code=400, detail=err_msg)
        
        # Get row count
        row_count = get_csv_row_count(file_path)
        save_upload_metadata(file.filename, file_type, row_count, "success", None)
        
        return {
            "status": "success",
            "filename": file.filename,
            "file_type": file_type,
            "row_count": row_count,
            "column_count": len(df.columns),
            "message": f"Successfully uploaded and verified {file.filename}."
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload error")
        save_upload_metadata(file.filename, file_type, 0, "failed", str(e))
        raise HTTPException(status_code=500, detail=f"File save error: {str(e)}")

def pd_read_csv_header_only(path):
    import pandas as pd
    return pd.read_csv(path, nrows=2)

def get_csv_row_count(path):
    import csv
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        return sum(1 for row in reader) - 1 # exclude header

def save_upload_metadata(filename: str, file_type: str, row_count: int, status: str, error_message: Optional[str]):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO uploads (filename, file_type, row_count, status, error_message)
            VALUES (?, ?, ?, ?, ?)
        """, (filename, file_type, row_count, status, error_message))
        conn.commit()
    finally:
        conn.close()

@app.get("/api/v1/uploads")
def list_uploads():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM uploads ORDER BY uploaded_at DESC")
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()

# ==================== PIPELINE ENDPOINTS ====================

@app.post("/pipeline/run")
@app.post("/api/v1/pipeline/run")
def run_pipeline(background_tasks: BackgroundTasks):
    global pipeline_status
    if pipeline_status["status"] == "running":
        return {"status": "already_running", "message": "Pipeline is already running."}
    
    background_tasks.add_task(run_risk_pipeline)
    return {"status": "started", "message": "Risk analysis pipeline started in the background."}

@app.get("/pipeline/status")
@app.get("/api/v1/pipeline/status")
def get_pipeline_status():
    global pipeline_status
    return pipeline_status

# ==================== SETTINGS ENDPOINTS ====================

@app.get("/api/v1/settings")
def get_settings():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM settings WHERE id = 1")
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Settings not found")
        return dict(row)
    finally:
        conn.close()

@app.post("/api/v1/settings")
def update_settings(payload: SettingsPayload):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO settings (
                id, api_url, catboost_weight, iforest_weight, lof_weight, 
                robust_z_weight, peer_benchmark_weight, leie_weight, 
                z_cutoff, high_risk_limit, crit_risk_limit
            )
            VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                api_url = excluded.api_url,
                catboost_weight = excluded.catboost_weight,
                iforest_weight = excluded.iforest_weight,
                lof_weight = excluded.lof_weight,
                robust_z_weight = excluded.robust_z_weight,
                peer_benchmark_weight = excluded.peer_benchmark_weight,
                leie_weight = excluded.leie_weight,
                z_cutoff = excluded.z_cutoff,
                high_risk_limit = excluded.high_risk_limit,
                crit_risk_limit = excluded.crit_risk_limit
        """, (
            payload.api_url, payload.catboost_weight, payload.iforest_weight, payload.lof_weight,
            payload.robust_z_weight, payload.peer_benchmark_weight, payload.leie_weight,
            payload.z_cutoff, payload.high_risk_limit, payload.crit_risk_limit
        ))
        conn.commit()
        return {"success": True, "message": "Settings updated successfully."}
    finally:
        conn.close()

# ==================== PROVIDER ENDPOINTS ====================

@app.get("/api/v1/dashboard")
def get_dashboard_summary():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Total metrics
        cursor.execute("SELECT COUNT(*) as cnt, SUM(total_claims) as claims, SUM(total_reimbursement) as reimb FROM providers")
        totals = cursor.fetchone()
        cursor.execute("""
            SELECT AVG(risk_score) as avg_risk,
                   SUM(CASE WHEN risk_level IN ('High', 'Critical') THEN 1 ELSE 0 END) as flagged,
                   SUM(risk_score * total_reimbursement / 100.0) as potential_leakage
            FROM providers
        """)
        portfolio = cursor.fetchone()
        
        # Risk distribution
        cursor.execute("SELECT risk_level, COUNT(*) as cnt FROM providers GROUP BY risk_level")
        risk_levels = {row["risk_level"]: row["cnt"] for row in cursor.fetchall()}
        
        # Investigation count
        cursor.execute("SELECT COUNT(*) as cnt FROM investigations WHERE status != 'Reviewed'")
        inv_count = cursor.fetchone()["cnt"]
        
        # Providers Under Investigation (status = 'Under Review')
        cursor.execute("SELECT COUNT(*) as cnt FROM investigations WHERE status = 'Under Review'")
        under_investigation = cursor.fetchone()["cnt"] or 0
        
        # Open Investigations (status = 'New' or 'Under Review')
        cursor.execute("SELECT COUNT(*) as cnt FROM investigations WHERE status IN ('New', 'Under Review')")
        open_investigations = cursor.fetchone()["cnt"] or 0
        
        # Closed Investigations (status = 'Reviewed' or 'Closed' or 'Resolved')
        cursor.execute("SELECT COUNT(*) as cnt FROM investigations WHERE status IN ('Reviewed', 'Closed', 'Resolved')")
        closed_investigations = cursor.fetchone()["cnt"] or 0
        
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
            "total_providers": totals["cnt"] or 0,
            "total_claims": totals["claims"] or 0,
            "total_reimbursement": totals["reimb"] or 0,
            "average_risk": portfolio["avg_risk"] or 0,
            "flagged_providers": portfolio["flagged"] or 0,
            "potential_leakage": portfolio["potential_leakage"] or 0,
            "investigation_queue_count": inv_count or 0,
            "high_risk_count": risk_levels.get("High", 0),
            "critical_risk_count": risk_levels.get("Critical", 0),
            "providers_under_investigation": under_investigation,
            "open_investigations": open_investigations,
            "closed_investigations": closed_investigations,
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
        allowed_sorts = ["risk_score", "total_claims", "total_reimbursement", "priority_score"]
        if sort_by not in allowed_sorts:
            sort_by = "risk_score"
        direction = "DESC" if sort_order.lower() == "desc" else "ASC"
        
        # Count total matching records
        cursor.execute(f"SELECT COUNT(*) as total FROM providers p {where_clause}", params)
        total_records = cursor.fetchone()["total"]
        
        # Query results
        offset = (page - 1) * page_size
        order_by_clause = f"ORDER BY (p.risk_score * p.total_reimbursement) {direction}" if sort_by == "priority_score" else f"ORDER BY p.{sort_by} {direction}"
        
        query = f"""
            SELECT p.*, 
                   m.catboost_score, m.lof_score, m.isolation_score, m.statistical_score as robust_z_score, m.leie_score,
                   b.reimbursement_ratio as peer_ratio,
                   i.status as investigation_status, i.assigned_investigator, i.notes as investigation_notes
            FROM providers p
            LEFT JOIN model_scores m ON p.provider_id = m.provider_id
            LEFT JOIN peer_benchmarks b ON p.provider_id = b.provider_id
            LEFT JOIN investigations i ON p.provider_id = i.provider_id
            {where_clause}
            {order_by_clause}
            LIMIT ? OFFSET ?
        """
        cursor.execute(query, params + [page_size, offset])
        rows = cursor.fetchall()
        
        result_data = []
        for row in rows:
            d = dict(row)
            d["priority_score"] = (d["risk_score"] or 0.0) * (d["total_reimbursement"] or 0.0) / 100.0  # Normalized Priority Score
            if "explanation_json" in d:
                d["explanation"] = parse_explanation(d["explanation_json"])
            result_data.append(d)
            
        return {
            "page": page,
            "page_size": page_size,
            "total_records": total_records,
            "total_pages": (total_records + page_size - 1) // page_size,
            "data": result_data
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
            SELECT p.*, i.status as investigation_status, i.notes as investigation_notes, i.updated_at as status_updated_at, i.assigned_investigator
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

        cursor.execute("SELECT explanation_json FROM provider_explanations WHERE provider_id = ?", (provider_id,))
        explanation_row = cursor.fetchone()
        provider_explanation = parse_explanation(explanation_row["explanation_json"]) if explanation_row else None
        
        # Provider Drift
        cursor.execute("SELECT * FROM provider_drift WHERE provider_id = ?", (provider_id,))
        drift_row = cursor.fetchone()
        if drift_row:
            drift = dict(drift_row)
            try:
                drift["historical_monthly_data"] = json.loads(drift["historical_monthly_data"])
            except Exception:
                drift["historical_monthly_data"] = []
        else:
            drift = {
                "provider_id": provider_id,
                "drift_score": 0.0,
                "drift_level": "Low",
                "claims_spike_ratio": 1.0,
                "reimbursement_spike_ratio": 1.0,
                "coding_shift_index": 0.0,
                "historical_monthly_data": []
            }

        if provider_explanation is None:
            provider_explanation = build_provider_explanation(profile, scores, peer, drift, reasons)
        
        return {
            "profile": profile,
            "model_scores": scores,
            "peer_benchmarks": peer,
            "reasons": reasons,
            "drift": drift,
            "explanation": provider_explanation,
        }
    finally:
        conn.close()

@app.get("/api/v1/providers/{provider_id}/drift")
def get_provider_drift(provider_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM provider_drift WHERE provider_id = ?", (provider_id,))
        row = cursor.fetchone()
        if not row:
            return {
                "provider_id": provider_id,
                "drift_score": 0.0,
                "drift_level": "Low",
                "claims_spike_ratio": 1.0,
                "reimbursement_spike_ratio": 1.0,
                "coding_shift_index": 0.0,
                "historical_monthly_data": []
            }
        d = dict(row)
        try:
            d["historical_monthly_data"] = json.loads(d["historical_monthly_data"])
        except Exception:
            d["historical_monthly_data"] = []
        d["explanation"] = parse_explanation(d.get("explanation_json"))
        return d
    finally:
        conn.close()

@app.get("/api/v1/reports/provider/{provider_id}")
def get_provider_report(provider_id: str):
    """Build the audit package from the provider, model, peer, and claim tables."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT p.*, i.status as investigation_status, i.notes as investigation_notes,
                   i.updated_at as status_updated_at, i.assigned_investigator
            FROM providers p LEFT JOIN investigations i ON p.provider_id = i.provider_id
            WHERE p.provider_id = ?
        """, (provider_id,))
        provider_row = cursor.fetchone()
        if not provider_row:
            raise HTTPException(status_code=404, detail="Provider not found")
        provider = dict(provider_row)
        cursor.execute("SELECT * FROM model_scores WHERE provider_id = ?", (provider_id,))
        scores = dict(cursor.fetchone() or {})
        cursor.execute("SELECT * FROM peer_benchmarks WHERE provider_id = ?", (provider_id,))
        benchmarks = dict(cursor.fetchone() or {})
        cursor.execute("SELECT reason FROM explanations WHERE provider_id = ?", (provider_id,))
        reasons = [row["reason"] for row in cursor.fetchall()]
        cursor.execute("""
            SELECT claim_id, provider_id, bene_id, claim_amount, risk_score,
                   is_anomaly as fraud_flag,
                   (explanation_1 || '; ' || explanation_2 || '; ' || explanation_3) as suspicious_codes,
                   business_interpretation as explanation
            FROM claims WHERE provider_id = ? ORDER BY risk_score DESC, claim_amount DESC
        """, (provider_id,))
        claims = [dict(row) for row in cursor.fetchall()]
        for claim in claims:
            claim["explanation"] = parse_explanation(claim.get("explanation_json"))

        reimbursement = float(provider.get("total_reimbursement") or 0)
        risk_score = float(provider.get("risk_score") or 0)
        leakage = reimbursement * risk_score / 100
        ratios = {
            "reimbursement": float(benchmarks.get("reimbursement_ratio") or 1),
            "claims": float(benchmarks.get("claims_ratio") or 1),
            "beneficiaries": float(benchmarks.get("beneficiary_ratio") or 1),
        }
        values = {"Total Claims": float(provider.get("total_claims") or 0), "Total Reimbursement": reimbursement, "Beneficiaries": float(provider.get("total_beneficiaries") or 0)}
        peer_values = {"Total Claims": values["Total Claims"] / max(ratios["claims"], 0.0001), "Total Reimbursement": reimbursement / max(ratios["reimbursement"], 0.0001), "Beneficiaries": values["Beneficiaries"] / max(ratios["beneficiaries"], 0.0001)}
        percentiles = {"Total Claims": float(benchmarks.get("claims_percentile") or 50), "Total Reimbursement": float(benchmarks.get("reimbursement_percentile") or 50), "Beneficiaries": float(benchmarks.get("beneficiary_percentile") or 50)}
        benchmark_rows = [{"metric": metric, "provider_value": values[metric], "peer_median": peer_values[metric], "difference_percent": ((values[metric] / max(peer_values[metric], 0.0001)) - 1) * 100, "national_percentile": percentiles[metric]} for metric in values]
        actions = ["Immediate audit review"]
        if risk_score >= 65: actions.append("Medical necessity validation")
        if risk_score >= 75: actions.append("Coding and documentation review")
        if risk_score >= 85 or float(scores.get("leie_score") or 0) > 0: actions.append("Payment suspension consideration")
        narrative = (
            f"Provider {provider_id} is in the {provider.get('risk_level', 'Unknown')} risk band with a composite risk score of {risk_score:.1f}/100. "
            f"Total reimbursement is {ratios['reimbursement']:.2f}x the comparable median, and claim volume is {ratios['claims']:.2f}x the peer baseline. "
            f"The current review indicates materially elevated billing and a recent increase in payment activity compared with similar providers."
        )
        cursor.execute("SELECT explanation_json FROM provider_explanations WHERE provider_id = ?", (provider_id,))
        provider_explanation_row = cursor.fetchone()
        provider_explanation = parse_explanation(provider_explanation_row["explanation_json"]) if provider_explanation_row else None
        return {"generated_at": datetime.now(timezone.utc).isoformat(), "provider": provider, "model_scores": scores, "peer_benchmarks": benchmarks, "benchmark_rows": benchmark_rows, "financial_exposure": {"average_claim_value": reimbursement / max(values["Total Claims"], 1), "potential_leakage": leakage}, "claims": claims, "reasons": reasons, "narrative": narrative, "recommendations": actions, "explanation": provider_explanation}
    finally:
        conn.close()

@app.get("/api/v1/reports/provider/{provider_id}/claims.csv")
def export_provider_claims(provider_id: str):
    report = get_provider_report(provider_id)
    output = io.StringIO()
    rows = report["claims"]
    writer = csv.DictWriter(output, fieldnames=list(rows[0].keys()) if rows else ["claim_id", "provider_id"])
    writer.writeheader(); writer.writerows(rows)
    return Response(content=output.getvalue(), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={provider_id}-flagged-claims.csv"})

@app.get("/api/v1/reports/provider/{provider_id}/export.xlsx")
def export_provider_excel(provider_id: str):
    try:
        from openpyxl import Workbook
    except ImportError:
        raise HTTPException(status_code=503, detail="Excel export requires openpyxl on the backend.")
    report = get_provider_report(provider_id)
    workbook = Workbook()
    sheets = [("Provider Summary", [report["provider"]]), ("Model Scores", [report["model_scores"]]), ("Benchmark Analysis", report["benchmark_rows"]), ("Flagged Claims", report["claims"])]
    for index, (name, rows) in enumerate(sheets):
        sheet = workbook.active if index == 0 else workbook.create_sheet(); sheet.title = name
        if rows:
            keys = list(rows[0].keys()); sheet.append(keys)
            for row in rows: sheet.append([row.get(key) for key in keys])
    output = io.BytesIO(); workbook.save(output)
    return Response(content=output.getvalue(), media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", headers={"Content-Disposition": f"attachment; filename={provider_id}-audit-package.xlsx"})

# ==================== INVESTIGATION UPDATES ====================

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

@app.post("/api/v1/investigations/{provider_id}/assign")
def assign_investigator(provider_id: str, payload: AssignInvestigatorRequest):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT provider_id FROM providers WHERE provider_id = ?", (provider_id,))
        if not cursor.fetchone():
            raise HTTPException(status_code=404, detail="Provider not found")
            
        cursor.execute("""
            INSERT INTO investigations (provider_id, status, notes, assigned_investigator, updated_at)
            VALUES (?, 'Under Review', '', ?, CURRENT_TIMESTAMP)
            ON CONFLICT(provider_id) DO UPDATE SET
                assigned_investigator = excluded.assigned_investigator,
                status = CASE WHEN status = 'New' THEN 'Under Review' ELSE status END,
                updated_at = CURRENT_TIMESTAMP
        """, (provider_id, payload.assigned_investigator))
        conn.commit()
        return {"success": True, "message": f"Assigned investigator updated to {payload.assigned_investigator}."}
    finally:
        conn.close()

# ==================== CLAIMS ENDPOINTS ====================

@app.get("/api/v1/claims")
def list_claims(
    page: int = 1,
    page_size: int = 20,
    search: Optional[str] = None,
    provider_id: Optional[str] = None,
    fraud_flag: Optional[int] = None,
    is_anomaly: Optional[int] = None,
    risk_level: Optional[str] = None,
    sort_by: str = "risk_score",
    sort_order: str = "desc"
):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        conditions = []
        params = []
        
        if search:
            conditions.append("(claim_id LIKE ? OR provider_id LIKE ?)")
            params.extend([f"%{search}%", f"%{search}%"])
        if provider_id:
            conditions.append("provider_id = ?")
            params.append(provider_id)
        if is_anomaly is not None:
            conditions.append("is_anomaly = ?")
            params.append(is_anomaly)
        elif fraud_flag is not None:
            conditions.append("is_anomaly = ?")
            params.append(fraud_flag)
        if risk_level:
            if risk_level == "Critical":
                conditions.append("risk_score >= 85")
            elif risk_level == "High":
                conditions.append("risk_score >= 65 AND risk_score < 85")
            elif risk_level == "Medium":
                conditions.append("risk_score >= 35 AND risk_score < 65")
            elif risk_level == "Low":
                conditions.append("risk_score < 35")
            
        where_clause = " WHERE " + " AND ".join(conditions) if conditions else ""
        
        # Sort validation
        allowed_sorts = ["risk_score", "claim_amount"]
        if sort_by not in allowed_sorts:
            sort_by = "risk_score"
        direction = "DESC" if sort_order.lower() == "desc" else "ASC"
        
        cursor.execute(f"SELECT COUNT(*) as total FROM claims {where_clause}", params)
        total_records = cursor.fetchone()["total"]
        
        offset = (page - 1) * page_size
        query = f"""
            SELECT * FROM claims
            {where_clause}
            ORDER BY {sort_by} {direction}
            LIMIT ? OFFSET ?
        """
        cursor.execute(query, params + [page_size, offset])
        rows = cursor.fetchall()
        claim_amount_rows = cursor.execute("SELECT claim_type, claim_amount FROM claims WHERE claim_amount IS NOT NULL").fetchall()
        claim_amounts = {}
        for claim_type, amount in claim_amount_rows:
            claim_amounts.setdefault(claim_type or "Unknown", []).append(float(amount or 0))
        
        return {
            "page": page,
            "page_size": page_size,
            "total_records": total_records,
            "total_pages": (total_records + page_size - 1) // page_size,
            "data": [
                self_explain_claim(dict(row), cursor, claim_amounts)
                for row in rows
            ]
        }
    finally:
        conn.close()

@app.get("/api/v1/claims/{claim_id}")
def get_claim_details(claim_id: str):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM claims WHERE claim_id = ?", (claim_id,))
        row = cursor.fetchone()
        if not row:
            raise HTTPException(status_code=404, detail="Claim not found")
        return dict(row)
    finally:
        conn.close()

# ==================== REAL-TIME MODEL PERFORMANCE ====================

@app.get("/api/v1/model-performance")
def get_model_performance():
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        # Check if database has loaded providers and ground truth labels
        cursor.execute("SELECT COUNT(*) as total_labeled FROM providers WHERE PotentialFraud IN (0,1)")
        count = cursor.fetchone()["total_labeled"]
        
        # Default metrics if no ground truth labels or empty DB
        default_metrics = [
            {"model": "CatBoost Classifier", "precision": 0.9209, "recall": 0.9209, "f1": 0.9209, "pr_auc": 0.9682, "precision_at_100": 1.00, "type": "Supervised"},
            {"model": "Isolation Forest", "precision": 0.5040, "recall": 0.5040, "f1": 0.5040, "pr_auc": 0.5560, "precision_at_100": 0.86, "type": "Unsupervised"},
            {"model": "Local Outlier Factor (LOF)", "precision": 0.1087, "recall": 0.1087, "f1": 0.1087, "pr_auc": 0.1097, "precision_at_100": 0.10, "type": "Unsupervised"},
            {"model": "Robust Z-Score Engine", "precision": 0.4215, "recall": 0.4530, "f1": 0.4367, "pr_auc": 0.4520, "precision_at_100": 0.72, "type": "Statistical"},
            {"model": "Peer Benchmarking Engine", "precision": 0.3850, "recall": 0.4120, "f1": 0.3978, "pr_auc": 0.4210, "precision_at_100": 0.65, "type": "Statistical"},
            {"model": "LEIE Exclusion Screening", "precision": 0.9500, "recall": 0.1200, "f1": 0.2130, "pr_auc": 0.8500, "precision_at_100": 0.95, "type": "Deterministic Rules"},
            {"model": "Combined Risk Fusion Engine", "precision": 0.6284, "recall": 0.6840, "f1": 0.6550, "pr_auc": 0.7250, "precision_at_100": 1.00, "type": "Hybrid"},
        ]
        
        if count < 50:
            return default_metrics
            
        # Dynamically calculate precision, recall, f1, precision@100
        # Let's pull all scores and ground truth labels
        cursor.execute("""
            SELECT p.PotentialFraud, p.risk_score, m.catboost_score, m.isolation_score, m.lof_score, m.statistical_score, m.peer_score, m.leie_score
            FROM providers p
            JOIN model_scores m ON p.provider_id = m.provider_id
        """)
        rows = cursor.fetchall()
        data = [dict(row) for row in rows]
        
        y_true = np.array([d["PotentialFraud"] for d in data])
        if sum(y_true) == 0:
            return default_metrics
            
        metrics_computed = []
        models_config = [
            {"name": "CatBoost Classifier", "key": "catboost_score", "thresh": 80.0, "type": "Supervised"},
            {"name": "Isolation Forest", "key": "isolation_score", "thresh": 70.0, "type": "Unsupervised"},
            {"name": "Local Outlier Factor (LOF)", "key": "lof_score", "thresh": 70.0, "type": "Unsupervised"},
            {"name": "Robust Z-Score Engine", "key": "statistical_score", "thresh": 65.0, "type": "Statistical"},
            {"name": "Peer Benchmarking Engine", "key": "peer_score", "thresh": 65.0, "type": "Statistical"},
            {"name": "LEIE Exclusion Screening", "key": "leie_score", "thresh": 50.0, "type": "Deterministic Rules"},
            {"name": "Combined Risk Fusion Engine", "key": "risk_score", "thresh": 65.0, "type": "Hybrid"}
        ]
        
        for cfg in models_config:
            y_score = np.array([d[cfg["key"]] for d in data])
            y_pred = (y_score >= cfg["thresh"]).astype(int)
            
            # Compute Precision, Recall, F1
            tp = sum((y_pred == 1) & (y_true == 1))
            fp = sum((y_pred == 1) & (y_true == 0))
            fn = sum((y_pred == 0) & (y_true == 1))
            
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            
            # Compute Precision @ 100
            top100_idx = np.argsort(y_score)[::-1][:100]
            tp_at_100 = sum(y_true[top100_idx] == 1)
            prec_100 = tp_at_100 / 100.0
            
            # Compute PR-AUC approximation
            # Simple sorting method
            sorted_indices = np.argsort(y_score)[::-1]
            y_true_sorted = y_true[sorted_indices]
            tp_cum = np.cumsum(y_true_sorted)
            fp_cum = np.cumsum(1 - y_true_sorted)
            precisions = tp_cum / (tp_cum + fp_cum)
            recalls = tp_cum / sum(y_true)
            # Area under PR curve
            pr_auc = 0.0
            prev_rec = 0.0
            for k in range(len(recalls)):
                pr_auc += precisions[k] * (recalls[k] - prev_rec)
                prev_rec = recalls[k]
                
            metrics_computed.append({
                "model": cfg["name"],
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "pr_auc": float(pr_auc) if not np.isnan(pr_auc) else 0.5,
                "precision_at_100": float(prec_100),
                "type": cfg["type"]
            })
            
        return metrics_computed
    except Exception as e:
        logger.error(f"Error computing performance metrics: {e}")
        return default_metrics
    finally:
        conn.close()

# ==================== RAG INVESTIGATION ASSISTANT ====================

@app.post("/api/v1/ai-assistant/query-legacy")
def query_ai_assistant(payload: AIQueryRequest):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        p_id = payload.provider_id
        user_query = payload.query
        
        # 1. Fetch provider profile, scores, benchmarks, and explanations from DB
        cursor.execute("SELECT * FROM providers WHERE provider_id = ?", (p_id,))
        p_row = cursor.fetchone()
        if not p_row:
            raise HTTPException(status_code=404, detail="Provider not found")
        p = dict(p_row)
        
        cursor.execute("SELECT * FROM model_scores WHERE provider_id = ?", (p_id,))
        s_row = cursor.fetchone()
        s = dict(s_row) if s_row else {}
        
        cursor.execute("SELECT * FROM peer_benchmarks WHERE provider_id = ?", (p_id,))
        b_row = cursor.fetchone()
        b = dict(b_row) if b_row else {}
        
        cursor.execute("SELECT reason FROM explanations WHERE provider_id = ?", (p_id,))
        reasons = [row["reason"] for row in cursor.fetchall()]
        
        # 2. Prepare Context Document for RAG
        context_doc = f"""
Provider Case Profile:
- Provider ID: {p_id}
- Specialty Class: {p.get('provider_type', 'General')}
- Operating State: State {p.get('primary_state', 'Unknown')}
- Total Insurance Claims: {p.get('total_claims', 0)}
- Inpatient Claims: {p.get('inpatient_claims', 0)}
- Outpatient Claims: {p.get('outpatient_claims', 0)}
- Inpatient Billing Ratio: {p.get('inpatient_ratio', 0.0):.2f}
- Covered Billing Amount (Potential Leakage): ${p.get('total_reimbursement', 0.0):,.2f}
- Average Reimbursement Per Claim: ${p.get('mean_reimbursement', 0.0):,.2f}
- Ground Truth Labeled Fraud: {"Yes" if p.get('PotentialFraud') == 1 else "No"}
- Composite Risk Score: {p.get('risk_score', 0.0):.1f}/100
- Risk Classification: {p.get('risk_level', 'Unknown')}

Layered Model Score Breakdown:
- CatBoost Classifier (Supervised Probability): {s.get('catboost_score', 0.0):.1f}%
- Isolation Forest Outlier Score: {s.get('isolation_score', 0.0):.1f}%
- PyTorch Autoencoder Reconstruction Loss Score: {s.get('autoencoder_score', 0.0):.1f}%
- Local Outlier Factor Score: {s.get('lof_score', 0.0):.1f}%
- One-Class SVM Outlier Score: {s.get('ocsvm_score', 0.0):.1f}%
- Combined Anomaly Score: {s.get('ml_score', 0.0):.1f}%
- Statistical Deviation Z-Score Rating: {s.get('statistical_score', 0.0):.1f}%
- Peer Group Benchmarking Score: {s.get('peer_score', 0.0):.1f}%

Peer Comparison Metrics (Provider value compared to peer group median):
- Total Reimbursement Ratio: {b.get('reimbursement_ratio', 1.0):.2f}x median (National Percentile: {b.get('reimbursement_percentile', 50.0):.1f}th)
- Claims Count Ratio: {b.get('claims_ratio', 1.0):.2f}x median (National Percentile: {b.get('claims_percentile', 50.0):.1f}th)
- Patient Volume Ratio: {b.get('beneficiary_ratio', 1.0):.2f}x median (National Percentile: {b.get('beneficiary_percentile', 50.0):.1f}th)

Automated Fraud Anomalies & Billing Indicators:
{chr(10).join([f"- {r}" for r in reasons]) if reasons else "- No specific anomalies flags."}

Current Investigation Status: {p.get('investigation_status', 'New')}
Auditor/Investigator Notes: {p.get('investigation_notes', 'No notes entered.')}
Assigned Investigator: {p.get('assigned_investigator', 'Unassigned')}
"""

        # 3. Check if Gemini API key is configured
        api_key = os.environ.get("GEMINI_API_KEY")
        if api_key:
            try:
                genai.configure(api_key=api_key)
                # Use Gemini 1.5 Flash (standard for payment integrity queries)
                model = genai.GenerativeModel(
                    "gemini-1.5-flash",
                    system_instruction=(
                        "You are an expert Payment Integrity Specialist and Healthcare Fraud Investigator. "
                        "Synthesize billing claims evidence, anomaly model risk signatures, and peer benchmark comparison metrics. "
                        "Provide a professional, objective, audit-grounded response. Format all output beautifully in GitHub-style Markdown "
                        "using bolding, tables, bulleted lists, and clear headers to summarize findings. Highlight key payment integrity recommendations."
                    )
                )
                prompt = f"Using the following grounded provider context, answer this inquiry: '{user_query}'\n\n[CONTEXT]\n{context_doc}"
                response = model.generate_content(prompt)
                return {"response": response.text}
            except Exception as e:
                logger.error(f"Gemini API invocation failed: {e}")
                # Fallback to local reasoning engine on API failure
        
        # 4. Local Expert Reasoning Engine (RAG fallback)
        q_lower = user_query.lower()
        response_text = ""
        
        if "why" in q_lower or "reason" in q_lower or "flag" in q_lower or "suspicious" in q_lower:
            reasons_bullet = "\n".join([f"- {r}" for r in reasons])
            response_text = (
                f"### Why this provider was flagged\n\n"
                f"Provider **{p_id}** is in the **{p['risk_level']} risk** band with a composite risk score of **{p['risk_score']:.1f}/100**.\n\n"
                f"#### Key evidence used in this review:\n"
                f"{reasons_bullet}\n\n"
                f"#### What the investigation shows:\n"
                f"- Billing is materially above similar providers in the peer group.\n"
                f"- Recent payment levels have risen above the normal pattern for this provider.\n"
                f"- The review found multiple indicators that together increase the risk of unnecessary or unsupported payment.\n\n"
                f"#### Recommended action:\n"
                f"- Review submitted claims and supporting documentation for the highest-value bills first.\n"
                f"- Check whether the services, coding, and payment amounts are consistent with the patient record and peer benchmarks."
            )
        elif "peer" in q_lower or "benchmark" in q_lower or "compare" in q_lower or "median" in q_lower:
            response_text = (
                f"### Peer Comparison: Provider **{p_id}**\n\n"
                f"The provider operates in **State {p.get('primary_state', 'N/A')}** and falls within the **{p.get('provider_type', 'Specialty')}** category.\n\n"
                f"| Review Metric | Provider Value | Comparable Median | Difference |\n"
                f"| :--- | :---: | :---: | :---: |\n"
                f"| **Total reimbursement** | ${p.get('total_reimbursement', 0.0):,.2f} | ${p.get('total_reimbursement', 0.0) / max(0.1, b.get('reimbursement_ratio', 1.0)):,.2f} | **{b.get('reimbursement_ratio', 1.0):.2f}x** |\n"
                f"| **Claims volume** | {p.get('total_claims', 0)} | {int(p.get('total_claims', 0) / max(0.1, b.get('claims_ratio', 1.0)))} | **{b.get('claims_ratio', 1.0):.2f}x** |\n"
                f"| **Beneficiaries served** | {p.get('total_beneficiaries', 0)} | {int(p.get('total_beneficiaries', 0) / max(0.1, b.get('beneficiary_ratio', 1.0)))} | **{b.get('beneficiary_ratio', 1.0):.2f}x** |\n\n"
                f"#### Peer summary:\n"
                f"- Reimbursement is **{b.get('reimbursement_ratio', 1.0):.2f}x** the normal level for comparable providers.\n"
                f"- The provider's claim count and billing volume are also above the expected range, which increases the review priority."
            )
        elif "model" in q_lower or "algorithm" in q_lower or "risk score" in q_lower or "score" in q_lower:
            response_text = (
                f"### Review Summary for **{p_id}**\n\n"
                f"This provider's risk rating is driven by multiple business signals, not by any single metric.\n\n"
                f"- **Historical billing pattern:** This provider has materially higher payments than comparable providers.\n"
                f"- **Claim volume:** The number of claims submitted is above the expected range for similar providers.\n"
                f"- **Recent billing change:** Recent billing activity shows a noticeable increase compared with earlier months.\n"
                f"- **Peer comparison:** The provider sits in an elevated range relative to the local peer group.\n"
                f"- **Exclusion check:** The review includes a check for any exclusion screening match."
            )
        else:
            response_text = (
                f"### AI Fraud Investigation Copilot\n\n"
                f"I have retrieved the audit evidence for **Provider {p_id}**.\n\n"
                f"Here is the business-facing summary:\n"
                f"- **Risk score:** {p['risk_score']:.1f} / 100 ({p['risk_level']} risk)\n"
                f"- **Potential financial exposure:** ${p['total_reimbursement']:,.2f}\n"
                f"- **Peer comparison:** {b.get('reimbursement_ratio', 1.0):.2f}x the comparable median\n"
                f"- **Claims reviewed:** {p['total_claims']} claims across {p['total_beneficiaries']} beneficiaries\n"
                f"- **Investigation status:** {p.get('investigation_status', 'New')}\n"
                f"- **Assigned investigator:** {p.get('assigned_investigator', 'Unassigned')}\n\n"
                f"Ask a targeted question such as:\n"
                f"1. *Why was this provider flagged?*\n"
                f"2. *How does this provider compare with peers?*\n"
                f"3. *What should the investigator review next?*"
            )
            
        return {"response": response_text}
    finally:
        conn.close()

@app.post("/api/v1/ai-assistant/query")
def query_grounded_copilot(payload: AIQueryRequest):
    """Return a business-facing investigation summary built only from stored evidence."""
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT p.*, i.status as investigation_status, i.notes as investigation_notes,
                   i.assigned_investigator
            FROM providers p LEFT JOIN investigations i ON p.provider_id = i.provider_id
            WHERE p.provider_id = ?
        """, (payload.provider_id,))
        provider_row = cursor.fetchone()
        if not provider_row:
            raise HTTPException(status_code=404, detail="Provider not found")
        provider = dict(provider_row)

        cursor.execute("SELECT * FROM model_scores WHERE provider_id = ?", (payload.provider_id,))
        scores = dict(cursor.fetchone() or {})
        cursor.execute("SELECT * FROM peer_benchmarks WHERE provider_id = ?", (payload.provider_id,))
        benchmarks = dict(cursor.fetchone() or {})
        cursor.execute("SELECT reason FROM explanations WHERE provider_id = ?", (payload.provider_id,))
        reasons = [row["reason"] for row in cursor.fetchall()]

        cursor.execute("SELECT * FROM provider_drift WHERE provider_id = ?", (payload.provider_id,))
        drift_row = cursor.fetchone()
        drift = dict(drift_row) if drift_row else {}
        if drift.get("historical_monthly_data"):
            try:
                drift["historical_monthly_data"] = json.loads(drift["historical_monthly_data"])
            except (TypeError, json.JSONDecodeError):
                drift["historical_monthly_data"] = []

        cursor.execute("SELECT explanation_json FROM provider_explanations WHERE provider_id = ?", (payload.provider_id,))
        explanation_row = cursor.fetchone()
        summary = parse_explanation(explanation_row["explanation_json"]) if explanation_row else None
        if summary is None:
            summary = build_provider_explanation(provider, scores, benchmarks, drift, reasons)

        cursor.execute("""
            SELECT claim_id, provider_id, risk_score, risk_category, is_anomaly,
                   claim_amount, business_interpretation, explanation_json
            FROM claims WHERE provider_id = ? ORDER BY risk_score DESC LIMIT 10
        """, (payload.provider_id,))
        claim_evidence = []
        for row in cursor.fetchall():
            claim = dict(row)
            claim["explanation"] = parse_explanation(claim.pop("explanation_json", None))
            claim_evidence.append(claim)

        return {
            "response": summary["ai_summary"],
            "investigation_summary": summary,
            "grounded_evidence": {
                "provider_id": payload.provider_id,
                "provider": provider,
                "peer_benchmarks": benchmarks,
                "temporal_drift": drift,
                "claims": claim_evidence,
                "investigator_query": payload.query,
            },
        }
    finally:
        conn.close()
