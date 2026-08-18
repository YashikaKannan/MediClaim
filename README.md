# MediClaim Fraud Intelligence Platform

An end-to-end healthcare claims fraud detection and audit intelligence platform combining statistical peer benchmarking, multi-signal anomaly engines, and a conversational AI audit assistant.

---

## 🚀 Quick Start Instructions

To run the application locally on your machine, follow these steps in order:

### 1. Run the Machine Learning & Feature Engineering Pipeline
If you need to recalculate features, retrain the models, or rebuild the SQLite database from the raw CSV data:
```powershell
# From the project root directory
python train_models.py
```
This generates the models in `backend/app/ml_models` and populates the database at `backend/app/db/mediclaim.db`.

---

### 2. Start the FastAPI Backend Server
Run the REST API server which loads the models and DB:
```powershell
# 1. Navigate to the backend folder
cd backend

# 2. Run the Uvicorn server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
* **API Documentation:** View endpoints at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### 3. Start the React Frontend Dashboard (Vite)
Open another terminal shell and run the React development server:
```powershell
# 1. Navigate to the frontend folder
cd frontend

# 2. Start the Vite server
npm run dev
```
* **Interactive Dashboard:** Open [http://localhost:5173/](http://localhost:5173/) in your web browser.

---

## 🛠️ Project Structure

```
├── datas/                                # Claim CSV datasets (Inpatient, Outpatient, Beneficiary)
├── backend/
│   └── app/
│       ├── main.py                       # FastAPI application & REST endpoints
│       ├── db/
│       │   └── mediclaim.db              # SQLite Database containing aggregated providers
│       └── ml_models/                    # Trained model binaries and scalers (.pth, .joblib, .cbm)
├── frontend/
│   ├── src/
│   │   ├── pages/                        # Dashboard pages (Dashboard, Queue, Assistant, etc.)
│   │   ├── components/                   # Sidebar and Header components
│   │   ├── App.tsx                       # Main React App with Client Router
│   │   └── index.css                     # Global Theme CSS & Design System
│   └── package.json                      # Frontend dependencies
├── train_models.py                       # CatBoost, PyTorch Autoencoder, and Z-Score training script
└── test_features.py                      # Feature engineering pipeline
```
