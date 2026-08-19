# MEDICLAIM — Complete Streamlit Frontend

This folder is a complete Streamlit/Python reimplementation of the specified MEDICLAIM enterprise payment-integrity frontend.

## Included

- Login + demo role differentiation
- User profile + logout
- Blue/white/navy enterprise branding
- Dashboard
- Risk distribution with percentages
- Financial leakage visualizations
- Risk prioritization
- Investigator-centric dynamic search
- Combined multi-filter investigation queue
- Provider investigation
- Claim investigation
- Peer analysis
- Provider score breakdown
- SHAP explanation interface
- Isolation Forest model adapter
- AI Investigation Assistant
- Reports: CSV, Excel, PDF
- Model Governance
- Data quality monitoring
- Business validation
- System health
- Audit log
- No Settings page
- No editable thresholds
- No workflow configuration

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Real model connection

Place the trained model at:

`models/isolation_forest.pkl`

and, if applicable, the preprocessing object at:

`models/preprocessor.pkl`

The adapter in `models/model_adapter.py` exposes:

- `load_model()`
- `preprocess_data()`
- `predict()`
- `calculate_anomaly_score()`
- `generate_risk_score()`
- `generate_shap_explanation()`

The SHAP UI is currently a clearly labeled demo adapter until a real explainer is connected. This avoids presenting fabricated explanations as real model output.

## Data

The existing React frontend's mock JSON data has been reused in `data/mock_data/`.

Governance data is in `data/governance.json`.

The supplied blue reference image is retained as `assets/blue_reference.png` for visual reference; it is not used as a giant page background.
