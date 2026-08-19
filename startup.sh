#!/usr/bin/env bash
set -e
python generate_provider_master.py
streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true
