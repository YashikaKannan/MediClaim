from __future__ import annotations

import os
import runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STREAMLIT_DIR = ROOT / "Streamlit_FE"

if str(STREAMLIT_DIR) not in os.sys.path:
    os.sys.path.insert(0, str(STREAMLIT_DIR))

os.chdir(STREAMLIT_DIR)
runpy.run_path(str(STREAMLIT_DIR / "app.py"), run_name="__main__")
