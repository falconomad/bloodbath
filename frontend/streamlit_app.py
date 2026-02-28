"""Entrypoint wrapper for the dashboard UI.

Run with: `streamlit run frontend/streamlit_app.py`
"""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[1] / "app.py"), run_name="__main__")
