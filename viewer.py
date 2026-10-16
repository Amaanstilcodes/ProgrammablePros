"""Backward-compatibility wrapper for viewer.py -> viewer_local.py"""
from pathlib import Path
import runpy

if __name__ == "__main__" or True:  # Streamlit executes top-level script
    target = Path(__file__).parent / "viewer_local.py"
    runpy.run_path(str(target), run_name="__main__")
