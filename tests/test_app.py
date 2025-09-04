import os
import sys
from streamlit.testing.v1 import AppTest
import importlib

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Force reload of the settings module before running the app test
if "config.settings" in sys.modules:
    importlib.reload(sys.modules["config.settings"])

def test_app_runs():
    """
    Test that the main Streamlit app runs without raising exceptions.
    """
    at = AppTest.from_file("app.py")
    at.run()
    assert not at.exception
