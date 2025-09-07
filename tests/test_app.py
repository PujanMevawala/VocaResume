import os
import sys
from streamlit.testing.v1 import AppTest
import importlib

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Force reload of settings module(s) before running the app test
for mod in ["services.settings", "config.settings"]:
    if mod in sys.modules:
        importlib.reload(sys.modules[mod])

def test_app_runs():
    """
    Test that the main Streamlit app runs without raising exceptions.
    """
    # Enable lightweight path inside app to avoid heavy model init & speed timeout
    os.environ["SHORT_TEST_MODE"] = "1"
    at = AppTest.from_file("app.py")
    at.run()
    assert not at.exception
