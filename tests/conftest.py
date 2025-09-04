import pytest
import os
import sys
import importlib

# Add the project root to the Python path before any other imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(scope="session", autouse=True)
def reload_modules():
    """
    This fixture automatically runs for the entire test session.
    It reloads key modules that might be cached with outdated versions,
    especially the settings module which is frequently changed.
    """
    if "config.settings" in sys.modules:
        importlib.reload(sys.modules["config.settings"])
    
    # You can add other modules to reload here if needed in the future
    # For example:
    # if "utils.file_utils" in sys.modules:
    #     importlib.reload(sys.modules["utils.file_utils"])

    # Yield to let the tests run with the reloaded modules
    yield

    # Teardown can go here if needed, but not necessary for this case
