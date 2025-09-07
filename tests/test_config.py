import os
import sys
import pytest
from unittest.mock import patch
import importlib
# Updated import path after consolidation
try:
    from services import settings as _settings
except ImportError:
    # Fallback for backward compatibility if shim present
    from config import settings as _settings

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

importlib.reload(_settings)
settings = _settings

# This is a workaround for pytest's module caching.
# By reading the file directly, we ensure we are testing the *current* state
# of the settings file, not a cached version.
def get_settings_from_file():
    # Adjust to new location
    new_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'services', 'settings.py'))
    legacy_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config', 'settings.py'))
    settings_path = new_path if os.path.exists(new_path) else legacy_path
    with open(settings_path, 'r') as f:
        content = f.read()
    
    # Execute the file content in a new dictionary to capture the variables
    settings_vars = {}
    exec(content, settings_vars)
    return settings_vars

settings_dict = get_settings_from_file()
LOGGING_CONFIG = settings_dict.get('LOGGING_CONFIG', {})
AVAILABLE_MODELS = settings_dict.get('AVAILABLE_MODELS', {})

def test_api_keys_load_from_env():
    """
    Test that API keys are loaded correctly from environment variables.
    This test still needs to patch os.environ and reload the original module.
    """
    try:
        from services import settings
    except ImportError:
        from config import settings
    import importlib

    with patch.dict(os.environ, {
        "GROQ_API_KEY": "test_groq_key",
        "GOOGLE_API_KEY": "test_google_key",
        "PPLX_API_KEY": "test_pplx_key"
    }):
        # Reload the original settings module to test the dotenv loading logic
        importlib.reload(settings)

        assert settings.GROQ_API_KEY == "test_groq_key"
        assert settings.GOOGLE_API_KEY == "test_google_key"
        assert settings.PPLX_API_KEY == "test_pplx_key"

def test_available_models_structure():
    """
    Test that the AVAILABLE_MODELS dictionary has the expected structure.
    """
    assert isinstance(AVAILABLE_MODELS, dict)
    for key, val in AVAILABLE_MODELS.items():
        assert isinstance(key, str)
        assert isinstance(val, dict)
        assert 'provider' in val and 'model' in val

def test_logging_config_structure():
    """
    Test that the LOGGING_CONFIG dictionary has the expected structure for logging.config.dictConfig.
    """
    assert isinstance(LOGGING_CONFIG, dict)
    assert 'version' in LOGGING_CONFIG
    assert 'formatters' in LOGGING_CONFIG
    assert 'handlers' in LOGGING_CONFIG
    assert 'loggers' in LOGGING_CONFIG
