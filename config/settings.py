"""Central application settings for VocaResume (formerly TalentAlign AI).

Provides model registry and API key loading. Logging reduced to simple
level map to keep footprint small in Streamlit environment.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PPLX_API_KEY = os.getenv("PPLX_API_KEY") or os.getenv("PERPLEXITY_API_KEY")

# Available Models
AVAILABLE_MODELS = {
    # Google
    "Gemini 2.5 Pro": {"provider": "google", "model": "gemini-2.5-pro"},
    "Gemini 2.5 Flash": {"provider": "google", "model": "gemini-2.5-flash"},
    "Gemini 1.5 Flash": {"provider": "google", "model": "gemini-1.5-flash"},

    # Groq
    "LLaMA 4 Maverick 17B": {"provider": "groq", "model": "meta-llama/llama-4-maverick-17b-128e-instruct"},
    "LLaMA 3.1 8B": {"provider": "groq", "model": "llama-3.1-8b-instant"},
    "LLaMA 3.3 70B-Versatile": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    "DeepSeek R1 Distill LLaMA 70B": {"provider": "groq", "model": "deepseek-r1-distill-llama-70b"},

    # Perplexity (best models)
    "Perplexity Sonar Reasoning Pro": {"provider": "perplexity", "model": "sonar-reasoning-pro"},
    "Perplexity Sonar Large": {"provider": "perplexity", "model": "sonar-large"}
}

# Structured logging configuration (compatible with logging.config.dictConfig) expected by tests
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        }
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
            "stream": "ext://sys.stdout"
        }
    },
    "loggers": {
        "": {"handlers": ["console"], "level": "INFO"},
        "litellm": {"handlers": ["console"], "level": "CRITICAL", "propagate": False},
        "crewai": {"handlers": ["console"], "level": "WARNING", "propagate": False},
        "httpx": {"handlers": ["console"], "level": "WARNING", "propagate": False},
    }
}
