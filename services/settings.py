"""Unified settings (migrated from config/settings.py).
Backward-compatible placeholder; original tests may import config.settings.
We'll keep a thin shim there until tests updated.
"""
from __future__ import annotations
import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PPLX_API_KEY = os.getenv("PPLX_API_KEY") or os.getenv("PERPLEXITY_API_KEY")

AVAILABLE_MODELS = {
    "Gemini 2.5 Pro": {"provider": "google", "model": "gemini-2.5-pro"},
    "Gemini 2.5 Flash": {"provider": "google", "model": "gemini-2.5-flash"},
    "Gemini 1.5 Flash": {"provider": "google", "model": "gemini-1.5-flash"},
    "LLaMA 4 Maverick 17B": {"provider": "groq", "model": "meta-llama/llama-4-maverick-17b-128e-instruct"},
    "LLaMA 3.1 8B": {"provider": "groq", "model": "llama-3.1-8b-instant"},
    "LLaMA 3.3 70B-Versatile": {"provider": "groq", "model": "llama-3.3-70b-versatile"},
    "DeepSeek R1 Distill LLaMA 70B": {"provider": "groq", "model": "deepseek-r1-distill-llama-70b"},
    "Perplexity Sonar Reasoning Pro": {"provider": "perplexity", "model": "sonar-reasoning-pro"},
    "Perplexity Sonar Large": {"provider": "perplexity", "model": "sonar-large"}
}

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"standard": {"format": "%(asctime)s | %(levelname)s | %(name)s | %(message)s"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "standard", "level": "INFO", "stream": "ext://sys.stdout"}},
    "loggers": {
        "": {"handlers": ["console"], "level": "INFO"},
        "litellm": {"handlers": ["console"], "level": "CRITICAL", "propagate": False},
        "crewai": {"handlers": ["console"], "level": "WARNING", "propagate": False},
        "httpx": {"handlers": ["console"], "level": "WARNING", "propagate": False},
    }
}
