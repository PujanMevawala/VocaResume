"""Backward compatibility shim after folder consolidation.

Re-exports services.settings so older import paths still work:
	from config import settings
"""
try:  # pragma: no cover
	from services import settings  # type: ignore
except Exception:  # pragma: no cover
	pass
