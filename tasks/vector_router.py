"""Vector-based task routing using Chroma + SentenceTransformers.

We index:
  - Resume text (doc_type=resume)
  - Job description (doc_type=job_desc)
  - Canonical task labels & descriptions (doc_type=task_label)
  - Historical user queries (doc_type=query) (optional incremental learning)

At query time we embed the incoming query and perform similarity search over
task labels + (optionally) resume & JD for contextual grounding. The top task
label drives selection of CrewAI task index.

Public functions:
  init_router(persist_dir: str | None) -> VectorRouter

VectorRouter methods:
  ingest_resume(text: str)
  ingest_job_description(text: str)
  ensure_task_labels()
  route(query: str) -> {"task_index": int, "label": str, "score": float, "alt": list}
  add_query_history(query: str)

Task label mapping order must match tasks in task_factory.create_tasks:
  0 analysis, 1 interview, 2 suggestions, 3 job_fit
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Any
import hashlib
import os

try:  # Lazy imports (tests without chroma should still import module)
	import chromadb  # type: ignore
	from chromadb.utils import embedding_functions  # type: ignore
	_CHROMA_AVAILABLE = True
except Exception:  # pragma: no cover
	_CHROMA_AVAILABLE = False

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

TASK_LABELS = [
	(0, "analysis", "Comprehensive resume analysis vs job description with strengths, gaps, recommendations."),
	(1, "interview", "Generate technical interview questions based on resume content and experience."),
	(2, "suggestions", "Provide actionable resume improvement and optimization suggestions."),
	(3, "job_fit", "Evaluate job fit score and suitability against the provided job description."),
]


def _hash(text: str) -> str:
	return hashlib.sha1(text.encode('utf-8')).hexdigest()[:16]


@dataclass
class RouteResult:
	task_index: int
	label: str
	score: float
	alt: List[Dict[str, Any]]


class VectorRouter:
	def __init__(self, persist_dir: str | None = None):
		# Pre-initialize tracking attributes so early returns still leave object consistent
		self._fallback_keyword = False
		self._routing_backend = "keyword"
		self._warned_fallback = False
		self._route_counts: Dict[str,int] = {lbl:0 for _,lbl,_ in TASK_LABELS}
		if not _CHROMA_AVAILABLE:
			raise RuntimeError("Chroma not installed; ensure requirements updated.")
		# Use PersistentClient for durability (avoids 'no such table: collections')
		chroma_dir = persist_dir or os.getenv("CHROMA_DIR")
		try:
			if chroma_dir:
				from chromadb import PersistentClient  # type: ignore
				os.makedirs(chroma_dir, exist_ok=True)
				self.client = PersistentClient(path=chroma_dir)  # type: ignore
			else:
				from chromadb import EphemeralClient  # type: ignore
				self.client = EphemeralClient()  # type: ignore
		except Exception as e:  # pragma: no cover
			# Attempt fallback to ephemeral if persistent failed
			try:
				from chromadb import EphemeralClient  # type: ignore
				self.client = EphemeralClient()  # type: ignore
				import warnings
				warnings.warn(f"Persistent Chroma init failed ({e}); using Ephemeral in-memory store.")
			except Exception as e2:
				# Final fallback: mark as disabled and provide stub routing
				self.client = None  # type: ignore
				self.collection = None  # type: ignore
				self._fallback_keyword = True
				import warnings
				warnings.warn(f"Chroma unavailable ({e}; {e2}). Falling back to keyword routing only.")
				return
		# Embedding function with meta-tensor fallback
		class _EmbedWrapper:
			def __init__(self, inner, model_name: str):
				self._inner = inner
				self._model_name = model_name
			def __call__(self, input: list[str]):  # Updated signature per Chroma 0.4.16+
				return self._inner(input)
			def name(self) -> str:
				return self._model_name

		try:
			base_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)  
			if not hasattr(base_fn, 'name') or not callable(getattr(base_fn, 'name')):
				self.embed_fn = _EmbedWrapper(base_fn, EMBED_MODEL)
			else:
				self.embed_fn = base_fn  # type: ignore
		except Exception as e:  # meta tensor or model weight issue
			# Fallback: load a tiny model explicitly on CPU or stub encoder
			try:
				from sentence_transformers import SentenceTransformer  # type: ignore
				cpu_model_name = os.getenv("EMBED_FALLBACK_MODEL", "all-MiniLM-L6-v2")
				_st_model = SentenceTransformer(cpu_model_name, device="cpu")
				def _cpu_encode(texts: list[str]):
					return _st_model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()
				self.embed_fn = _EmbedWrapper(_cpu_encode, cpu_model_name)  # type: ignore
			except Exception as e2:
				# Final minimal fallback: hash-based pseudo embedding (not semantic but prevents crash)
				def _poor_hash_vec(txts: list[str]):
					out = []
					for t in txts:
						h = hashlib.sha1(t.encode('utf-8')).digest()
						vec = [((h[i % len(h)] / 255.0) - 0.5) for i in range(32)]
						out.append(vec)
					return out
				self.embed_fn = _EmbedWrapper(_poor_hash_vec, "hash32")  # type: ignore
				import warnings
				warnings.warn(f"SentenceTransformer unavailable ({e}; {e2}). Using hash embeddings (reduced accuracy).")
		try:
			# Prefer get_collection first to avoid embedding function mismatch warning
			try:
				self.collection = self.client.get_collection("vocaresume_router")  # type: ignore
			except Exception:
				self.collection = self.client.create_collection("vocaresume_router", embedding_function=self.embed_fn)  # type: ignore
			# If existing collection lacks embeddings due to prior bad function, recreate
			if hasattr(self.collection, 'embedding_function') and self.collection.embedding_function is None:  # type: ignore
				self.client.delete_collection("vocaresume_router")  # type: ignore
				self.collection = self.client.create_collection("vocaresume_router", embedding_function=self.embed_fn)  # type: ignore
		except Exception as e:
			# Attempt alternate client construction if _type related internal store error
			if "_type" in str(e) or "tenant" in str(e).lower():
				try:
					from chromadb.config import Settings  # type: ignore
					# Rebuild generic client (in-memory) and retry
					self.client = chromadb.Client(Settings(anonymized_telemetry=False))  # type: ignore
					self.collection = self.client.create_collection("vocaresume_router", embedding_function=self.embed_fn)  # type: ignore
				except Exception as e_alt:
					self.client = None  # type: ignore
					self.collection = None  # type: ignore
					self._fallback_keyword = True
					import warnings
					warnings.warn(f"Chroma collection unavailable after alternate attempt ({e_alt}). Using keyword routing.")
					return
			# As last resort: delete and recreate if conflict about embedding function name signature
			if 'conflict' in str(e).lower() or 'name' in str(e).lower():
				try:
					self.client.delete_collection("vocaresume_router")  # type: ignore
					self.collection = self.client.create_collection("vocaresume_router", embedding_function=self.embed_fn)  # type: ignore
				except Exception as e2:
					raise RuntimeError(f"Failed to recreate collection: {e2}")
			else:
				raise RuntimeError(f"Failed to get/create collection: {e}")

		self._fallback_keyword = False
		self._routing_backend = "chroma" if self.client and getattr(self, 'collection', None) else "keyword"

	def _upsert(self, docs: List[str], metadatas: List[Dict[str, Any]], ids: List[str]):
		"""Safe upsert with fallback switching.

		If the underlying collection is unavailable (None) switch to keyword mode.
		"""
		if getattr(self, 'collection', None) is None:
			self._fallback_keyword = True
			self._routing_backend = "keyword"
			return
		try:
			self.collection.upsert(documents=docs, metadatas=metadatas, ids=ids)  # type: ignore
		except Exception as e:  # pragma: no cover
			if not self._warned_fallback:
				import warnings
				warnings.warn(f"Upsert failed ({e}); switching to keyword routing")
				self._warned_fallback = True
			self._fallback_keyword = True
			self._routing_backend = "keyword"

	def ingest_resume(self, text: str):
		if not text: return
		self._upsert([text[:12000]], [{"doc_type": "resume"}], [f"resume-{_hash(text)}"])

	def ingest_job_description(self, text: str):
		if not text: return
		self._upsert([text[:12000]], [{"doc_type": "job_desc"}], [f"job-{_hash(text)}"])

	def ensure_task_labels(self):
		if getattr(self, '_fallback_keyword', False):
			return
		for idx, label, desc in TASK_LABELS:
			doc_id = f"task-{idx}-{label}"
			self._upsert([f"{label}: {desc}"], [{"doc_type": "task_label", "task_index": idx, "label": label}], [doc_id])

	def add_query_history(self, query: str):
		if not query: return
		self._upsert([query], [{"doc_type": "query"}], [f"q-{_hash(query)}"])

	def route(self, query: str, k: int = 4) -> RouteResult:
		if not query:
			return RouteResult(0, "analysis", 0.0, [])
		if getattr(self, '_fallback_keyword', False):  # simple keyword heuristic if vector routing disabled
			q = query.lower()
			if any(w in q for w in ["interview","question"]):
				self._route_counts['interview'] += 1
				return RouteResult(1, "interview", 0.5, [])
			if any(w in q for w in ["suggest","improve","optimiz"]):
				self._route_counts['suggestions'] += 1
				return RouteResult(2, "suggestions", 0.5, [])
			if any(w in q for w in ["fit","score","match"]):
				self._route_counts['job_fit'] += 1
				return RouteResult(3, "job_fit", 0.5, [])
			self._route_counts['analysis'] += 1
			return RouteResult(0, "analysis", 0.5, [])
		self.ensure_task_labels()
		if getattr(self, 'collection', None) is None:
			# Collection disappeared mid-run; degrade gracefully
			self._fallback_keyword = True
			self._routing_backend = "keyword"
			return self.route(query, k)
		try:
			results = self.collection.query(query_texts=[query], n_results=k)  # type: ignore
		except Exception as e:
			if not self._warned_fallback:
				import warnings
				warnings.warn(f"Vector query failed ({e}); switching to keyword routing.")
				self._warned_fallback = True
			self._fallback_keyword = True
			self._routing_backend = "keyword"
			return self.route(query, k)
		# Process vector results
		ids = results.get('ids', [[]])[0]
		metas = results.get('metadatas', [[]])[0]
		# distances may vary; some versions return distances or embeddings
		dists = results.get('distances', [[]])[0] or results.get('embeddings', [])
		candidates: List[Dict[str, Any]] = []
		for i, m, d in zip(ids, metas, dists):
			if not m:
				continue
			if m.get('doc_type') == 'task_label':
				score = 1 - d if isinstance(d, (int, float)) else 0.0
				candidates.append({
					"task_index": m.get('task_index', 0),
					"label": m.get('label', 'analysis'),
					"score": score
				})
		if not candidates:
			self._route_counts['analysis'] += 1
			return RouteResult(0, "analysis", 0.0, [])
		candidates.sort(key=lambda x: x['score'], reverse=True)
		top = candidates[0]
		alt = candidates[1:]
		self.add_query_history(query)
		self._route_counts[top['label']] = self._route_counts.get(top['label'],0) + 1
		return RouteResult(top['task_index'], top['label'], top['score'], alt)

	def routing_backend(self) -> str:
		"""Return active backend mode (chroma|keyword)."""
		return getattr(self, '_routing_backend', 'keyword')

	def stats(self) -> Dict[str,int]:
		"""Return cumulative route counts by label."""
		return dict(self._route_counts)


def init_router(persist_dir: str | None = None) -> VectorRouter:
	return VectorRouter(persist_dir=persist_dir)

__all__ = ["init_router", "VectorRouter", "RouteResult", "TASK_LABELS"]
