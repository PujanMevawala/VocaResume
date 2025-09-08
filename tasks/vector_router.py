"""Lightweight semantic task router using ChromaDB (if available).

Purpose: Provide more accurate mapping from a free-form user query to one of the
existing task indices (0=analysis,1=interview,2=suggestions,3=job_fit).

If chromadb or sentence-transformers not installed, gracefully degrade and the
keyword fallback in `get_task_from_query` will be used.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Optional

try:
    import chromadb  # type: ignore
    from chromadb.utils import embedding_functions  # type: ignore
    _CHROMA_AVAILABLE = True
except Exception:  # pragma: no cover
    _CHROMA_AVAILABLE = False

_TASK_PROTOTYPES = [
    (0, "analysis", "Comprehensive resume vs job description analysis with strengths, gaps, and recommendations"),
    (1, "interview", "In-depth technical interview questions strictly derived from the candidate resume technologies and implementations"),
    (2, "suggestions", "Actionable resume improvement suggestions and optimization guidance"),
    (3, "job_fit", "Job fit scoring and suitability assessment with a quantified score and reasoning")
]

@dataclass
class RouteMatch:
    index: int
    label: str
    score: float

class VectorTaskRouter:
    def __init__(self, collection_name: str = "vocaresume_tasks"):
        if not _CHROMA_AVAILABLE:
            raise RuntimeError("Chroma not available")
        client = chromadb.Client()  # in-memory default
        self._embed = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        if collection_name in [c.name for c in client.list_collections()]:  # type: ignore
            self.col = client.get_collection(collection_name)  # type: ignore
        else:  # seed collection
            self.col = client.create_collection(collection_name, embedding_function=self._embed)  # type: ignore
            self._seed()

    def _seed(self):
        ids = [f"task_{i}" for i,_,_ in _TASK_PROTOTYPES]
        metadatas = [{"index": i, "label": label} for i,label,_ in _TASK_PROTOTYPES]
        documents = [desc for _,_,desc in _TASK_PROTOTYPES]
        self.col.add(ids=ids, metadatas=metadatas, documents=documents)  # type: ignore

    def route(self, query: str) -> Optional[Dict]:
        if not query.strip():
            return None
        r = self.col.query(query_texts=[query], n_results=1)  # type: ignore
        if not r or not r.get('metadatas'):
            return None
        meta = r['metadatas'][0][0]
        dist = r.get('distances', [[1]])[0][0]
        score = 1 - float(dist)
        return {"index": meta['index'], "label": meta['label'], "score": score}

# Convenience factory

def build_router_if_possible():
    if not _CHROMA_AVAILABLE:
        return None
    try:
        return VectorTaskRouter()
    except Exception:
        return None
