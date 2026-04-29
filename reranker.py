"""
Cross-encoder reranker for financial RAG.

Reranks retrieved documents using a cross-encoder model for more accurate
relevance scoring. Falls back to a lightweight scoring function when the
cross-encoder model is unavailable.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple

from langchain.schema import Document

logger = logging.getLogger(__name__)

# ── Cross-encoder loader (lazy) ──────────────────────────────────────────
_reranker_model = None
_RERANKER_MODEL_NAME = "BAAI/bge-reranker-v2-m3"


def _load_reranker():
    """Lazy-load the cross-encoder model."""
    global _reranker_model
    if _reranker_model is not None:
        return _reranker_model
    try:
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder(_RERANKER_MODEL_NAME)
        logger.info(f"Loaded reranker: {_RERANKER_MODEL_NAME}")
    except Exception as e:
        logger.warning(f"Failed to load reranker model '{_RERANKER_MODEL_NAME}': {e}")
        _reranker_model = False  # sentinel
    return _reranker_model


# ── Lightweight keyword-based scoring (fallback) ─────────────────────────
FINANCIAL_KEYWORDS = {
    "revenue", "income", "profit", "loss", "asset", "liability", "equity",
    "cash", "debt", "expense", "margin", "ratio", "earnings", "share",
    "dividend", "tax", "depreciation", "amortization", "inventory",
    "receivable", "payable", "goodwill", "intangible", "risk", "factor",
    "management", "strategy", "competition", "regulation", "compliance",
    "growth", "decline", "trend", "forecast", "guidance", "outlook",
    "liquidity", "solvency", "capital", "investment", "operating",
    "financing", "segment", "subsidiary", "acquisition", "divestiture",
}


def _keyword_score(query: str, text: str) -> float:
    """Simple keyword overlap score as fallback reranker."""
    query_lower = query.lower()
    text_lower = text.lower()
    query_terms = set(re.sub(r'[^\w\s]', ' ', query_lower).split())
    text_terms = set(re.sub(r'[^\w\s]', ' ', text_lower).split())

    # Exact term overlap
    overlap = len(query_terms & text_terms)
    if overlap == 0:
        return 0.0

    # Bonus for financial keyword matches
    fin_overlap = len(query_terms & FINANCIAL_KEYWORDS & text_terms)

    return overlap + fin_overlap * 0.5


def rerank_documents(
    query: str,
    documents: List[Document],
    top_k: int = 5,
    use_cross_encoder: bool = True,
) -> List[Tuple[Document, float]]:
    """Rerank documents by relevance to the query.

    Args:
        query: The user's question.
        documents: List of retrieved documents.
        top_k: Number of top documents to return.
        use_cross_encoder: Whether to attempt cross-encoder reranking.

    Returns:
        List of (document, score) tuples sorted by relevance descending.
    """
    if not documents:
        return []

    scored: List[Tuple[Document, float]] = []

    if use_cross_encoder:
        model = _load_reranker()
        if model:
            try:
                pairs = [[query, doc.page_content[:512]] for doc in documents]
                scores = model.predict(pairs)
                for doc, score in zip(documents, scores):
                    scored.append((doc, float(score)))
                scored.sort(key=lambda x: x[1], reverse=True)
                result = scored[:top_k]
                logger.debug(f"Cross-encoder reranked {len(documents)} -> {top_k} docs")
                return result
            except Exception as e:
                logger.warning(f"Cross-encoder reranking failed: {e}, falling back to keyword scoring")

    # Fallback: keyword scoring
    import re
    for doc in documents:
        score = _keyword_score(query, doc.page_content)
        scored.append((doc, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    result = scored[:top_k]
    logger.debug(f"Keyword reranked {len(documents)} -> {top_k} docs")
    return result


def rerank_texts(
    query: str,
    texts: List[str],
    metadata: Optional[List[Dict]] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Rerank plain text strings with optional metadata.

    Returns list of dicts: {"text": str, "score": float, "metadata": dict}
    """
    docs = [Document(page_content=t, metadata=(metadata[i] if metadata else {}))
            for i, t in enumerate(texts)]
    results = rerank_documents(query, docs, top_k=top_k)

    return [
        {
            "text": doc.page_content,
            "score": score,
            "metadata": doc.metadata,
        }
        for doc, score in results
    ]
