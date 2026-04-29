"""
Query enhancement module for financial RAG.

Provides:
1. HyDE (Hypothetical Document Embedding) — generates a synthetic answer
   paragraph to use as the retrieval query instead of the raw question.
2. Query rewriting — expands abbreviations, normalizes financial terms.
3. Multi-query expansion — generates multiple reformulations for broader recall.
"""
import re
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# ── Financial abbreviation expansion ─────────────────────────────────────
FINANCIAL_ABBREVIATIONS = {
    r"\bYoY\b": "year over year",
    r"\bQoQ\b": "quarter over quarter",
    r"\bEBITDA\b": "Earnings Before Interest, Taxes, Depreciation, and Amortization",
    r"\bEPS\b": "earnings per share",
    r"\bP/E\b": "price to earnings",
    r"\bROE\b": "return on equity",
    r"\bROA\b": "return on assets",
    r"\bROI\b": "return on investment",
    r"\bEBIT\b": "Earnings Before Interest and Taxes",
    r"\bGAAP\b": "Generally Accepted Accounting Principles",
    r"\bFCF\b": "free cash flow",
    r"\bCAPEX\b": "capital expenditures",
    r"\bCAGR\b": "compound annual growth rate",
    r"\bD\&A\b": "depreciation and amortization",
    r"\bDCF\b": "discounted cash flow",
    r"\bSG\&A\b": "Selling, General and Administrative",
    r"\bCOGS\b": "Cost of Goods Sold",
    r"\bR\&D\b": "research and development",
    r"\bOpEx\b": "operating expenses",
    r"\bYTD\b": "year to date",
    r"\bLTM\b": "last twelve months",
    r"\bNRV\b": "net realizable value",
    r"\bNOPAT\b": "net operating profit after tax",
    r"\bWACC\b": "weighted average cost of capital",
    r"\bIPO\b": "initial public offering",
    r"\bM\&A\b": "mergers and acquisitions",
}

COMPANY_ABBREVIATIONS = {
    r"\bMSFT\b": "Microsoft",
    r"\bAAPL\b": "Apple",
    r"\bGOOGL?\b": "Alphabet (Google)",
    r"\bAMZN\b": "Amazon",
    r"\bMETA\b": "Meta (Facebook)",
    r"\bNVDA\b": "Nvidia",
    r"\bTSLA\b": "Tesla",
    r"\bJPM\b": "JPMorgan Chase",
    r"\bBAC\b": "Bank of America",
    r"\bWMT\b": "Walmart",
    r"\bJNJ\b": "Johnson & Johnson",
    r"\bVZ\b": "Verizon",
    r"\bT\b(?!\w)": "AT&T",
    r"\bDIS\b": "Walt Disney",
    r"\bKO\b": "Coca-Cola",
}


def expand_abbreviations(query: str) -> str:
    """Expand financial and company abbreviations in the query."""
    expanded = query
    for pattern, replacement in FINANCIAL_ABBREVIATIONS.items():
        expanded = re.sub(pattern, replacement, expanded)
    for pattern, replacement in COMPANY_ABBREVIATIONS.items():
        expanded = re.sub(pattern, replacement, expanded)
    if expanded != query:
        logger.debug(f"Expanded abbreviations: {query} -> {expanded}")
    return expanded


def normalize_financial_query(query: str) -> str:
    """Normalize financial terms for better retrieval."""
    normalized = query.strip()

    # Standardize comparative periods
    normalized = re.sub(r"this\s+year\s+(over|vs|versus)\s+last\s+year",
                        "year over year change", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"this\s+quarter\s+(over|vs|versus)\s+last\s+quarter",
                        "quarter over quarter change", normalized, flags=re.IGNORECASE)

    # Normalize currency references
    normalized = re.sub(r"\bin\s+USD\b", "in US dollars", normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"\$\s*(\d+)\s*([BMK])\b",
                        lambda m: f"${m.group(1)} {'billion' if m.group(2) == 'B' else 'million' if m.group(2) == 'M' else 'thousand'}",
                        normalized)

    return normalized


def generate_hypothetical_answer(query: str, llm=None) -> Optional[str]:
    """Generate a hypothetical answer paragraph for HyDE retrieval.

    Uses the LLM if provided, otherwise returns None (caller falls back to raw query).
    """
    if not llm:
        return None

    hyde_prompt = (
        "You are a financial analyst. Given a question about a company's 10-K report, "
        "write a concise, factual paragraph that would answer it. "
        "Use specific financial terms and numbers where appropriate. "
        "Do not mention that you lack real data — write as if you have the report.\n\n"
        f"Question: {query}\n\nHypothetical Answer:"
    )

    try:
        # Wrap for ChatGoogleGenerativeAI / ChatOpenAI message format
        from langchain.schema import HumanMessage
        result = llm([HumanMessage(content=hyde_prompt)])
        answer = result.content if hasattr(result, 'content') else str(result)
        logger.debug(f"HyDE generated: {answer[:100]}...")
        return answer
    except Exception as e:
        logger.warning(f"HyDE generation failed: {e}")
        return None


def generate_multi_queries(query: str, num_queries: int = 3) -> List[str]:
    """Generate multiple query reformulations for broader recall.

    Uses rule-based reformulation (no LLM call needed).
    """
    queries = [query]
    expanded = expand_abbreviations(query)

    if expanded != query:
        queries.append(expanded)

    normalized = normalize_financial_query(query)
    if normalized != query and normalized not in queries:
        queries.append(normalized)

    # Reformulate as a search for specific metric
    metric_patterns = [
        (r"(?:what\s+is|what\s+are)\s+(.+?)(?:\?|$)", r"\1"),
        (r"(?:how\s+much|how\s+many)\s+(.+?)(?:\?|$)", r"\1"),
        (r"(?:tell\s+me\s+about)\s+(.+?)(?:\?|$)", r"\1"),
    ]
    for pattern, replacement in metric_patterns:
        m = re.search(pattern, query, re.IGNORECASE)
        if m:
            reformulated = re.sub(pattern, replacement, query, flags=re.IGNORECASE).strip()
            if reformulated and reformulated not in queries:
                queries.append(reformulated)

    return queries[:num_queries]


def enhance_query(query: str, llm=None, use_hyde: bool = False) -> dict:
    """Full query enhancement pipeline.

    Returns:
        dict with:
        - original: raw query
        - expanded: abbreviation-expanded query
        - normalized: financially normalized query
        - hyde: hypothetical document (if use_hyde and llm provided)
        - multi_queries: list of reformulations for multi-query retrieval
        - best: the single best query string for standard retrieval
    """
    expanded = expand_abbreviations(query)
    normalized = normalize_financial_query(expanded)

    result = {
        "original": query,
        "expanded": expanded,
        "normalized": normalized,
        "hyde": None,
        "multi_queries": generate_multi_queries(normalized),
        "best": normalized,
    }

    if use_hyde and llm:
        hyde = generate_hypothetical_answer(query, llm)
        if hyde:
            result["hyde"] = hyde
            result["best"] = hyde  # HyDE replaces the query for retrieval

    return result
