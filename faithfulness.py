"""
Faithfulness / hallucination detection for financial RAG.

Verifies that LLM-generated claims are supported by the retrieved source documents.
Uses NLI-based verification when a model is available, plus rule-based checks.
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field

from langchain.schema import Document

logger = logging.getLogger(__name__)

# ── NLI model (lazy loaded) ─────────────────────────────────────────────
_nli_model = None
_NLI_MODEL_NAME = "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"  # Zero-shot NLI


def _load_nli_model():
    """Lazy-load the NLI model for faithfulness checking."""
    global _nli_model
    if _nli_model is not None:
        return _nli_model
    try:
        from transformers import pipeline
        _nli_model = pipeline(
            "zero-shot-classification",
            model=_NLI_MODEL_NAME,
            device=-1,  # CPU
        )
        logger.info(f"Loaded NLI model: {_NLI_MODEL_NAME}")
    except Exception as e:
        logger.warning(f"Failed to load NLI model '{_NLI_MODEL_NAME}': {e}")
        _nli_model = False
    return _nli_model


@dataclass
class Claim:
    """A single claim extracted from the generated answer."""
    text: str
    position: int  # char position in original text


@dataclass
class ClaimVerification:
    """Verification result for a single claim."""
    claim: str
    verdict: str  # "SUPPORTED" | "CONTRADICTED" | "NOT_VERIFIABLE"
    confidence: float
    evidence: Optional[str] = None


@dataclass
class FaithfulnessResult:
    """Overall faithfulness assessment."""
    total_claims: int
    supported: int
    contradicted: int
    not_verifiable: int
    faithfulness_score: float  # 0.0 - 1.0
    verdicts: List[ClaimVerification] = field(default_factory=list)
    summary: str = ""


# ── Claim extraction ────────────────────────────────────────────────────
_NUMERIC_PATTERN = re.compile(
    r'[\$€£]?\s*[\d,]+(?:\.\d+)?\s*(?:billion|million|thousand|%|percent|B|M|K)?',
    re.IGNORECASE
)


def extract_claims(text: str) -> List[Claim]:
    """Extract factual claims from generated text.

    Splits on sentence boundaries and keeps sentences with potential
    factual content (numbers, financial terms, comparisons).
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    claims = []
    pos = 0

    fin_terms = {"revenue", "profit", "loss", "increased", "decreased",
                 "grew", "declined", "margin", "ratio", "percent", "%",
                 "billion", "million", "total", "net", "operating"}

    for sent in sentences:
        sent = sent.strip()
        if not sent or len(sent) < 10:
            pos += len(sent) + 1
            continue

        # Only keep sentences with factual indicators
        has_number = bool(_NUMERIC_PATTERN.search(sent))
        has_fin_term = any(term in sent.lower() for term in fin_terms)

        if has_number or has_fin_term:
            claims.append(Claim(text=sent, position=pos))

        pos += len(sent) + 1

    return claims


# ── Evidence retrieval ──────────────────────────────────────────────────
def find_evidence(claim: str, source_texts: List[str]) -> Optional[Tuple[str, float]]:
    """Find the most relevant evidence for a claim in the source texts.

    Returns (best_evidence_text, relevance_score) or None.
    """
    best_score = 0.0
    best_text = None

    claim_lower = claim.lower()
    claim_words = set(re.sub(r'[^\w\s]', ' ', claim_lower).split())

    for source in source_texts:
        source_lower = source.lower()
        # Word overlap score
        source_words = set(re.sub(r'[^\w\s]', ' ', source_lower).split())
        overlap = len(claim_words & source_words)
        if len(claim_words) > 0:
            score = overlap / len(claim_words)
        else:
            score = 0.0

        # Bonus for numeric overlap
        claim_numbers = set(_NUMERIC_PATTERN.findall(claim))
        source_numbers = set(_NUMERIC_PATTERN.findall(source))
        if claim_numbers and source_numbers:
            num_overlap = len(claim_numbers & source_numbers)
            score += num_overlap * 0.1

        if score > best_score:
            best_score = score
            best_text = source[:500]  # Truncate to first 500 chars

    if best_text and best_score > 0.1:
        return (best_text, best_score)
    return None


# ── NLI verification ────────────────────────────────────────────────────
def verify_with_nli(claim: str, evidence: str) -> Tuple[str, float]:
    """Use NLI model to verify a claim against evidence.

    Returns (verdict, confidence).
    """
    model = _load_nli_model()
    if not model:
        return ("NOT_VERIFIABLE", 0.0)

    try:
        result = model(
            f"{evidence} </s> {claim}",
            ["entailment", "contradiction", "neutral"],
            multi_label=False,
        )
        labels = result["labels"]
        scores = result["scores"]

        label_map = {"entailment": "SUPPORTED", "contradiction": "CONTRADICTED", "neutral": "NOT_VERIFIABLE"}
        verdict = label_map.get(labels[0], "NOT_VERIFIABLE")
        confidence = scores[0]
        return (verdict, float(confidence))
    except Exception as e:
        logger.warning(f"NLI verification failed: {e}")
        return ("NOT_VERIFIABLE", 0.0)


# ── Rule-based verification ─────────────────────────────────────────────
def verify_rule_based(claim: str, source_texts: List[str]) -> Tuple[str, float]:
    """Rule-based claim verification using lexical overlap.

    For financial claims, checks if the key numbers and metrics mentioned
    in the claim actually appear in the source documents.
    """
    evidence = find_evidence(claim, source_texts)
    if not evidence:
        return ("NOT_VERIFIABLE", 0.0)

    evidence_text, score = evidence

    # Check specific numbers
    claim_numbers = re.findall(r'[\d,]+(?:\.\d+)?', claim)
    if claim_numbers:
        matched = sum(1 for n in claim_numbers if n in evidence_text)
        ratio = matched / len(claim_numbers)
        if ratio >= 0.8:
            return ("SUPPORTED", ratio)
        elif ratio <= 0.2:
            return ("CONTRADICTED", 1.0 - ratio)

    # Score from word overlap
    if score >= 0.5:
        return ("SUPPORTED", score)
    elif score >= 0.3:
        return ("NOT_VERIFIABLE", score)
    else:
        return ("NOT_VERIFIABLE", score)


# ── Main faithfulness check ─────────────────────────────────────────────
def check_faithfulness(
    answer: str,
    source_documents: List[Document],
    use_nli: bool = True,
) -> FaithfulnessResult:
    """Check faithfulness of a generated answer against source documents.

    Args:
        answer: The LLM-generated answer text.
        source_documents: The retrieved source documents.
        use_nli: Whether to use NLI model (slower but more accurate).

    Returns:
        FaithfulnessResult with per-claim verdicts and aggregate score.
    """
    source_texts = [doc.page_content for doc in source_documents]

    # Extract claims
    claims = extract_claims(answer)
    if not claims:
        return FaithfulnessResult(
            total_claims=0,
            supported=0,
            contradicted=0,
            not_verifiable=0,
            faithfulness_score=1.0,
            summary="No factual claims detected to verify.",
        )

    # Verify each claim
    verdicts: List[ClaimVerification] = []
    for claim in claims:
        if use_nli:
            evidence = find_evidence(claim.text, source_texts)
            if evidence:
                verdict, conf = verify_with_nli(claim.text, evidence[0])
            else:
                verdict, conf = "NOT_VERIFIABLE", 0.0
        else:
            verdict, conf = verify_rule_based(claim.text, source_texts)

        verdicts.append(ClaimVerification(
            claim=claim.text,
            verdict=verdict,
            confidence=conf,
            evidence=evidence if use_nli else None,
        ))

    # Aggregate
    total = len(verdicts)
    supported = sum(1 for v in verdicts if v.verdict == "SUPPORTED")
    contradicted = sum(1 for v in verdicts if v.verdict == "CONTRADICTED")
    not_verifiable = sum(1 for v in verdicts if v.verdict == "NOT_VERIFIABLE")
    faithfulness_score = supported / total if total > 0 else 1.0

    summary_parts = [
        f"Faithfulness: {faithfulness_score:.0%} ({supported}/{total} claims supported)"
    ]
    if contradicted:
        summary_parts.append(f", {contradicted} contradicted")
        for v in verdicts:
            if v.verdict == "CONTRADICTED":
                summary_parts.append(f"\n  - ⚠️ {v.claim[:80]}...")

    return FaithfulnessResult(
        total_claims=total,
        supported=supported,
        contradicted=contradicted,
        not_verifiable=not_verifiable,
        faithfulness_score=faithfulness_score,
        verdicts=verdicts,
        summary="".join(summary_parts),
    )


def format_faithfulness_report(result: FaithfulnessResult) -> str:
    """Format faithfulness check as a readable report."""
    lines = [
        "## Faithfulness Verification",
        f"Score: {result.faithfulness_score:.0%}",
        f"Claims: {result.total_claims} total, {result.supported} supported, "
        f"{result.contradicted} contradicted, {result.not_verifiable} not verifiable",
    ]

    if result.contradicted > 0:
        lines.append("\n### ⚠️ Potentially Hallucinated Claims")
        for v in result.verdicts:
            if v.verdict == "CONTRADICTED":
                lines.append(f"- {v.claim[:120]}")

    if result.not_verifiable > 0:
        lines.append("\n### Unverifiable Claims")
        for v in result.verdicts:
            if v.verdict == "NOT_VERIFIABLE":
                lines.append(f"- {v.claim[:120]}")

    return "\n".join(lines)
