"""
RAG evaluation pipeline for financial RAG system.

Integrates RAGAS metrics for systematic evaluation:
- Faithfulness: is the answer supported by retrieved context?
- Answer Relevancy: how relevant is the answer to the question?
- Context Precision: what fraction of retrieved docs are actually relevant?
- Context Recall: are all necessary facts present in the retrieved context?

Usage:
    python evaluation_pipeline.py                          # run default eval
    python evaluation_pipeline.py --questions qs.json      # custom questions
    python evaluation_pipeline.py --sample 20              # sample of test set
"""
import os
import sys
import json
import logging
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class EvalSample:
    """A single evaluation sample."""
    question: str
    ground_truth: str  # ideal answer
    contexts: List[str] = field(default_factory=list)
    answer: str = ""
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0


# ── Default test questions for 10-K financial analysis ──────────────────
DEFAULT_TEST_QUESTIONS = [
    # Revenue-related
    {
        "question": "What was the total revenue for the fiscal year?",
        "ground_truth": "The total revenue should be reported in the income statement section of the 10-K filing."
    },
    {
        "question": "How has revenue changed compared to the previous year?",
        "ground_truth": "The year-over-year revenue change can be found by comparing the current and prior year income statements."
    },
    # Risk-related
    {
        "question": "What are the key risk factors disclosed by the company?",
        "ground_truth": "Key risk factors are disclosed in Item 1A (Risk Factors) of the 10-K filing."
    },
    {
        "question": "Is there any going concern uncertainty mentioned?",
        "ground_truth": "Going concern uncertainties, if any, are typically discussed in the Risk Factors section or the auditor's report."
    },
    # Financial health
    {
        "question": "What is the company's debt-to-equity ratio?",
        "ground_truth": "The debt-to-equity ratio can be calculated from the balance sheet, using total liabilities divided by shareholders' equity."
    },
    {
        "question": "What is the current ratio?",
        "ground_truth": "The current ratio is calculated as current assets divided by current liabilities, reported on the balance sheet."
    },
    {
        "question": "What was net income for the period?",
        "ground_truth": "Net income is reported on the income statement in the financial statements section (Item 8) of the 10-K."
    },
    # Segment / MD&A
    {
        "question": "What are the company's operating segments?",
        "ground_truth": "Operating segments are described in the notes to the financial statements and the MD&A section."
    },
    {
        "question": "What is management's outlook for the next fiscal year?",
        "ground_truth": "Management's outlook is typically discussed in the MD&A (Item 7) section of the 10-K."
    },
    # Competition / market
    {
        "question": "Who are the company's main competitors?",
        "ground_truth": "Competitors are usually listed in the Business section (Item 1) and the Risk Factors section (Item 1A)."
    },
    {
        "question": "What is the company's market share or competitive position?",
        "ground_truth": "Market share and competitive position are typically discussed in Item 1 (Business) of the 10-K."
    },
    # Cash flow
    {
        "question": "What was free cash flow for the year?",
        "ground_truth": "Free cash flow can be derived from the cash flow statement, typically as operating cash flow minus capital expenditures."
    },
    {
        "question": "How much cash and cash equivalents does the company hold?",
        "ground_truth": "Cash and cash equivalents are reported on the balance sheet as current assets."
    },
]


def run_ragas_evaluation(
    questions: Optional[List[Dict[str, str]]] = None,
    rag_core=None,
    sample_size: Optional[int] = None,
) -> Dict[str, Any]:
    """Run full RAGAS evaluation on the RAG system.

    Args:
        questions: List of {"question": str, "ground_truth": str} dicts.
        rag_core: An initialized RAGChatbotCore instance.
        sample_size: Random sample of questions to use.

    Returns:
        dict with average metrics and per-sample results.
    """
    if questions is None:
        questions = DEFAULT_TEST_QUESTIONS

    if sample_size:
        import random
        questions = random.sample(questions, min(sample_size, len(questions)))

    if rag_core is None or not rag_core.qa_chain:
        logger.info("No RAG core provided, running context-only evaluation")
        return _run_context_only_eval(questions)

    results = []
    for q in questions:
        logger.info(f"Evaluating: {q['question'][:60]}...")
        sample = EvalSample(
            question=q["question"],
            ground_truth=q["ground_truth"],
        )

        try:
            result = rag_core.ask_question(q["question"])
            sample.answer = result.get("answer", "")

            # Collect source contexts
            source_docs = result.get("source_documents", [])
            if source_docs:
                sample.contexts = [doc.page_content[:1000] for doc in source_docs[:5]]

            # Store raw data for RAGAS processing
            results.append(sample)

        except Exception as e:
            logger.error(f"Error evaluating '{q['question'][:40]}': {e}")
            results.append(sample)

        time.sleep(0.5)  # Rate limiting

    return _compute_ragas_metrics(results, questions)


def _compute_ragas_metrics(results: List[EvalSample], questions: List[Dict]) -> Dict[str, Any]:
    """Compute RAGAS metrics using available scoring methods.

    When the full ragas library is available, uses it. Otherwise falls back
    to heuristic-based scoring.
    """
    try:
        from datasets import Dataset

        # Prepare data for RAGAS
        data = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        gt_lookup = {q["question"]: q["ground_truth"] for q in questions}
        for r in results:
            data["question"].append(r.question)
            data["answer"].append(r.answer)
            data["contexts"].append(r.contexts if r.contexts else ["No context retrieved"])
            data["ground_truth"].append(gt_lookup.get(r.question, ""))

        dataset = Dataset.from_dict(data)

        # Try to use RAGAS metrics
        metrics = _compute_with_ragas(dataset)
        return metrics

    except ImportError:
        logger.warning("RAGAS library not fully available, using heuristic evaluation")
        return _heuristic_evaluation(results, questions)
    except Exception as e:
        logger.error(f"RAGAS computation failed: {e}")
        return _heuristic_evaluation(results, questions)


def _compute_with_ragas(dataset) -> Dict[str, Any]:
    """Compute RAGAS metrics from the dataset."""
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    )

    score = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    result = score.to_pandas().to_dict(orient="records")

    avg = {
        "faithfulness": sum(r.get("faithfulness", 0) for r in result) / len(result) if result else 0,
        "answer_relevancy": sum(r.get("answer_relevancy", 0) for r in result) / len(result) if result else 0,
        "context_precision": sum(r.get("context_precision", 0) for r in result) / len(result) if result else 0,
        "context_recall": sum(r.get("context_recall", 0) for r in result) / len(result) if result else 0,
    }

    return {
        "average": avg,
        "samples": result,
        "num_samples": len(result),
        "method": "ragas",
    }


def _heuristic_evaluation(results: List[EvalSample], questions: List[Dict]) -> Dict[str, Any]:
    """Fallback heuristic evaluation when RAGAS library is unavailable."""
    total = len(results)
    if total == 0:
        return {"average": {}, "samples": [], "num_samples": 0, "method": "heuristic"}

    gt_lookup = {q["question"]: q["ground_truth"] for q in questions}
    scores = []

    for r in results:
        gt = gt_lookup.get(r.question, "").lower()
        answer_lower = r.answer.lower()
        contexts_text = " ".join(r.contexts).lower()

        # Faithfulness heuristic: do key terms from answer appear in context?
        answer_terms = set(answer_lower.split())
        context_terms = set(contexts_text.split())
        overlap = len(answer_terms & context_terms)
        faithfulness = min(1.0, overlap / max(1, len(answer_terms)))

        # Answer relevancy: does answer overlap with ground truth?
        gt_terms = set(gt.split())
        rel_overlap = len(answer_terms & gt_terms)
        relevancy = min(1.0, rel_overlap / max(1, len(gt_terms)))

        # Context precision: does context contain ground truth terms?
        ctx_gt_overlap = len(context_terms & gt_terms)
        precision = min(1.0, ctx_gt_overlap / max(1, len(gt_terms)))

        # Context recall: are ground truth terms present in context?
        recall = min(1.0, ctx_gt_overlap / max(1, len(gt_terms)))

        scores.append({
            "question": r.question,
            "faithfulness": round(faithfulness, 3),
            "answer_relevancy": round(relevancy, 3),
            "context_precision": round(precision, 3),
            "context_recall": round(recall, 3),
        })

    avg = {
        "faithfulness": round(sum(s["faithfulness"] for s in scores) / total, 3),
        "answer_relevancy": round(sum(s["answer_relevancy"] for s in scores) / total, 3),
        "context_precision": round(sum(s["context_precision"] for s in scores) / total, 3),
        "context_recall": round(sum(s["context_recall"] for s in scores) / total, 3),
    }

    return {
        "average": avg,
        "samples": scores,
        "num_samples": total,
        "method": "heuristic",
    }


def _run_context_only_eval(questions: List[Dict]) -> Dict[str, Any]:
    """Run evaluation without a QA chain (context quality only)."""
    logger.info("Running context-only evaluation (no QA chain)")
    results = [
        EvalSample(question=q["question"], ground_truth=q["ground_truth"])
        for q in questions
    ]
    return _heuristic_evaluation(results, questions)


def print_eval_report(metrics: Dict[str, Any]):
    """Pretty-print evaluation results."""
    avg = metrics.get("average", {})
    print("\n" + "=" * 60)
    print("  RAG EVALUATION REPORT")
    print("=" * 60)
    print(f"  Method:       {metrics.get('method', 'unknown')}")
    print(f"  Samples:      {metrics.get('num_samples', 0)}")
    print(f"  Faithfulness:    {avg.get('faithfulness', 0):.1%}")
    print(f"  Answer Relevancy: {avg.get('answer_relevancy', 0):.1%}")
    print(f"  Context Precision:{avg.get('context_precision', 0):.1%}")
    print(f"  Context Recall:   {avg.get('context_recall', 0):.1%}")
    print("=" * 60)

    # Per-sample breakdown
    if metrics.get("samples"):
        print("\n  Per-Sample Breakdown:")
        print(f"  {'Question':<40} {'Faith':>6} {'Rel':>6} {'Prec':>6} {'Recall':>6}")
        print("  " + "-" * 64)
        best = worst = metrics["samples"][0] if metrics["samples"] else None
        for s in metrics["samples"]:
            q_short = s["question"][:38]
            print(f"  {q_short:<40} {s['faithfulness']:>5.0%} {s['answer_relevancy']:>5.0%} "
                  f"{s['context_precision']:>5.0%} {s['context_recall']:>5.0%}")
            if s["faithfulness"] > (best["faithfulness"] if best else 0):
                best = s

        if best:
            print(f"\n  Best: {best['question'][:50]} (faithfulness={best['faithfulness']:.0%})")

    print("=" * 60)


# ── CLI entry point ─────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate RAG system performance")
    parser.add_argument("--questions", type=str, help="JSON file with questions")
    parser.add_argument("--sample", type=int, default=None, help="Number of questions to sample")
    parser.add_argument("--context-only", action="store_true", help="Skip QA chain, eval context only")
    args = parser.parse_args()

    test_questions = None
    if args.questions and os.path.exists(args.questions):
        with open(args.questions) as f:
            test_questions = json.load(f)

    rag_core = None
    if not args.context_only:
        sys.path.insert(0, os.path.dirname(__file__))
        from RAG_core import RAGChatbotCore
        rag_core = RAGChatbotCore()
        # Try loading persisted vector store
        chroma_path = "/tmp/F_web/chroma_db"
        if os.path.exists(chroma_path):
            try:
                rag_core.load_persisted_vector_store(chroma_path)
                rag_core.create_qa_chain()
                logger.info("Loaded persisted vector store for evaluation")
            except Exception as e:
                logger.warning(f"Could not load vector store: {e}")

    metrics = run_ragas_evaluation(
        questions=test_questions,
        rag_core=rag_core,
        sample_size=args.sample,
    )
    print_eval_report(metrics)

    # Save results
    output = "eval_results.json"
    with open(output, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    print(f"\nResults saved to {output}")
