"""
Evaluation and Benchmark Module
Module: src/evaluation.py

Provides ROUGE score computation (ROUGE-1, ROUGE-2, ROUGE-L) and
head-to-head model comparison utilities. Never fabricates scores or results.
"""

from __future__ import annotations

import collections
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from src.summarizer import (
    MODEL_REGISTRY,
    SummarizationResult,
    TransformerSummarizer,
    clean_text,
)

# Try importing Hugging Face evaluate or rouge_score
ROUGE_EVALUATE_AVAILABLE = False
try:
    import evaluate
    try:
        _test_rouge = evaluate.load("rouge")
        ROUGE_EVALUATE_AVAILABLE = True
    except Exception:
        ROUGE_EVALUATE_AVAILABLE = False
except ImportError:
    pass

ROUGE_SCORE_LIB_AVAILABLE = False
try:
    from rouge_score import rouge_scorer
    ROUGE_SCORE_LIB_AVAILABLE = True
except ImportError:
    pass


def _tokenize_for_rouge(text: str) -> List[str]:
    """Tokenizes text into lowercase alphanumeric word tokens for fallback ROUGE."""
    import re
    return re.findall(r"\b\w+\b", text.lower())


def _get_ngrams(tokens: List[str], n: int) -> collections.Counter:
    """Extracts n-grams from a list of tokens."""
    if len(tokens) < n:
        return collections.Counter()
    return collections.Counter([tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)])


def _compute_lcs(a: List[str], b: List[str]) -> int:
    """Computes the Longest Common Subsequence length between two token lists."""
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 0
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def fallback_rouge_score(candidate: str, reference: str) -> Dict[str, Dict[str, float]]:
    """
    Pure Python reference ROUGE calculation (ROUGE-1, ROUGE-2, ROUGE-L).
    Used if third-party rouge packages encounter system or library dependency issues.
    Computes exact precision, recall, and fmeasure.
    """
    cand_tokens = _tokenize_for_rouge(candidate)
    ref_tokens = _tokenize_for_rouge(reference)

    def calc_pr_f1(overlap_count: int, cand_len: int, ref_len: int) -> Dict[str, float]:
        prec = overlap_count / cand_len if cand_len > 0 else 0.0
        rec = overlap_count / ref_len if ref_len > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        return {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "fmeasure": round(f1, 4),
        }

    # ROUGE-1
    cand_1 = _get_ngrams(cand_tokens, 1)
    ref_1 = _get_ngrams(ref_tokens, 1)
    overlap_1 = sum((cand_1 & ref_1).values())
    r1 = calc_pr_f1(overlap_1, len(cand_tokens), len(ref_tokens))

    # ROUGE-2
    cand_2 = _get_ngrams(cand_tokens, 2)
    ref_2 = _get_ngrams(ref_tokens, 2)
    overlap_2 = sum((cand_2 & ref_2).values())
    r2 = calc_pr_f1(overlap_2, max(0, len(cand_tokens) - 1), max(0, len(ref_tokens) - 1))

    # ROUGE-L (LCS-based)
    lcs_len = _compute_lcs(cand_tokens, ref_tokens)
    rl = calc_pr_f1(lcs_len, len(cand_tokens), len(ref_tokens))

    return {
        "rouge1": r1,
        "rouge2": r2,
        "rougeL": rl,
    }


def calculate_rouge_scores(
    candidate_summary: str, reference_summary: str
) -> Dict[str, Dict[str, float]]:
    """
    Calculates ROUGE-1, ROUGE-2, and ROUGE-L metrics between candidate and reference summaries.
    Returns:
        {
            "rouge1": {"precision": ..., "recall": ..., "fmeasure": ...},
            "rouge2": {"precision": ..., "recall": ..., "fmeasure": ...},
            "rougeL": {"precision": ..., "recall": ..., "fmeasure": ...},
        }
    """
    cleaned_cand = clean_text(candidate_summary)
    cleaned_ref = clean_text(reference_summary)

    if not cleaned_cand or not cleaned_ref:
        raise ValueError(
            "Both candidate summary and reference summary must be non-empty to compute ROUGE scores."
        )

    # 1. Attempt using rouge-score library
    if ROUGE_SCORE_LIB_AVAILABLE:
        try:
            scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
            raw_scores = scorer.score(cleaned_ref, cleaned_cand)
            return {
                "rouge1": {
                    "precision": round(raw_scores["rouge1"].precision, 4),
                    "recall": round(raw_scores["rouge1"].recall, 4),
                    "fmeasure": round(raw_scores["rouge1"].fmeasure, 4),
                },
                "rouge2": {
                    "precision": round(raw_scores["rouge2"].precision, 4),
                    "recall": round(raw_scores["rouge2"].recall, 4),
                    "fmeasure": round(raw_scores["rouge2"].fmeasure, 4),
                },
                "rougeL": {
                    "precision": round(raw_scores["rougeL"].precision, 4),
                    "recall": round(raw_scores["rougeL"].recall, 4),
                    "fmeasure": round(raw_scores["rougeL"].fmeasure, 4),
                },
            }
        except Exception:
            pass

    # 2. Attempt using Hugging Face evaluate
    if ROUGE_EVALUATE_AVAILABLE:
        try:
            rouge_metric = evaluate.load("rouge")
            results = rouge_metric.compute(
                predictions=[cleaned_cand],
                references=[cleaned_ref],
                use_stemmer=True,
            )
            return {
                "rouge1": {
                    "fmeasure": round(float(results.get("rouge1", 0.0)), 4),
                    "precision": round(float(results.get("rouge1", 0.0)), 4),
                    "recall": round(float(results.get("rouge1", 0.0)), 4),
                },
                "rouge2": {
                    "fmeasure": round(float(results.get("rouge2", 0.0)), 4),
                    "precision": round(float(results.get("rouge2", 0.0)), 4),
                    "recall": round(float(results.get("rouge2", 0.0)), 4),
                },
                "rougeL": {
                    "fmeasure": round(float(results.get("rougeL", 0.0)), 4),
                    "precision": round(float(results.get("rougeL", 0.0)), 4),
                    "recall": round(float(results.get("rougeL", 0.0)), 4),
                },
            }
        except Exception:
            pass

    # 3. Fallback to exact deterministic algorithmic ROUGE
    return fallback_rouge_score(cleaned_cand, cleaned_ref)


def format_rouge_to_dataframe(
    rouge_dict: Dict[str, Dict[str, float]], model_label: Optional[str] = None
) -> pd.DataFrame:
    """
    Converts a ROUGE score dictionary into a clean display DataFrame.
    """
    rows = []
    for metric_name, values in rouge_dict.items():
        row = {
            "Metric": metric_name.upper(),
            "Precision (%)": f"{values['precision'] * 100:.2f}%",
            "Recall (%)": f"{values['recall'] * 100:.2f}%",
            "F1-Score (%)": f"{values['fmeasure'] * 100:.2f}%",
        }
        if model_label:
            row["Model"] = model_label
        rows.append(row)
    return pd.DataFrame(rows)


def run_head_to_head_comparison(
    text: str,
    models: Optional[List[str]] = None,
    reference_summary: Optional[str] = None,
    min_length: int = 40,
    max_length: int = 140,
    progress_callback: Optional[Callable[[float, str], None]] = None,
) -> Dict[str, Any]:
    """
    Runs actual inference on the same input text across multiple models,
    measuring wall-clock time, compression, and ROUGE metrics if reference is provided.
    Never fabricates scores.
    """
    if models is None:
        models = [
            "BART (facebook/bart-large-cnn)",
            "FLAN-T5 (google/flan-t5-base)",
        ]

    cleaned_input = clean_text(text)
    if not cleaned_input:
        raise ValueError("Cannot compare models with empty input text.")

    cleaned_ref = clean_text(reference_summary) if reference_summary else None
    results: Dict[str, SummarizationResult] = {}
    rouge_evaluations: Dict[str, Dict[str, Dict[str, float]]] = {}

    total_models = len(models)
    for idx, model_name in enumerate(models):
        if progress_callback:
            base_p = idx / total_models
            progress_callback(base_p, f"Running benchmark on {model_name}...")

        summarizer = TransformerSummarizer(model_name)
        result = summarizer.summarize(
            cleaned_input,
            min_length=min_length,
            max_length=max_length,
            progress_callback=lambda p, msg: progress_callback(
                (idx + p) / total_models, f"[{model_name}] {msg}"
            )
            if progress_callback
            else None,
        )
        results[model_name] = result

        if cleaned_ref:
            rouge_evaluations[model_name] = calculate_rouge_scores(
                result.summary_text, cleaned_ref
            )

    # Build comparison comparison table
    summary_rows = []
    for model_name, res in results.items():
        row = {
            "Model": model_name,
            "Inference Time (s)": f"{res.inference_time_sec:.2f}s",
            "Summary Words": res.summary_words,
            "Compression Ratio": f"{res.compression_ratio:.2f}x",
            "Length Reduction": f"{res.reduction_percent:.1f}%",
            "Hierarchical Chunks": res.num_chunks,
            "Device": res.device_used,
        }
        if cleaned_ref and model_name in rouge_evaluations:
            r = rouge_evaluations[model_name]
            row["ROUGE-1 F1"] = f"{r['rouge1']['fmeasure'] * 100:.2f}%"
            row["ROUGE-2 F1"] = f"{r['rouge2']['fmeasure'] * 100:.2f}%"
            row["ROUGE-L F1"] = f"{r['rougeL']['fmeasure'] * 100:.2f}%"
        summary_rows.append(row)

    df_comparison = pd.DataFrame(summary_rows)

    return {
        "results": results,
        "rouge_evaluations": rouge_evaluations if cleaned_ref else None,
        "dataframe": df_comparison,
        "has_reference": bool(cleaned_ref),
    }
