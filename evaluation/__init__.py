"""Evaluation framework package for SW403 RAG system."""

from evaluation.metrics import (
    RetrievalResult,
    GroundTruth,
    EvaluationMetrics,
    calculate_partial_credit,
    evaluate_query,
    aggregate_metrics,
    compare_systems_ttest
)

from evaluation.runner import PrototypeRunner, EvaluationRunner
from evaluation.analyzer import ErrorAnalyzer, ComparativeAnalyzer, ErrorCase
from evaluation.visualize import EvaluationVisualizer

__all__ = [
    "RetrievalResult",
    "GroundTruth",
    "EvaluationMetrics",
    "calculate_partial_credit",
    "evaluate_query",
    "aggregate_metrics",
    "compare_systems_ttest",
    "PrototypeRunner",
    "EvaluationRunner",
    "ErrorAnalyzer",
    "ComparativeAnalyzer",
    "ErrorCase",
    "EvaluationVisualizer"
]
