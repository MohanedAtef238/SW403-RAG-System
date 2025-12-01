"""
Evaluation metrics for RAG system performance comparison.

Implements partial credit scoring, information retrieval metrics (MRR, NDCG, P@K, R@K),
and statistical analysis (t-tests, confidence intervals) for P1 vs P2 comparison.
"""

from typing import List, Dict, Tuple, Any
import numpy as np
from scipy import stats
from dataclasses import dataclass


@dataclass
class RetrievalResult:
    """Single retrieval result with scoring information."""
    function_name: str
    file_path: str
    score: float  # similarity score
    rank: int  # position in results


@dataclass
class GroundTruth:
    """Ground truth for a single query."""
    query_id: int
    query: str
    category: str
    expected_function: str
    expected_file: str


@dataclass
class EvaluationMetrics:
    """Comprehensive metrics for a single query evaluation."""
    query_id: int
    query: str
    category: str
    
    # Retrieval success
    exact_match: bool  # Top result is exact match
    found_in_top_k: bool  # Correct result appears in top K
    rank: int  # Rank of correct result (0 if not found)
    
    # Partial credit scoring (file=0.3, module=0.5, exact=1.0)
    partial_credit_score: float
    
    # IR metrics
    reciprocal_rank: float  # 1/rank if found, else 0
    precision_at_k: float  # Relevant items in top K / K
    recall_at_k: float  # Relevant items found / total relevant
    ndcg_at_k: float  # Normalized DCG
    
    # Retrieval details
    top_result_function: str
    top_result_file: str
    top_result_score: float


def calculate_partial_credit(
    predicted_function: str,
    predicted_file: str,
    expected_function: str,
    expected_file: str
) -> float:
    """
    Calculate partial credit score based on match granularity.
    
    Scoring:
    - Exact function match: 1.0
    - Same class/module (within same file): 0.5
    - Same file (different function): 0.3
    - Wrong file: 0.0
    
    Args:
        predicted_function: Predicted function name
        predicted_file: Predicted file path
        expected_function: Ground truth function name
        expected_file: Ground truth file path
    
    Returns:
        Partial credit score between 0.0 and 1.0
    """
    # Normalize file paths for comparison (allow absolute vs relative by suffix match)
    pred_file_norm = predicted_file.replace('\\', '/').lower()
    exp_file_norm = expected_file.replace('\\', '/').lower()

    def same_file(a: str, b: str) -> bool:
        # Accept exact or suffix match (handles absolute vs relative paths)
        return a == b or a.endswith('/' + b) or a.endswith(b)
    
    # Exact match
    if predicted_function == expected_function and same_file(pred_file_norm, exp_file_norm):
        return 1.0
    
    # Same file, different function
    if same_file(pred_file_norm, exp_file_norm):
        # Check if same class (simple heuristic: both contain '.')
        if '.' in predicted_function and '.' in expected_function:
            pred_class = predicted_function.split('.')[0]
            exp_class = expected_function.split('.')[0]
            if pred_class == exp_class:
                return 0.5
        return 0.3
    
    # Different file
    return 0.0


def calculate_reciprocal_rank(results: List[RetrievalResult], ground_truth: GroundTruth) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) for a single query.
    
    Args:
        results: List of retrieval results ordered by rank
        ground_truth: Ground truth for the query
    
    Returns:
        Reciprocal rank (1/rank if found, else 0.0)
    """
    def same_file(a: str, b: str) -> bool:
        a_n = a.replace('\\', '/').lower()
        b_n = b.replace('\\', '/').lower()
        return a_n == b_n or a_n.endswith('/' + b_n) or a_n.endswith(b_n)

    for result in results:
        if (result.function_name == ground_truth.expected_function and
            same_file(result.file_path, ground_truth.expected_file)):
            return 1.0 / result.rank
    return 0.0


def calculate_precision_at_k(results: List[RetrievalResult], ground_truth: GroundTruth, k: int) -> float:
    """
    Calculate Precision@K - fraction of top K results that are relevant.
    
    Args:
        results: List of retrieval results ordered by rank
        ground_truth: Ground truth for the query
        k: Number of top results to consider
    
    Returns:
        Precision@K value between 0.0 and 1.0
    """
    if k == 0:
        return 0.0
    
    top_k_results = results[:k]
    def same_file(a: str, b: str) -> bool:
        a_n = a.replace('\\', '/').lower()
        b_n = b.replace('\\', '/').lower()
        return a_n == b_n or a_n.endswith('/' + b_n) or a_n.endswith(b_n)

    relevant_count = sum(
        1 for r in top_k_results
        if r.function_name == ground_truth.expected_function and
           same_file(r.file_path, ground_truth.expected_file)
    )
    return relevant_count / k


def calculate_recall_at_k(results: List[RetrievalResult], ground_truth: GroundTruth, k: int) -> float:
    """
    Calculate Recall@K - fraction of relevant items found in top K.
    
    Since we have single ground truth per query, this is binary (0.0 or 1.0).
    
    Args:
        results: List of retrieval results ordered by rank
        ground_truth: Ground truth for the query
        k: Number of top results to consider
    
    Returns:
        Recall@K value (0.0 or 1.0 for single ground truth)
    """
    top_k_results = results[:k]
    def same_file(a: str, b: str) -> bool:
        a_n = a.replace('\\', '/').lower()
        b_n = b.replace('\\', '/').lower()
        return a_n == b_n or a_n.endswith('/' + b_n) or a_n.endswith(b_n)

    for r in top_k_results:
        if (r.function_name == ground_truth.expected_function and
            same_file(r.file_path, ground_truth.expected_file)):
            return 1.0
    return 0.0


def calculate_ndcg_at_k(results: List[RetrievalResult], ground_truth: GroundTruth, k: int) -> float:
    """
    Calculate Normalized Discounted Cumulative Gain (NDCG@K).
    
    Uses partial credit scoring for relevance grades.
    
    Args:
        results: List of retrieval results ordered by rank
        ground_truth: Ground truth for the query
        k: Number of top results to consider
    
    Returns:
        NDCG@K value between 0.0 and 1.0
    """
    if k == 0:
        return 0.0
    
    top_k_results = results[:k]
    
    # Calculate DCG
    dcg = 0.0
    for i, result in enumerate(top_k_results):
        relevance = calculate_partial_credit(
            result.function_name,
            result.file_path,
            ground_truth.expected_function,
            ground_truth.expected_file
        )
        # DCG formula: sum(rel_i / log2(i+2)) for i in [0, k-1]
        dcg += relevance / np.log2(i + 2)
    
    # Calculate IDCG (best possible DCG - perfect ranking)
    idcg = 1.0 / np.log2(2)  # Perfect result at rank 1
    
    # Normalize
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_query(
    results: List[RetrievalResult],
    ground_truth: GroundTruth,
    k: int = 5
) -> EvaluationMetrics:
    """
    Evaluate a single query with comprehensive metrics.
    
    Args:
        results: List of retrieval results ordered by rank
        ground_truth: Ground truth for the query
        k: Number of top results to consider (default: 5)
    
    Returns:
        EvaluationMetrics object with all calculated metrics
    """
    if not results:
        return EvaluationMetrics(
            query_id=ground_truth.query_id,
            query=ground_truth.query,
            category=ground_truth.category,
            exact_match=False,
            found_in_top_k=False,
            rank=0,
            partial_credit_score=0.0,
            reciprocal_rank=0.0,
            precision_at_k=0.0,
            recall_at_k=0.0,
            ndcg_at_k=0.0,
            top_result_function="",
            top_result_file="",
            top_result_score=0.0
        )
    
    top_result = results[0]
    
    # Check exact match
    def same_file(a: str, b: str) -> bool:
        a_n = a.replace('\\', '/').lower()
        b_n = b.replace('\\', '/').lower()
        return a_n == b_n or a_n.endswith('/' + b_n) or a_n.endswith(b_n)

    exact_match = (
        top_result.function_name == ground_truth.expected_function and
        same_file(top_result.file_path, ground_truth.expected_file)
    )
    
    # Find rank of correct result
    rank = 0
    for result in results:
        if (result.function_name == ground_truth.expected_function and
            same_file(result.file_path, ground_truth.expected_file)):
            rank = result.rank
            break
    
    found_in_top_k = 0 < rank <= k
    
    # Calculate partial credit for top result
    partial_credit = calculate_partial_credit(
        top_result.function_name,
        top_result.file_path,
        ground_truth.expected_function,
        ground_truth.expected_file
    )
    
    # Calculate IR metrics
    mrr = calculate_reciprocal_rank(results, ground_truth)
    p_at_k = calculate_precision_at_k(results, ground_truth, k)
    r_at_k = calculate_recall_at_k(results, ground_truth, k)
    ndcg = calculate_ndcg_at_k(results, ground_truth, k)
    
    return EvaluationMetrics(
        query_id=ground_truth.query_id,
        query=ground_truth.query,
        category=ground_truth.category,
        exact_match=exact_match,
        found_in_top_k=found_in_top_k,
        rank=rank,
        partial_credit_score=partial_credit,
        reciprocal_rank=mrr,
        precision_at_k=p_at_k,
        recall_at_k=r_at_k,
        ndcg_at_k=ndcg,
        top_result_function=top_result.function_name,
        top_result_file=top_result.file_path,
        top_result_score=top_result.score
    )


def aggregate_metrics(metrics_list: List[EvaluationMetrics]) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple queries.
    
    Args:
        metrics_list: List of EvaluationMetrics for all queries
    
    Returns:
        Dictionary with aggregated metrics and per-category breakdowns
    """
    if not metrics_list:
        return {}
    
    # Overall metrics
    exact_match_rate = np.mean([m.exact_match for m in metrics_list])
    found_rate = np.mean([m.found_in_top_k for m in metrics_list])
    avg_partial_credit = np.mean([m.partial_credit_score for m in metrics_list])
    mrr = np.mean([m.reciprocal_rank for m in metrics_list])
    avg_precision = np.mean([m.precision_at_k for m in metrics_list])
    avg_recall = np.mean([m.recall_at_k for m in metrics_list])
    avg_ndcg = np.mean([m.ndcg_at_k for m in metrics_list])
    
    # Per-category breakdown
    categories = set(m.category for m in metrics_list)
    category_metrics = {}
    
    for category in categories:
        cat_metrics = [m for m in metrics_list if m.category == category]
        category_metrics[category] = {
            "count": len(cat_metrics),
            "exact_match_rate": np.mean([m.exact_match for m in cat_metrics]),
            "found_rate": np.mean([m.found_in_top_k for m in cat_metrics]),
            "avg_partial_credit": np.mean([m.partial_credit_score for m in cat_metrics]),
            "mrr": np.mean([m.reciprocal_rank for m in cat_metrics]),
            "avg_precision": np.mean([m.precision_at_k for m in cat_metrics]),
            "avg_recall": np.mean([m.recall_at_k for m in cat_metrics]),
            "avg_ndcg": np.mean([m.ndcg_at_k for m in cat_metrics])
        }
    
    return {
        "overall": {
            "total_queries": len(metrics_list),
            "exact_match_rate": exact_match_rate,
            "found_in_top_k_rate": found_rate,
            "avg_partial_credit": avg_partial_credit,
            "mrr": mrr,
            "avg_precision_at_k": avg_precision,
            "avg_recall_at_k": avg_recall,
            "avg_ndcg_at_k": avg_ndcg
        },
        "by_category": category_metrics
    }


def compare_systems_ttest(
    metrics_p1: List[EvaluationMetrics],
    metrics_p2: List[EvaluationMetrics],
    metric_name: str = "partial_credit_score"
) -> Dict[str, Any]:
    """
    Perform paired t-test to compare P1 vs P2 on a specific metric.
    
    Args:
        metrics_p1: Metrics from Prototype 1
        metrics_p2: Metrics from Prototype 2
        metric_name: Name of metric to compare (e.g., "partial_credit_score", "reciprocal_rank")
    
    Returns:
        Dictionary with t-test results and confidence intervals
    """
    # Extract metric values
    values_p1 = [getattr(m, metric_name) for m in metrics_p1]
    values_p2 = [getattr(m, metric_name) for m in metrics_p2]
    
    # Ensure same number of queries
    assert len(values_p1) == len(values_p2), "Metrics lists must have same length for paired t-test"
    
    # Calculate means and std devs
    mean_p1 = np.mean(values_p1)
    mean_p2 = np.mean(values_p2)
    std_p1 = np.std(values_p1, ddof=1)
    std_p2 = np.std(values_p2, ddof=1)
    
    # Paired t-test
    t_statistic, p_value = stats.ttest_rel(values_p2, values_p1)
    
    # Calculate 95% confidence intervals
    n = len(values_p1)
    confidence_level = 0.95
    degrees_freedom = n - 1
    t_critical = stats.t.ppf((1 + confidence_level) / 2, degrees_freedom)
    
    ci_p1 = (
        mean_p1 - t_critical * std_p1 / np.sqrt(n),
        mean_p1 + t_critical * std_p1 / np.sqrt(n)
    )
    ci_p2 = (
        mean_p2 - t_critical * std_p2 / np.sqrt(n),
        mean_p2 + t_critical * std_p2 / np.sqrt(n)
    )
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((std_p1**2 + std_p2**2) / 2)
    cohens_d = (mean_p2 - mean_p1) / pooled_std if pooled_std > 0 else 0.0
    
    # Statistical significance
    is_significant = p_value < 0.05
    
    return {
        "metric": metric_name,
        "p1": {
            "mean": mean_p1,
            "std": std_p1,
            "ci_95": ci_p1
        },
        "p2": {
            "mean": mean_p2,
            "std": std_p2,
            "ci_95": ci_p2
        },
        "comparison": {
            "t_statistic": t_statistic,
            "p_value": p_value,
            "is_significant": is_significant,
            "cohens_d": cohens_d,
            "improvement": mean_p2 - mean_p1,
            "improvement_pct": ((mean_p2 - mean_p1) / mean_p1 * 100) if mean_p1 > 0 else 0.0
        }
    }
