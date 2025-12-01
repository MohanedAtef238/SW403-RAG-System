"""
Error analysis and comparative evaluation for P1 vs P2.

Classifies errors, detects hallucinations, and performs statistical comparisons.
"""

import json
from typing import List, Dict, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import pandas as pd

from evaluation.metrics import EvaluationMetrics, compare_systems_ttest


@dataclass
class ErrorCase:
    """Detailed error case for analysis."""
    query_id: int
    query: str
    category: str
    prototype: str
    
    expected_function: str
    expected_file: str
    
    predicted_function: str
    predicted_file: str
    predicted_score: float
    
    error_type: str
    is_hallucination: bool
    root_cause_hypothesis: str


class ErrorAnalyzer:
    """Analyzes errors and generates detailed error reports."""
    
    ERROR_TYPES = {
        "NOT_FOUND": "Expected function not retrieved in top-K results",
        "WRONG_FUNCTION_SAME_FILE": "Correct file but wrong function",
        "WRONG_FILE": "Incorrect file entirely",
        "HALLUCINATION": "Returned function does not exist in corpus",
        "LOW_CONFIDENCE": "Correct result but very low similarity score"
    }
    
    def __init__(self, corpus_functions: List[str] = None): # type: ignore
        """
        Initialize error analyzer.
        
        Args:
            corpus_functions: List of all valid function names in corpus (for hallucination detection)
        """
        self.corpus_functions = set(corpus_functions) if corpus_functions else set()
    
    def classify_error(
        self,
        metric: EvaluationMetrics,
        ground_truth_file: str
    ) -> str:
        """
        Classify error type based on evaluation metrics.
        
        Args:
            metric: EvaluationMetrics for a query
            ground_truth_file: Expected file path from ground truth
        
        Returns:
            Error type string
        """
        # Perfect match - no error
        if metric.exact_match:
            return "CORRECT"
        
        # Not found in results
        if metric.rank == 0:
            return "NOT_FOUND"
        
        # Check if it's a hallucination (if corpus provided)
        if self.corpus_functions and metric.top_result_function not in self.corpus_functions:
            return "HALLUCINATION"
        
        # Same file, different function
        pred_file_norm = metric.top_result_file.replace('\\', '/').lower()
        gt_file_norm = ground_truth_file.replace('\\', '/').lower()
        
        if pred_file_norm == gt_file_norm:
            return "WRONG_FUNCTION_SAME_FILE"
        
        # Correct result but low confidence
        if metric.found_in_top_k and metric.top_result_score < 0.5:
            return "LOW_CONFIDENCE"
        
        # Different file
        return "WRONG_FILE"
    
    def is_hallucination(self, function_name: str) -> bool:
        """
        Check if a function name is a hallucination (not in corpus).
        
        Args:
            function_name: Predicted function name
        
        Returns:
            True if hallucination, False otherwise
        """
        if not self.corpus_functions:
            return False
        return function_name not in self.corpus_functions
    
    def generate_root_cause_hypothesis(
        self,
        error_type: str,
        metric: EvaluationMetrics,
        category: str
    ) -> str:
        """
        Generate hypothesis for why the error occurred.
        
        Args:
            error_type: Classified error type
            metric: EvaluationMetrics for the query
            category: Query category (simple_lookup, local_context, global_relational)
        
        Returns:
            Root cause hypothesis string
        """
        hypotheses = {
            "NOT_FOUND": f"Function not indexed or query embedding mismatch. Category: {category}",
            "WRONG_FUNCTION_SAME_FILE": "Correct file but insufficient function-level discrimination",
            "WRONG_FILE": "Embedding captured wrong semantic context or file-level noise",
            "HALLUCINATION": "Model generated non-existent function (possible chunking artifact)",
            "LOW_CONFIDENCE": "Weak semantic match - query may be too specific or use different terminology"
        }
        
        base_hypothesis = hypotheses.get(error_type, "Unknown error type")
        
        # Add category-specific insights
        if category == "global_relational" and error_type in ["NOT_FOUND", "WRONG_FILE"]:
            base_hypothesis += " | Global context queries require cross-function understanding"
        elif category == "local_context" and error_type == "WRONG_FUNCTION_SAME_FILE":
            base_hypothesis += " | Metadata richness may be insufficient (params/return types)"
        
        return base_hypothesis
    
    def analyze_prototype_errors(
        self,
        metrics: List[EvaluationMetrics],
        ground_truth_data: List[Dict[str, Any]],
        prototype_name: str
    ) -> List[ErrorCase]:
        """
        Analyze all errors for a single prototype.
        
        Args:
            metrics: List of EvaluationMetrics
            ground_truth_data: List of ground truth query dictionaries
            prototype_name: Name of prototype ("P1" or "P2")
        
        Returns:
            List of ErrorCase objects
        """
        error_cases = []
        
        # Create ground truth lookup
        gt_lookup = {gt["id"]: gt for gt in ground_truth_data}
        
        for metric in metrics:
            gt = gt_lookup[metric.query_id]
            error_type = self.classify_error(metric, gt["expected_file"])
            
            # Only include actual errors
            if error_type != "CORRECT":
                error_cases.append(ErrorCase(
                    query_id=metric.query_id,
                    query=metric.query,
                    category=metric.category,
                    prototype=prototype_name,
                    expected_function=gt["expected_function"],
                    expected_file=gt["expected_file"],
                    predicted_function=metric.top_result_function,
                    predicted_file=metric.top_result_file,
                    predicted_score=metric.top_result_score,
                    error_type=error_type,
                    is_hallucination=self.is_hallucination(metric.top_result_function),
                    root_cause_hypothesis=self.generate_root_cause_hypothesis(
                        error_type, metric, metric.category
                    )
                ))
        
        return error_cases
    
    def generate_error_table(
        self,
        error_cases: List[ErrorCase],
        output_path: Path
    ):
        """
        Generate error analysis table in CSV and Markdown formats.
        
        Args:
            error_cases: List of ErrorCase objects
            output_path: Base path for output files (without extension)
        """
        # Convert to DataFrame
        df = pd.DataFrame([
            {
                "Query ID": ec.query_id,
                "Query": ec.query,
                "Category": ec.category,
                "Prototype": ec.prototype,
                "Expected": ec.expected_function,
                "Predicted": ec.predicted_function,
                "Expected File": ec.expected_file,
                "Predicted File": ec.predicted_file,
                "Score": f"{ec.predicted_score:.3f}",
                "Error Type": ec.error_type,
                "Hallucination": "Yes" if ec.is_hallucination else "No",
                "Root Cause Hypothesis": ec.root_cause_hypothesis
            }
            for ec in error_cases
        ])
        
        # Save as CSV
        csv_path = output_path.with_suffix(".csv")
        df.to_csv(csv_path, index=False)
        print(f"Error analysis CSV saved to: {csv_path}")
        
        # Save as Markdown
        md_path = output_path.with_suffix(".md")
        with open(md_path, "w") as f:
            f.write("# Error Analysis Table\n\n")
            f.write(df.to_markdown(index=False))
        print(f"Error analysis Markdown saved to: {md_path}")
        
        return df


class ComparativeAnalyzer:
    """Performs statistical comparison between P1 and P2."""
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize comparative analyzer.
        
        Args:
            results_dir: Directory containing evaluation results
        """
        self.results_dir = Path(results_dir)
        self.p1_metrics: List[EvaluationMetrics] = []
        self.p2_metrics: List[EvaluationMetrics] = []
        
        self.load_results()
    
    def load_results(self):
        """Load P1 and P2 results from JSON files."""
        # Load P1
        p1_path = self.results_dir / "p1_results.json"
        if p1_path.exists():
            with open(p1_path, "r") as f:
                data = json.load(f)
                self.p1_metrics = [EvaluationMetrics(**m) for m in data["metrics"]]
            print(f"Loaded {len(self.p1_metrics)} P1 metrics")
        
        # Load P2
        p2_path = self.results_dir / "p2_results.json"
        if p2_path.exists():
            with open(p2_path, "r") as f:
                data = json.load(f)
                self.p2_metrics = [EvaluationMetrics(**m) for m in data["metrics"]]
            print(f"Loaded {len(self.p2_metrics)} P2 metrics")
    
    def compare_all_metrics(self) -> Dict[str, Any]:
        """
        Run statistical comparisons on all key metrics.
        
        Returns:
            Dictionary with comparison results for all metrics
        """
        metrics_to_compare = [
            "partial_credit_score",
            "reciprocal_rank",
            "precision_at_k",
            "recall_at_k",
            "ndcg_at_k"
        ]
        
        comparisons = {}
        for metric_name in metrics_to_compare:
            print(f"\nComparing {metric_name}...")
            comparisons[metric_name] = compare_systems_ttest(
                self.p1_metrics,
                self.p2_metrics,
                metric_name
            )
        
        return comparisons
    
    def generate_comparison_report(self, output_path: Path):
        """
        Generate comprehensive comparison report.
        
        Args:
            output_path: Path for output JSON file
        """
        comparisons = self.compare_all_metrics()
        
        report = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "n_queries": len(self.p1_metrics),
            "comparisons": comparisons,
            "summary": self._generate_summary(comparisons)
        }
        
        # Save JSON
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        print(f"\nComparison report saved to: {output_path}")
        
        return report
    
    def _generate_summary(self, comparisons: Dict[str, Any]) -> Dict[str, Any]:
        """Generate human-readable summary of comparisons."""
        summary = {
            "statistically_significant": [],
            "not_significant": [],
            "p2_improvements": {},
            "p2_regressions": {}
        }
        
        for metric_name, comparison in comparisons.items():
            comp_data = comparison["comparison"]
            
            if comp_data["is_significant"]:
                summary["statistically_significant"].append(metric_name)
            else:
                summary["not_significant"].append(metric_name)
            
            improvement = comp_data["improvement"]
            if improvement > 0:
                summary["p2_improvements"][metric_name] = {
                    "improvement": improvement,
                    "improvement_pct": comp_data["improvement_pct"]
                }
            elif improvement < 0:
                summary["p2_regressions"][metric_name] = {
                    "regression": abs(improvement),
                    "regression_pct": abs(comp_data["improvement_pct"])
                }
        
        return summary


def main():
    """Main entry point for error analysis and comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze errors and compare P1 vs P2")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing evaluation results"
    )
    parser.add_argument(
        "--ground-truth",
        default="shared_data/ground_truth.json",
        help="Path to ground truth JSON file"
    )
    
    args = parser.parse_args()
    
    results_dir = Path(args.results_dir)
    
    # Load ground truth
    with open(args.ground_truth, "r") as f:
        gt_data = json.load(f)
    
    # Load results
    with open(results_dir / "p1_results.json", "r") as f:
        p1_data = json.load(f)
        p1_metrics = [EvaluationMetrics(**m) for m in p1_data["metrics"]]
    
    with open(results_dir / "p2_results.json", "r") as f:
        p2_data = json.load(f)
        p2_metrics = [EvaluationMetrics(**m) for m in p2_data["metrics"]]
    
    # Initialize error analyzer
    analyzer = ErrorAnalyzer()
    
    # Analyze P1 errors
    print("\nAnalyzing P1 errors...")
    p1_errors = analyzer.analyze_prototype_errors(p1_metrics, gt_data["queries"], "P1")
    print(f"Found {len(p1_errors)} P1 errors")
    
    # Analyze P2 errors
    print("\nAnalyzing P2 errors...")
    p2_errors = analyzer.analyze_prototype_errors(p2_metrics, gt_data["queries"], "P2")
    print(f"Found {len(p2_errors)} P2 errors")
    
    # Generate combined error table
    all_errors = p1_errors + p2_errors
    analyzer.generate_error_table(all_errors, results_dir / "error_analysis")
    
    # Statistical comparison
    print("\n" + "="*60)
    print("STATISTICAL COMPARISON: P1 vs P2")
    print("="*60)
    
    comp_analyzer = ComparativeAnalyzer(args.results_dir)
    comp_analyzer.generate_comparison_report(results_dir / "statistical_comparison.json")


if __name__ == "__main__":
    main()
