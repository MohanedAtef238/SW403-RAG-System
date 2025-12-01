"""
Batch query runner for evaluating RAG prototypes.

Executes ground truth queries against P1 and P2 APIs, collects results,
and calculates evaluation metrics.
"""

import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
import requests
from dataclasses import asdict

from evaluation.metrics import (
    RetrievalResult,
    GroundTruth,
    EvaluationMetrics,
    evaluate_query,
    aggregate_metrics
)


class PrototypeRunner:
    """Runs queries against a single prototype API."""
    
    def __init__(self, base_url: str, prototype_name: str):
        """
        Initialize runner for a prototype.
        
        Args:
            base_url: Base URL for the API (e.g., "http://localhost:8001")
            prototype_name: Name for logging (e.g., "P1", "P2")
        """
        self.base_url = base_url
        self.prototype_name = prototype_name
    
    def check_health(self) -> bool:
        """Check if the prototype API is accessible."""
        try:
            # Services expose /health; root path may not exist
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            print(f"[{self.prototype_name}] Health check failed: {e}")
            return False
    
    def query(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.0) -> Dict[str, Any]:
        """
        Execute a single query against the prototype API.
        
        Args:
            query_text: Query string
            top_k: Number of results to return
            similarity_threshold: Minimum similarity score
        
        Returns:
            API response as dictionary
        """
        try:
            response = requests.post(
                f"{self.base_url}/query",
                json={
                    "query": query_text,
                    "top_k": top_k,
                    "similarity_threshold": similarity_threshold
                },
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"[{self.prototype_name}] Query failed: {e}")
            return {"results": [], "error": str(e)}
    
    def parse_results(self, api_response: Dict[str, Any]) -> List[RetrievalResult]:
        """
        Parse API response into RetrievalResult objects.
        
        Args:
            api_response: Response from /query endpoint
        
        Returns:
            List of RetrievalResult objects
        """
        results = []
        api_results = api_response.get("results", [])
        
        for rank, item in enumerate(api_results, start=1):
            results.append(RetrievalResult(
                function_name=item.get("function_name", ""),
                file_path=item.get("file_path", ""),
                # P1/P2 return 'similarity_score'; fall back to 'score' if present
                score=item.get("similarity_score", item.get("score", 0.0)),
                rank=rank
            ))
        
        return results


class EvaluationRunner:
    """Orchestrates evaluation across ground truth queries and prototypes."""
    
    def __init__(
        self,
        ground_truth_path: str,
        p1_url: str = "http://localhost:8001",
        p2_url: str = "http://localhost:8002",
        top_k: int = 5
    ):
        """
        Initialize evaluation runner.
        
        Args:
            ground_truth_path: Path to ground_truth.json file
            p1_url: Base URL for P1 API
            p2_url: Base URL for P2 API
            top_k: Number of results to retrieve per query
        """
        self.ground_truth_path = Path(ground_truth_path)
        self.top_k = top_k
        
        self.p1_runner = PrototypeRunner(p1_url, "P1")
        self.p2_runner = PrototypeRunner(p2_url, "P2")
        
        self.ground_truth_queries: List[GroundTruth] = []
        self.load_ground_truth()
    
    def load_ground_truth(self):
        """Load ground truth queries from JSON file."""
        with open(self.ground_truth_path, "r") as f:
            data = json.load(f)
        
        for query_item in data["queries"]:
            self.ground_truth_queries.append(GroundTruth(
                query_id=query_item["id"],
                query=query_item["query"],
                category=query_item["category"],
                expected_function=query_item["expected_function"],
                expected_file=query_item["expected_file"]
            ))
        
        print(f"Loaded {len(self.ground_truth_queries)} ground truth queries")
    
    def check_services(self) -> bool:
        """Check if both prototype services are running."""
        p1_ok = self.p1_runner.check_health()
        p2_ok = self.p2_runner.check_health()
        
        if not p1_ok:
            print("[ERROR] P1 service is not accessible")
        if not p2_ok:
            print("[ERROR] P2 service is not accessible")
        
        return p1_ok and p2_ok
    
    def run_evaluation(
        self,
        output_dir: str = "results",
        run_p1: bool = True,
        run_p2: bool = True
    ) -> Dict[str, Any]:
        """
        Run complete evaluation on both prototypes.
        
        Args:
            output_dir: Directory to save results
            run_p1: Whether to evaluate P1
            run_p2: Whether to evaluate P2
        
        Returns:
            Dictionary with evaluation results for both prototypes
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ground_truth_file": str(self.ground_truth_path),
            "top_k": self.top_k,
            "total_queries": len(self.ground_truth_queries)
        }
        
        # Run P1 evaluation
        if run_p1:
            print("\n" + "="*60)
            print("EVALUATING P1 (Regex-based Chunking)")
            print("="*60)
            p1_metrics = self.evaluate_prototype(self.p1_runner)
            p1_aggregated = aggregate_metrics(p1_metrics)
            
            results["p1"] = {
                "metrics": [asdict(m) for m in p1_metrics],
                "aggregated": p1_aggregated
            }
            
            # Save P1 results
            with open(output_path / "p1_results.json", "w") as f:
                json.dump(results["p1"], f, indent=2)
            print(f"\n[P1] Results saved to {output_path / 'p1_results.json'}")
        
        # Run P2 evaluation
        if run_p2:
            print("\n" + "="*60)
            print("EVALUATING P2 (AST-based Semantic Chunking)")
            print("="*60)
            p2_metrics = self.evaluate_prototype(self.p2_runner)
            p2_aggregated = aggregate_metrics(p2_metrics)
            
            results["p2"] = {
                "metrics": [asdict(m) for m in p2_metrics],
                "aggregated": p2_aggregated
            }
            
            # Save P2 results
            with open(output_path / "p2_results.json", "w") as f:
                json.dump(results["p2"], f, indent=2)
            print(f"\n[P2] Results saved to {output_path / 'p2_results.json'}")
        
        # Save combined results
        with open(output_path / "evaluation_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n[DONE] Combined results saved to {output_path / 'evaluation_results.json'}")
        
        return results
    
    def evaluate_prototype(self, runner: PrototypeRunner) -> List[EvaluationMetrics]:
        """
        Evaluate a single prototype on all ground truth queries.
        
        Args:
            runner: PrototypeRunner for the prototype to evaluate
        
        Returns:
            List of EvaluationMetrics for all queries
        """
        all_metrics = []
        
        for i, gt in enumerate(self.ground_truth_queries, start=1):
            print(f"\n[{runner.prototype_name}] Query {i}/{len(self.ground_truth_queries)}: {gt.query[:60]}...")
            
            # Execute query
            start_time = time.time()
            api_response = runner.query(gt.query, top_k=self.top_k)
            query_time = time.time() - start_time
            
            # Parse results
            retrieval_results = runner.parse_results(api_response)
            
            # Evaluate
            metrics = evaluate_query(retrieval_results, gt, k=self.top_k)
            
            # Log results
            print(f"  ✓ Completed in {query_time:.2f}s")
            print(f"  - Exact match: {metrics.exact_match}")
            print(f"  - Partial credit: {metrics.partial_credit_score:.2f}")
            print(f"  - Found in top-{self.top_k}: {metrics.found_in_top_k}")
            if retrieval_results:
                print(f"  - Top result: {metrics.top_result_function} ({metrics.top_result_score:.3f})")
            
            all_metrics.append(metrics)
        
        # Print summary
        print(f"\n[{runner.prototype_name}] SUMMARY")
        print(f"  - Exact matches: {sum(m.exact_match for m in all_metrics)}/{len(all_metrics)}")
        print(f"  - Found in top-{self.top_k}: {sum(m.found_in_top_k for m in all_metrics)}/{len(all_metrics)}")
        print(f"  - Avg partial credit: {sum(m.partial_credit_score for m in all_metrics) / len(all_metrics):.3f}")
        print(f"  - MRR: {sum(m.reciprocal_rank for m in all_metrics) / len(all_metrics):.3f}")
        
        return all_metrics


def main():
    """Main entry point for running evaluations."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RAG system evaluation")
    parser.add_argument(
        "--ground-truth",
        default="shared_data/ground_truth.json",
        help="Path to ground truth JSON file"
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--p1-url",
        default="http://localhost:8001",
        help="P1 API base URL"
    )
    parser.add_argument(
        "--p2-url",
        default="http://localhost:8002",
        help="P2 API base URL"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results to retrieve per query"
    )
    parser.add_argument(
        "--skip-p1",
        action="store_true",
        help="Skip P1 evaluation"
    )
    parser.add_argument(
        "--skip-p2",
        action="store_true",
        help="Skip P2 evaluation"
    )
    
    args = parser.parse_args()
    
    # Initialize runner
    runner = EvaluationRunner(
        ground_truth_path=args.ground_truth,
        p1_url=args.p1_url,
        p2_url=args.p2_url,
        top_k=args.top_k
    )
    
    # Check services
    if not runner.check_services():
        print("\n[ERROR] One or more services are not accessible. Start services with:")
        print("  docker compose up --build")
        return
    
    # Run evaluation
    runner.run_evaluation(
        output_dir=args.output_dir,
        run_p1=not args.skip_p1,
        run_p2=not args.skip_p2
    )


if __name__ == "__main__":
    main()
