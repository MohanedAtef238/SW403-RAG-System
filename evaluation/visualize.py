"""
Visualization tools for evaluation results.

Generates charts and graphs for P1 vs P2 comparison, category breakdowns,
and error distributions.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from evaluation.metrics import EvaluationMetrics


class EvaluationVisualizer:
    """Creates visualizations for evaluation results."""
    
    def __init__(self, results_dir: str = "results"):
        """
        Initialize visualizer.
        
        Args:
            results_dir: Directory containing evaluation results
        """
        self.results_dir = Path(results_dir)
        self.output_dir = self.results_dir / "figures"
        self.output_dir.mkdir(exist_ok=True)
        
        # Set style
        sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 10
        
        # Load data
        self.p1_data = self._load_results("p1_results.json")
        self.p2_data = self._load_results("p2_results.json")
    
    def _load_results(self, filename: str) -> Dict[str, Any]:
        """Load results from JSON file."""
        path = self.results_dir / filename
        if path.exists():
            with open(path, "r") as f:
                return json.load(f)
        return {}
    
    def plot_accuracy_by_category(self):
        """
        Plot accuracy comparison by query category.
        
        Creates bar chart showing exact match rates for each category.
        """
        if not self.p1_data or not self.p2_data:
            print("Missing data for accuracy comparison")
            return
        
        categories = list(self.p1_data["aggregated"]["by_category"].keys())
        
        p1_accuracy = [
            self.p1_data["aggregated"]["by_category"][cat]["exact_match_rate"]
            for cat in categories
        ]
        p2_accuracy = [
            self.p2_data["aggregated"]["by_category"][cat]["exact_match_rate"]
            for cat in categories
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, p1_accuracy, width, label='P1 (Regex)', color='#FF6B6B')
        bars2 = ax.bar(x + width/2, p2_accuracy, width, label='P2 (AST)', color='#4ECDC4')
        
        ax.set_xlabel('Query Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Exact Match Rate', fontsize=12, fontweight='bold')
        ax.set_title('Accuracy Comparison by Query Category', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([cat.replace('_', ' ').title() for cat in categories])
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_path = self.output_dir / "accuracy_by_category.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_metrics_comparison(self):
        """
        Plot comparison of all key metrics.
        
        Creates grouped bar chart for MRR, Precision@K, Recall@K, NDCG@K.
        """
        if not self.p1_data or not self.p2_data:
            print("Missing data for metrics comparison")
            return
        
        metrics = {
            'MRR': ('mrr', 'Mean Reciprocal Rank'),
            'Precision@5': ('avg_precision_at_k', 'Precision@5'),
            'Recall@5': ('avg_recall_at_k', 'Recall@5'),
            'NDCG@5': ('avg_ndcg_at_k', 'NDCG@5')
        }
        
        metric_names = list(metrics.keys())
        p1_values = [self.p1_data["aggregated"]["overall"][metrics[m][0]] for m in metric_names]
        p2_values = [self.p2_data["aggregated"]["overall"][metrics[m][0]] for m in metric_names]
        
        x = np.arange(len(metric_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, p1_values, width, label='P1 (Regex)', color='#FF6B6B')
        bars2 = ax.bar(x + width/2, p2_values, width, label='P2 (AST)', color='#4ECDC4')
        
        ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Information Retrieval Metrics Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names)
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}',
                       ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_path = self.output_dir / "metrics_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_similarity_distributions(self):
        """
        Plot similarity score distributions for P1 vs P2.
        
        Creates overlapping histograms showing score distributions.
        """
        if not self.p1_data or not self.p2_data:
            print("Missing data for similarity distributions")
            return
        
        p1_scores = [m["top_result_score"] for m in self.p1_data["metrics"] if m["top_result_score"] > 0]
        p2_scores = [m["top_result_score"] for m in self.p2_data["metrics"] if m["top_result_score"] > 0]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.hist(p1_scores, bins=20, alpha=0.6, label='P1 (Regex)', color='#FF6B6B', edgecolor='black')
        ax.hist(p2_scores, bins=20, alpha=0.6, label='P2 (AST)', color='#4ECDC4', edgecolor='black')
        
        ax.set_xlabel('Similarity Score', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title('Top Result Similarity Score Distribution', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "similarity_distributions.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_partial_credit_boxplot(self):
        """
        Plot boxplot of partial credit scores by category.
        
        Shows distribution and outliers for each category.
        """
        if not self.p1_data or not self.p2_data:
            print("Missing data for boxplot")
            return
        
        # Prepare data
        data = []
        for metrics, prototype in [(self.p1_data["metrics"], "P1"), (self.p2_data["metrics"], "P2")]:
            for m in metrics:
                data.append({
                    "Prototype": prototype,
                    "Category": m["category"].replace('_', ' ').title(),
                    "Partial Credit": m["partial_credit_score"]
                })
        
        df = pd.DataFrame(data)
        
        fig, ax = plt.subplots(figsize=(14, 6))
        sns.boxplot(data=df, x="Category", y="Partial Credit", hue="Prototype",
                   palette={"P1": "#FF6B6B", "P2": "#4ECDC4"}, ax=ax)
        
        ax.set_xlabel('Query Category', fontsize=12, fontweight='bold')
        ax.set_ylabel('Partial Credit Score', fontsize=12, fontweight='bold')
        ax.set_title('Partial Credit Score Distribution by Category', fontsize=14, fontweight='bold')
        ax.set_ylim(-0.05, 1.05)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "partial_credit_boxplot.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def plot_confidence_intervals(self):
        """
        Plot confidence intervals for key metrics.
        
        Shows mean ± 95% CI for P1 and P2.
        """
        comp_path = self.results_dir / "statistical_comparison.json"
        if not comp_path.exists():
            print("Statistical comparison not found. Run analyzer.py first.")
            return
        
        with open(comp_path, "r") as f:
            comp_data = json.load(f)
        
        metrics = ["partial_credit_score", "reciprocal_rank", "ndcg_at_k"]
        metric_labels = ["Partial Credit", "MRR", "NDCG@5"]
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(metrics))
        width = 0.35
        
        for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
            comp = comp_data["comparisons"][metric]
            
            # P1
            p1_mean = comp["p1"]["mean"]
            p1_ci = comp["p1"]["ci_95"]
            p1_err = [[p1_mean - p1_ci[0]], [p1_ci[1] - p1_mean]]
            
            # P2
            p2_mean = comp["p2"]["mean"]
            p2_ci = comp["p2"]["ci_95"]
            p2_err = [[p2_mean - p2_ci[0]], [p2_ci[1] - p2_mean]]
            
            ax.errorbar(x[i] - width/2, p1_mean, yerr=p1_err, fmt='o', 
                       markersize=8, capsize=5, label='P1' if i == 0 else '', color='#FF6B6B')
            ax.errorbar(x[i] + width/2, p2_mean, yerr=p2_err, fmt='o',
                       markersize=8, capsize=5, label='P2' if i == 0 else '', color='#4ECDC4')
        
        ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Mean Scores with 95% Confidence Intervals', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "confidence_intervals.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
        plt.close()
    
    def generate_all_visualizations(self):
        """Generate all visualization plots."""
        print("\nGenerating visualizations...")
        print("="*60)
        
        self.plot_accuracy_by_category()
        self.plot_metrics_comparison()
        self.plot_similarity_distributions()
        self.plot_partial_credit_boxplot()
        self.plot_confidence_intervals()
        
        print("="*60)
        print(f"All visualizations saved to: {self.output_dir}")


def main():
    """Main entry point for visualization generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate evaluation visualizations")
    parser.add_argument(
        "--results-dir",
        default="results",
        help="Directory containing evaluation results"
    )
    
    args = parser.parse_args()
    
    visualizer = EvaluationVisualizer(args.results_dir)
    visualizer.generate_all_visualizations()


if __name__ == "__main__":
    main()
