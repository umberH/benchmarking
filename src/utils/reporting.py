"""
Reporting utilities for XAI benchmarking
"""

import logging
from pathlib import Path
from typing import Dict, Any, List
import json
from contextlib import suppress
import importlib

# Optional pandas dependency for prettier tables
pd = None
try:  # pragma: no cover - optional import
    pd = importlib.import_module('pandas')  # type: ignore
except Exception:
    pd = None


class ReportGenerator:
    """Generate reports from benchmark results"""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
    
    def generate_reports(self, results: Dict[str, Any]):
        """Generate comprehensive reports"""
        self.logger.info("Generating benchmark reports")
        
        # Save detailed results
        self._save_detailed_results(results)
        
        # Generate summary report
        self._generate_summary_report(results)
        
        # Generate comparison report
        self._generate_comparison_report(results)

        # Print summary tables to console
        self.print_summary_tables(results)
    
    def _save_detailed_results(self, results: Dict[str, Any]):
        """Save detailed results to JSON"""
        results_file = self.output_dir / "detailed_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Detailed results saved to {results_file}")
    
    def _generate_summary_report(self, results: Dict[str, Any]):
        """Generate summary report"""
        summary = {
            'experiment_info': results.get('experiment_info', {}),
            'summary': {
                'n_datasets': len(results.get('dataset_results', {})),
                'n_models': len(results.get('model_results', {})),
                'n_explanations': len(results.get('explanation_results', {})),
                'n_evaluations': len(results.get('evaluation_results', {}))
            }
        }
        
        summary_file = self.output_dir / "summary_report.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Summary report saved to {summary_file}")
    
    def _generate_comparison_report(self, results: Dict[str, Any]):
        """Generate comparison report"""
        comparison = {
            'model_performance': self._extract_model_performance(results),
            'explanation_metrics': self._extract_explanation_metrics(results)
        }
        
        comparison_file = self.output_dir / "comparison_report.json"
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
        
        self.logger.info(f"Comparison report saved to {comparison_file}")

    # -------- Pretty tables (console + CSV) --------
    def _build_model_performance_rows(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for key, model_result in results.get('model_results', {}).items():
            perf = model_result.get('performance', {})
            rows.append({
                'Key': key,
                'Model': model_result.get('model_info', {}).get('name', ''),
                'Accuracy': round(perf.get('accuracy', 0.0), 4),
                'F1': round(perf.get('f1_score', 0.0), 4),
                'Training Time (s)': round(model_result.get('training_time', 0.0), 3),
            })
        return rows

    def _build_explanation_metric_rows(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for key, eval_result in results.get('evaluation_results', {}).items():
            rows.append({
                'Key': key,
                'Time/Explanation (s)': round(eval_result.get('time_complexity', 0.0), 4),
                'Faithfulness': round(eval_result.get('faithfulness', 0.0), 4),
                'Monotonicity': round(eval_result.get('monotonicity', 0.0), 4),
                'Completeness': round(eval_result.get('completeness', 0.0), 4),
                'Stability': round(eval_result.get('stability', 0.0), 4),
                'Consistency': round(eval_result.get('consistency', 0.0), 4),
                'Sparsity': round(eval_result.get('sparsity', 0.0), 4),
                'Simplicity': round(eval_result.get('simplicity', 0.0), 4),
            })
        # Interactive runs (if present)
        for run in results.get('interactive_runs', []) or []:
            metrics = run.get('evaluation_results', {})
            rows.append({
                'Key': f"{run.get('dataset','')}_{run.get('model','')}_{run.get('explanation_method','')}",
                'Time/Explanation (s)': round(metrics.get('time_complexity', 0.0), 4),
                'Faithfulness': round(metrics.get('faithfulness', 0.0), 4),
                'Monotonicity': round(metrics.get('monotonicity', 0.0), 4),
                'Completeness': round(metrics.get('completeness', 0.0), 4),
                'Stability': round(metrics.get('stability', 0.0), 4),
                'Consistency': round(metrics.get('consistency', 0.0), 4),
                'Sparsity': round(metrics.get('sparsity', 0.0), 4),
                'Simplicity': round(metrics.get('simplicity', 0.0), 4),
            })
        return rows

    def print_summary_tables(self, results: Dict[str, Any]):
        """Pretty-print summary tables in the console and save CSVs."""
        # Model Performance table
        model_rows = self._build_model_performance_rows(results)
        if model_rows:
            if pd is not None:
                df_models = pd.DataFrame(model_rows)
                print("\n=== Model Performance ===")
                print(df_models.to_string(index=False))
                with suppress(Exception):
                    df_models.to_csv(self.output_dir / 'model_performance.csv', index=False)
            else:
                print("\n=== Model Performance ===")
                headers = model_rows[0].keys()
                print(" | ".join(headers))
                for r in model_rows:
                    print(" | ".join(str(r[h]) for h in headers))
        else:
            self.logger.info("No model performance entries to display")

        # Explanation Metrics table
        exp_rows = self._build_explanation_metric_rows(results)
        if exp_rows:
            if pd is not None:
                df_exp = pd.DataFrame(exp_rows)
                print("\n=== Explanation Metrics ===")
                print(df_exp.to_string(index=False))
                with suppress(Exception):
                    df_exp.to_csv(self.output_dir / 'explanation_metrics.csv', index=False)
            else:
                print("\n=== Explanation Metrics ===")
                headers = exp_rows[0].keys()
                print(" | ".join(headers))
                for r in exp_rows:
                    print(" | ".join(str(r[h]) for h in headers))
        else:
            self.logger.info("No explanation metrics entries to display")
    
    def _extract_model_performance(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract model performance metrics"""
        model_performance = {}
        
        for model_key, model_result in results.get('model_results', {}).items():
            performance = model_result.get('performance', {})
            model_performance[model_key] = {
                'accuracy': performance.get('accuracy', 0.0),
                'f1_score': performance.get('f1_score', 0.0),
                'training_time': model_result.get('training_time', 0.0)
            }
        
        return model_performance
    
    def _extract_explanation_metrics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract explanation evaluation metrics"""
        explanation_metrics = {}
        
        for eval_key, eval_result in results.get('evaluation_results', {}).items():
            explanation_metrics[eval_key] = {
                'time_complexity': eval_result.get('time_complexity', 0.0),
                'faithfulness': eval_result.get('faithfulness', 0.0),
                'stability': eval_result.get('stability', 0.0),
                'sparsity': eval_result.get('sparsity', 0.0)
            }
        
        return explanation_metrics 