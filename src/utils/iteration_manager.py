"""
Utility module for managing and analyzing separate iteration result files
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from collections import defaultdict

try:
    import pandas as pd
except ImportError:
    pd = None


class IterationManager:
    def get_all_iterations(self) -> List[Dict[str, Any]]:
        """
        Return all loaded iteration dicts (with filename info) for pretty-printing and analysis.
        """
        iteration_files = self.get_iteration_files()
        iterations = []
        for iteration_file in iteration_files:
            iteration_data = self.load_iteration(iteration_file)
            if iteration_data:
                iteration_data['filename'] = str(iteration_file.name)
            iterations.append(iteration_data)
        return iterations
    """
    Manager for handling separate iteration result files
    """
    
    def __init__(self, results_dir: Path):
        """
        Initialize iteration manager
        
        Args:
            results_dir: Directory containing iteration results
        """
        self.results_dir = Path(results_dir)
        self.logger = logging.getLogger(__name__)
        
        # Ensure results directory exists
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def get_iteration_files(self, pattern: str = "iteration_*.json") -> List[Path]:
        """
        Get all iteration files matching the pattern
        
        Args:
            pattern: Glob pattern to match files
            
        Returns:
            List of iteration file paths
        """
        return sorted(self.results_dir.glob(pattern))
    
    def load_iteration(self, iteration_file: Path) -> Dict[str, Any]:
        """
        Load a single iteration result
        
        Args:
            iteration_file: Path to iteration file
            
        Returns:
            Loaded iteration data
        """
        try:
            with open(iteration_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Failed to load iteration file {iteration_file}: {e}")
            return {}
    
    def load_all_iterations(self) -> List[Dict[str, Any]]:
        """
        Load all iteration results
        
        Returns:
            List of all iteration data
        """
        iteration_files = self.get_iteration_files()
        iterations = []
        
        for iteration_file in iteration_files:
            iteration_data = self.load_iteration(iteration_file)
            if iteration_data:
                iterations.append(iteration_data)
        
        return iterations
    
    def get_iteration_summary(self) -> Dict[str, Any]:
        """
        Get summary of all iterations
        
        Returns:
            Summary statistics and metadata
        """
        iterations = self.load_all_iterations()
        
        if not iterations:
            return {
                'total_iterations': 0,
                'datasets': [],
                'models': [],
                'methods': [],
                'validation_stats': {}
            }
        
        # Extract unique values
        datasets = set()
        models = set()
        methods = set()
        validation_statuses = []
        
        for iteration in iterations:
            result_data = iteration.get('result_data', {})
            datasets.add(result_data.get('dataset', 'unknown'))
            models.add(result_data.get('model', 'unknown'))
            methods.add(result_data.get('explanation_method', 'unknown'))
            validation_statuses.append(result_data.get('validation_status', False))
        
        # Calculate validation statistics
        valid_count = sum(validation_statuses)
        total_count = len(validation_statuses)
        
        return {
            'total_iterations': len(iterations),
            'datasets': sorted(list(datasets)),
            'models': sorted(list(models)),
            'methods': sorted(list(methods)),
            'validation_stats': {
                'valid_count': valid_count,
                'invalid_count': total_count - valid_count,
                'valid_ratio': valid_count / total_count if total_count > 0 else 0
            },
            'date_range': {
                'earliest': min(it.get('iteration_info', {}).get('timestamp', '') for it in iterations),
                'latest': max(it.get('iteration_info', {}).get('timestamp', '') for it in iterations)
            }
        }
    
    def filter_iterations(self, 
                         dataset: Optional[str] = None,
                         model: Optional[str] = None,
                         method: Optional[str] = None,
                         validation_status: Optional[bool] = None) -> List[Dict[str, Any]]:
        """
        Filter iterations based on criteria
        
        Args:
            dataset: Filter by dataset name
            model: Filter by model name
            method: Filter by explanation method
            validation_status: Filter by validation status
            
        Returns:
            Filtered list of iterations
        """
        iterations = self.load_all_iterations()
        filtered = []
        
        for iteration in iterations:
            result_data = iteration.get('result_data', {})
            
            # Apply filters
            if dataset and result_data.get('dataset') != dataset:
                continue
            if model and result_data.get('model') != model:
                continue
            if method and result_data.get('explanation_method') != method:
                continue
            if validation_status is not None and result_data.get('validation_status') != validation_status:
                continue
            
            filtered.append(iteration)
        
        return filtered
    
    def create_metrics_dataframe(self) -> pd.DataFrame:
        """
        Create a pandas DataFrame with all evaluation metrics
        
        Returns:
            DataFrame with metrics from all iterations
        """
        iterations = self.load_all_iterations()
        
        if not iterations:
            return pd.DataFrame()
        
        rows = []
        for iteration in iterations:
            iteration_info = iteration.get('iteration_info', {})
            result_data = iteration.get('result_data', {})
            evaluation_results = result_data.get('evaluation_results', {})
            
            row = {
                'iteration_number': iteration_info.get('iteration_number'),
                'timestamp': iteration_info.get('timestamp'),
                'iteration_key': iteration_info.get('iteration_key'),
                'dataset': result_data.get('dataset'),
                'model': result_data.get('model'),
                'explanation_method': result_data.get('explanation_method'),
                'validation_status': result_data.get('validation_status', False)
            }
            
            # Add evaluation metrics
            if isinstance(evaluation_results, dict):
                for metric_name, metric_value in evaluation_results.items():
                    row[f'metric_{metric_name}'] = metric_value
            
            # Add model performance metrics
            model_performance = result_data.get('model_performance', {})
            if isinstance(model_performance, dict):
                for perf_name, perf_value in model_performance.items():
                    row[f'perf_{perf_name}'] = perf_value
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def create_comparison_report(self, output_file: Optional[Path] = None) -> str:
        """
        Create a comprehensive comparison report
        
        Args:
            output_file: Optional file to save the report
            
        Returns:
            Formatted report string
        """
        summary = self.get_iteration_summary()
        df = self.create_metrics_dataframe()
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("XAI BENCHMARKING - ITERATION COMPARISON REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary statistics
        report_lines.append("📊 SUMMARY STATISTICS")
        report_lines.append(f"  - Total iterations: {summary['total_iterations']}")
        report_lines.append(f"  - Datasets: {', '.join(summary['datasets'])}")
        report_lines.append(f"  - Models: {', '.join(summary['models'])}")
        report_lines.append(f"  - Methods: {', '.join(summary['methods'])}")
        report_lines.append("")
        
        # Validation statistics
        val_stats = summary['validation_stats']
        report_lines.append("🔍 VALIDATION STATISTICS")
        report_lines.append(f"  - Valid iterations: {val_stats['valid_count']}")
        report_lines.append(f"  - Invalid iterations: {val_stats['invalid_count']}")
        report_lines.append(f"  - Validation success rate: {val_stats['valid_ratio']:.2%}")
        report_lines.append("")
        
        if not df.empty:
            # Performance comparison by dataset
            report_lines.append("📈 PERFORMANCE BY DATASET")
            for dataset in summary['datasets']:
                dataset_df = df[df['dataset'] == dataset]
                if not dataset_df.empty:
                    report_lines.append(f"  {dataset}:")
                    # Find best performing model-method combination
                    metric_cols = [col for col in df.columns if col.startswith('metric_')]
                    if metric_cols:
                        best_idx = dataset_df[metric_cols[0]].idxmax()
                        best_row = dataset_df.loc[best_idx]
                        report_lines.append(f"    Best: {best_row['model']} + {best_row['explanation_method']}")
                        report_lines.append(f"    Score: {best_row[metric_cols[0]]:.4f}")
            report_lines.append("")
            
            # Performance comparison by model
            report_lines.append("🤖 PERFORMANCE BY MODEL")
            for model in summary['models']:
                model_df = df[df['model'] == model]
                if not model_df.empty:
                    report_lines.append(f"  {model}:")
                    metric_cols = [col for col in df.columns if col.startswith('metric_')]
                    if metric_cols:
                        avg_score = model_df[metric_cols[0]].mean()
                        report_lines.append(f"    Average score: {avg_score:.4f}")
            report_lines.append("")
            
            # Performance comparison by method
            report_lines.append("🔍 PERFORMANCE BY EXPLANATION METHOD")
            for method in summary['methods']:
                method_df = df[df['explanation_method'] == method]
                if not method_df.empty:
                    report_lines.append(f"  {method}:")
                    metric_cols = [col for col in df.columns if col.startswith('metric_')]
                    if metric_cols:
                        avg_score = method_df[metric_cols[0]].mean()
                        report_lines.append(f"    Average score: {avg_score:.4f}")
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        # Save report if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report)
            self.logger.info(f"Comparison report saved to {output_file}")
        
        return report
    
    def export_to_csv(self, output_file: Path):
        """
        Export all iteration data to CSV
        
        Args:
            output_file: Output CSV file path
        """
        df = self.create_metrics_dataframe()
        
        if not df.empty:
            df.to_csv(output_file, index=False)
            self.logger.info(f"Iteration data exported to {output_file}")
        else:
            self.logger.warning("No iteration data to export")
    
    def get_failed_iterations(self) -> List[Dict[str, Any]]:
        """
        Get iterations that failed validation
        
        Returns:
            List of failed iterations
        """
        return self.filter_iterations(validation_status=False)
    
    def get_best_performing_combinations(self, metric: str = 'faithfulness', top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Get top performing dataset-model-method combinations
        
        Args:
            metric: Metric to rank by
            top_k: Number of top combinations to return
            
        Returns:
            List of top performing combinations
        """
        df = self.create_metrics_dataframe()
        
        if df.empty:
            return []
        
        metric_col = f'metric_{metric}'
        if metric_col not in df.columns:
            self.logger.warning(f"Metric '{metric}' not found in data")
            return []
        
        # Sort by metric and get top k
        top_df = df.nlargest(top_k, metric_col)
        
        results = []
        for _, row in top_df.iterrows():
            results.append({
                'dataset': row['dataset'],
                'model': row['model'],
                'explanation_method': row['explanation_method'],
                'score': row[metric_col],
                'iteration_key': row['iteration_key']
            })
        
        return results
    
    def cleanup_old_iterations(self, days_old: int = 30):
        """
        Clean up iteration files older than specified days
        
        Args:
            days_old: Remove files older than this many days
        """
        from datetime import timedelta
        
        cutoff_date = datetime.now() - timedelta(days=days_old)
        iteration_files = self.get_iteration_files()
        
        removed_count = 0
        for iteration_file in iteration_files:
            try:
                # Get file modification time
                mtime = datetime.fromtimestamp(iteration_file.stat().st_mtime)
                if mtime < cutoff_date:
                    iteration_file.unlink()
                    removed_count += 1
                    self.logger.info(f"Removed old iteration file: {iteration_file}")
            except Exception as e:
                self.logger.error(f"Failed to remove {iteration_file}: {e}")
        
        self.logger.info(f"Cleaned up {removed_count} old iteration files")


def analyze_iterations(results_dir: Path) -> str:
    """
    Convenience function to analyze iterations and generate report
    
    Args:
        results_dir: Directory containing iteration results
        
    Returns:
        Analysis report
    """
    manager = IterationManager(results_dir)
    return manager.create_comparison_report()


def export_iterations_to_csv(results_dir: Path, output_file: Path):
    """
    Convenience function to export iterations to CSV
    
    Args:
        results_dir: Directory containing iteration results
        output_file: Output CSV file path
    """
    manager = IterationManager(results_dir)
    manager.export_to_csv(output_file) 