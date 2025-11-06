"""
Comprehensive Text Monotonicity Evaluation Script
Evaluates monotonicity for all text datasets and stores results in experiment format
"""

import sys
import os
import json
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.evaluation.monotonicity_evaluator import MonotonicityEvaluator
from src.data.data_manager import DataManager
from src.models.model_factory import ModelFactory
from src.explanations.explanation_factory import ExplanationFactory
from src.utils.config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TextMonotonicityEvaluator:
    """Evaluate monotonicity for all text datasets"""

    def __init__(self, config_path='configs/default_config.yaml', experiment_dir=None):
        """
        Initialize the evaluator

        Args:
            config_path: Path to configuration file
            experiment_dir: Directory to store results (if None, uses latest experiment)
        """
        # Load configuration
        self.config = load_config(config_path)

        # Set up experiment directory
        if experiment_dir:
            self.experiment_dir = Path(experiment_dir)
        else:
            # Use latest experiment directory
            results_dir = Path('results')
            experiment_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith('experiment_')])
            if experiment_dirs:
                self.experiment_dir = experiment_dirs[-1]
            else:
                # Create new experiment directory
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                self.experiment_dir = results_dir / f'experiment_{timestamp}'

        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        self.monotonicity_dir = self.experiment_dir / 'monotonicity_results'
        self.monotonicity_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Using experiment directory: {self.experiment_dir}")

        # Initialize components
        self.data_manager = DataManager(self.config)
        self.model_factory = ModelFactory(self.config.get('models', {}))
        self.explanation_factory = ExplanationFactory(self.config.get('explanations', {}))
        self.monotonicity_evaluator = MonotonicityEvaluator()

        # Get text datasets from config
        self.text_datasets = self._get_text_datasets()

    def _get_text_datasets(self):
        """Extract text dataset names from configuration"""
        text_datasets = []
        data_config = self.config.get('data', {})

        for dataset in data_config.get('text_datasets', []):
            if dataset.get('mandatory', False):
                text_datasets.append(dataset['name'])

        # Also include optional datasets if requested
        include_optional = self.config.get('include_optional_datasets', False)
        if include_optional:
            for dataset in data_config.get('text_datasets', []):
                if not dataset.get('mandatory', False):
                    text_datasets.append(dataset['name'])

        logger.info(f"Found {len(text_datasets)} text datasets: {text_datasets}")
        return text_datasets

    def evaluate_dataset(self, dataset_name, model_names=None, explanation_methods=None):
        """
        Evaluate monotonicity for a single text dataset

        Args:
            dataset_name: Name of the dataset
            model_names: List of model names to use (if None, uses all configured)
            explanation_methods: List of explanation methods (if None, uses all configured)

        Returns:
            Dictionary of results
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating dataset: {dataset_name}")
        logger.info(f"{'='*80}")

        results = {
            'dataset': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'models': {},
            'summary': {}
        }

        try:
            # Load dataset
            logger.info(f"Loading dataset: {dataset_name}")
            dataset = self.data_manager.load_dataset(dataset_name, data_type='text')

            if dataset is None:
                logger.error(f"Failed to load dataset: {dataset_name}")
                results['error'] = f"Failed to load dataset: {dataset_name}"
                return results

            X_train, X_test, y_train, y_test = dataset
            logger.info(f"Dataset loaded: Train={len(X_train)}, Test={len(X_test)} samples")

            # Get model names
            if model_names is None:
                model_names = list(self.config.get('models', {}).keys())

            # Get explanation methods
            if explanation_methods is None:
                explanation_methods = list(self.config.get('explanations', {}).keys())

            logger.info(f"Models to evaluate: {model_names}")
            logger.info(f"Explanation methods: {explanation_methods}")

            # Evaluate each model
            for model_name in model_names:
                logger.info(f"\n--- Training model: {model_name} ---")
                model_results = {
                    'explanations': {},
                    'model_performance': {}
                }

                try:
                    # Create and train model
                    model = self.model_factory.create_model(model_name, data_type='text')
                    model.fit(X_train, y_train)

                    # Get model performance
                    train_score = model.score(X_train, y_train)
                    test_score = model.score(X_test, y_test)
                    model_results['model_performance'] = {
                        'train_accuracy': float(train_score),
                        'test_accuracy': float(test_score)
                    }
                    logger.info(f"Model accuracy: Train={train_score:.3f}, Test={test_score:.3f}")

                    # Evaluate each explanation method
                    for exp_method in explanation_methods:
                        logger.info(f"  Evaluating explanation method: {exp_method}")

                        try:
                            # Generate explanations
                            explainer = self.explanation_factory.create_explainer(
                                exp_method,
                                model=model,
                                data_type='text',
                                training_data=(X_train, y_train)
                            )

                            # Generate explanations for test samples (limit to avoid long runtime)
                            max_samples = min(50, len(X_test))  # Limit to 50 samples for efficiency
                            explanations = []

                            for i in range(max_samples):
                                try:
                                    explanation = explainer.explain(X_test[i])
                                    if explanation:
                                        explanations.append(explanation)
                                except Exception as e:
                                    logger.warning(f"    Failed to explain sample {i}: {str(e)}")

                            logger.info(f"    Generated {len(explanations)} explanations")

                            if explanations:
                                # Evaluate monotonicity
                                monotonicity_result = self.monotonicity_evaluator.evaluate_monotonicity(
                                    explanations,
                                    model=model,
                                    data_type='text'
                                )

                                model_results['explanations'][exp_method] = {
                                    'monotonicity': monotonicity_result.get('monotonicity', 0.0),
                                    'num_explanations': len(explanations),
                                    'details': monotonicity_result
                                }

                                logger.info(f"    Monotonicity score: {monotonicity_result.get('monotonicity', 0.0):.4f}")
                            else:
                                logger.warning(f"    No valid explanations generated for {exp_method}")
                                model_results['explanations'][exp_method] = {
                                    'monotonicity': 0.0,
                                    'num_explanations': 0,
                                    'error': 'No valid explanations generated'
                                }

                        except Exception as e:
                            logger.error(f"    Error with explanation method {exp_method}: {str(e)}")
                            model_results['explanations'][exp_method] = {
                                'monotonicity': 0.0,
                                'error': str(e)
                            }

                    results['models'][model_name] = model_results

                except Exception as e:
                    logger.error(f"  Error with model {model_name}: {str(e)}")
                    results['models'][model_name] = {'error': str(e)}

            # Calculate summary statistics
            all_scores = []
            for model_name, model_data in results['models'].items():
                if 'explanations' in model_data:
                    for exp_name, exp_data in model_data['explanations'].items():
                        if 'monotonicity' in exp_data:
                            all_scores.append(exp_data['monotonicity'])

            if all_scores:
                results['summary'] = {
                    'mean_monotonicity': float(np.mean(all_scores)),
                    'std_monotonicity': float(np.std(all_scores)),
                    'min_monotonicity': float(np.min(all_scores)),
                    'max_monotonicity': float(np.max(all_scores)),
                    'num_evaluations': len(all_scores)
                }

        except Exception as e:
            logger.error(f"Error evaluating dataset {dataset_name}: {str(e)}")
            results['error'] = str(e)

        return results

    def evaluate_all_datasets(self, save_intermediate=True):
        """
        Evaluate monotonicity for all text datasets

        Args:
            save_intermediate: Save results after each dataset

        Returns:
            Dictionary of all results
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING COMPREHENSIVE TEXT MONOTONICITY EVALUATION")
        logger.info("="*80)

        all_results = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config,
            'datasets': {},
            'overall_summary': {}
        }

        # Evaluate each dataset
        for dataset_name in self.text_datasets:
            dataset_results = self.evaluate_dataset(dataset_name)
            all_results['datasets'][dataset_name] = dataset_results

            # Save intermediate results
            if save_intermediate:
                self.save_results(all_results, suffix='_intermediate')

        # Calculate overall summary
        all_scores = []
        dataset_summaries = []

        for dataset_name, dataset_data in all_results['datasets'].items():
            if 'summary' in dataset_data and dataset_data['summary']:
                dataset_summaries.append({
                    'dataset': dataset_name,
                    **dataset_data['summary']
                })
                if 'mean_monotonicity' in dataset_data['summary']:
                    all_scores.append(dataset_data['summary']['mean_monotonicity'])

        if all_scores:
            all_results['overall_summary'] = {
                'num_datasets': len(self.text_datasets),
                'datasets_evaluated': len(all_scores),
                'overall_mean_monotonicity': float(np.mean(all_scores)),
                'overall_std_monotonicity': float(np.std(all_scores)),
                'dataset_summaries': dataset_summaries
            }

        # Save final results
        self.save_results(all_results)

        logger.info("\n" + "="*80)
        logger.info("EVALUATION COMPLETE")
        logger.info(f"Results saved to: {self.monotonicity_dir}")
        logger.info("="*80)

        return all_results

    def save_results(self, results, suffix=''):
        """
        Save results to experiment directory

        Args:
            results: Results dictionary
            suffix: Optional suffix for filename
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save JSON results
        json_path = self.monotonicity_dir / f'text_monotonicity_results{suffix}_{timestamp}.json'
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to: {json_path}")

        # Save pickle for complete data preservation
        pickle_path = self.monotonicity_dir / f'text_monotonicity_results{suffix}_{timestamp}.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)

        # Create summary report
        self.create_summary_report(results, suffix)

    def create_summary_report(self, results, suffix=''):
        """
        Create a markdown summary report

        Args:
            results: Results dictionary
            suffix: Optional suffix for filename
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.monotonicity_dir / f'text_monotonicity_report{suffix}_{timestamp}.md'

        with open(report_path, 'w') as f:
            f.write("# Text Dataset Monotonicity Evaluation Report\n\n")
            f.write(f"**Generated:** {results['timestamp']}\n\n")
            f.write(f"**Experiment Directory:** {self.experiment_dir}\n\n")

            # Overall summary
            if 'overall_summary' in results and results['overall_summary']:
                f.write("## Overall Summary\n\n")
                summary = results['overall_summary']
                f.write(f"- **Datasets Evaluated:** {summary.get('datasets_evaluated', 0)}/{summary.get('num_datasets', 0)}\n")
                f.write(f"- **Mean Monotonicity:** {summary.get('overall_mean_monotonicity', 0):.4f}\n")
                f.write(f"- **Std Monotonicity:** {summary.get('overall_std_monotonicity', 0):.4f}\n\n")

            # Dataset-specific results
            f.write("## Dataset Results\n\n")
            for dataset_name, dataset_data in results['datasets'].items():
                f.write(f"### {dataset_name}\n\n")

                if 'error' in dataset_data:
                    f.write(f"**Error:** {dataset_data['error']}\n\n")
                    continue

                if 'summary' in dataset_data and dataset_data['summary']:
                    summary = dataset_data['summary']
                    f.write(f"- **Mean Monotonicity:** {summary.get('mean_monotonicity', 0):.4f}\n")
                    f.write(f"- **Std Monotonicity:** {summary.get('std_monotonicity', 0):.4f}\n")
                    f.write(f"- **Range:** [{summary.get('min_monotonicity', 0):.4f}, {summary.get('max_monotonicity', 0):.4f}]\n")
                    f.write(f"- **Evaluations:** {summary.get('num_evaluations', 0)}\n\n")

                # Model-specific results
                if 'models' in dataset_data:
                    f.write("#### Model Results\n\n")
                    for model_name, model_data in dataset_data['models'].items():
                        f.write(f"**{model_name}**\n")

                        if 'error' in model_data:
                            f.write(f"- Error: {model_data['error']}\n")
                            continue

                        if 'model_performance' in model_data:
                            perf = model_data['model_performance']
                            f.write(f"- Accuracy: Train={perf.get('train_accuracy', 0):.3f}, Test={perf.get('test_accuracy', 0):.3f}\n")

                        if 'explanations' in model_data:
                            f.write("- Explanation Methods:\n")
                            for exp_name, exp_data in model_data['explanations'].items():
                                mono_score = exp_data.get('monotonicity', 0)
                                num_exp = exp_data.get('num_explanations', 0)
                                f.write(f"  - {exp_name}: {mono_score:.4f} ({num_exp} samples)\n")
                        f.write("\n")

        logger.info(f"Report saved to: {report_path}")


def main():
    """Main function to run the evaluation"""
    import argparse

    parser = argparse.ArgumentParser(description='Evaluate monotonicity for text datasets')
    parser.add_argument('--config', default='configs/default_config.yaml', help='Path to configuration file')
    parser.add_argument('--experiment', default=None, help='Experiment directory to use (default: latest)')
    parser.add_argument('--dataset', default=None, help='Evaluate specific dataset only')
    parser.add_argument('--models', nargs='+', default=None, help='Specific models to evaluate')
    parser.add_argument('--methods', nargs='+', default=None, help='Specific explanation methods to evaluate')

    args = parser.parse_args()

    # Create evaluator
    evaluator = TextMonotonicityEvaluator(
        config_path=args.config,
        experiment_dir=args.experiment
    )

    # Run evaluation
    if args.dataset:
        # Evaluate single dataset
        results = evaluator.evaluate_dataset(
            args.dataset,
            model_names=args.models,
            explanation_methods=args.methods
        )
        evaluator.save_results({'datasets': {args.dataset: results}})
    else:
        # Evaluate all datasets
        results = evaluator.evaluate_all_datasets()

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

    if 'overall_summary' in results and results['overall_summary']:
        summary = results['overall_summary']
        print(f"\nOverall Results:")
        print(f"  Datasets Evaluated: {summary.get('datasets_evaluated', 0)}")
        print(f"  Mean Monotonicity: {summary.get('overall_mean_monotonicity', 0):.4f}")
        print(f"  Std Monotonicity: {summary.get('overall_std_monotonicity', 0):.4f}")

    print(f"\nResults saved to: {evaluator.monotonicity_dir}")


if __name__ == "__main__":
    main()