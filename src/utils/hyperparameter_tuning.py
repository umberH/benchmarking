"""
Hyperparameter tuning utilities for XAI benchmarking
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
from datetime import datetime
import numpy as np
import warnings

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    from sklearn.model_selection import GridSearchCV, cross_val_score
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class HyperparameterTuner:
    """
    Comprehensive hyperparameter tuning for XAI benchmarking models
    """
    
    def __init__(self, config: Dict[str, Any] = None, output_dir: Path = None):
        """
        Initialize hyperparameter tuner
        
        Args:
            config: Configuration dictionary with tuning parameters
            output_dir: Directory to save tuning results
        """
        self.config = config or {}
        self.output_dir = Path(output_dir) if output_dir else Path("tuning_results")
        self.logger = logging.getLogger(__name__)
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Default tuning configuration
        self.tuning_config = {
            'cv_folds': 5,
            'scoring': 'accuracy',
            'n_jobs': -1,
            'verbose': 1,
            'save_best_params': True,
            'save_all_results': True,
            'optimization_method': 'grid_search',  # 'grid_search' or 'optuna'
            'n_trials': 100,  # for optuna
            'timeout': 3600,  # 1 hour timeout
        }
        
        # Update with config if provided
        if 'hyperparameter_tuning' in self.config:
            self.tuning_config.update(self.config['hyperparameter_tuning'])
        
        # Store tuning results
        self.tuning_results = {}
    
    def get_model_param_grid(self, model_name: str, dataset_type: str) -> Dict[str, List]:
        """
        Get parameter grid for a specific model and dataset type
        
        Args:
            model_name: Name of the model
            dataset_type: Type of dataset (tabular, image, text)
            
        Returns:
            Parameter grid for grid search
        """
        # Default parameter grids
        param_grids = {
            'decision_tree': {
                'max_depth': [3, 5, 7, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy'],
                'splitter': ['best', 'random']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5, 10],
                'subsample': [0.8, 0.9, 1.0]
            },
            'mlp': {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50)],
                'activation': ['relu', 'tanh'],
                'solver': ['adam', 'sgd'],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive'],
                'max_iter': [500, 1000]
            },
            'cnn': {
                'num_layers': [2, 3, 4],
                'num_filters': [32, 64, 128],
                'kernel_size': [3, 5],
                'dropout_rate': [0.1, 0.2, 0.3],
                'learning_rate': [0.001, 0.01],
                'batch_size': [32, 64]
            },
            'vit': {
                'patch_size': [8, 16],
                'num_layers': [6, 12],
                'num_heads': [8, 12],
                'mlp_ratio': [2, 4],
                'dropout_rate': [0.1, 0.2],
                'learning_rate': [0.0001, 0.001]
            },
            'bert': {
                'max_length': [128, 256],
                'batch_size': [16, 32],
                'learning_rate': [2e-5, 3e-5, 5e-5],
                'epochs': [3, 5],
                'dropout_rate': [0.1, 0.2]
            },
            'lstm': {
                'hidden_size': [64, 128, 256],
                'num_layers': [1, 2],
                'dropout_rate': [0.1, 0.2, 0.3],
                'learning_rate': [0.001, 0.01],
                'batch_size': [32, 64]
            }
        }
        
        # Get base grid for model
        base_grid = param_grids.get(model_name, {})
        
        # Customize based on dataset type
        if dataset_type == 'tabular':
            # For tabular data, focus on tree-based and MLP parameters
            if model_name in ['decision_tree', 'random_forest', 'gradient_boosting']:
                return base_grid
            elif model_name == 'mlp':
                # Reduce complexity for tabular data
                return {
                    'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                    'activation': ['relu', 'tanh'],
                    'alpha': [0.0001, 0.001, 0.01],
                    'max_iter': [500, 1000]
                }
        
        elif dataset_type == 'image':
            # For image data, focus on CNN and ViT parameters
            if model_name in ['cnn', 'vit']:
                return base_grid
        
        elif dataset_type == 'text':
            # For text data, focus on BERT and LSTM parameters
            if model_name in ['bert', 'lstm']:
                return base_grid
        
        return base_grid
    
    def create_model_for_tuning(self, model_name: str, dataset_type: str):
        """
        Create a model instance for hyperparameter tuning
        
        Args:
            model_name: Name of the model
            dataset_type: Type of dataset
            
        Returns:
            Model instance
        """
        if not SKLEARN_AVAILABLE:
            self.logger.error("scikit-learn not available for hyperparameter tuning")
            return None
            
        # Import model classes
        try:
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
            from sklearn.neural_network import MLPClassifier
            
            model_map = {
                'decision_tree': DecisionTreeClassifier,
                'random_forest': RandomForestClassifier,
                'gradient_boosting': GradientBoostingClassifier,
                'mlp': MLPClassifier
            }
            
            model_class = model_map.get(model_name)
            if model_class:
                return model_class()
            
            # For custom models (CNN, ViT, BERT, LSTM), we'll need to implement
            # a scikit-learn compatible interface
            self.logger.warning(f"Custom model {model_name} not yet implemented for tuning")
            return None
            
        except ImportError as e:
            self.logger.error(f"Failed to import sklearn models: {e}")
            return None
    
    def tune_model(self, model_name: str, dataset, dataset_name: str) -> Dict[str, Any]:
        """
        Perform hyperparameter tuning for a specific model and dataset
        
        Args:
            model_name: Name of the model to tune
            dataset: Dataset object with get_data() method
            dataset_name: Name of the dataset
            
        Returns:
            Tuning results dictionary
        """
        self.logger.info(f"Starting hyperparameter tuning for {model_name} on {dataset_name}")
        
        # Get dataset info
        dataset_info = dataset.get_info()
        dataset_type = dataset_info.get('type', 'tabular')
        
        # Get data
        X_train, X_test, y_train, y_test = dataset.get_data()
        
        # Create model
        model = self.create_model_for_tuning(model_name, dataset_type)
        if model is None:
            return {
                'success': False,
                'error': f'Model {model_name} not supported for tuning'
            }
        
        # Get parameter grid
        param_grid = self.get_model_param_grid(model_name, dataset_type)
        if not param_grid:
            return {
                'success': False,
                'error': f'No parameter grid defined for {model_name}'
            }
        
        # Perform grid search
        start_time = time.time()
        
        try:
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grid,
                cv=self.tuning_config['cv_folds'],
                scoring=self.tuning_config['scoring'],
                n_jobs=self.tuning_config['n_jobs'],
                verbose=self.tuning_config['verbose']
            )
            
            # Fit the grid search
            grid_search.fit(X_train, y_train)
            
            tuning_time = time.time() - start_time
            
            # Collect results
            results = {
                'success': True,
                'model_name': model_name,
                'dataset_name': dataset_name,
                'dataset_type': dataset_type,
                'best_params': grid_search.best_params_,
                'best_score': grid_search.best_score_,
                'best_estimator': grid_search.best_estimator_,
                'cv_results': grid_search.cv_results_,
                'tuning_time': tuning_time,
                'n_combinations': len(grid_search.cv_results_['params']),
                'timestamp': datetime.now().isoformat()
            }
            
            # Evaluate on test set
            y_pred = grid_search.predict(X_test)
            results['test_accuracy'] = accuracy_score(y_test, y_pred)
            results['test_f1'] = f1_score(y_test, y_pred, average='weighted')
            
            # Try ROC AUC if binary classification
            if len(np.unique(y_test)) == 2:
                try:
                    y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
                    results['test_roc_auc'] = roc_auc_score(y_test, y_pred_proba)
                except:
                    results['test_roc_auc'] = None
            
            self.logger.info(f"Tuning completed for {model_name} on {dataset_name}")
            self.logger.info(f"Best score: {grid_search.best_score_:.4f}")
            self.logger.info(f"Best params: {grid_search.best_params_}")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Tuning failed for {model_name} on {dataset_name}: {e}")
            return {
                'success': False,
                'error': str(e),
                'model_name': model_name,
                'dataset_name': dataset_name,
                'timestamp': datetime.now().isoformat()
            }
    
    def tune_all_models(self, datasets: Dict[str, Any]) -> Dict[str, Any]:
        """
        Tune hyperparameters for all models on all datasets
        
        Args:
            datasets: Dictionary of dataset objects
            
        Returns:
            Dictionary of all tuning results
        """
        self.logger.info("Starting comprehensive hyperparameter tuning")
        
        all_results = {}
        
        for dataset_name, dataset in datasets.items():
            self.logger.info(f"Processing dataset: {dataset_name}")
            
            dataset_info = dataset.get_info()
            dataset_type = dataset_info.get('type', 'tabular')
            
            # Get compatible models for this dataset type
            compatible_models = self.get_compatible_models(dataset_type)
            
            dataset_results = {}
            for model_name in compatible_models:
                result = self.tune_model(model_name, dataset, dataset_name)
                dataset_results[model_name] = result
                
                # Save individual result
                if result['success']:
                    self.save_tuning_result(dataset_name, model_name, result)
            
            all_results[dataset_name] = dataset_results
        
        # Save comprehensive results
        self.save_all_tuning_results(all_results)
        
        return all_results
    
    def get_compatible_models(self, dataset_type: str) -> List[str]:
        """
        Get list of models compatible with dataset type
        
        Args:
            dataset_type: Type of dataset
            
        Returns:
            List of compatible model names
        """
        compatibility_map = {
            'tabular': ['decision_tree', 'random_forest', 'gradient_boosting', 'mlp'],
            'image': ['cnn', 'vit'],
            'text': ['bert', 'lstm']
        }
        
        return compatibility_map.get(dataset_type, [])
    
    def save_tuning_result(self, dataset_name: str, model_name: str, result: Dict[str, Any]):
        """
        Save individual tuning result
        
        Args:
            dataset_name: Name of the dataset
            model_name: Name of the model
            result: Tuning result dictionary
        """
        if not result['success']:
            return
        
        # Create filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"tuning_{dataset_name}_{model_name}_{timestamp}.json"
        filepath = self.output_dir / filename
        
        # Prepare data for saving (remove non-serializable objects)
        save_data = {
            'model_name': result['model_name'],
            'dataset_name': result['dataset_name'],
            'dataset_type': result['dataset_type'],
            'best_params': result['best_params'],
            'best_score': result['best_score'],
            'tuning_time': result['tuning_time'],
            'n_combinations': result['n_combinations'],
            'test_accuracy': result['test_accuracy'],
            'test_f1': result['test_f1'],
            'test_roc_auc': result['test_roc_auc'],
            'timestamp': result['timestamp']
        }
        
        # Save to file
        with open(filepath, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)
        
        self.logger.info(f"Saved tuning result to {filepath}")
    
    def save_all_tuning_results(self, results: Dict[str, Any]):
        """
        Save comprehensive tuning results
        
        Args:
            results: All tuning results
        """
        # Create summary
        summary = {
            'tuning_summary': {
                'total_datasets': len(results),
                'total_models': sum(len(dataset_results) for dataset_results in results.values()),
                'successful_tunings': 0,
                'failed_tunings': 0,
                'timestamp': datetime.now().isoformat()
            },
            'best_parameters': {},
            'performance_summary': {},
            'detailed_results': results
        }
        
        # Count successes and failures
        for dataset_name, dataset_results in results.items():
            for model_name, result in dataset_results.items():
                if result['success']:
                    summary['tuning_summary']['successful_tunings'] += 1
                    
                    # Store best parameters
                    key = f"{dataset_name}_{model_name}"
                    summary['best_parameters'][key] = result['best_params']
                    
                    # Store performance
                    summary['performance_summary'][key] = {
                        'cv_score': result['best_score'],
                        'test_accuracy': result['test_accuracy'],
                        'test_f1': result['test_f1'],
                        'test_roc_auc': result.get('test_roc_auc'),
                        'tuning_time': result['tuning_time']
                    }
                else:
                    summary['tuning_summary']['failed_tunings'] += 1
        
        # Save summary
        summary_file = self.output_dir / "tuning_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Saved tuning summary to {summary_file}")
    
    def load_best_parameters(self, dataset_name: str, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Load best parameters for a specific dataset-model combination
        
        Args:
            dataset_name: Name of the dataset
            model_name: Name of the model
            
        Returns:
            Best parameters dictionary or None if not found
        """
        summary_file = self.output_dir / "tuning_summary.json"
        
        if not summary_file.exists():
            return None
        
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            key = f"{dataset_name}_{model_name}"
            return summary['best_parameters'].get(key)
            
        except Exception as e:
            self.logger.error(f"Failed to load best parameters: {e}")
            return None
    
    def load_all_best_parameters(self) -> Dict[str, Dict[str, Any]]:
        """
        Load all best parameters
        
        Returns:
            Dictionary of all best parameters
        """
        summary_file = self.output_dir / "tuning_summary.json"
        
        if not summary_file.exists():
            return {}
        
        try:
            with open(summary_file, 'r') as f:
                summary = json.load(f)
            
            return summary.get('best_parameters', {})
            
        except Exception as e:
            self.logger.error(f"Failed to load all best parameters: {e}")
            return {}
    
    def generate_tuning_report(self, results: Dict[str, Any]) -> str:
        """
        Generate a comprehensive tuning report
        
        Args:
            results: Tuning results
            
        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("HYPERPARAMETER TUNING REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary statistics
        total_datasets = len(results)
        total_models = sum(len(dataset_results) for dataset_results in results.values())
        successful_tunings = 0
        failed_tunings = 0
        
        for dataset_results in results.values():
            for result in dataset_results.values():
                if result['success']:
                    successful_tunings += 1
                else:
                    failed_tunings += 1
        
        report_lines.append("📊 SUMMARY STATISTICS")
        report_lines.append(f"  - Total datasets: {total_datasets}")
        report_lines.append(f"  - Total models: {total_models}")
        report_lines.append(f"  - Successful tunings: {successful_tunings}")
        report_lines.append(f"  - Failed tunings: {failed_tunings}")
        report_lines.append(f"  - Success rate: {successful_tunings/total_models:.2%}")
        report_lines.append("")
        
        # Best performers
        report_lines.append("🏆 BEST PERFORMERS")
        best_performers = []
        
        for dataset_name, dataset_results in results.items():
            for model_name, result in dataset_results.items():
                if result['success']:
                    best_performers.append({
                        'dataset': dataset_name,
                        'model': model_name,
                        'cv_score': result['best_score'],
                        'test_accuracy': result['test_accuracy'],
                        'test_f1': result['test_f1']
                    })
        
        # Sort by CV score
        best_performers.sort(key=lambda x: x['cv_score'], reverse=True)
        
        for i, performer in enumerate(best_performers[:5], 1):
            report_lines.append(f"  {i}. {performer['dataset']} + {performer['model']}")
            report_lines.append(f"     CV Score: {performer['cv_score']:.4f}")
            report_lines.append(f"     Test Accuracy: {performer['test_accuracy']:.4f}")
            report_lines.append(f"     Test F1: {performer['test_f1']:.4f}")
            report_lines.append("")
        
        # Detailed results by dataset
        report_lines.append("📈 DETAILED RESULTS BY DATASET")
        for dataset_name, dataset_results in results.items():
            report_lines.append(f"  {dataset_name}:")
            
            for model_name, result in dataset_results.items():
                if result['success']:
                    report_lines.append(f"    {model_name}:")
                    report_lines.append(f"      CV Score: {result['best_score']:.4f}")
                    report_lines.append(f"      Test Accuracy: {result['test_accuracy']:.4f}")
                    report_lines.append(f"      Tuning Time: {result['tuning_time']:.2f}s")
                    report_lines.append(f"      Best Params: {result['best_params']}")
                else:
                    report_lines.append(f"    {model_name}: FAILED - {result['error']}")
            report_lines.append("")
        
        return "\n".join(report_lines)


def run_hyperparameter_tuning(config: Dict[str, Any], datasets: Dict[str, Any], output_dir: Path = None) -> Dict[str, Any]:
    """
    Convenience function to run hyperparameter tuning
    
    Args:
        config: Configuration dictionary
        datasets: Dictionary of dataset objects
        output_dir: Output directory for results
        
    Returns:
        Tuning results
    """
    tuner = HyperparameterTuner(config, output_dir)
    return tuner.tune_all_models(datasets)


def load_tuned_parameters(output_dir: Path, dataset_name: str, model_name: str) -> Optional[Dict[str, Any]]:
    """
    Convenience function to load tuned parameters
    
    Args:
        output_dir: Directory containing tuning results
        dataset_name: Name of the dataset
        model_name: Name of the model
        
    Returns:
        Best parameters or None
    """
    tuner = HyperparameterTuner(output_dir=output_dir)
    return tuner.load_best_parameters(dataset_name, model_name) 