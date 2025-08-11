"""
Main benchmark class for XAI evaluation
"""

import logging
from pathlib import Path
from typing import Dict, Any
import time
import json
from datetime import datetime

from .data.data_manager import DataManager
from .models.model_factory import ModelFactory
from .explanations.explanation_factory import ExplanationFactory
from .evaluation.evaluator import Evaluator
from .utils.reporting import ReportGenerator
from .utils.data_validation import DataValidator, validate_all_datasets
from .utils.hyperparameter_tuning import HyperparameterTuner, load_tuned_parameters


class XAIBenchmark:
    """
    Main class for XAI benchmarking experiments
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        """
        Initialize the benchmark
        
        Args:
            config: Configuration dictionary
            output_dir: Output directory for results
        """
        self.config = config
        self.output_dir = output_dir
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        # Pass the full config so DataManager can access the 'data' section internally
        self.data_manager = DataManager(config)
        self.model_factory = ModelFactory(config.get('models', {}))
        self.explanation_factory = ExplanationFactory(config.get('explanations', {}))
        self.evaluator = Evaluator(config.get('evaluation', {}))
        self.report_generator = ReportGenerator(output_dir)
        self.data_validator = DataValidator(config)
        
        # Initialize hyperparameter tuner
        self.hyperparameter_tuner = HyperparameterTuner(
            config, 
            output_dir / "tuning_results"
        )
        
        # Results storage
        self.results = {
            'experiment_info': {
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'config': config
            },
            'dataset_results': {},
            'model_results': {},
            'explanation_results': {},
            'evaluation_results': {},
            'validation_results': {},
            'hyperparameter_info': {}
        }
        
        # Iteration counter for separate file saving
        self.iteration_counter = 0
        
        # Create results directory structure
        self.results_dir = output_dir / "iterations"
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create summary results file
        self.summary_file = output_dir / "benchmark_results.json"
    
    def run_hyperparameter_tuning(self, datasets: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Run hyperparameter tuning for all models and datasets
        
        Args:
            datasets: Optional dictionary of datasets (if None, loads all mandatory)
            
        Returns:
            Tuning results
        """
        self.logger.info("Starting hyperparameter tuning")
        
        if datasets is None:
            datasets = self.data_manager.load_datasets()
        
        if not datasets:
            self.logger.error("No datasets available for tuning")
            return {}
        
        # Run tuning
        tuning_results = self.hyperparameter_tuner.tune_all_models(datasets)
        
        # Store tuning info in results
        self.results['hyperparameter_info'] = {
            'tuning_performed': True,
            'tuning_timestamp': datetime.now().isoformat(),
            'tuning_results_summary': {
                'total_datasets': len(tuning_results),
                'total_models': sum(len(dataset_results) for dataset_results in tuning_results.values()),
                'successful_tunings': sum(
                    1 for dataset_results in tuning_results.values() 
                    for result in dataset_results.values() 
                    if result['success']
                )
            }
        }
        
        self.logger.info("Hyperparameter tuning completed")
        return tuning_results
    
    def get_tuned_parameters(self, dataset_name: str, model_name: str) -> Dict[str, Any]:
        """
        Get tuned parameters for a specific dataset-model combination
        
        Args:
            dataset_name: Name of the dataset
            model_name: Name of the model
            
        Returns:
            Tuned parameters or empty dict if not found
        """
        return self.hyperparameter_tuner.load_best_parameters(dataset_name, model_name) or {}
    
    def run_full_pipeline(self, use_tuned_params: bool = False):
        # Step 1: Load and preprocess data
        self.logger.info("Step 1: Loading and preprocessing data")
        datasets = self.data_manager.load_datasets()

        # Interactive dataset selection (one-time loading)
        dataset_names = list(datasets.keys())
        print("\n=== DATASET SELECTION ===")
        print("Available datasets:")
        for i, name in enumerate(dataset_names):
            print(f"  [{i+1}] {name}")
        selected = input("Select dataset(s) by number (comma-separated, Enter for all): ")
        if selected.strip():
            indices = [int(x.strip())-1 for x in selected.split(",") if x.strip().isdigit() and 0 < int(x.strip()) <= len(dataset_names)]
            dataset_names = [dataset_names[i] for i in indices]
        # Only keep selected datasets, but do not reload
        datasets = {name: datasets[name] for name in dataset_names}

        # Extra validation: log all loaded datasets and their types/samples
        self.logger.info("[DEBUG] Listing all loaded datasets and their types/samples:")
        for name, ds in datasets.items():
            self.logger.info(f"[DEBUG] Loaded dataset '{name}': type={type(ds)}, info={ds.get_info()}")
            try:
                X_train, X_test, y_train, y_test = ds.get_data()
                self.logger.info(f"[DEBUG] Sample X_train for '{name}': {str(X_train[:2])[:300]}")
            except Exception as e:
                self.logger.warning(f"[DEBUG] Could not get data sample for '{name}': {e}")
        """Run the complete benchmarking pipeline"""
        self.logger.info("Starting full XAI benchmarking pipeline")
        
        # Step 1: Load and preprocess data
        self.logger.info("Step 1: Loading and preprocessing data")
        datasets = self.data_manager.load_datasets()
        
        # Step 1.5: Validate datasets (temporarily disabled)
        self.logger.info("Step 1.5: Skipping dataset validation (temporarily disabled)")
        validation_results = {name: {'is_valid': True, 'warnings': [], 'errors': []} for name in datasets.keys()}
        self.results['validation_results'] = validation_results
        
        for dataset_name, dataset in datasets.items():
            self.logger.info(f"Processing dataset: {dataset_name}")
            self.results['dataset_results'][dataset_name] = {
                'info': dataset.get_info(),
                'preprocessing_time': dataset.preprocessing_time,
                'validation_status': validation_results.get(dataset_name, {}).get('is_valid', False)
            }
        
        # Step 2: Train models (with optional tuned parameters)
        self.logger.info("Step 2: Training models")
        models = {}
        for dataset_name, dataset in datasets.items():
            models[dataset_name] = {}
            # Determine dataset type (tabular, image, text) based on config
            if dataset_name in [d['name'] for d in self.config['data'].get('tabular_datasets', [])]:
                valid_model_section = 'tabular'
                dataset_type = 'tabular'
            elif dataset_name in [d['name'] for d in self.config['data'].get('image_datasets', [])]:
                valid_model_section = 'image'
                dataset_type = 'image'
            elif dataset_name in [d['name'] for d in self.config['data'].get('text_datasets', [])]:
                valid_model_section = 'text'
                dataset_type = 'text'
            else:
                self.logger.warning(f"Dataset '{dataset_name}' not found in config data sections. Skipping.")
                continue
            # Interactive model selection
            valid_models = [m['name'] for m in self.config['models'].get(valid_model_section, [])]
            print(f"\n=== MODEL SELECTION for dataset '{dataset_name}' ===")
            print(f"Available models:")
            for i, name in enumerate(valid_models):
                print(f"  [{i+1}] {name}")
            selected = input(f"Select model(s) for '{dataset_name}' by number (comma-separated, Enter for all): ")
            if selected.strip():
                indices = [int(x.strip())-1 for x in selected.split(",") if x.strip().isdigit() and 0 < int(x.strip()) <= len(valid_models)]
                selected_models = [valid_models[i] for i in indices]
            else:
                selected_models = valid_models
            for model_name in selected_models:
                model_config = next((m for m in self.config['models'].get(valid_model_section, []) if m['name'] == model_name), None)
                if model_config is None:
                    self.logger.error(f"Model '{model_name}' not found in section '{valid_model_section}'.")
                    continue
                self.logger.info(f"Training {model_name} on {dataset_name} (type: {dataset_type})")
                # Apply tuned parameters if requested and available
                model_config_to_use = dict(model_config)  # Copy to avoid mutating config
                if use_tuned_params:
                    tuned_params = self.get_tuned_parameters(dataset_name, model_name)
                    if tuned_params:
                        self.logger.info(f"Using tuned parameters for {model_name}: {tuned_params}")
                        model_config_to_use.update(tuned_params)
                    else:
                        self.logger.warning(f"No tuned parameters found for {model_name} on {dataset_name}")
                model = self.model_factory.create_model(model_config_to_use, dataset, model_name)
                model.train(dataset)
                self.logger.info(f"Model '{model_name}' trained on '{dataset_name}'. Training time: {getattr(model, 'training_time', 'N/A')}")
                performance = model.evaluate(dataset)
                self.logger.info(f"Performance metrics for '{model_name}' on '{dataset_name}': {performance}")
                models[dataset_name][model_name] = model
                self.results['model_results'][f"{dataset_name}_{model_name}"] = {
                    'training_time': model.training_time,
                    'performance': performance,
                    'model_info': model.get_info(),
                    'used_tuned_params': use_tuned_params and bool(self.get_tuned_parameters(dataset_name, model_name))
                }
        
        # Step 3: Generate explanations
        self.logger.info("Step 3: Generating explanations")
        explanations = {}
        for dataset_name, dataset_models in models.items():
            explanations[dataset_name] = {}
            for model_name, model in dataset_models.items():
                explanations[dataset_name][model_name] = {}
                # Gather all explanation methods from all groups, and add 'type' key
                all_methods = []
                for group in ['feature_attribution', 'example_based', 'concept_based', 'perturbation']:
                    for method in self.config['explanations'].get(group, []):
                        method_with_type = dict(method)
                        method_with_type['type'] = group
                        all_methods.append(method_with_type)
                # Interactive explanation selection
                print(f"\n=== EXPLANATION METHOD SELECTION for model '{model_name}' on dataset '{dataset_name}' ===")
                print(f"Available explanation methods:")
                for i, m in enumerate(all_methods):
                    print(f"  [{i+1}] {m['name']} ({m['type']})")
                selected = input(f"Select explanation method(s) for '{model_name}' on '{dataset_name}' by number (comma-separated, Enter for all): ")
                if selected.strip():
                    indices = [int(x.strip())-1 for x in selected.split(",") if x.strip().isdigit() and 0 < int(x.strip()) <= len(all_methods)]
                    selected_methods = [all_methods[i] for i in indices]
                else:
                    selected_methods = all_methods
                for explanation_config in selected_methods:
                    method_name = explanation_config['name']
                    # Get explainer class to check compatibility
                    explainer_class = self.explanation_factory.explainer_registry.get(method_name)
                    if explainer_class is None:
                        self.logger.warning(f"No explainer class found for method '{method_name}', skipping.")
                        continue
                    # Check supported data types
                    supported_data_types = getattr(explainer_class, 'supported_data_types', ['tabular', 'image', 'text'])
                    supported_model_types = getattr(explainer_class, 'supported_model_types', ['all'])
                    # Infer dataset type
                    if dataset_name in [d['name'] for d in self.config['data'].get('tabular_datasets', [])]:
                        dataset_type = 'tabular'
                    elif dataset_name in [d['name'] for d in self.config['data'].get('image_datasets', [])]:
                        dataset_type = 'image'
                    elif dataset_name in [d['name'] for d in self.config['data'].get('text_datasets', [])]:
                        dataset_type = 'text'
                    else:
                        self.logger.warning(f"Unknown dataset type for '{dataset_name}', skipping explanation.")
                        continue
                    # Check compatibility
                    if dataset_type not in supported_data_types:
                        self.logger.info(f"Skipping {method_name} for {model_name} on {dataset_name}: not supported for dataset type '{dataset_type}'.")
                        continue
                    if supported_model_types != ['all'] and model_name not in supported_model_types:
                        self.logger.info(f"Skipping {method_name} for {model_name} on {dataset_name}: not supported for model type '{model_name}'.")
                        continue
                    # Debug: Log data sample and types before explanation
                    try:
                        X_train, X_test, y_train, y_test = dataset.get_data()
                        self.logger.info(f"[DEBUG] About to explain: dataset='{dataset_name}', model='{model_name}', method='{method_name}', dataset_type='{dataset_type}'")
                        self.logger.info(f"[DEBUG] X_test type: {type(X_test)}, sample: {str(X_test[:2])[:300]}")
                        self.logger.info(f"[DEBUG] y_test type: {type(y_test)}, sample: {str(y_test[:10])[:300]}")
                        # Validate that the model was trained on this dataset
                        if hasattr(model, 'dataset_name'):
                            if model.dataset_name != dataset_name:
                                self.logger.error(f"[DEBUG] Model '{model_name}' was trained on '{model.dataset_name}', but now used with dataset '{dataset_name}'!")
                        # Validate that the dataset is the expected type
                        if dataset_type == 'tabular' and not hasattr(X_train, 'shape'):
                            self.logger.error(f"[DEBUG] Expected tabular data for '{dataset_name}', but got type {type(X_train)}")
                        if dataset_type == 'text' and not isinstance(X_train, list):
                            self.logger.error(f"[DEBUG] Expected text data for '{dataset_name}', but got type {type(X_train)}")
                        if dataset_type == 'image' and not (hasattr(X_train, 'shape') and len(getattr(X_train, 'shape', [])) in [3, 4]):
                            self.logger.error(f"[DEBUG] Expected image data for '{dataset_name}', but got type {type(X_train)}")
                    except Exception as e:
                        self.logger.warning(f"[DEBUG] Could not get data sample for explanation: {e}")
                    self.logger.info(f"Generating {method_name} explanations for {model_name} on {dataset_name}")
                    explainer = self.explanation_factory.create_explainer(
                        explanation_config, model, dataset
                    )
                    explanation_results = explainer.explain(dataset)
                    explanations[dataset_name][model_name][method_name] = explanation_results
                    self.results['explanation_results'][f"{dataset_name}_{model_name}_{method_name}"] = {
                        'generation_time': explanation_results.get('generation_time', 0),
                        'explanation_info': explanation_results.get('info', {})
                    }
        # Ensure results directory exists
        import os, json
        os.makedirs(self.output_dir, exist_ok=True)
        if not self.results_dir.exists():
            self.results_dir.mkdir(parents=True, exist_ok=True)
        # Save results summary at the end (even if partial)
        try:
            with open(self.summary_file, 'w') as f:
                json.dump(self.results, f, indent=2)
            self.logger.info(f"Results saved to {self.summary_file}")
        except Exception as e:
            self.logger.error(f"Failed to save results: {e}")
        
        # Step 4: Evaluate explanations
        self.logger.info("Step 4: Evaluating explanations")
        all_evaluation_results = {}
        for dataset_name, dataset_models in models.items():
            for model_name, model in dataset_models.items():
                for method_name in explanations[dataset_name][model_name].keys():
                    self.logger.info(f"Evaluating {method_name} explanations for {model_name} on {dataset_name}")
                    
                    explanation_results = explanations[dataset_name][model_name][method_name]
                    evaluation_results = self.evaluator.evaluate(
                        model, 
                        explanation_results, 
                        datasets[dataset_name]
                    )
                    
                    result_key = f"{dataset_name}_{model_name}_{method_name}"
                    self.results['evaluation_results'][result_key] = evaluation_results
                    all_evaluation_results[result_key] = {
                        'metrics': evaluation_results,
                        'method': method_name,
                        'model': model_name,
                        'dataset': dataset_name
                    }
        
        # Step 5: Statistical significance testing
        self.logger.info("Step 5: Running statistical significance tests")
        statistical_results = self.evaluator.run_statistical_analysis(all_evaluation_results)
        self.results['statistical_results'] = statistical_results
        
        # Step 6: Generate reports
        self.logger.info("Step 6: Generating reports")
        self.report_generator.generate_reports(self.results)
        
        # Save results
        self._save_results("full_pipeline")
        
        self.logger.info("Benchmarking pipeline completed successfully!")
    
    def run_interactive(self, use_tuned_params: bool = False):
        """Run interactive benchmarking mode with robust, always-on prompts and non-fatal validation."""
        self.logger.info("Starting interactive XAI benchmarking (robust mode)")

        # Load all datasets once and cache
        available_datasets = self.data_manager._get_mandatory_datasets()
        if not available_datasets:
            self.logger.error("No mandatory datasets available")
            return


        # Interactive dataset selection with 'All' option
        dataset_names = list(available_datasets.keys())
        print("\nAvailable datasets:")
        print("  [0] All")
        for i, name in enumerate(dataset_names, 1):
            print(f"  [{i}] {name}")
        selected = input("Select dataset(s) by number (comma-separated, 0 for all, Enter for all): ")
        if selected.strip():
            if selected.strip() == '0':
                selected_datasets = dataset_names
            else:
                indices = [int(x.strip())-1 for x in selected.split(",") if x.strip().isdigit() and 0 < int(x.strip()) <= len(dataset_names)]
                selected_datasets = [dataset_names[i] for i in indices]
        else:
            selected_datasets = dataset_names

        # Load selected datasets once
        datasets = {name: self.data_manager.load_dataset(name) for name in selected_datasets}

        # Validate all selected datasets, but do not skip on error; prompt user for action
        for name, ds in datasets.items():
            validation_result = self.data_validator.validate_dataset(ds, name)
            self.results.setdefault('validation_results', {})[name] = validation_result
            if not validation_result['is_valid']:
                print(f"\n[WARNING] Dataset '{name}' failed validation:")
                for error in validation_result['errors']:
                    print(f"  - {error}")
                for warning in validation_result['warnings']:
                    print(f"  [Warning] {warning}")
                while True:
                    user_choice = input(f"Proceed with dataset '{name}' anyway? (y = yes, s = skip, f = fix and retry): ").strip().lower()
                    if user_choice == 'y':
                        print(f"Proceeding with dataset '{name}' despite validation errors.")
                        break
                    elif user_choice == 's':
                        print(f"Skipping dataset '{name}' as per user request.")
                        datasets[name] = None
                        break
                    elif user_choice == 'f':
                        print(f"Please fix the dataset and restart the pipeline.")
                        return
                    else:
                        print("Invalid input. Please enter 'y', 's', or 'f'.")
        # Remove skipped datasets
        datasets = {k: v for k, v in datasets.items() if v is not None}
        if not datasets:
            print("No valid datasets selected. Exiting.")
            return

        # Model selection per dataset
        available_models = self.model_factory.get_available_models()
        if not available_models:
            self.logger.error("No models available from configuration")
            return
        # Normalize: convert string entries to dicts with at least 'name' and 'type'
        normalized_models = []
        for m in available_models:
            if isinstance(m, dict):
                normalized_models.append(m)
            elif isinstance(m, str):
                normalized_models.append({'name': m, 'type': m})
            else:
                continue
        selected_models_per_dataset = {}
        for dataset_name, dataset in datasets.items():
            dataset_type = dataset.get_info().get('type')
            compatible_models = []
            for m in normalized_models:
                model_type_key = m.get('type')
                model_class = self.model_factory.model_registry.get(model_type_key)
                if not model_class:
                    continue
                supported = getattr(model_class, 'supported_data_types', [])
                if not supported or (dataset_type in supported):
                    compatible_models.append(m)
            # DEBUG: Print all compatible models for this dataset
            print(f"[DEBUG] Compatible models for dataset '{dataset_name}' (type: {dataset_type}): {[m.get('name') for m in compatible_models]}")
            if not compatible_models:
                print(f"No compatible models found for dataset type '{dataset_type}' - skipping {dataset_name}")
                continue
            print(f"\nAvailable models for dataset '{dataset_name}':")
            print("  [0] All")
            for i, m in enumerate(compatible_models, 1):
                print(f"  [{i}] {m.get('name')} ({m.get('type')})")
            selected = input(f"Select model(s) for '{dataset_name}' by number (comma-separated, 0 for all, Enter for all): ")
            if selected.strip():
                if selected.strip() == '0':
                    selected_models = compatible_models
                else:
                    indices = [int(x.strip())-1 for x in selected.split(",") if x.strip().isdigit() and 0 < int(x.strip()) <= len(compatible_models)]
                    selected_models = [compatible_models[i] for i in indices]
            else:
                selected_models = compatible_models
            selected_models_per_dataset[dataset_name] = selected_models

        # Explanation method selection per model/dataset
        available_explanations = self.explanation_factory.get_available_methods()
        if not available_explanations:
            self.logger.error("No explanation methods available from configuration")
            return
        # Prompt for evaluation metrics to compute
        all_eval_metrics = [
            'time_complexity', 'faithfulness', 'monotonicity', 'completeness',
            'stability', 'consistency', 'sparsity', 'simplicity'
        ]
        print("\nAvailable evaluation metrics:")
        print("  [0] All")
        for i, m in enumerate(all_eval_metrics, 1):
            print(f"  [{i}] {m}")
        selected = input("Select evaluation metric(s) by number (comma-separated, 0 for all, Enter for all): ")
        if selected.strip() == '' or selected.strip() == '0':
            selected_eval_metrics = all_eval_metrics
        else:
            indices = [int(x.strip())-1 for x in selected.split(",") if x.strip().isdigit() and 0 < int(x.strip()) <= len(all_eval_metrics)]
            selected_eval_metrics = [all_eval_metrics[i] for i in indices]

        for dataset_name, dataset in datasets.items():
            dataset_type = dataset.get_info().get('type')
            for model_config in selected_models_per_dataset.get(dataset_name, []):
                # Apply tuned parameters if requested and available
                model_config_to_use = dict(model_config)
                if use_tuned_params:
                    tuned_params = self.get_tuned_parameters(dataset_name, model_config['name'])
                    if tuned_params:
                        print(f"Using tuned parameters for {model_config['name']}: {tuned_params}")
                        model_config_to_use.update(tuned_params)
                # Train model (pass model type key, not dataset type)
                model = self.model_factory.create_model(model_config_to_use, dataset, model_config_to_use['type'])
                model.train(dataset)
                print(f"\nTrained model: {model_config['name']} on {dataset_name}")
                # Filter compatible explanation methods
                model_type_key = model_config.get('type')
                explanation_candidates = []
                for e in available_explanations:
                    method_type_key = e.get('type')
                    explainer_class = self.explanation_factory.explainer_registry.get(method_type_key)
                    if not explainer_class:
                        continue
                    data_supported = getattr(explainer_class, 'supported_data_types', [])
                    model_supported = getattr(explainer_class, 'supported_model_types', ['all'])
                    if (not data_supported or dataset_type in data_supported) and (model_supported == ['all'] or model_type_key in model_supported):
                        explanation_candidates.append(e)
                if not explanation_candidates:
                    print(f"No compatible explanation methods for model '{model_type_key}' and dataset type '{dataset_type}'")
                    continue
                print(f"\nAvailable explanation methods for model '{model_config['name']}' on dataset '{dataset_name}':")
                print("  [0] All")
                for i, e in enumerate(explanation_candidates, 1):
                    print(f"  [{i}] {e.get('name')} ({e.get('type')})")
                selected = input(f"Select explanation method(s) for '{model_config['name']}' on '{dataset_name}' by number (comma-separated, 0 for all, Enter for all): ")
                if selected.strip():
                    if selected.strip() == '0':
                        selected_explanations = explanation_candidates
                    else:
                        indices = [int(x.strip())-1 for x in selected.split(",") if x.strip().isdigit() and 0 < int(x.strip()) <= len(explanation_candidates)]
                        selected_explanations = [explanation_candidates[i] for i in indices]
                else:
                    selected_explanations = explanation_candidates
                for explanation_config in selected_explanations:
                    # Generate explanations
                    explainer = self.explanation_factory.create_explainer(explanation_config, model, dataset)
                    explanation_results = explainer.explain(dataset)
                    print(f"\nGenerated explanations using: {explanation_config['name']} for {dataset_name}/{model_config['name']}")
                    # Thorough debug: print the raw explanation_results
                    import pprint
                    print("\n[DEBUG] Raw explanation_results:")
                    pprint.pprint(explanation_results, depth=4, compact=False, width=120)
                    # Evaluate explanations (filter by selected metrics)
                    all_results = self.evaluator.evaluate(model, explanation_results, dataset)
                    evaluation_results = {k: v for k, v in all_results.items() if k in selected_eval_metrics}
                    # Pretty-print evaluation results for readability
                    print("\nEvaluation results:")
                    if isinstance(evaluation_results, dict):
                        for k, v in evaluation_results.items():
                            # Format floats nicely, handle numpy types
                            try:
                                import numpy as np
                                if isinstance(v, float) or (hasattr(np, 'floating') and isinstance(v, np.floating)):
                                    print(f"  {k:15}: {float(v):.4f}")
                                else:
                                    print(f"  {k:15}: {v}")
                            except Exception:
                                print(f"  {k:15}: {v}")
                    else:
                        print(evaluation_results)
                    # Save results incrementally
                    self.results.setdefault('interactive_runs', [])
                    self.results['interactive_runs'].append({
                        'dataset': dataset_name,
                        'model': model_config['name'],
                        'explanation_method': explanation_config['name'],
                        'evaluation_results': evaluation_results,
                        'used_tuned_params': use_tuned_params and bool(self.get_tuned_parameters(dataset_name, model_config['name']))
                    })
                    # Save individual iteration result
                    iteration_key = f"{dataset_name}_{model_config['name']}_{explanation_config['name']}"
                    self._save_iteration_result(iteration_key, {
                        'dataset': dataset_name,
                        'model': model_config['name'],
                        'explanation_method': explanation_config['name'],
                        'evaluation_results': evaluation_results,
                        'explanation_results': explanation_results,
                        'model_performance': model.evaluate(dataset),
                        'validation_status': self.results.get('validation_results', {}).get(dataset_name, {}).get('is_valid', False),
                        'used_tuned_params': use_tuned_params and bool(self.get_tuned_parameters(dataset_name, model_config['name']))
                    })
        print("\nInteractive run completed.")

    def run_all(self, use_tuned_params: bool = False):
        """Run all datasets x compatible models x compatible explanation methods with incremental saving."""
        self.logger.info("Starting run_all across datasets x models x methods")

        mandatory_dataset_names = list(self.data_manager._get_mandatory_datasets().keys())
        if not mandatory_dataset_names:
            self.logger.error("No mandatory datasets available")
            return

        available_models = self.model_factory.get_available_models()
        available_methods = self.explanation_factory.get_available_methods()
        if not available_models or not available_methods:
            self.logger.error("Models or explanation methods not available from configuration")
            return

        for dataset_name in mandatory_dataset_names:
            try:
                dataset = self.data_manager.load_dataset(dataset_name)
                
                # Validate dataset (temporarily disabled for debugging)
                # validation_result = self.data_validator.validate_dataset(dataset, dataset_name)
                # self.results.setdefault('validation_results', {})[dataset_name] = validation_result
                # 
                # if not validation_result['is_valid']:
                #     self.logger.warning(f"Dataset {dataset_name} failed validation, skipping...")
                #     continue
                
                # Create a dummy validation result for now
                validation_result = {'is_valid': True, 'warnings': [], 'errors': []}
                
                dataset_type = dataset.get_info().get('type')
            except Exception as e:
                self.logger.error(f"Failed to load dataset '{dataset_name}': {e}")
                continue

            # Filter models by dataset compatibility
            compatible_models = []
            for m in available_models:
                model_type_key = m.get('type')
                model_class = self.model_factory.model_registry.get(model_type_key)
                if not model_class:
                    continue
                supported = getattr(model_class, 'supported_data_types', [])
                if not supported or (dataset_type in supported):
                    compatible_models.append(m)

            if not compatible_models:
                self.logger.warning(f"No compatible models for dataset '{dataset_name}' (type {dataset_type}); skipping")
                continue

            for model_config in compatible_models:
                # Apply tuned parameters if requested and available
                if use_tuned_params:
                    tuned_params = self.get_tuned_parameters(dataset_name, model_config.get('name'))
                    if tuned_params:
                        self.logger.info(f"Using tuned parameters for {model_config.get('name')}: {tuned_params}")
                        model_config = {**model_config, **tuned_params}
                
                try:
                    model = self.model_factory.create_model(model_config, dataset)
                    model.train(dataset)
                except Exception as e:
                    self.logger.error(f"Failed to train model '{model_config.get('name')}' on '{dataset_name}': {e}")
                    continue

                # Filter methods by compatibility
                model_type_key = model_config.get('type')
                compatible_methods = []
                for econf in available_methods:
                    method_type_key = econf.get('type')
                    explainer_class = self.explanation_factory.explainer_registry.get(method_type_key)
                    if not explainer_class:
                        continue
                    data_supported = getattr(explainer_class, 'supported_data_types', [])
                    model_supported = getattr(explainer_class, 'supported_model_types', ['all'])
                    if (not data_supported or dataset_type in data_supported) and (model_supported == ['all'] or model_type_key in model_supported):
                        compatible_methods.append(econf)

                if not compatible_methods:
                    self.logger.warning(f"No compatible methods for model '{model_type_key}' on dataset '{dataset_name}'")
                    continue

                for method_config in compatible_methods:
                    try:
                        explainer = self.explanation_factory.create_explainer(method_config, model, dataset)
                        explanation_results = explainer.explain(dataset)
                        evaluation_results = self.evaluator.evaluate(model, explanation_results, dataset)
                    except Exception as e:
                        self.logger.error(f"Failed {dataset_name}/{model_config.get('name')}/{method_config.get('name')}: {e}")
                        continue

                    # Save incrementally per combination
                    key = f"{dataset_name}_{model_config.get('name')}_{method_config.get('name')}"
                    self.results.setdefault('evaluation_results', {})[key] = evaluation_results
                    self.results.setdefault('explanation_results', {})[key] = {
                        'generation_time': explanation_results.get('generation_time', 0),
                        'explanation_info': explanation_results.get('info', {})
                    }
                    
                    # Save individual iteration result
                    self._save_iteration_result(key, {
                        'dataset': dataset_name,
                        'model': model_config.get('name'),
                        'explanation_method': method_config.get('name'),
                        'evaluation_results': evaluation_results,
                        'explanation_results': explanation_results,
                        'model_performance': model.evaluate(dataset),
                        'validation_status': validation_result.get('is_valid', False),
                        'validation_warnings': validation_result.get('warnings', []),
                        'validation_errors': validation_result.get('errors', []),
                        'used_tuned_params': use_tuned_params and bool(self.get_tuned_parameters(dataset_name, model_config.get('name')))
                    })

        self.logger.info("run_all completed")
    
    def _save_iteration_result(self, iteration_key: str, result_data: Dict[str, Any]):
        """
        Save individual iteration result to a separate folder structure
        
        Args:
            iteration_key: Unique key for this iteration (e.g., "dataset_model_method")
            result_data: Result data to save
        """
        self.iteration_counter += 1
        
        # Create timestamp for unique filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Parse iteration key to create folder structure
        # Format: dataset_model_method
        parts = iteration_key.split('_')
        if len(parts) >= 3:
            dataset_name = parts[0]
            model_name = parts[1]
            method_name = '_'.join(parts[2:])  # Handle method names with underscores
        else:
            dataset_name = "unknown"
            model_name = "unknown"
            method_name = "unknown"
        
        # Create folder structure: results/iterations/dataset/model/method/
        iteration_folder = self.results_dir / "iterations" / dataset_name / model_name / method_name
        iteration_folder.mkdir(parents=True, exist_ok=True)
        
        # Create filename with iteration info
        filename = f"iteration_{self.iteration_counter:04d}_{timestamp}.json"
        iteration_file = iteration_folder / filename
        
        # Prepare iteration result with metadata
        iteration_result = {
            'iteration_info': {
                'iteration_number': self.iteration_counter,
                'timestamp': datetime.now().isoformat(),
                'iteration_key': iteration_key,
                'filename': filename,
                'folder_path': str(iteration_folder)
            },
            'experiment_info': {
                'config': self.config,
                'benchmark_timestamp': self.results['experiment_info']['timestamp']
            },
            'result_data': result_data
        }
        
        # Save to individual file
        with open(iteration_file, 'w') as f:
            json.dump(iteration_result, f, indent=2, default=str)
        
        self.logger.info(f"Saved iteration result to {iteration_file}")
        
        # Also save a copy in the main results directory for easy access
        main_iteration_file = self.results_dir / f"iteration_{self.iteration_counter:04d}_{timestamp}_{iteration_key}.json"
        with open(main_iteration_file, 'w') as f:
            json.dump(iteration_result, f, indent=2, default=str)
    
    def _save_results(self, run_type: str = "general"):
        """Save results to file"""
        # Update experiment info
        self.results['experiment_info']['run_type'] = run_type
        self.results['experiment_info']['total_iterations'] = self.iteration_counter
        
        # Save summary results
        with open(self.summary_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info(f"Summary results saved to {self.summary_file}")
        
        # Create iteration summary
        self._create_iteration_summary()
    
    def _create_iteration_summary(self):
        """Create a summary of all iterations"""
        summary_file = self.results_dir / "iterations_summary.json"
        
        # Collect all iteration files from both main directory and organized folders
        main_iteration_files = list(self.results_dir.glob("iteration_*.json"))
        organized_iteration_files = list((self.results_dir / "iterations").rglob("*.json")) if (self.results_dir / "iterations").exists() else []
        
        # Combine and deduplicate files
        all_iteration_files = list(set(main_iteration_files + organized_iteration_files))
        
        summary = {
            'summary_info': {
                'total_iterations': len(all_iteration_files),
                'generated_at': datetime.now().isoformat(),
                'results_directory': str(self.results_dir),
                'organized_folder': str(self.results_dir / "iterations")
            },
            'iterations': []
        }
        
        for iteration_file in sorted(all_iteration_files):
            try:
                with open(iteration_file, 'r') as f:
                    iteration_data = json.load(f)
                
                summary['iterations'].append({
                    'filename': iteration_file.name,
                    'file_path': str(iteration_file),
                    'iteration_key': iteration_data['iteration_info']['iteration_key'],
                    'timestamp': iteration_data['iteration_info']['timestamp'],
                    'dataset': iteration_data['result_data']['dataset'],
                    'model': iteration_data['result_data']['model'],
                    'explanation_method': iteration_data['result_data']['explanation_method'],
                    'validation_status': iteration_data['result_data'].get('validation_status', False),
                    'used_tuned_params': iteration_data['result_data'].get('used_tuned_params', False)
                })
            except Exception as e:
                self.logger.warning(f"Could not read iteration file {iteration_file}: {e}")
        
        # Save summary
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self.logger.info(f"Iteration summary saved to {summary_file}")
        
        # Create folder structure summary
        self._create_folder_structure_summary()
    
    def _create_folder_structure_summary(self):
        """Create a summary of the organized folder structure"""
        if not (self.results_dir / "iterations").exists():
            return
            
        structure_file = self.results_dir / "folder_structure_summary.json"
        
        structure = {
            'organized_structure': {},
            'generated_at': datetime.now().isoformat()
        }
        
        iterations_dir = self.results_dir / "iterations"
        
        # Walk through the organized folder structure
        for dataset_dir in iterations_dir.iterdir():
            if dataset_dir.is_dir():
                dataset_name = dataset_dir.name
                structure['organized_structure'][dataset_name] = {}
                
                for model_dir in dataset_dir.iterdir():
                    if model_dir.is_dir():
                        model_name = model_dir.name
                        structure['organized_structure'][dataset_name][model_name] = {}
                        
                        for method_dir in model_dir.iterdir():
                            if method_dir.is_dir():
                                method_name = method_dir.name
                                method_files = list(method_dir.glob("*.json"))
                                structure['organized_structure'][dataset_name][model_name][method_name] = {
                                    'file_count': len(method_files),
                                    'files': [f.name for f in method_files]
                                }
        
        # Save structure summary
        with open(structure_file, 'w') as f:
            json.dump(structure, f, indent=2, default=str)
        
        self.logger.info(f"Folder structure summary saved to {structure_file}") 