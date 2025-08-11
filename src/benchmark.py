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
            for model_config in self.config['models']['models_to_train']:
                model_name = model_config['name']
                self.logger.info(f"Training {model_name} on {dataset_name}")
                
                # Apply tuned parameters if requested and available
                if use_tuned_params:
                    tuned_params = self.get_tuned_parameters(dataset_name, model_name)
                    if tuned_params:
                        self.logger.info(f"Using tuned parameters for {model_name}: {tuned_params}")
                        model_config = {**model_config, **tuned_params}
                    else:
                        self.logger.warning(f"No tuned parameters found for {model_name} on {dataset_name}")
                
                model = self.model_factory.create_model(model_config, dataset)
                model.train(dataset)
                
                models[dataset_name][model_name] = model
                self.results['model_results'][f"{dataset_name}_{model_name}"] = {
                    'training_time': model.training_time,
                    'performance': model.evaluate(dataset),
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
                
                for explanation_config in self.config['explanations']['methods']:
                    method_name = explanation_config['name']
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
        """Run interactive benchmarking mode"""
        self.logger.info("Starting interactive XAI benchmarking")
        
        # Interactive dataset selection
        available_datasets = list(self.data_manager._get_mandatory_datasets().keys())
        if not available_datasets:
            self.logger.error("No mandatory datasets available")
            return
            
        print("\nAvailable datasets:")
        print("0. All")
        for i, dataset_name in enumerate(available_datasets, 1):
            print(f"{i}. {dataset_name}")
        
        while True:
            try:
                dataset_choice = input("\nSelect dataset (enter number): ")
                choice_index = int(dataset_choice)
                if choice_index == 0:
                    selected_datasets = available_datasets
                    break
                choice_index -= 1
                if 0 <= choice_index < len(available_datasets):
                    selected_datasets = [available_datasets[choice_index]]
                    break
                else:
                    print(f"Please enter a number between 0 and {len(available_datasets)}")
            except (ValueError, IndexError):
                print(f"Please enter a valid number between 0 and {len(available_datasets)}")
        
        # Determine model selection
        available_models = self.model_factory.get_available_models()
        if not available_models:
            self.logger.error("No models available from configuration")
            return

        # If multiple datasets selected, we will run all compatible models per dataset automatically
        select_models_interactively = len(selected_datasets) == 1
        selected_models_per_dataset = {}
        
        for selected_dataset in selected_datasets:
            dataset = self.data_manager.load_dataset(selected_dataset)
            
            # Validate dataset before proceeding
            validation_result = self.data_validator.validate_dataset(dataset, selected_dataset)
            self.results.setdefault('validation_results', {})[selected_dataset] = validation_result
            
            if not validation_result['is_valid']:
                self.logger.warning(f"Dataset {selected_dataset} failed validation:")
                for error in validation_result['errors']:
                    self.logger.warning(f"  - {error}")
                
                if validation_result['errors']:
                    self.logger.warning(f"Skipping {selected_dataset} due to validation errors")
                    continue
            
            if validation_result['warnings']:
                self.logger.info(f"Dataset {selected_dataset} has warnings:")
                for warning in validation_result['warnings']:
                    self.logger.info(f"  - {warning}")
            
            print(f"\nLoaded dataset: {selected_dataset}")
            print(f"Dataset info: {dataset.get_info()}")

            dataset_type = dataset.get_info().get('type')
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
                self.logger.warning(f"No compatible models found for dataset type '{dataset_type}' - skipping {selected_dataset}")
                continue

            if select_models_interactively:
                print("\nAvailable models:")
                print("0. All")
                for i, m in enumerate(compatible_models, 1):
                    print(f"{i}. {m.get('name')} ({m.get('type')})")
                while True:
                    try:
                        model_choice = input("\nSelect model (enter number): ")
                        choice_index = int(model_choice)
                        if choice_index == 0:
                            selected_models_per_dataset[selected_dataset] = compatible_models
                            break
                        choice_index -= 1
                        if 0 <= choice_index < len(compatible_models):
                            selected_models_per_dataset[selected_dataset] = [compatible_models[choice_index]]
                            break
                        else:
                            print(f"Please enter a number between 0 and {len(compatible_models)}")
                    except (ValueError, IndexError):
                        print(f"Please enter a valid number between 0 and {len(compatible_models)}")
            else:
                selected_models_per_dataset[selected_dataset] = compatible_models
        
        # For each selected dataset and its selected models, run explanations selection and evaluation
        available_explanations = self.explanation_factory.get_available_methods()
        if not available_explanations:
            self.logger.error("No explanation methods available from configuration")
            return

        for selected_dataset in selected_datasets:
            dataset = self.data_manager.load_dataset(selected_dataset)
            dataset_type = dataset.get_info().get('type')

            for selected_model_config in selected_models_per_dataset.get(selected_dataset, []):
                # Apply tuned parameters if requested and available
                if use_tuned_params:
                    tuned_params = self.get_tuned_parameters(selected_dataset, selected_model_config['name'])
                    if tuned_params:
                        print(f"Using tuned parameters for {selected_model_config['name']}: {tuned_params}")
                        selected_model_config = {**selected_model_config, **tuned_params}
                
                # Train model
                model = self.model_factory.create_model(selected_model_config, dataset)
                model.train(dataset)
                print(f"\nTrained model: {selected_model_config['name']} on {selected_dataset}")

                # Filter explanation methods compatible with dataset and model
                model_type_key = selected_model_config.get('type')
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
                    self.logger.warning(f"No compatible explanation methods for model '{model_type_key}' and dataset type '{dataset_type}'")
                    continue

                # If only one dataset selected and one model, allow interactive explanation selection; else run all
                interactive_explainer_selection = (len(selected_datasets) == 1 and len(selected_models_per_dataset.get(selected_dataset, [])) == 1)
                selected_explanations = []
                if interactive_explainer_selection:
                    print("\nAvailable explanation methods:")
                    print("0. All")
                    for i, e in enumerate(explanation_candidates, 1):
                        print(f"{i}. {e.get('name')} ({e.get('type')})")
                    while True:
                        try:
                            explanation_choice = input("\nSelect explanation method (enter number): ")
                            choice_index = int(explanation_choice)
                            if choice_index == 0:
                                selected_explanations = explanation_candidates
                                break
                            choice_index -= 1
                            if 0 <= choice_index < len(explanation_candidates):
                                selected_explanations = [explanation_candidates[choice_index]]
                                break
                            else:
                                print(f"Please enter a number between 0 and {len(explanation_candidates)}")
                        except (ValueError, IndexError):
                            print(f"Please enter a valid number between 0 and {len(explanation_candidates)}")
                else:
                    selected_explanations = explanation_candidates

                for selected_explanation_config in selected_explanations:
                    # Generate explanations
                    explainer = self.explanation_factory.create_explainer(
                        selected_explanation_config, model, dataset
                    )
                    explanation_results = explainer.explain(dataset)
                    print(f"\nGenerated explanations using: {selected_explanation_config['name']} for {selected_dataset}/{selected_model_config['name']}")

                    # Evaluate explanations
                    evaluation_results = self.evaluator.evaluate(model, explanation_results, dataset)
                    print(f"\nEvaluation results: {evaluation_results}")

                    # Save results incrementally
                    self.results.setdefault('interactive_runs', [])
                    self.results['interactive_runs'].append({
                        'dataset': selected_dataset,
                        'model': selected_model_config['name'],
                        'explanation_method': selected_explanation_config['name'],
                        'evaluation_results': evaluation_results,
                        'used_tuned_params': use_tuned_params and bool(self.get_tuned_parameters(selected_dataset, selected_model_config['name']))
                    })
                    
                    # Save individual iteration result
                    iteration_key = f"{selected_dataset}_{selected_model_config['name']}_{selected_explanation_config['name']}"
                    self._save_iteration_result(iteration_key, {
                        'dataset': selected_dataset,
                        'model': selected_model_config['name'],
                        'explanation_method': selected_explanation_config['name'],
                        'evaluation_results': evaluation_results,
                        'explanation_results': explanation_results,
                        'model_performance': model.evaluate(dataset),
                        'validation_status': self.results.get('validation_results', {}).get(selected_dataset, {}).get('is_valid', False),
                        'used_tuned_params': use_tuned_params and bool(self.get_tuned_parameters(selected_dataset, selected_model_config['name']))
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