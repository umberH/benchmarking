"""
Smart Explanation Factory with Intelligent Method Selection
Enhanced factory that automatically selects appropriate explanation methods based on data and model characteristics
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
import time

from .base_explainer import BaseExplainer
from .explanation_factory import ExplanationFactory
from ..utils.method_selector import SmartMethodSelector, SelectionCriteria, MethodSelectionResult
from ..utils.evaluation_profiles import EvaluationProfileManager
from ..utils.method_compatibility import MethodCompatibilityMatrix, Priority


class SmartExplanationFactory(ExplanationFactory):
    """
    Enhanced explanation factory with intelligent method selection and routing
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize smart explanation factory
        
        Args:
            config: Configuration dictionary for explanations
        """
        super().__init__(config)
        self.method_selector = SmartMethodSelector()
        self.evaluation_manager = EvaluationProfileManager()
        self.compatibility_matrix = MethodCompatibilityMatrix()
        
        # Performance tracking
        self.method_performance_history = {}
        self.selection_history = []
        
        # Configuration options
        self.auto_selection_enabled = config.get('auto_selection', {}).get('enabled', True)
        self.fallback_enabled = config.get('fallback', {}).get('enabled', True)
        self.performance_tracking = config.get('performance_tracking', {}).get('enabled', True)
    
    def select_and_create_explainers(self, model, dataset, 
                                   user_preferences: Optional[List[str]] = None,
                                   selection_criteria: Optional[SelectionCriteria] = None) -> Dict[str, Any]:
        """
        Intelligently select and create appropriate explainers for the given model and dataset
        
        Args:
            model: Trained model instance
            dataset: Dataset instance
            user_preferences: Optional list of preferred methods
            selection_criteria: Optional criteria for method selection
            
        Returns:
            Dictionary containing explainers, selection info, and evaluation plan
        """
        start_time = time.time()
        
        # Step 1: Select appropriate methods
        if selection_criteria is None:
            selection_criteria = self._get_default_selection_criteria()
        
        selection_result = self.method_selector.select_methods(
            dataset, model, selection_criteria, user_preferences
        )
        
        # Step 2: Create explainers for selected methods
        explainers = {}
        creation_errors = {}
        
        for method_name in selection_result.selected_methods:
            try:
                explainer = self._create_explainer_safe(method_name, model, dataset)
                explainers[method_name] = explainer
                self.logger.info(f"Successfully created explainer: {method_name}")
            except Exception as e:
                creation_errors[method_name] = str(e)
                self.logger.error(f"Failed to create explainer {method_name}: {e}")
                
                # Try fallback methods if enabled
                if self.fallback_enabled and method_name in selection_result.fallback_methods:
                    fallback_success = self._try_fallback_methods(
                        method_name, selection_result.fallback_methods[method_name],
                        model, dataset, explainers, creation_errors
                    )
                    if not fallback_success:
                        self.logger.warning(f"All fallback methods failed for {method_name}")
        
        # Step 3: Create evaluation plan
        evaluation_plan = self._create_evaluation_plan(explainers, selection_result)
        
        # Step 4: Track selection and performance
        selection_time = time.time() - start_time
        selection_info = self._create_selection_info(
            selection_result, explainers, creation_errors, selection_time
        )
        
        if self.performance_tracking:
            self._track_selection_performance(selection_info)
        
        return {
            'explainers': explainers,
            'selection_info': selection_info,
            'evaluation_plan': evaluation_plan,
            'creation_errors': creation_errors
        }
    
    def _get_default_selection_criteria(self) -> SelectionCriteria:
        """Get default selection criteria from config"""
        auto_config = self.config.get('auto_selection', {})
        
        return SelectionCriteria(
            max_methods=auto_config.get('max_methods', 3),
            max_priority=Priority(auto_config.get('max_priority', Priority.MEDIUM.value)),
            allow_special_setup=auto_config.get('allow_special_setup', False),
            allow_high_cost=auto_config.get('allow_high_cost', True),
            require_differentiable=auto_config.get('require_differentiable'),
            include_research_methods=auto_config.get('include_research_methods', False)
        )
    
    def _create_explainer_safe(self, method_name: str, model, dataset) -> BaseExplainer:
        """Safely create an explainer with enhanced error handling"""
        # Validate compatibility first
        is_compatible, reason = self.method_selector.validate_method_compatibility(
            method_name, dataset, model
        )
        
        if not is_compatible:
            raise ValueError(f"Method {method_name} is not compatible: {reason}")
        
        # Create explainer configuration
        method_config = {
            'name': method_name,
            'type': method_name,
            'params': self.config.get('method_params', {}).get(method_name, {})
        }
        
        # Create explainer
        return self.create_explainer(method_config, model, dataset)
    
    def _try_fallback_methods(self, failed_method: str, fallback_methods: List[str],
                            model, dataset, explainers: Dict, errors: Dict) -> bool:
        """Try fallback methods when primary method fails"""
        for fallback_method in fallback_methods:
            try:
                explainer = self._create_explainer_safe(fallback_method, model, dataset)
                explainers[fallback_method] = explainer
                self.logger.info(f"Successfully created fallback explainer {fallback_method} "
                               f"for failed method {failed_method}")
                return True
            except Exception as e:
                errors[fallback_method] = f"Fallback for {failed_method}: {str(e)}"
                self.logger.warning(f"Fallback method {fallback_method} also failed: {e}")
        
        return False
    
    def _create_evaluation_plan(self, explainers: Dict[str, BaseExplainer], 
                              selection_result: MethodSelectionResult) -> Dict[str, List[str]]:
        """Create evaluation plan for the selected explainers"""
        evaluation_plan = {}
        
        # Get method categories for better metric selection
        method_categories = {}
        for method_name in explainers.keys():
            method_info = self.compatibility_matrix.get_method_info(method_name)
            if method_info:
                method_categories[method_name] = method_info.category.value
        
        # Create plan using evaluation manager
        evaluation_plan = self.evaluation_manager.create_evaluation_plan(
            list(explainers.keys()), method_categories
        )
        
        # Add comparative metrics for cross-method comparison
        if len(explainers) > 1:
            comparative_metrics = self.evaluation_manager.get_comparative_metrics(
                list(explainers.keys())
            )
            evaluation_plan['_comparative_metrics'] = comparative_metrics
        
        return evaluation_plan
    
    def _create_selection_info(self, selection_result: MethodSelectionResult,
                             explainers: Dict, creation_errors: Dict,
                             selection_time: float) -> Dict[str, Any]:
        """Create comprehensive selection information"""
        return {
            'data_type_info': {
                'detected_type': selection_result.data_type_info.data_type,
                'shape': selection_result.data_type_info.data_shape,
                'features': selection_result.data_type_info.feature_count,
                'is_high_dimensional': selection_result.data_type_info.is_high_dimensional,
                'has_spatial_structure': selection_result.data_type_info.has_spatial_structure,
                'detection_confidence': selection_result.data_type_info.confidence
            },
            'model_type': selection_result.model_type,
            'selected_methods': selection_result.selected_methods,
            'successfully_created': list(explainers.keys()),
            'creation_failures': creation_errors,
            'recommendations': selection_result.recommendations,
            'excluded_methods': selection_result.excluded_methods,
            'selection_rationale': selection_result.selection_rationale,
            'selection_time_seconds': selection_time,
            'auto_selection_used': self.auto_selection_enabled
        }
    
    def _track_selection_performance(self, selection_info: Dict[str, Any]):
        """Track selection performance for future optimization"""
        entry = {
            'timestamp': time.time(),
            'data_type': selection_info['data_type_info']['detected_type'],
            'model_type': selection_info['model_type'],
            'selected_methods': selection_info['selected_methods'],
            'success_rate': len(selection_info['successfully_created']) / 
                          len(selection_info['selected_methods']) if selection_info['selected_methods'] else 0,
            'selection_time': selection_info['selection_time_seconds']
        }
        
        self.selection_history.append(entry)
        
        # Keep only recent history (last 100 selections)
        if len(self.selection_history) > 100:
            self.selection_history = self.selection_history[-100:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the smart selection system"""
        if not self.selection_history:
            return {'message': 'No selection history available'}
        
        # Calculate statistics
        total_selections = len(self.selection_history)
        avg_success_rate = sum(entry['success_rate'] for entry in self.selection_history) / total_selections
        avg_selection_time = sum(entry['selection_time'] for entry in self.selection_history) / total_selections
        
        # Method popularity
        method_counts = {}
        for entry in self.selection_history:
            for method in entry['selected_methods']:
                method_counts[method] = method_counts.get(method, 0) + 1
        
        most_popular_methods = sorted(method_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Data type distribution
        data_type_counts = {}
        for entry in self.selection_history:
            data_type = entry['data_type']
            data_type_counts[data_type] = data_type_counts.get(data_type, 0) + 1
        
        return {
            'total_selections': total_selections,
            'average_success_rate': avg_success_rate,
            'average_selection_time_seconds': avg_selection_time,
            'most_popular_methods': most_popular_methods,
            'data_type_distribution': data_type_counts
        }
    
    def get_method_recommendations_for_dataset(self, dataset, model) -> Dict[str, Any]:
        """Get method recommendations without creating explainers"""
        return self.method_selector.get_method_recommendations_summary(dataset, model)
    
    def validate_method_for_dataset(self, method_name: str, dataset, model) -> Tuple[bool, str]:
        """Validate if a method is suitable for the given dataset and model"""
        return self.method_selector.validate_method_compatibility(method_name, dataset, model)
    
    def create_explainer_with_validation(self, method_name: str, model, dataset) -> Tuple[Optional[BaseExplainer], str]:
        """
        Create an explainer with full validation and error handling
        
        Returns:
            Tuple of (explainer_instance_or_None, status_message)
        """
        try:
            # Validate compatibility
            is_compatible, reason = self.validate_method_for_dataset(method_name, dataset, model)
            if not is_compatible:
                return None, f"Incompatible: {reason}"
            
            # Create explainer
            explainer = self._create_explainer_safe(method_name, model, dataset)
            return explainer, f"Successfully created {method_name} explainer"
            
        except Exception as e:
            self.logger.error(f"Failed to create {method_name}: {e}")
            return None, f"Creation failed: {str(e)}"
    
    def get_evaluation_metrics_for_methods(self, methods: List[str]) -> Dict[str, List[str]]:
        """Get appropriate evaluation metrics for a list of methods"""
        return self.evaluation_manager.create_evaluation_plan(methods)
    
    def get_optimal_method_set(self, dataset, model, max_methods: int = 3) -> List[str]:
        """Get optimal method set for benchmarking purposes"""
        data_info = self.method_selector.data_detector.detect_data_type(dataset)
        model_type = self.method_selector.data_detector.detect_model_type(model)
        
        return self.compatibility_matrix.suggest_optimal_method_set(
            data_info.data_type, model_type, max_methods
        )