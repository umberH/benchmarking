"""
Smart Method Selection Logic
Orchestrates the selection of appropriate explanation methods based on data and model characteristics
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from .data_type_detector import DataTypeDetector, DataTypeInfo
from .method_compatibility import MethodCompatibilityMatrix, Priority


@dataclass
class SelectionCriteria:
    """Criteria for method selection"""
    max_methods: int = 3
    max_priority: Priority = Priority.MEDIUM
    allow_special_setup: bool = False
    allow_high_cost: bool = True
    require_differentiable: Optional[bool] = None
    include_research_methods: bool = False


@dataclass
class MethodSelectionResult:
    """Result of method selection process"""
    selected_methods: List[str]
    data_type_info: DataTypeInfo
    model_type: str
    recommendations: Dict[str, List[str]]
    fallback_methods: Dict[str, List[str]]
    excluded_methods: Dict[str, str]  # method -> reason for exclusion
    selection_rationale: str


class SmartMethodSelector:
    """
    Intelligent method selection orchestrator
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.data_detector = DataTypeDetector()
        self.compatibility_matrix = MethodCompatibilityMatrix()
    
    def select_methods(self, dataset, model, 
                      criteria: Optional[SelectionCriteria] = None,
                      user_preferences: Optional[List[str]] = None) -> MethodSelectionResult:
        """
        Select optimal explanation methods for the given dataset and model
        
        Args:
            dataset: Dataset instance
            model: Model instance
            criteria: Selection criteria and constraints
            user_preferences: Optional list of preferred methods
            
        Returns:
            MethodSelectionResult with selected methods and rationale
        """
        if criteria is None:
            criteria = SelectionCriteria()
        
        # Step 1: Detect data type and model characteristics
        data_info = self.data_detector.detect_data_type(dataset)
        model_type = self.data_detector.detect_model_type(model)
        
        self.logger.info(f"Detected data type: {data_info.data_type} "
                        f"(confidence: {data_info.confidence:.2f})")
        self.logger.info(f"Detected model type: {model_type}")
        
        # Step 2: Get compatible methods
        max_priority = Priority.RESEARCH if criteria.include_research_methods else criteria.max_priority
        compatible_methods = self.compatibility_matrix.get_compatible_methods(
            data_info.data_type, model_type, max_priority
        )
        
        # Step 3: Apply constraints and filters
        filtered_methods = self.compatibility_matrix.filter_methods_by_constraints(
            compatible_methods,
            allow_special_setup=criteria.allow_special_setup,
            allow_high_cost=criteria.allow_high_cost,
            require_differentiable=criteria.require_differentiable
        )
        
        # Step 4: Handle user preferences
        if user_preferences:
            # Validate user preferences against compatibility
            valid_preferences = [m for m in user_preferences if m in filtered_methods]
            invalid_preferences = [m for m in user_preferences if m not in filtered_methods]
            
            if invalid_preferences:
                self.logger.warning(f"Some preferred methods are not compatible: {invalid_preferences}")
            
            # Prioritize user preferences
            remaining_slots = max(0, criteria.max_methods - len(valid_preferences))
            auto_selected = [m for m in filtered_methods if m not in valid_preferences][:remaining_slots]
            selected_methods = valid_preferences + auto_selected
        else:
            # Use automatic optimal selection
            selected_methods = self.compatibility_matrix.suggest_optimal_method_set(
                data_info.data_type, model_type, criteria.max_methods
            )
            # Filter selected methods through constraints
            selected_methods = [m for m in selected_methods if m in filtered_methods]
        
        # Step 5: Generate recommendations and fallbacks
        recommendations = self.compatibility_matrix.get_recommended_methods(
            data_info.data_type, model_type
        )
        
        fallback_methods = {}
        for method in selected_methods:
            fallbacks = self.compatibility_matrix.get_fallback_methods(
                data_info.data_type, model_type, method
            )
            if fallbacks:
                fallback_methods[method] = fallbacks
        
        # Step 6: Identify excluded methods and reasons
        excluded_methods = {}
        all_methods = list(self.compatibility_matrix.methods.keys())
        
        for method in all_methods:
            if method in selected_methods:
                continue
            
            method_info = self.compatibility_matrix.get_method_info(method)
            if method_info is None:
                continue
            
            # Determine exclusion reason
            if data_info.data_type not in method_info.supported_data_types:
                excluded_methods[method] = f"Incompatible data type (requires {method_info.supported_data_types})"
            elif ('all' not in method_info.supported_model_types and 
                  model_type not in method_info.supported_model_types):
                excluded_methods[method] = f"Incompatible model type (requires {method_info.supported_model_types})"
            elif method_info.priority_by_data_type.get(data_info.data_type, Priority.RESEARCH).value > max_priority.value:
                excluded_methods[method] = f"Priority too low for current settings"
            elif method_info.requires_special_setup and not criteria.allow_special_setup:
                excluded_methods[method] = "Requires special setup (disabled)"
            elif method_info.computational_cost == "high" and not criteria.allow_high_cost:
                excluded_methods[method] = "Too computationally expensive"
            elif (criteria.require_differentiable is not None and 
                  method_info.requires_differentiable_model != criteria.require_differentiable):
                excluded_methods[method] = "Differentiability requirement mismatch"
            else:
                excluded_methods[method] = "Not selected (lower priority or max methods reached)"
        
        # Step 7: Generate selection rationale
        rationale = self._generate_selection_rationale(
            data_info, model_type, selected_methods, criteria
        )
        
        result = MethodSelectionResult(
            selected_methods=selected_methods,
            data_type_info=data_info,
            model_type=model_type,
            recommendations=recommendations,
            fallback_methods=fallback_methods,
            excluded_methods=excluded_methods,
            selection_rationale=rationale
        )
        
        self.logger.info(f"Selected {len(selected_methods)} methods: {selected_methods}")
        return result
    
    def _generate_selection_rationale(self, data_info: DataTypeInfo, model_type: str,
                                    selected_methods: List[str], 
                                    criteria: SelectionCriteria) -> str:
        """Generate human-readable rationale for method selection"""
        rationale_parts = []
        
        # Data type rationale
        rationale_parts.append(f"Detected {data_info.data_type} data with {data_info.feature_count} features")
        
        if data_info.has_spatial_structure:
            rationale_parts.append("Data has spatial structure (image-like)")
        if data_info.has_sequential_structure:
            rationale_parts.append("Data has sequential structure")
        if data_info.is_high_dimensional:
            rationale_parts.append("High-dimensional data detected")
        
        # Model type rationale
        rationale_parts.append(f"Model type: {model_type}")
        
        # Method selection rationale
        if len(selected_methods) > 0:
            method_categories = {}
            for method in selected_methods:
                method_info = self.compatibility_matrix.get_method_info(method)
                if method_info:
                    category = method_info.category.value
                    if category not in method_categories:
                        method_categories[category] = []
                    method_categories[category].append(method)
            
            rationale_parts.append("Selected methods provide:")
            for category, methods in method_categories.items():
                rationale_parts.append(f"  - {category.replace('_', ' ').title()}: {', '.join(methods)}")
        
        # Constraints rationale
        constraint_info = []
        if not criteria.allow_special_setup:
            constraint_info.append("excluding methods requiring special setup")
        if not criteria.allow_high_cost:
            constraint_info.append("excluding high-cost methods")
        if criteria.require_differentiable is not None:
            if criteria.require_differentiable:
                constraint_info.append("requiring differentiable methods only")
            else:
                constraint_info.append("excluding differentiable-only methods")
        
        if constraint_info:
            rationale_parts.append(f"Applied constraints: {', '.join(constraint_info)}")
        
        return ". ".join(rationale_parts) + "."
    
    def validate_method_compatibility(self, method_name: str, dataset, model) -> Tuple[bool, str]:
        """
        Validate if a specific method is compatible with the dataset and model
        
        Args:
            method_name: Name of the method to validate
            dataset: Dataset instance
            model: Model instance
            
        Returns:
            Tuple of (is_compatible, reason)
        """
        data_info = self.data_detector.detect_data_type(dataset)
        model_type = self.data_detector.detect_model_type(model)
        
        method_info = self.compatibility_matrix.get_method_info(method_name)
        if method_info is None:
            return False, f"Unknown method: {method_name}"
        
        # Check data type compatibility
        if data_info.data_type not in method_info.supported_data_types:
            return False, f"Method '{method_name}' does not support {data_info.data_type} data"
        
        # Check model type compatibility
        if ('all' not in method_info.supported_model_types and 
            model_type not in method_info.supported_model_types):
            return False, f"Method '{method_name}' does not support {model_type} models"
        
        # Check if method requires differentiable model
        if method_info.requires_differentiable_model:
            if model_type in ['decision_tree', 'random_forest', 'gradient_boosting', 'svm']:
                return False, f"Method '{method_name}' requires a differentiable model, but got {model_type}"
        
        return True, f"Method '{method_name}' is compatible"
    
    def get_method_recommendations_summary(self, dataset, model) -> Dict[str, Any]:
        """
        Get a summary of method recommendations for the dataset and model
        
        Returns:
            Dictionary with recommendation summary
        """
        data_info = self.data_detector.detect_data_type(dataset)
        model_type = self.data_detector.detect_model_type(model)
        
        recommendations = self.compatibility_matrix.get_recommended_methods(
            data_info.data_type, model_type
        )
        
        # Get method information for each recommended method
        detailed_recommendations = {}
        for category, methods in recommendations.items():
            detailed_recommendations[category] = []
            for method in methods:
                method_info = self.compatibility_matrix.get_method_info(method)
                if method_info:
                    detailed_recommendations[category].append({
                        'name': method,
                        'description': method_info.description,
                        'computational_cost': method_info.computational_cost,
                        'requires_special_setup': method_info.requires_special_setup
                    })
        
        return {
            'data_type': data_info.data_type,
            'model_type': model_type,
            'data_characteristics': {
                'shape': data_info.data_shape,
                'is_high_dimensional': data_info.is_high_dimensional,
                'has_spatial_structure': data_info.has_spatial_structure,
                'has_sequential_structure': data_info.has_sequential_structure
            },
            'recommendations': detailed_recommendations,
            'optimal_method_set': self.compatibility_matrix.suggest_optimal_method_set(
                data_info.data_type, model_type, 3
            )
        }