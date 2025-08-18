"""
Method Compatibility Matrix and Filtering
Determines which explanation methods are suitable for given data types and models
"""

from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging


class MethodCategory(Enum):
    """Categories of explanation methods"""
    FEATURE_ATTRIBUTION = "feature_attribution"
    EXAMPLE_BASED = "example_based"
    CONCEPT_BASED = "concept_based"
    PERTURBATION_BASED = "perturbation_based"


class Priority(Enum):
    """Priority levels for method selection"""
    HIGH = 1      # Highly recommended, proven effective
    MEDIUM = 2    # Good option, situational
    LOW = 3       # Experimental or limited use cases
    RESEARCH = 4  # Research methods, may require special setup


@dataclass
class MethodInfo:
    """Information about an explanation method"""
    name: str
    category: MethodCategory
    supported_data_types: Set[str]
    supported_model_types: Set[str]
    priority_by_data_type: Dict[str, Priority]
    requires_differentiable_model: bool = False
    requires_special_setup: bool = False
    computational_cost: str = "medium"  # low, medium, high
    description: str = ""


class MethodCompatibilityMatrix:
    """
    Central registry for method compatibility and selection logic
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.methods = self._initialize_method_registry()
    
    def _initialize_method_registry(self) -> Dict[str, MethodInfo]:
        """Initialize the comprehensive method registry"""
        methods = {}
        
        # Feature Attribution Methods
        methods['shap'] = MethodInfo(
            name='shap',
            category=MethodCategory.FEATURE_ATTRIBUTION,
            supported_data_types={'tabular', 'text'},
            supported_model_types={'decision_tree', 'random_forest', 'gradient_boosting', 'mlp'},
            priority_by_data_type={
                'tabular': Priority.HIGH,
                'image': Priority.LOW,  # Limited support
                'text': Priority.HIGH   # Now fully supported
            },
            computational_cost="medium",
            description="SHapley Additive exPlanations - theoretically grounded feature attributions"
        )
        
        methods['lime'] = MethodInfo(
            name='lime',
            category=MethodCategory.FEATURE_ATTRIBUTION,
            supported_data_types={'tabular', 'text', 'image'},
            supported_model_types={'all'},
            priority_by_data_type={
                'tabular': Priority.HIGH,
                'text': Priority.HIGH,
                'image': Priority.MEDIUM  # Works but not optimal for spatial data
            },
            computational_cost="medium",
            description="Local Interpretable Model-agnostic Explanations"
        )
        
        methods['integrated_gradients'] = MethodInfo(
            name='integrated_gradients',
            category=MethodCategory.FEATURE_ATTRIBUTION,
            supported_data_types={'tabular', 'image', 'text'},
            supported_model_types={'mlp', 'cnn', 'vit'},
            priority_by_data_type={
                'tabular': Priority.MEDIUM,
                'image': Priority.MEDIUM,
                'text': Priority.MEDIUM  # Now fully supported
            },
            requires_differentiable_model=True,
            computational_cost="high",
            description="Integrated Gradients for differentiable models"
        )
        
        methods['feature_ablation'] = MethodInfo(
            name='feature_ablation',
            category=MethodCategory.PERTURBATION_BASED,
            supported_data_types={'tabular'},
            supported_model_types={'all'},
            priority_by_data_type={
                'tabular': Priority.HIGH,
                'image': Priority.LOW,
                'text': Priority.LOW
            },
            computational_cost="low",
            description="Simple feature ablation analysis"
        )
        
        # Example-Based Methods
        methods['prototype'] = MethodInfo(
            name='prototype',
            category=MethodCategory.EXAMPLE_BASED,
            supported_data_types={'tabular', 'image'},
            supported_model_types={'all'},
            priority_by_data_type={
                'tabular': Priority.MEDIUM,
                'image': Priority.HIGH,
                'text': Priority.LOW
            },
            computational_cost="medium",
            description="Prototype-based explanations using nearest neighbors"
        )
        
        methods['counterfactual'] = MethodInfo(
            name='counterfactual',
            category=MethodCategory.EXAMPLE_BASED,
            supported_data_types={'tabular', 'image'},
            supported_model_types={'all'},
            priority_by_data_type={
                'tabular': Priority.MEDIUM,
                'image': Priority.MEDIUM,
                'text': Priority.LOW
            },
            computational_cost="high",
            description="Counterfactual explanations showing minimal changes"
        )
        
        # Perturbation-Based Methods
        methods['occlusion'] = MethodInfo(
            name='occlusion',
            category=MethodCategory.PERTURBATION_BASED,
            supported_data_types={'image'},
            supported_model_types={'cnn', 'vit'},
            priority_by_data_type={
                'image': Priority.HIGH,
                'tabular': Priority.LOW,  # Not applicable
                'text': Priority.LOW      # Not applicable
            },
            computational_cost="high",
            description="Occlusion-based spatial importance analysis"
        )
        
        # Concept-Based Methods
        methods['tcav'] = MethodInfo(
            name='tcav',
            category=MethodCategory.CONCEPT_BASED,
            supported_data_types={'image'},
            supported_model_types={'cnn', 'vit'},
            priority_by_data_type={
                'image': Priority.RESEARCH,
                'tabular': Priority.LOW,
                'text': Priority.LOW
            },
            requires_special_setup=True,
            computational_cost="high",
            description="Testing with Concept Activation Vectors"
        )
        
        methods['concept_bottleneck'] = MethodInfo(
            name='concept_bottleneck',
            category=MethodCategory.CONCEPT_BASED,
            supported_data_types={'image'},
            supported_model_types={'cnn', 'vit'},
            priority_by_data_type={
                'image': Priority.RESEARCH,
                'tabular': Priority.LOW,
                'text': Priority.LOW
            },
            requires_special_setup=True,
            computational_cost="high",
            description="Concept Bottleneck Models for interpretable predictions"
        )
        
        # Text-Specific Methods
        methods['text_occlusion'] = MethodInfo(
            name='text_occlusion',
            category=MethodCategory.PERTURBATION_BASED,
            supported_data_types={'text'},
            supported_model_types={'all'},
            priority_by_data_type={
                'text': Priority.HIGH,
                'tabular': Priority.LOW,  # Not applicable
                'image': Priority.LOW     # Not applicable
            },
            computational_cost="medium",
            description="Text occlusion-based word importance analysis"
        )
        
        methods['attention_visualization'] = MethodInfo(
            name='attention_visualization',
            category=MethodCategory.FEATURE_ATTRIBUTION,
            supported_data_types={'text'},
            supported_model_types={'all'},
            priority_by_data_type={
                'text': Priority.HIGH,
                'tabular': Priority.LOW,  # Not applicable
                'image': Priority.LOW     # Not applicable
            },
            computational_cost="low",
            description="Attention-based explanations for text models"
        )
        
        # Advanced Methods
        methods['causal_shap'] = MethodInfo(
            name='causal_shap',
            category=MethodCategory.FEATURE_ATTRIBUTION,
            supported_data_types={'tabular'},
            supported_model_types={'all'},
            priority_by_data_type={
                'tabular': Priority.HIGH,
                'image': Priority.LOW,
                'text': Priority.LOW
            },
            computational_cost="high",
            description="Causal SHAP considers feature dependencies and causal structure"
        )
        
        methods['shapley_flow'] = MethodInfo(
            name='shapley_flow',
            category=MethodCategory.FEATURE_ATTRIBUTION,
            supported_data_types={'tabular', 'image'},
            supported_model_types={'decision_tree', 'random_forest', 'mlp', 'cnn'},
            priority_by_data_type={
                'tabular': Priority.MEDIUM,
                'image': Priority.MEDIUM,
                'text': Priority.LOW
            },
            computational_cost="high",
            description="Shapley Flow traces value propagation through model layers"
        )
        
        methods['bayesian_rule_list'] = MethodInfo(
            name='bayesian_rule_list',
            category=MethodCategory.CONCEPT_BASED,
            supported_data_types={'tabular'},
            supported_model_types={'all'},
            priority_by_data_type={
                'tabular': Priority.HIGH,
                'image': Priority.LOW,
                'text': Priority.LOW
            },
            computational_cost="medium",
            description="Bayesian Rule Lists provide interpretable if-then rule explanations"
        )
        
        methods['influence_functions'] = MethodInfo(
            name='influence_functions',
            category=MethodCategory.EXAMPLE_BASED,
            supported_data_types={'tabular', 'image', 'text'},
            supported_model_types={'mlp', 'logistic_regression', 'svm'},
            priority_by_data_type={
                'tabular': Priority.MEDIUM,
                'image': Priority.MEDIUM,
                'text': Priority.MEDIUM
            },
            computational_cost="high",
            description="Influence Functions identify training examples that most affect predictions"
        )
        
        methods['shap_interactive'] = MethodInfo(
            name='shap_interactive',
            category=MethodCategory.FEATURE_ATTRIBUTION,
            supported_data_types={'tabular'},
            supported_model_types={'all'},
            priority_by_data_type={
                'tabular': Priority.HIGH,
                'image': Priority.LOW,
                'text': Priority.LOW
            },
            computational_cost="high",
            description="SHAP Interactive includes both main effects and pairwise feature interactions"
        )
        
        methods['corels'] = MethodInfo(
            name='corels',
            category=MethodCategory.CONCEPT_BASED,
            supported_data_types={'tabular'},
            supported_model_types={'all'},
            priority_by_data_type={
                'tabular': Priority.HIGH,
                'image': Priority.LOW,
                'text': Priority.LOW
            },
            computational_cost="high",
            description="CORELS provides certifiably optimal rule lists with provable guarantees"
        )
        
        return methods
    
    def get_compatible_methods(self, data_type: str, model_type: str, 
                             max_priority: Priority = Priority.RESEARCH) -> List[str]:
        """
        Get list of compatible methods for given data type and model
        
        Args:
            data_type: Type of data ('tabular', 'image', 'text')
            model_type: Type of model ('cnn', 'mlp', etc.)
            max_priority: Maximum priority level to include
            
        Returns:
            List of method names sorted by priority (best first)
        """
        compatible = []
        
        for method_name, method_info in self.methods.items():
            # Check data type compatibility
            if data_type not in method_info.supported_data_types:
                continue
            
            # Check model type compatibility
            if 'all' not in method_info.supported_model_types and \
               model_type not in method_info.supported_model_types:
                continue
            
            # Check priority threshold
            method_priority = method_info.priority_by_data_type.get(data_type, Priority.RESEARCH)
            if method_priority.value > max_priority.value:
                continue
            
            compatible.append((method_name, method_priority))
        
        # Sort by priority (lower number = higher priority)
        compatible.sort(key=lambda x: x[1].value)
        return [name for name, _ in compatible]
    
    def get_recommended_methods(self, data_type: str, model_type: str) -> Dict[str, List[str]]:
        """
        Get methods categorized by recommendation level
        
        Returns:
            Dict with keys: 'highly_recommended', 'recommended', 'optional', 'research'
        """
        all_methods = self.get_compatible_methods(data_type, model_type, Priority.RESEARCH)
        
        categorized = {
            'highly_recommended': [],
            'recommended': [],
            'optional': [],
            'research': []
        }
        
        for method_name in all_methods:
            method_info = self.methods[method_name]
            priority = method_info.priority_by_data_type.get(data_type, Priority.RESEARCH)
            
            if priority == Priority.HIGH:
                categorized['highly_recommended'].append(method_name)
            elif priority == Priority.MEDIUM:
                categorized['recommended'].append(method_name)
            elif priority == Priority.LOW:
                categorized['optional'].append(method_name)
            else:
                categorized['research'].append(method_name)
        
        return categorized
    
    def filter_methods_by_constraints(self, methods: List[str], 
                                    allow_special_setup: bool = False,
                                    allow_high_cost: bool = True,
                                    require_differentiable: bool = None) -> List[str]:
        """
        Filter methods based on computational and setup constraints
        
        Args:
            methods: List of method names to filter
            allow_special_setup: Whether to include methods requiring special setup
            allow_high_cost: Whether to include computationally expensive methods
            require_differentiable: If True, only differentiable methods; if False, exclude them
            
        Returns:
            Filtered list of method names
        """
        filtered = []
        
        for method_name in methods:
            if method_name not in self.methods:
                continue
                
            method_info = self.methods[method_name]
            
            # Check special setup requirement
            if method_info.requires_special_setup and not allow_special_setup:
                continue
            
            # Check computational cost
            if method_info.computational_cost == "high" and not allow_high_cost:
                continue
            
            # Check differentiability requirement
            if require_differentiable is not None:
                if require_differentiable and not method_info.requires_differentiable_model:
                    continue
                if not require_differentiable and method_info.requires_differentiable_model:
                    continue
            
            filtered.append(method_name)
        
        return filtered
    
    def get_method_info(self, method_name: str) -> Optional[MethodInfo]:
        """Get detailed information about a specific method"""
        return self.methods.get(method_name)
    
    def get_evaluation_metrics_for_method(self, method_name: str) -> List[str]:
        """
        Get appropriate evaluation metrics for a given method
        
        Args:
            method_name: Name of the explanation method
            
        Returns:
            List of suitable evaluation metric names
        """
        if method_name not in self.methods:
            return []
        
        method_info = self.methods[method_name]
        
        # Base metrics for all methods
        base_metrics = ['time_complexity']
        
        if method_info.category == MethodCategory.FEATURE_ATTRIBUTION:
            return base_metrics + [
                'faithfulness', 'monotonicity', 'stability', 'consistency',
                'sparsity', 'simplicity'
            ] + (['completeness'] if method_name in ['shap', 'integrated_gradients'] else [])
        
        elif method_info.category == MethodCategory.EXAMPLE_BASED:
            return base_metrics + [
                'faithfulness', 'stability', 'consistency'
            ]
        
        elif method_info.category == MethodCategory.PERTURBATION_BASED:
            if method_name == 'occlusion':
                return base_metrics + [
                    'faithfulness', 'stability', 'sparsity', 'simplicity'
                ]
            else:  # feature_ablation
                return base_metrics + [
                    'faithfulness', 'monotonicity', 'stability', 'consistency',
                    'sparsity', 'simplicity'
                ]
        
        elif method_info.category == MethodCategory.CONCEPT_BASED:
            return base_metrics + [
                'faithfulness', 'stability'
            ]
        
        return base_metrics
    
    def get_fallback_methods(self, data_type: str, model_type: str, 
                           failed_method: str) -> List[str]:
        """
        Get fallback methods when a primary method fails
        
        Args:
            data_type: Type of data
            model_type: Type of model
            failed_method: Method that failed
            
        Returns:
            List of fallback method names
        """
        # Define fallback chains
        fallback_chains = {
            'shap': ['lime', 'feature_ablation'],
            'lime': ['feature_ablation', 'prototype'],
            'integrated_gradients': ['lime', 'feature_ablation'],
            'occlusion': ['lime', 'prototype'],
            'tcav': ['occlusion', 'prototype'],
            'concept_bottleneck': ['occlusion', 'prototype'],
        }
        
        fallbacks = fallback_chains.get(failed_method, [])
        
        # Filter to only compatible methods
        compatible = self.get_compatible_methods(data_type, model_type)
        return [method for method in fallbacks if method in compatible]
    
    def suggest_optimal_method_set(self, data_type: str, model_type: str,
                                 max_methods: int = 3) -> List[str]:
        """
        Suggest an optimal set of complementary methods
        
        Args:
            data_type: Type of data
            model_type: Type of model
            max_methods: Maximum number of methods to suggest
            
        Returns:
            List of recommended method names
        """
        recommendations = self.get_recommended_methods(data_type, model_type)
        
        # Start with highly recommended methods
        selected = recommendations['highly_recommended'][:max_methods]
        
        # Fill remaining slots with recommended methods
        remaining = max_methods - len(selected)
        if remaining > 0:
            selected.extend(recommendations['recommended'][:remaining])
        
        # Ensure diversity across categories
        if len(selected) < max_methods:
            categories_present = set()
            for method in selected:
                if method in self.methods:
                    categories_present.add(self.methods[method].category)
            
            # Add methods from different categories
            remaining = max_methods - len(selected)
            for method in recommendations['optional'][:remaining]:
                if method in self.methods:
                    method_category = self.methods[method].category
                    if method_category not in categories_present:
                        selected.append(method)
                        categories_present.add(method_category)
                        if len(selected) >= max_methods:
                            break
        
        return selected[:max_methods]