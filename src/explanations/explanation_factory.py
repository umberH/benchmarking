"""
Explanation factory for creating different XAI methods
"""

import logging
from typing import Dict, Any, List

from .base_explainer import BaseExplainer
from .feature_attribution import SHAPExplainer, LIMEExplainer, IntegratedGradientsExplainer
from .example_based import PrototypeExplainer, CounterfactualExplainer
from .concept_based import TCAVExplainer, ConceptBottleneckExplainer
from .perturbation import OcclusionExplainer, FeatureAblationExplainer, TextOcclusionExplainer
from .attention_based import AttentionVisualizationExplainer
from .advanced_methods import (
    CausalSHAPExplainer, ShapleyFlowExplainer, BayesianRuleListExplainer,
    InfluenceFunctionExplainer, SHAPInteractivePlotsExplainer, CORELSExplainer
)


class ExplanationFactory:
    """
    Factory class for creating XAI explanation methods
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize explanation factory
        
        Args:
            config: Configuration dictionary for explanations
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Explanation method registry
        self.explainer_registry = {
            # Feature attribution methods
            'shap': SHAPExplainer,
            'lime': LIMEExplainer,
            'integrated_gradients': IntegratedGradientsExplainer,
            
            # Example-based methods
            'prototype': PrototypeExplainer,
            'counterfactual': CounterfactualExplainer,
            
            # Concept-based methods
            'tcav': TCAVExplainer,
            'concept_bottleneck': ConceptBottleneckExplainer,
            
            # Perturbation methods
            'occlusion': OcclusionExplainer,
            'feature_ablation': FeatureAblationExplainer,
            'text_occlusion': TextOcclusionExplainer,
            
            # Attention-based methods
            'attention_visualization': AttentionVisualizationExplainer,
            
            # Advanced methods
            'causal_shap': CausalSHAPExplainer,
            'shapley_flow': ShapleyFlowExplainer,
            'bayesian_rule_list': BayesianRuleListExplainer,
            'influence_functions': InfluenceFunctionExplainer,
            'shap_interactive': SHAPInteractivePlotsExplainer,
            'corels': CORELSExplainer,
        }
    
    def get_available_methods(self) -> List[Dict[str, Any]]:
        """Get list of available explanation method configurations"""
        methods = self.config.get('methods')
        if methods:
            return methods

        # Fallback: collect from categorized sections
        flattened: List[Dict[str, Any]] = []
        categories = (
            'feature_attribution',
            'example_based',
            'concept_based',
            'perturbation',
        )
        for category in categories:
            for item in self.config.get(category, []) or []:
                name = item.get('name')
                if not name:
                    continue
                flattened.append({
                    'name': name,
                    'type': name,  # registry key matches the name in default config
                    'category': category,
                    'description': item.get('description', ''),
                    'params': item.get('params', {})
                })
        # Also add methods from registry that aren't in config
        for method_name, explainer_class in self.explainer_registry.items():
            # Check if method is already in flattened list
            if not any(method['name'] == method_name for method in flattened):
                # Add method from registry
                method_info = self.get_method_info(method_name)
                flattened.append({
                    'name': method_name,
                    'type': method_name,
                    'category': self._infer_category_from_registry(method_name),
                    'description': method_info.get('description', ''),
                    'params': {}
                })
        
        return flattened
    
    def _infer_category_from_registry(self, method_name: str) -> str:
        """Infer category for registry methods"""
        if method_name in ['shap', 'lime', 'integrated_gradients', 'attention_visualization', 
                          'causal_shap', 'shapley_flow', 'shap_interactive']:
            return 'feature_attribution'
        elif method_name in ['prototype', 'counterfactual', 'influence_functions']:
            return 'example_based'
        elif method_name in ['tcav', 'concept_bottleneck', 'bayesian_rule_list', 'corels']:
            return 'concept_based'
        elif method_name in ['occlusion', 'feature_ablation', 'text_occlusion']:
            return 'perturbation'
        else:
            return 'advanced'
    
    def create_explainer(self, explanation_config: Dict[str, Any], model, dataset) -> BaseExplainer:
        """
        Create an explainer instance
        
        Args:
            explanation_config: Explanation method configuration
            model: Trained model instance
            dataset: Dataset instance
        Returns:
            Explainer instance
        """
        method_name = explanation_config['name']
        if method_name not in self.explainer_registry:
            raise ValueError(f"Unknown explanation method name: {method_name}")
        self.logger.info(f"Creating explainer: {method_name}")
        explainer_class = self.explainer_registry[method_name]
        explainer = explainer_class(explanation_config, model, dataset)
        return explainer
    
    def get_method_info(self, method_name: str) -> Dict[str, Any]:
        """Get information about an explanation method by name"""
        if method_name not in self.explainer_registry:
            raise ValueError(f"Unknown explanation method name: {method_name}")
        explainer_class = self.explainer_registry[method_name]
        return {
            'name': method_name,
            'class_name': explainer_class.__name__,
            'description': explainer_class.__doc__ or '',
            'supported_data_types': getattr(explainer_class, 'supported_data_types', []),
            'supported_model_types': getattr(explainer_class, 'supported_model_types', [])
        }