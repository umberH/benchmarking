"""
Example-based explanation methods
"""

from .base_explainer import BaseExplainer


class PrototypeExplainer(BaseExplainer):
    """Prototype-based explainer (placeholder)"""
    
    supported_data_types = ['tabular', 'image']
    supported_model_types = ['all']
    
    def explain(self, dataset) -> dict:
        """Generate prototype explanations"""
        return {
            'explanations': [],
            'generation_time': 0.0,
            'method': 'prototype',
            'info': {'n_explanations': 0}
        }


class CounterfactualExplainer(BaseExplainer):
    """Counterfactual explainer (placeholder)"""
    
    supported_data_types = ['tabular']
    supported_model_types = ['all']
    
    def explain(self, dataset) -> dict:
        """Generate counterfactual explanations"""
        return {
            'explanations': [],
            'generation_time': 0.0,
            'method': 'counterfactual',
            'info': {'n_explanations': 0}
        } 