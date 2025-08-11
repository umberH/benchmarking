"""
Concept-based explanation methods
"""

from .base_explainer import BaseExplainer


class TCAVExplainer(BaseExplainer):
    """TCAV (Testing with Concept Activation Vectors) explainer (placeholder)"""
    
    supported_data_types = ['image']
    supported_model_types = ['cnn', 'vit']
    
    def explain(self, dataset) -> dict:
        """Generate TCAV explanations"""
        return {
            'explanations': [],
            'generation_time': 0.0,
            'method': 'tcav',
            'info': {'n_explanations': 0}
        }


class ConceptBottleneckExplainer(BaseExplainer):
    """Concept Bottleneck explainer (placeholder)"""
    
    supported_data_types = ['image']
    supported_model_types = ['cnn', 'vit']
    
    def explain(self, dataset) -> dict:
        """Generate concept bottleneck explanations"""
        return {
            'explanations': [],
            'generation_time': 0.0,
            'method': 'concept_bottleneck',
            'info': {'n_explanations': 0}
        } 