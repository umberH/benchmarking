"""
Perturbation-based explanation methods
"""

from .base_explainer import BaseExplainer


class OcclusionExplainer(BaseExplainer):
    """Occlusion explainer (placeholder)"""
    
    supported_data_types = ['image']
    supported_model_types = ['cnn', 'vit']
    
    def explain(self, dataset) -> dict:
        """Generate occlusion explanations"""
        return {
            'explanations': [],
            'generation_time': 0.0,
            'method': 'occlusion',
            'info': {'n_explanations': 0}
        }


class FeatureAblationExplainer(BaseExplainer):
    """Feature ablation explainer (placeholder)"""
    
    supported_data_types = ['tabular']
    supported_model_types = ['all']
    
    def explain(self, dataset) -> dict:
        """Generate feature ablation explanations"""
        return {
            'explanations': [],
            'generation_time': 0.0,
            'method': 'feature_ablation',
            'info': {'n_explanations': 0}
        } 