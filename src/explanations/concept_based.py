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
        # Placeholder: check for concepts in config or dataset
        concepts = getattr(dataset, 'concepts', None)
        if not concepts or len(concepts) == 0:
            return {
                'explanations': [],
                'generation_time': 0.0,
                'method': 'tcav',
                'info': {
                    'n_explanations': 0,
                    'debug': 'No concepts provided for TCAV. Please specify concept sets for this dataset/model.'
                }
            }
        # If concepts exist, implement TCAV logic here (placeholder)
        # For now, just return a debug message
        return {
            'explanations': [],
            'generation_time': 0.0,
            'method': 'tcav',
            'info': {
                'n_explanations': 0,
                'debug': f'TCAV not implemented. Concepts provided: {concepts}'
            }
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