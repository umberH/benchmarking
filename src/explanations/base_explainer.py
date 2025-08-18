"""
Base explainer class for all XAI methods
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import numpy as np


class BaseExplainer(ABC):
    """
    Base class for all XAI explanation methods
    """
    
    def __init__(self, config: Dict[str, Any], model, dataset):
        """
        Initialize base explainer
        
        Args:
            config: Explanation method configuration
            model: Trained model instance
            dataset: Dataset instance
        """
        self.config = config
        self.model = model
        self.dataset = dataset
        self.logger = logging.getLogger(__name__)
        self.generation_time = 0.0
    
    @abstractmethod
    def explain(self, dataset) -> Dict[str, Any]:
        """
        Generate explanations for the dataset
        
        Args:
            dataset: Dataset to explain
            
        Returns:
            Dictionary containing explanation results
        """
        pass
    
    def explain_instance(self, instance) -> Dict[str, Any]:
        """
        Generate explanation for a single instance
        
        Args:
            instance: Single data instance
            
        Returns:
            Dictionary containing explanation for the instance
        """
        # Default implementation - override in subclasses
        start_time = time.time()
        
        # Create a mock explanation
        explanation = {
            'feature_importance': np.random.rand(instance.shape[0]) if hasattr(instance, 'shape') else [0.5],
            'prediction': self.model.predict([instance])[0] if hasattr(self.model, 'predict') else 0,
            'confidence': 0.8,
            'method': self.config['type']
        }
        
        self.generation_time = time.time() - start_time
        explanation['generation_time'] = self.generation_time
        
        return explanation
    
    def get_info(self) -> Dict[str, Any]:
        """Get explainer information"""
        return {
            'name': self.config['name'],
            'type': self.config['type'],
            'config': self.config,
            'supported_data_types': getattr(self, 'supported_data_types', []),
            'supported_model_types': getattr(self, 'supported_model_types', [])
        }
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure when explanation cannot be generated"""
        return {
            'explanations': [],
            'generation_time': 0.0,
            'method': self.config.get('type', 'unknown'),
            'info': {
                'n_explanations': 0,
                'error': 'Explanation method not compatible with this data type',
                'supported_data_types': getattr(self, 'supported_data_types', []),
                'supported_model_types': getattr(self, 'supported_model_types', [])
            }
        } 