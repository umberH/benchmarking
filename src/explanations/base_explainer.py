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

    def _limit_test_samples(self, X_test, y_test):
        """
        Limit test samples based on config max_test_samples setting.

        Args:
            X_test: Full test set
            y_test: Full test labels

        Returns:
            Limited X_test, y_test
        """
        # Debug logging
        print(f"\n[LIMIT DEBUG] === _limit_test_samples called ===")
        print(f"[LIMIT DEBUG] Config type: {type(self.config)}")
        print(f"[LIMIT DEBUG] Config keys: {list(self.config.keys()) if isinstance(self.config, dict) else 'Not a dict'}")
        print(f"[LIMIT DEBUG] X_test length: {len(X_test)}")

        # Check config structure
        if 'experiment' in self.config:
            print(f"[LIMIT DEBUG] 'experiment' found in config")
            exp_config = self.config.get('experiment', {})
            print(f"[LIMIT DEBUG] experiment keys: {list(exp_config.keys()) if isinstance(exp_config, dict) else 'Not a dict'}")
            if 'explanation' in exp_config:
                print(f"[LIMIT DEBUG] 'explanation' found in experiment")
                expl_config = exp_config.get('explanation', {})
                print(f"[LIMIT DEBUG] explanation keys: {list(expl_config.keys()) if isinstance(expl_config, dict) else 'Not a dict'}")
                if 'max_test_samples' in expl_config:
                    print(f"[LIMIT DEBUG] max_test_samples = {expl_config['max_test_samples']}")
                else:
                    print(f"[LIMIT DEBUG] 'max_test_samples' NOT found in explanation config")
            else:
                print(f"[LIMIT DEBUG] 'explanation' NOT found in experiment config")
        else:
            print(f"[LIMIT DEBUG] 'experiment' NOT found in config")

        max_test_samples = self.config.get('experiment', {}).get('explanation', {}).get('max_test_samples', None)
        print(f"[LIMIT DEBUG] Final max_test_samples value: {max_test_samples}")
        print(f"[LIMIT DEBUG] Will limit: {max_test_samples is not None and len(X_test) > max_test_samples}")

        if max_test_samples is not None and len(X_test) > max_test_samples:
            self.logger.info(f"Limiting test set from {len(X_test)} to {max_test_samples} samples for explanation generation")
            print(f"[LIMIT DEBUG] LIMITING from {len(X_test)} to {max_test_samples}")
            if isinstance(X_test, list):
                return X_test[:max_test_samples], y_test[:max_test_samples]
            else:
                return X_test[:max_test_samples], y_test[:max_test_samples]

        print(f"[LIMIT DEBUG] NOT LIMITING - returning full test set")
        return X_test, y_test
    
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