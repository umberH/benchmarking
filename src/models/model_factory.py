"""
Model factory for creating different types of ML models
"""

import logging
from typing import Dict, Any, List

from .base_model import BaseModel
from .tabular_models import DecisionTreeModel, RandomForestModel, GradientBoostingModel, MLPModel
from .image_models import CNNModel, ViTModel
from .text_models import BERTModel, LSTMModel


class ModelFactory:
    """
    Factory class for creating ML models
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize model factory
        
        Args:
            config: Configuration dictionary for models
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model registry
        self.model_registry = {
            # Tabular models
            'decision_tree': DecisionTreeModel,
            'random_forest': RandomForestModel,
            'gradient_boosting': GradientBoostingModel,
            'mlp': MLPModel,
            
            # Image models
            'cnn': CNNModel,
            'vit': ViTModel,
            
            # Text models
            'bert': BERTModel,
            'lstm': LSTMModel,
        }
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available model configurations"""
        models_to_train = self.config.get('models_to_train')
        if models_to_train:
            return models_to_train

        # Fallback: build from grouped sections (tabular/image/text)
        flattened: List[Dict[str, Any]] = []
        for data_type in ('tabular', 'image', 'text'):
            for item in self.config.get(data_type, []) or []:
                name = item.get('name')
                if not name:
                    continue
                flattened.append({
                    'name': name,
                    'type': name,  # registry key matches the name in default config
                    'description': item.get('description', ''),
                    'data_type': data_type,
                    'params': item.get('params', {})
                })
        return flattened
    
    def create_model(self, model_config: Dict[str, Any], dataset, model_type: str) -> BaseModel:
        """
        Create a model instance
        
        Args:
            model_config: Model configuration
            dataset: Dataset instance
            model_type: Type of the model (tabular, image, text)
        Returns:
            Model instance
        """
        model_name = model_config['name']
        if model_type not in self.model_registry:
            raise ValueError(f"Unknown model type: {model_type}")
        self.logger.info(f"Creating model: {model_name} (type: {model_type})")
        model_class = self.model_registry[model_type]
        model = model_class(model_config, dataset)
        return model
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """Get information about a model type"""
        if model_type not in self.model_registry:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_class = self.model_registry[model_type]
        return {
            'type': model_type,
            'name': model_class.__name__,
            'description': model_class.__doc__ or '',
            'supported_data_types': getattr(model_class, 'supported_data_types', [])
        } 