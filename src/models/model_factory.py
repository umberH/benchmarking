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
        """Get list of available model configurations with correct registry keys and data types"""
        models_to_train = self.config.get('models_to_train')
        # If models_to_train is a list of strings, expand to full config dicts
        if models_to_train and all(isinstance(m, str) for m in models_to_train):
            expanded = []
            # Search all model sections for each name
            for model_name in models_to_train:
                found = False
                for data_type in ('tabular', 'image', 'text'):
                    for item in self.config.get(data_type, []) or []:
                        if item.get('name', '').lower() == model_name.lower():
                            # Map name to registry key
                            name_lower = model_name.lower()
                            reg_key = name_lower if name_lower in self.model_registry else None
                            if not reg_key:
                                for k in self.model_registry:
                                    if k in name_lower:
                                        reg_key = k
                                        break
                            if not reg_key:
                                continue
                            model_class = self.model_registry[reg_key]
                            expanded.append({
                                'name': model_name,
                                'type': reg_key,
                                'description': item.get('description', ''),
                                'data_type': data_type,
                                'params': item.get('params', {}),
                                'supported_data_types': getattr(model_class, 'supported_data_types', [])
                            })
                            found = True
                            break
                    if found:
                        break
            return expanded
        # If models_to_train is a list of dicts, ensure 'type' is valid
        if models_to_train and all(isinstance(m, dict) for m in models_to_train):
            for m in models_to_train:
                if 'type' in m and m['type'] not in self.model_registry:
                    # Try to map from name to registry key
                    name_lower = m['name'].lower()
                    if name_lower in self.model_registry:
                        m['type'] = name_lower
            return models_to_train

        # Fallback: build from grouped sections (tabular/image/text)
        flattened: List[Dict[str, Any]] = []
        for data_type in ('tabular', 'image', 'text'):
            for item in self.config.get(data_type, []) or []:
                name = item.get('name')
                if not name:
                    continue
                # Map name to registry key
                name_lower = name.lower()
                if name_lower in self.model_registry:
                    reg_key = name_lower
                else:
                    # Try to match by partials (e.g., 'cnn' in 'cnnmodel')
                    reg_key = None
                    for k in self.model_registry:
                        if k in name_lower:
                            reg_key = k
                            break
                    if not reg_key:
                        continue
                model_class = self.model_registry[reg_key]
                flattened.append({
                    'name': name,
                    'type': reg_key,
                    'description': item.get('description', ''),
                    'data_type': data_type,
                    'params': item.get('params', {}),
                    'supported_data_types': getattr(model_class, 'supported_data_types', [])
                })
        return flattened
    
    def create_model(self, model_config: Dict[str, Any], dataset, model_type: str = None) -> BaseModel:
        """
        Create a model instance
        
        Args:
            model_config: Model configuration
            dataset: Dataset instance
            model_type: Type of the model (registry key, e.g., 'cnn', 'bert', etc.)
        Returns:
            Model instance
        """
        model_name = model_config['name']
        # Use model_type if provided, else use config 'type', else map from name
        reg_key = model_type or model_config.get('type')
        if not reg_key or reg_key not in self.model_registry:
            # Try to map from name
            name_lower = model_name.lower()
            if name_lower in self.model_registry:
                reg_key = name_lower
            else:
                # Try partial match
                for k in self.model_registry:
                    if k in name_lower:
                        reg_key = k
                        break
        if not reg_key or reg_key not in self.model_registry:
            raise ValueError(f"Unknown model type: {reg_key} (from name: {model_name})")
        self.logger.info(f"Creating model: {model_name} (type: {reg_key})")
        model_class = self.model_registry[reg_key]
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