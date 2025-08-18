"""
Data Type Detection Module
Automatically detects data types and model architectures for intelligent method selection
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class DataTypeInfo:
    """Information about detected data type and characteristics"""
    data_type: str  # 'tabular', 'image', 'text'
    data_shape: Tuple[int, ...]
    feature_count: int
    is_high_dimensional: bool
    has_spatial_structure: bool
    has_sequential_structure: bool
    sample_range: Tuple[float, float]
    likely_normalized: bool
    confidence: float  # 0-1 confidence in detection


class DataTypeDetector:
    """
    Intelligent data type detection for explanation method selection
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def detect_data_type(self, dataset) -> DataTypeInfo:
        """
        Detect the primary data type of the dataset
        
        Args:
            dataset: Dataset instance with get_data() method
            
        Returns:
            DataTypeInfo: Comprehensive information about the detected data type
        """
        try:
            X_train, X_test, y_train, y_test = dataset.get_data()
            
            # Handle both numpy arrays and lists
            if isinstance(X_train, list):
                return self._detect_text_or_mixed(X_train, dataset)
            
            if not isinstance(X_train, np.ndarray):
                X_train = np.array(X_train)
            
            return self._analyze_array_data(X_train, dataset)
            
        except Exception as e:
            self.logger.error(f"Error in data type detection: {e}")
            return self._create_fallback_info()
    
    def _detect_text_or_mixed(self, data: List, dataset) -> DataTypeInfo:
        """Detect text data or mixed types"""
        if len(data) == 0:
            return self._create_fallback_info()
        
        # Check if elements are strings (text data)
        sample = data[0]
        if isinstance(sample, str):
            avg_length = np.mean([len(text) for text in data[:100]])
            return DataTypeInfo(
                data_type='text',
                data_shape=(len(data), int(avg_length)),
                feature_count=int(avg_length),
                is_high_dimensional=avg_length > 1000,
                has_spatial_structure=False,
                has_sequential_structure=True,
                sample_range=(0, 0),  # Not applicable for text
                likely_normalized=False,
                confidence=0.9
            )
        
        # Convert to numpy and analyze
        try:
            array_data = np.array(data)
            return self._analyze_array_data(array_data, dataset)
        except:
            return self._create_fallback_info()
    
    def _analyze_array_data(self, X: np.ndarray, dataset) -> DataTypeInfo:
        """Analyze numpy array data to determine type"""
        shape = X.shape
        ndim = X.ndim
        
        # Get data range for normalization detection
        data_min, data_max = float(X.min()), float(X.max())
        likely_normalized = self._is_likely_normalized(data_min, data_max)
        
        # Detect based on dimensionality and characteristics
        if ndim == 2:
            return self._analyze_2d_data(X, shape, data_min, data_max, likely_normalized, dataset)
        elif ndim >= 3:
            return self._analyze_high_dim_data(X, shape, data_min, data_max, likely_normalized, dataset)
        else:
            # 1D data - likely tabular features for single instance
            return DataTypeInfo(
                data_type='tabular',
                data_shape=shape,
                feature_count=shape[0],
                is_high_dimensional=shape[0] > 1000,
                has_spatial_structure=False,
                has_sequential_structure=False,
                sample_range=(data_min, data_max),
                likely_normalized=likely_normalized,
                confidence=0.7
            )
    
    def _analyze_2d_data(self, X: np.ndarray, shape: Tuple, 
                        data_min: float, data_max: float, 
                        likely_normalized: bool, dataset) -> DataTypeInfo:
        """Analyze 2D data - could be tabular or flattened images"""
        n_samples, n_features = shape
        
        # Check for common image sizes (flattened)
        image_sizes = {784: (28, 28), 3072: (32, 32, 3), 1024: (32, 32), 
                      4096: (64, 64), 12288: (64, 64, 3)}
        
        if n_features in image_sizes:
            # Likely flattened image data
            original_shape = image_sizes[n_features]
            return DataTypeInfo(
                data_type='image',
                data_shape=original_shape,
                feature_count=n_features,
                is_high_dimensional=True,
                has_spatial_structure=True,
                has_sequential_structure=False,
                sample_range=(data_min, data_max),
                likely_normalized=likely_normalized,
                confidence=0.8
            )
        
        # Check if data range suggests images (pixel values)
        if (likely_normalized and 0 <= data_min <= 1 and 0 <= data_max <= 1) or \
           (0 <= data_min <= 255 and 0 <= data_max <= 255):
            # Might be flattened images
            side_len = int(np.sqrt(n_features))
            if side_len * side_len == n_features:
                return DataTypeInfo(
                    data_type='image',
                    data_shape=(side_len, side_len),
                    feature_count=n_features,
                    is_high_dimensional=n_features > 100,
                    has_spatial_structure=True,
                    has_sequential_structure=False,
                    sample_range=(data_min, data_max),
                    likely_normalized=likely_normalized,
                    confidence=0.6
                )
        
        # Default to tabular
        return DataTypeInfo(
            data_type='tabular',
            data_shape=shape,
            feature_count=n_features,
            is_high_dimensional=n_features > 100,
            has_spatial_structure=False,
            has_sequential_structure=False,
            sample_range=(data_min, data_max),
            likely_normalized=likely_normalized,
            confidence=0.8
        )
    
    def _analyze_high_dim_data(self, X: np.ndarray, shape: Tuple,
                              data_min: float, data_max: float,
                              likely_normalized: bool, dataset) -> DataTypeInfo:
        """Analyze 3D+ data - likely images"""
        n_samples = shape[0]
        data_shape = shape[1:]  # Remove batch dimension
        
        # Check if it looks like image data
        is_image = self._looks_like_image_data(data_shape, data_min, data_max)
        
        if is_image:
            total_features = np.prod(data_shape)
            return DataTypeInfo(
                data_type='image',
                data_shape=data_shape,
                feature_count=total_features,
                is_high_dimensional=True,
                has_spatial_structure=True,
                has_sequential_structure=False,
                sample_range=(data_min, data_max),
                likely_normalized=likely_normalized,
                confidence=0.9
            )
        else:
            # High-dimensional tabular or sequential data
            total_features = np.prod(data_shape)
            has_sequential = len(data_shape) == 2  # (sequence_length, features)
            
            return DataTypeInfo(
                data_type='tabular',
                data_shape=data_shape,
                feature_count=total_features,
                is_high_dimensional=True,
                has_spatial_structure=False,
                has_sequential_structure=has_sequential,
                sample_range=(data_min, data_max),
                likely_normalized=likely_normalized,
                confidence=0.7
            )
    
    def _looks_like_image_data(self, shape: Tuple, data_min: float, data_max: float) -> bool:
        """Heuristics to determine if data looks like images"""
        # Check dimensionality
        if len(shape) < 2 or len(shape) > 4:
            return False
        
        # Check for typical image dimensions
        if len(shape) == 2:  # Grayscale (H, W)
            h, w = shape
            return 8 <= h <= 2048 and 8 <= w <= 2048
        elif len(shape) == 3:  # Either (H, W, C) or (C, H, W)
            if shape[0] in [1, 3, 4]:  # (C, H, W)
                c, h, w = shape
                return 8 <= h <= 2048 and 8 <= w <= 2048 and c <= 4
            elif shape[2] in [1, 3, 4]:  # (H, W, C)
                h, w, c = shape
                return 8 <= h <= 2048 and 8 <= w <= 2048 and c <= 4
        
        # Check data range (pixel values)
        if (0 <= data_min <= 1 and 0 <= data_max <= 1) or \
           (0 <= data_min <= 255 and 0 <= data_max <= 255):
            return True
        
        return False
    
    def _is_likely_normalized(self, data_min: float, data_max: float) -> bool:
        """Check if data appears to be normalized"""
        return abs(data_min) < 0.1 and abs(data_max - 1.0) < 0.1
    
    def _create_fallback_info(self) -> DataTypeInfo:
        """Create fallback info when detection fails"""
        return DataTypeInfo(
            data_type='tabular',  # Safe default
            data_shape=(0,),
            feature_count=0,
            is_high_dimensional=False,
            has_spatial_structure=False,
            has_sequential_structure=False,
            sample_range=(0.0, 1.0),
            likely_normalized=True,
            confidence=0.1
        )
    
    def detect_model_type(self, model) -> str:
        """
        Detect the type of model being used
        
        Args:
            model: Model instance
            
        Returns:
            str: Model type ('cnn', 'vit', 'mlp', 'decision_tree', etc.)
        """
        try:
            model_class = model.__class__.__name__.lower()
            
            # CNN detection
            if any(keyword in model_class for keyword in ['cnn', 'conv', 'resnet', 'densenet']):
                return 'cnn'
            
            # ViT detection
            if any(keyword in model_class for keyword in ['vit', 'transformer', 'attention']):
                return 'vit'
            
            # MLP detection
            if any(keyword in model_class for keyword in ['mlp', 'neural', 'perceptron']):
                return 'mlp'
            
            # Tree-based models
            if any(keyword in model_class for keyword in ['tree', 'forest', 'boost', 'xgb', 'lgb']):
                if 'forest' in model_class:
                    return 'random_forest'
                elif any(gb in model_class for gb in ['boost', 'xgb', 'lgb']):
                    return 'gradient_boosting'
                else:
                    return 'decision_tree'
            
            # SVM
            if 'svm' in model_class or 'support' in model_class:
                return 'svm'
            
            # Linear models
            if any(keyword in model_class for keyword in ['linear', 'logistic', 'ridge', 'lasso']):
                return 'linear'
            
            # Check if it's a PyTorch model with CNN layers
            if hasattr(model, 'model') and hasattr(model.model, 'modules'):
                import torch.nn as nn
                for module in model.model.modules():
                    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                        return 'cnn'
                    elif isinstance(module, nn.TransformerEncoder):
                        return 'vit'
                    elif isinstance(module, nn.Linear):
                        return 'mlp'
            
            # Default fallback
            return 'unknown'
            
        except Exception as e:
            logging.getLogger(__name__).warning(f"Model type detection failed: {e}")
            return 'unknown'