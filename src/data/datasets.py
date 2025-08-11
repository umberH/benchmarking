"""
Dataset classes for different data types
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.preprocessing import StandardScaler, LabelEncoder


class BaseDataset(ABC):
    """Base class for all datasets"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.preprocessing_time = 0.0
        self._preprocess()
    
    @abstractmethod
    def _preprocess(self):
        """Preprocess the dataset"""
        pass
    
    @abstractmethod
    def get_info(self) -> Dict[str, Any]:
        """Get dataset information"""
        pass
    
    @abstractmethod
    def get_data(self) -> Tuple[Any, Any, Any, Any]:
        """Get train/test data"""
        pass


class TabularDataset(BaseDataset):
    """Tabular dataset class"""
    
    def __init__(self, name: str, X_train: pd.DataFrame, X_test: pd.DataFrame, 
                 y_train: pd.Series, y_test: pd.Series, feature_names: List[str], 
                 config: Dict[str, Any]):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = feature_names
        self.scaler = None
        self.label_encoder = None
        
        super().__init__(name, config)
    
    def _preprocess(self):
        """Preprocess tabular data"""
        start_time = time.time()
        
        # Handle missing values
        self.X_train = self.X_train.fillna(self.X_train.mean())
        self.X_test = self.X_test.fillna(self.X_test.mean())
        
        # Encode categorical variables
        categorical_columns = self.X_train.select_dtypes(include=['object']).columns
        for col in categorical_columns:
            self.X_train[col] = self.X_train[col].astype('category').cat.codes
            self.X_test[col] = self.X_test[col].astype('category').cat.codes
        
        # Scale numerical features
        numerical_columns = self.X_train.select_dtypes(include=[np.number]).columns
        if len(numerical_columns) > 0:
            self.scaler = StandardScaler()
            self.X_train[numerical_columns] = self.scaler.fit_transform(self.X_train[numerical_columns])
            self.X_test[numerical_columns] = self.scaler.transform(self.X_test[numerical_columns])
        
        # Encode target variable
        self.label_encoder = LabelEncoder()
        self.y_train = self.label_encoder.fit_transform(self.y_train)
        self.y_test = self.label_encoder.transform(self.y_test)
        
        self.preprocessing_time = time.time() - start_time
        self.logger.info(f"Tabular dataset preprocessing completed in {self.preprocessing_time:.2f}s")
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset information"""
        return {
            'name': self.name,
            'type': 'tabular',
            'n_features': len(self.feature_names),
            'n_train_samples': len(self.X_train),
            'n_test_samples': len(self.X_test),
            'n_classes': len(np.unique(self.y_train)),
            'feature_names': self.feature_names,
            'class_distribution': {
                'train': np.bincount(self.y_train).tolist(),
                'test': np.bincount(self.y_test).tolist()
            }
        }
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get train/test data as numpy arrays"""
        return (
            self.X_train.values,
            self.X_test.values,
            self.y_train,
            self.y_test
        )


class ImageDataset(BaseDataset):
    """Image dataset class"""
    
    def __init__(self, name: str, data_path: Path = None, config: Dict[str, Any] = None, 
                 X_train: np.ndarray = None, X_test: np.ndarray = None, 
                 y_train: np.ndarray = None, y_test: np.ndarray = None):
        self.data_path = data_path
        self.image_size = config.get('image_size', (224, 224)) if config else (224, 224)
        self.train_images = []
        self.train_labels = []
        self.test_images = []
        self.test_labels = []
        self.class_names = []
        
        # If data is provided directly, use it
        if X_train is not None and X_test is not None and y_train is not None and y_test is not None:
            self.train_images = X_train
            self.test_images = X_test
            self.train_labels = y_train
            self.test_labels = y_test
            self.class_names = [f"class_{i}" for i in range(len(np.unique(y_train)))]
            config = config or {}
        
        super().__init__(name, config or {})
    
    def _preprocess(self):
        """Preprocess image data"""
        start_time = time.time()
        
        # If data is already provided, skip loading from files
        if len(self.train_images) > 0 and len(self.test_images) > 0:
            self.preprocessing_time = time.time() - start_time
            self.logger.info(f"Image dataset preprocessing completed in {self.preprocessing_time:.2f}s")
            return
        
        # Load images from directory structure
        if self.data_path is None:
            self.logger.warning("No data path provided for image dataset")
            return
            
        class_dirs = [d for d in self.data_path.iterdir() if d.is_dir()]
        self.class_names = [d.name for d in class_dirs]
        
        # Load training data (assuming 80% split)
        for class_idx, class_dir in enumerate(class_dirs):
            image_files = list(class_dir.glob('*.jpg')) + list(class_dir.glob('*.png'))
            
            # Split into train/test
            n_train = int(0.8 * len(image_files))
            train_files = image_files[:n_train]
            test_files = image_files[n_train:]
            
            # Load training images
            for img_file in train_files:
                try:
                    img = Image.open(img_file).convert('RGB')
                    img = img.resize(self.image_size)
                    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                    self.train_images.append(img_array)
                    self.train_labels.append(class_idx)
                except Exception as e:
                    self.logger.warning(f"Failed to load image {img_file}: {e}")
            
            # Load test images
            for img_file in test_files:
                try:
                    img = Image.open(img_file).convert('RGB')
                    img = img.resize(self.image_size)
                    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
                    self.test_images.append(img_array)
                    self.test_labels.append(class_idx)
                except Exception as e:
                    self.logger.warning(f"Failed to load image {img_file}: {e}")
        
        self.train_images = np.array(self.train_images)
        self.test_images = np.array(self.test_images)
        self.train_labels = np.array(self.train_labels)
        self.test_labels = np.array(self.test_labels)
        
        self.preprocessing_time = time.time() - start_time
        self.logger.info(f"Image dataset preprocessing completed in {self.preprocessing_time:.2f}s")
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset information"""
        return {
            'name': self.name,
            'type': 'image',
            'n_train_samples': len(self.train_images),
            'n_test_samples': len(self.test_images),
            'n_classes': len(self.class_names),
            'image_size': self.image_size,
            'class_names': self.class_names,
            'class_distribution': {
                'train': np.bincount(self.train_labels).tolist(),
                'test': np.bincount(self.test_labels).tolist()
            }
        }
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Get train/test data as numpy arrays"""
        return (
            self.train_images,
            self.test_images,
            self.train_labels,
            self.test_labels
        )


class TextDataset(BaseDataset):
    """Text dataset class"""
    
    def __init__(self, name: str, train_texts: List[str], test_texts: List[str],
                 train_targets: List[Any], test_targets: List[Any], config: Dict[str, Any]):
        self.train_texts = train_texts
        self.test_texts = test_texts
        self.train_targets = train_targets
        self.test_targets = test_targets
        self.max_length = config.get('max_length', 512)
        self.tokenizer = None
        
        super().__init__(name, config)
    
    def _preprocess(self):
        """Preprocess text data"""
        start_time = time.time()
        
        # Basic text preprocessing
        self.train_texts = [str(text).strip() for text in self.train_texts]
        self.test_texts = [str(text).strip() for text in self.test_texts]
        
        # Encode targets
        unique_targets = list(set(self.train_targets + self.test_targets))
        target_to_idx = {target: idx for idx, target in enumerate(unique_targets)}
        
        self.train_targets = [target_to_idx[target] for target in self.train_targets]
        self.test_targets = [target_to_idx[target] for target in self.test_targets]
        
        self.train_targets = np.array(self.train_targets)
        self.test_targets = np.array(self.test_targets)
        
        self.preprocessing_time = time.time() - start_time
        self.logger.info(f"Text dataset preprocessing completed in {self.preprocessing_time:.2f}s")
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset information"""
        return {
            'name': self.name,
            'type': 'text',
            'n_train_samples': len(self.train_texts),
            'n_test_samples': len(self.test_texts),
            'n_classes': len(np.unique(self.train_targets)),
            'max_length': self.max_length,
            'avg_text_length': {
                'train': np.mean([len(text) for text in self.train_texts]),
                'test': np.mean([len(text) for text in self.test_texts])
            },
            'class_distribution': {
                'train': np.bincount(self.train_targets).tolist(),
                'test': np.bincount(self.test_targets).tolist()
            }
        }
    
    def get_data(self) -> Tuple[List[str], List[str], np.ndarray, np.ndarray]:
        """Get train/test data"""
        return (
            self.train_texts,
            self.test_texts,
            self.train_targets,
            self.test_targets
        ) 