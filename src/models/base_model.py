"""
Base model class for all ML models
"""

import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import accuracy_score, classification_report


class BaseModel(ABC):
    """
    Base class for all ML models
    """
    
    def __init__(self, config: Dict[str, Any], dataset):
        """
        Initialize base model
        
        Args:
            config: Model configuration
            dataset: Dataset instance
        """
        self.config = config
        self.dataset = dataset
        self.logger = logging.getLogger(__name__)
        self.model = None
        self.training_time = 0.0
        self.is_trained = False
    
    @abstractmethod
    def _create_model(self) -> Any:
        """Create the underlying model"""
        pass
    
    @abstractmethod
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the model"""
        pass
    
    @abstractmethod
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        pass
    
    def train(self, dataset):
        """Train the model on the dataset"""
        self.logger.info(f"Training model: {self.config['name']}")
        
        start_time = time.time()
        
        # Get training data
        X_train, X_test, y_train, y_test = dataset.get_data()

        # Ensure at least two classes are present to avoid downstream solver errors
        try:
            unique_classes = np.unique(y_train)
            if unique_classes.size < 2:
                raise ValueError(
                    f"Training labels have only one class ({unique_classes.tolist()}); "
                    f"cannot train model '{self.config.get('name')}'."
                )
        except Exception:
            # If uniqueness check fails due to type, proceed and let underlying model raise
            pass
        
        # Create and train model
        self.model = self._create_model()
        self._train_model(X_train, y_train)
        
        self.training_time = time.time() - start_time
        self.is_trained = True
        
        self.logger.info(f"Model training completed in {self.training_time:.2f}s")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        return self._predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before making predictions")
        
        # Default implementation - override in subclasses
        predictions = self._predict(X)
        # Convert to one-hot encoding for probability-like output
        n_classes = len(np.unique(predictions))
        proba = np.zeros((len(predictions), n_classes))
        proba[np.arange(len(predictions)), predictions] = 1.0
        return proba
    
    def evaluate(self, dataset) -> Dict[str, float]:
        """Evaluate the model on the dataset"""
        if not self.is_trained:
            raise RuntimeError("Model must be trained before evaluation")
        
        X_train, X_test, y_train, y_test = dataset.get_data()
        
        # Make predictions
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score']
        }
    
    def get_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'name': self.config['name'],
            'type': self.config['type'],
            'is_trained': self.is_trained,
            'training_time': self.training_time,
            'config': self.config
        } 