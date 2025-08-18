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
        
        # Evaluate model performance immediately after training
        self.performance_metrics = self._evaluate_model(X_train, X_test, y_train, y_test)
        
        self.logger.info(f"Model training completed in {self.training_time:.2f}s")
        self.logger.info(f"Model performance - Train Acc: {self.performance_metrics['train_accuracy']:.4f}, Test Acc: {self.performance_metrics['test_accuracy']:.4f}")
        self.logger.info(f"Model complexity - Params: {self.performance_metrics['model_complexity']['n_parameters']}, Size: {self.performance_metrics['model_complexity']['model_size_mb']:.2f}MB")
    
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
    
    def _evaluate_model(self, X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Comprehensive model evaluation after training"""
        import sys
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        try:
            # Get predictions
            train_predictions = self.predict(X_train)
            test_predictions = self.predict(X_test)
            
            # Basic accuracy metrics
            train_accuracy = accuracy_score(y_train, train_predictions)
            test_accuracy = accuracy_score(y_test, test_predictions)
            
            # Additional classification metrics (handle multi-class)
            try:
                train_f1 = f1_score(y_train, train_predictions, average='weighted')
                test_f1 = f1_score(y_test, test_predictions, average='weighted')
                train_precision = precision_score(y_train, train_predictions, average='weighted')
                test_precision = precision_score(y_test, test_predictions, average='weighted')
                train_recall = recall_score(y_train, train_predictions, average='weighted')
                test_recall = recall_score(y_test, test_predictions, average='weighted')
            except Exception:
                # Fallback for edge cases
                train_f1 = test_f1 = train_precision = test_precision = train_recall = test_recall = 0.0
            
            # Confusion matrix analysis
            try:
                cm = confusion_matrix(y_test, test_predictions)
                n_classes = len(np.unique(y_test))
                class_accuracies = cm.diagonal() / cm.sum(axis=1) if cm.sum(axis=1).all() else [0.0] * n_classes
            except Exception:
                n_classes = len(np.unique(y_test))
                class_accuracies = [0.0] * n_classes
            
            # Model complexity metrics
            model_complexity = self._get_model_complexity()
            
            # Overfitting analysis
            overfitting_gap = train_accuracy - test_accuracy
            overfitting_severity = "high" if overfitting_gap > 0.2 else ("moderate" if overfitting_gap > 0.1 else "low")
            
            return {
                'train_accuracy': float(train_accuracy),
                'test_accuracy': float(test_accuracy),
                'train_f1': float(train_f1),
                'test_f1': float(test_f1),
                'train_precision': float(train_precision),
                'test_precision': float(test_precision),
                'train_recall': float(train_recall),
                'test_recall': float(test_recall),
                'overfitting_gap': float(overfitting_gap),
                'overfitting_severity': overfitting_severity,
                'class_accuracies': [float(acc) for acc in class_accuracies],
                'n_classes': int(n_classes),
                'n_train_samples': int(len(X_train)),
                'n_test_samples': int(len(X_test)),
                'training_time': float(self.training_time),
                'model_complexity': model_complexity
            }
        except Exception as e:
            self.logger.warning(f"Error during model evaluation: {e}")
            return {
                'train_accuracy': 0.0,
                'test_accuracy': 0.0,
                'error': str(e),
                'training_time': float(self.training_time),
                'model_complexity': self._get_model_complexity()
            }
    
    def _get_model_complexity(self) -> Dict[str, Any]:
        """Get model complexity metrics"""
        import sys
        
        try:
            # Try to get model parameters count
            n_parameters = 0
            model_size_bytes = 0
            
            if hasattr(self.model, 'get_params'):
                # Sklearn model
                params = self.model.get_params()
                n_parameters = len(params)
                model_size_bytes = sys.getsizeof(self.model)
                
            elif hasattr(self.model, 'parameters'):
                # PyTorch model
                n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                model_size_bytes = sum(p.numel() * p.element_size() for p in self.model.parameters())
                
            elif hasattr(self.model, 'count_params'):
                # TensorFlow/Keras model
                n_parameters = self.model.count_params()
                model_size_bytes = sys.getsizeof(self.model)
            else:
                # Generic fallback
                model_size_bytes = sys.getsizeof(self.model)
                n_parameters = 1  # Placeholder
                
            model_size_mb = model_size_bytes / (1024 * 1024)
            
            # Complexity classification
            if n_parameters < 1000:
                complexity_level = "simple"
            elif n_parameters < 100000:
                complexity_level = "moderate"
            else:
                complexity_level = "complex"
                
            return {
                'n_parameters': int(n_parameters),
                'model_size_bytes': int(model_size_bytes),
                'model_size_mb': float(model_size_mb),
                'complexity_level': complexity_level
            }
        except Exception as e:
            return {
                'n_parameters': 0,
                'model_size_bytes': 0,
                'model_size_mb': 0.0,
                'complexity_level': 'unknown',
                'error': str(e)
            }
    
    def evaluate(self, dataset) -> Dict[str, Any]:
        """Public method to evaluate model"""
        if hasattr(self, 'performance_metrics'):
            return self.performance_metrics
        else:
            # Fallback evaluation
            X_train, X_test, y_train, y_test = dataset.get_data()
            return self._evaluate_model(X_train, X_test, y_train, y_test)
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
            'type': self.config['name'],
            'is_trained': self.is_trained,
            'training_time': self.training_time,
            'config': self.config
        }