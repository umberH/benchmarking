"""
Tabular model implementations
"""

import numpy as np
from typing import Dict, Any
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

from .base_model import BaseModel


class DecisionTreeModel(BaseModel):
    """Decision Tree model for tabular data"""
    
    supported_data_types = ['tabular']
    
    def _create_model(self) -> DecisionTreeClassifier:
        """Create decision tree model"""
        params = self.config.get('params', {})
        return DecisionTreeClassifier(
            max_depth=params.get('max_depth', 10),
            min_samples_split=params.get('min_samples_split', 2),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            random_state=42,
            **params
        )
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the decision tree"""
        self.model.fit(X_train, y_train)
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions"""
        return self.model.predict_proba(X)


class RandomForestModel(BaseModel):
    """Random Forest model for tabular data"""
    
    supported_data_types = ['tabular']
    
    def _create_model(self) -> RandomForestClassifier:
        """Create random forest model"""
        params = self.config.get('params', {})
        return RandomForestClassifier(
            n_estimators=params.get('n_estimators', 100),
            max_depth=params.get('max_depth', 10),
            min_samples_split=params.get('min_samples_split', 2),
            min_samples_leaf=params.get('min_samples_leaf', 1),
            random_state=42,
            **params
        )
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the random forest"""
        self.model.fit(X_train, y_train)
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions"""
        return self.model.predict_proba(X)


class GradientBoostingModel(BaseModel):
    """Gradient Boosting model for tabular data"""
    
    supported_data_types = ['tabular']
    
    def _create_model(self) -> GradientBoostingClassifier:
        """Create gradient boosting model"""
        params = self.config.get('params', {})
        return GradientBoostingClassifier(
            n_estimators=params.get('n_estimators', 100),
            learning_rate=params.get('learning_rate', 0.1),
            max_depth=params.get('max_depth', 3),
            random_state=42,
            **params
        )
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the gradient boosting model"""
        self.model.fit(X_train, y_train)
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions"""
        return self.model.predict_proba(X)


class MLPModel(BaseModel):
    """Multi-layer Perceptron model for tabular data"""
    
    supported_data_types = ['tabular']
    
    def _create_model(self) -> MLPClassifier:
        """Create MLP model"""
        params = self.config.get('params', {})
        return MLPClassifier(
            hidden_layer_sizes=params.get('hidden_layer_sizes', (100, 50)),
            activation=params.get('activation', 'relu'),
            solver=params.get('solver', 'adam'),
            alpha=params.get('alpha', 0.0001),
            learning_rate=params.get('learning_rate', 'adaptive'),
            max_iter=params.get('max_iter', 200),
            random_state=42,
            **params
        )
    
    def _train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train the MLP model"""
        self.model.fit(X_train, y_train)
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Make probability predictions"""
        return self.model.predict_proba(X) 