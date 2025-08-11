"""
Feature attribution explanation methods
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List
from sklearn.ensemble import RandomForestRegressor

from .base_explainer import BaseExplainer


class SHAPExplainer(BaseExplainer):
    """SHAP (SHapley Additive exPlanations) explainer"""
    
    supported_data_types = ['tabular']
    supported_model_types = ['decision_tree', 'random_forest', 'gradient_boosting', 'mlp']
    
    def explain(self, dataset) -> Dict[str, Any]:
        """Generate SHAP explanations"""
        start_time = time.time()
        
        # Get test data
        X_train, X_test, y_train, y_test = dataset.get_data()
        
        # Handle different data types
        if isinstance(X_test, list):
            # Text data - SHAP not supported for text
            self.logger.warning("SHAP not supported for text data, skipping...")
            return {
                'explanations': [],
                'generation_time': 0.0,
                'method': 'shap',
                'info': {
                    'n_explanations': 0,
                    'feature_names': []
                }
            }
        
        # Simplified SHAP implementation using permutation importance
        explanations = []
        feature_names = dataset.feature_names if hasattr(dataset, 'feature_names') else [f'feature_{i}' for i in range(X_test.shape[1])]
        
        # Generate explanations for a subset of test instances
        n_explanations = min(100, len(X_test))  # Limit for performance
        test_subset = X_test[:n_explanations]
        
        for i, instance in enumerate(test_subset):
            # Calculate feature importance using permutation
            importance = self._calculate_permutation_importance(instance, X_train, y_train)
            
            explanation = {
                'instance_id': i,
                'feature_importance': importance,
                'feature_names': feature_names,
                'prediction': self.model.predict([instance])[0],
                'true_label': y_test[i] if i < len(y_test) else None
            }
            explanations.append(explanation)
        
        self.generation_time = time.time() - start_time
        
        return {
            'explanations': explanations,
            'generation_time': self.generation_time,
            'method': 'shap',
            'info': {
                'n_explanations': len(explanations),
                'feature_names': feature_names
            }
        }
    
    def _calculate_permutation_importance(self, instance, X_train, y_train):
        """Calculate feature importance using permutation method"""
        base_prediction = self.model.predict([instance])[0]
        importance = np.zeros(len(instance))
        
        for i in range(len(instance)):
            # Create perturbed instance
            perturbed_instance = instance.copy()
            perturbed_instance[i] = np.mean(X_train[:, i])  # Replace with mean value
            
            # Calculate new prediction
            new_prediction = self.model.predict([perturbed_instance])[0]
            
            # Importance is the change in prediction
            importance[i] = abs(base_prediction - new_prediction)
        
        return importance


class LIMEExplainer(BaseExplainer):
    """LIME (Local Interpretable Model-agnostic Explanations) explainer"""
    
    supported_data_types = ['tabular', 'text']
    supported_model_types = ['all']
    
    def explain(self, dataset) -> Dict[str, Any]:
        """Generate LIME explanations"""
        start_time = time.time()
        
        # Get test data
        X_train, X_test, y_train, y_test = dataset.get_data()
        
        # Handle different data types
        if isinstance(X_test, list):
            # Text data - use fixed-size word-based features
            max_words = 50  # Fixed size for all explanations
            feature_names = dataset.feature_names if hasattr(dataset, 'feature_names') else ['word_' + str(i) for i in range(max_words)]
            n_features = max_words
        else:
            # Tabular data
            feature_names = dataset.feature_names if hasattr(dataset, 'feature_names') else [f'feature_{i}' for i in range(X_test.shape[1])]
            n_features = X_test.shape[1]
        
        explanations = []
        
        # Generate explanations for a subset of test instances
        n_explanations = min(50, len(X_test))  # Limit for performance
        test_subset = X_test[:n_explanations]
        
        for i, instance in enumerate(test_subset):
            # Generate local explanations using linear approximation
            importance = self._generate_local_explanation(instance, X_train, y_train, n_features)
            
            explanation = {
                'instance_id': i,
                'feature_importance': importance,
                'feature_names': feature_names,
                'prediction': self.model.predict([instance])[0],
                'true_label': y_test[i] if i < len(y_test) else None
            }
            explanations.append(explanation)
        
        self.generation_time = time.time() - start_time
        
        return {
            'explanations': explanations,
            'generation_time': self.generation_time,
            'method': 'lime',
            'info': {
                'n_explanations': len(explanations),
                'feature_names': feature_names
            }
        }
    
    def _generate_local_explanation(self, instance, X_train, y_train, n_features=None):
        """Generate local explanation using linear approximation"""
        if isinstance(instance, str):
            # Text data - word-based importance using TF-IDF like approach
            max_words = n_features if n_features else 50
            words = instance.split()[:max_words]  # Take first max_words words
            
            if not words:
                return np.zeros(max_words)
            
            # Create word importance based on frequency and position
            word_importance = []
            for i, word in enumerate(words):
                # Base importance from word frequency in training data
                word_freq = 0
                if isinstance(X_train, list):
                    for text in X_train:
                        if isinstance(text, str) and word in text:
                            word_freq += 1
                
                # Position-based importance (earlier words more important)
                position_weight = 1.0 - (i / len(words))
                
                # Length-based importance (longer words might be more important)
                length_weight = min(len(word) / 10.0, 1.0)
                
                # Combine factors
                importance = (word_freq / max(len(X_train), 1)) * position_weight * length_weight
                word_importance.append(importance)
            
            # Pad or truncate to ensure consistent size
            if len(word_importance) < max_words:
                # Pad with zeros
                word_importance.extend([0.0] * (max_words - len(word_importance)))
            elif len(word_importance) > max_words:
                # Truncate
                word_importance = word_importance[:max_words]
            
            # Normalize importance scores
            word_importance = np.array(word_importance)
            if np.sum(word_importance) > 0:
                word_importance = word_importance / np.sum(word_importance)
            else:
                word_importance = np.ones_like(word_importance) / len(word_importance)
            
            return word_importance
        else:
            # Tabular data - generate perturbed samples around the instance
            n_samples = 100
            perturbed_samples = []
            
            for _ in range(n_samples):
                # Add random noise to create perturbed sample
                noise = np.random.normal(0, 0.1, len(instance))
                perturbed_sample = instance + noise
                perturbed_samples.append(perturbed_sample)
            
            perturbed_samples = np.array(perturbed_samples)
            
            # Get predictions for perturbed samples
            predictions = self.model.predict(perturbed_samples)
            
            # Fit a linear model to approximate the local behavior
            linear_model = RandomForestRegressor(n_estimators=10, random_state=42)
            linear_model.fit(perturbed_samples, predictions)
            
            # Feature importance from the linear model
            importance = linear_model.feature_importances_
            
            return importance


class IntegratedGradientsExplainer(BaseExplainer):
    """Integrated Gradients explainer (simplified implementation)"""
    
    supported_data_types = ['tabular']
    supported_model_types = ['mlp', 'cnn', 'vit']
    
    def explain(self, dataset) -> Dict[str, Any]:
        """Generate Integrated Gradients explanations"""
        start_time = time.time()
        
        # Get test data
        X_train, X_test, y_train, y_test = dataset.get_data()
        
        # Handle different data types
        if isinstance(X_test, list):
            # Text data - Integrated Gradients not supported for text
            self.logger.warning("Integrated Gradients not supported for text data, skipping...")
            return {
                'explanations': [],
                'generation_time': 0.0,
                'method': 'integrated_gradients',
                'info': {
                    'n_explanations': 0,
                    'feature_names': []
                }
            }
        
        explanations = []
        feature_names = dataset.feature_names if hasattr(dataset, 'feature_names') else [f'feature_{i}' for i in range(X_test.shape[1])]
        
        # Generate explanations for a subset of test instances
        n_explanations = min(30, len(X_test))  # Limit for performance
        test_subset = X_test[:n_explanations]
        
        for i, instance in enumerate(test_subset):
            # Calculate integrated gradients
            importance = self._calculate_integrated_gradients(instance, X_train)
            
            explanation = {
                'instance_id': i,
                'feature_importance': importance,
                'feature_names': feature_names,
                'prediction': self.model.predict([instance])[0],
                'true_label': y_test[i] if i < len(y_test) else None
            }
            explanations.append(explanation)
        
        self.generation_time = time.time() - start_time
        
        return {
            'explanations': explanations,
            'generation_time': self.generation_time,
            'method': 'integrated_gradients',
            'info': {
                'n_explanations': len(explanations),
                'feature_names': feature_names
            }
        }
    
    def _calculate_integrated_gradients(self, instance, X_train):
        """Calculate integrated gradients (simplified)"""
        # Simplified implementation using finite differences
        baseline = np.mean(X_train, axis=0)
        n_steps = 50
        
        importance = np.zeros(len(instance))
        
        for i in range(len(instance)):
            gradient_sum = 0
            
            for step in range(1, n_steps + 1):
                alpha = step / n_steps
                interpolated = baseline + alpha * (instance - baseline)
                
                # Calculate gradient using finite differences
                epsilon = 1e-6
                interpolated_plus = interpolated.copy()
                interpolated_plus[i] += epsilon
                interpolated_minus = interpolated.copy()
                interpolated_minus[i] -= epsilon
                
                pred_plus = self.model.predict([interpolated_plus])[0]
                pred_minus = self.model.predict([interpolated_minus])[0]
                
                gradient = (pred_plus - pred_minus) / (2 * epsilon)
                gradient_sum += gradient * (instance[i] - baseline[i]) / n_steps
            
            importance[i] = abs(gradient_sum)
        
        return importance 