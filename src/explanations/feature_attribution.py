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
    
    supported_data_types = ['tabular', 'text']
    supported_model_types = ['decision_tree', 'random_forest', 'gradient_boosting', 'mlp']
    
    def explain(self, dataset) -> Dict[str, Any]:
        """Generate SHAP explanations"""
        start_time = time.time()
        
        # Get test data
        X_train, X_test, y_train, y_test = dataset.get_data()
        
        # Handle different data types
        if isinstance(X_test, list):
            # Text data - use text-specific SHAP implementation
            return self._explain_text_data(X_test, X_train, y_test, y_train, start_time)
        
        # Simplified SHAP implementation using permutation importance
        explanations = []
        feature_names = dataset.feature_names if hasattr(dataset, 'feature_names') else [f'feature_{i}' for i in range(X_test.shape[1])]
        
        # Generate explanations for a subset of test instances
        n_explanations = min(100, len(X_test))  # Limit for performance
        test_subset = X_test[:n_explanations]
        
        for i, instance in enumerate(test_subset):
            # Calculate feature importance using permutation
            importance = self._calculate_permutation_importance(instance, X_train, y_train)
            
            # Get model predictions for faithfulness/monotonicity evaluation
            prediction = self.model.predict([instance])[0]
            
            # Create baseline (mean of training data)
            baseline = np.mean(X_train, axis=0)
            baseline_prediction = self.model.predict([baseline])[0]
            
            explanation = {
                'instance_id': i,
                'feature_importance': importance,
                'feature_names': feature_names,
                'prediction': prediction,
                'true_label': y_test[i] if i < len(y_test) else None,
                'input': instance.tolist() if hasattr(instance, 'tolist') else instance,
                'baseline_prediction': baseline_prediction
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
    
    def _explain_text_data(self, X_test, X_train, y_test, y_train, start_time) -> Dict[str, Any]:
        """Text-specific SHAP implementation using word-level Shapley values"""
        explanations = []
        
        # Generate explanations for a subset of test instances
        n_explanations = min(50, len(X_test))  # Limit for performance
        test_subset = X_test[:n_explanations]
        
        for i, text_instance in enumerate(test_subset):
            # Calculate word-level Shapley values for text
            word_importance, feature_names = self._calculate_text_shapley_values(
                text_instance, X_train, y_train
            )
            
            # Get model prediction
            prediction = self.model.predict([text_instance])[0]
            
            # Calculate baseline prediction (average of training data predictions)
            if len(X_train) > 100:
                # Sample for efficiency
                train_sample = X_train[:100]
            else:
                train_sample = X_train
            
            train_predictions = [self.model.predict([text])[0] for text in train_sample]
            baseline_prediction = np.mean(train_predictions)
            
            explanation = {
                'instance_id': i,
                'feature_importance': word_importance,
                'feature_names': feature_names,
                'prediction': prediction,
                'true_label': y_test[i] if i < len(y_test) else None,
                'input': list(range(len(word_importance))),  # Numerical indices for evaluation
                'text_content': text_instance,  # Store original text separately
                'baseline_prediction': baseline_prediction
            }
            explanations.append(explanation)
        
        self.generation_time = time.time() - start_time
        
        return {
            'explanations': explanations,
            'generation_time': self.generation_time,
            'method': 'shap',
            'info': {
                'n_explanations': len(explanations),
                'feature_names': explanations[0]['feature_names'] if explanations else []
            }
        }
    
    def _calculate_text_shapley_values(self, text_instance, X_train, y_train):
        """Calculate Shapley values for words in text using coalition-based approach"""
        # Ensure text_instance is a string
        if not isinstance(text_instance, str):
            # Handle numpy arrays or other data types
            if hasattr(text_instance, 'item'):
                text_instance = str(text_instance.item())
            elif hasattr(text_instance, 'tolist'):
                text_instance = ' '.join(str(x) for x in text_instance.tolist() if x)
            else:
                text_instance = str(text_instance)
        
        words = text_instance.split()
        n_words = len(words)
        
        if n_words == 0:
            return np.array([]), []
        
        # Limit to first 50 words for computational efficiency
        max_words = 50
        if n_words > max_words:
            words = words[:max_words]
            n_words = max_words
        
        # Get baseline prediction (empty text represented by training set average)
        if len(X_train) > 50:
            train_sample = X_train[:50]
        else:
            train_sample = X_train
        
        baseline_predictions = [self.model.predict([text])[0] for text in train_sample]
        baseline_prediction = np.mean(baseline_predictions)
        
        # Get full text prediction
        full_prediction = self.model.predict([text_instance])[0]
        
        # Calculate Shapley values using marginal contributions
        shapley_values = np.zeros(n_words)
        
        # For computational efficiency, use a sampling-based approach
        n_samples = min(100, 2 ** min(n_words, 10))  # Limit sampling
        
        for word_idx in range(n_words):
            marginal_contributions = []
            
            for _ in range(min(20, n_samples)):  # Sample coalitions
                # Generate random coalition (subset of other words)
                coalition_size = np.random.randint(0, n_words)
                coalition = np.random.choice(
                    [idx for idx in range(n_words) if idx != word_idx],
                    size=min(coalition_size, n_words - 1),
                    replace=False
                )
                
                # Coalition without current word
                coalition_without = list(coalition)
                text_without = ' '.join([words[idx] for idx in coalition_without])
                if not text_without.strip():
                    pred_without = baseline_prediction
                else:
                    pred_without = self.model.predict([text_without])[0]
                
                # Coalition with current word
                coalition_with = list(coalition) + [word_idx]
                text_with = ' '.join([words[idx] for idx in sorted(coalition_with)])
                pred_with = self.model.predict([text_with])[0]
                
                # Marginal contribution
                marginal_contribution = pred_with - pred_without
                marginal_contributions.append(marginal_contribution)
            
            # Average marginal contribution is the Shapley value
            shapley_values[word_idx] = np.mean(marginal_contributions)
        
        # Ensure Shapley values sum to the difference from baseline (efficiency property)
        total_effect = full_prediction - baseline_prediction
        current_sum = np.sum(shapley_values)
        if abs(current_sum) > 1e-8:
            shapley_values = shapley_values * (total_effect / current_sum)
        
        # Convert to absolute values for importance ranking
        importance_values = np.abs(shapley_values)
        
        # Create feature names
        feature_names = [f"word_{i}_{word}" for i, word in enumerate(words)]
        
        return importance_values, feature_names


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
            
            # Get model predictions for faithfulness/monotonicity evaluation
            prediction = self.model.predict([instance])[0]
            
            # Create baseline (mean of training data)
            if isinstance(X_train, np.ndarray):
                baseline = np.mean(X_train, axis=0)
                baseline_prediction = self.model.predict([baseline])[0]
            else:
                baseline = None
                baseline_prediction = None
            
            # Handle different data types for input field
            if isinstance(instance, str):
                # For text data, store numerical representation for evaluation
                input_data = list(range(len(importance)))  # Numerical indices for evaluation
                text_content = instance
            else:
                # For tabular data
                input_data = instance.tolist() if hasattr(instance, 'tolist') else instance
                text_content = None
            
            explanation = {
                'instance_id': i,
                'feature_importance': importance,
                'feature_names': feature_names,
                'prediction': prediction,
                'true_label': y_test[i] if i < len(y_test) else None,
                'input': input_data,
                'text_content': text_content,  # Store original text separately
                'baseline_prediction': baseline_prediction
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
            # Enhanced text data implementation with perturbation-based LIME
            return self._generate_text_lime_explanation(instance, X_train, y_train, n_features)
        else:
            # Tabular data - generate perturbed samples around the instance
            n_samples = 100
            perturbed_samples = []
            
            # Determine appropriate noise level based on data range
            data_std = np.std(X_train, axis=0)
            noise_scale = np.maximum(data_std * 0.1, 0.01)  # At least 1% of std or 0.01
            
            for _ in range(n_samples):
                # Add scaled random noise to create perturbed sample
                noise = np.random.normal(0, noise_scale, len(instance))
                perturbed_sample = instance + noise
                
                # For image data, ensure values stay in valid range [0, 1] or [0, 255]
                if np.max(instance) <= 1.0:  # Normalized images
                    perturbed_sample = np.clip(perturbed_sample, 0.0, 1.0)
                elif np.max(instance) <= 255:  # Pixel values
                    perturbed_sample = np.clip(perturbed_sample, 0, 255)
                
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
    
    def _generate_text_lime_explanation(self, text_instance, X_train, y_train, n_features=None):
        """Enhanced LIME implementation for text using perturbation and local linear models"""
        max_words = n_features if n_features else 50
        
        # Ensure text_instance is a string
        if not isinstance(text_instance, str):
            # Handle numpy arrays or other data types
            if hasattr(text_instance, 'item'):
                text_instance = str(text_instance.item())
            elif hasattr(text_instance, 'tolist'):
                text_instance = ' '.join(str(x) for x in text_instance.tolist() if x)
            else:
                text_instance = str(text_instance)
        
        words = text_instance.split()[:max_words]
        
        if not words:
            return np.zeros(max_words)
        
        # Get baseline prediction
        try:
            baseline_prediction = self.model.predict([text_instance])[0]
        except Exception:
            return np.ones(len(words)) / len(words)  # Fallback uniform distribution
        
        # Generate perturbed samples by masking words
        n_samples = 200  # Number of perturbations
        perturbed_texts = []
        presence_matrix = []  # Binary matrix indicating which words are present
        
        for _ in range(n_samples):
            # Randomly select which words to keep (binary mask)
            mask = np.random.binomial(1, 0.5, len(words))  # 50% probability to keep each word
            
            # Create perturbed text
            perturbed_words = [words[i] for i in range(len(words)) if mask[i] == 1]
            perturbed_text = ' '.join(perturbed_words)
            
            # Handle empty text case
            if not perturbed_text.strip():
                perturbed_text = ""  # Empty string
            
            perturbed_texts.append(perturbed_text)
            
            # Pad mask to max_words length for consistency
            padded_mask = np.zeros(max_words)
            padded_mask[:len(mask)] = mask
            presence_matrix.append(padded_mask)
        
        presence_matrix = np.array(presence_matrix)
        
        # Get predictions for perturbed samples
        predictions = []
        for perturbed_text in perturbed_texts:
            try:
                if perturbed_text.strip():
                    pred = self.model.predict([perturbed_text])[0]
                else:
                    # For empty text, use average of training predictions or neutral value
                    if len(X_train) > 50:
                        train_sample = X_train[:50]
                    else:
                        train_sample = X_train
                    
                    train_preds = []
                    for train_text in train_sample:
                        try:
                            train_preds.append(self.model.predict([train_text])[0])
                        except Exception:
                            continue
                    
                    pred = np.mean(train_preds) if train_preds else 0.5
                
                predictions.append(pred)
            except Exception:
                predictions.append(baseline_prediction)  # Fallback to baseline
        
        predictions = np.array(predictions)
        
        # Calculate distance weights (closer perturbations get higher weight)
        original_presence = np.ones(max_words)
        original_presence[len(words):] = 0  # Pad with zeros
        
        distances = []
        for presence in presence_matrix:
            # Cosine similarity between original and perturbed presence vectors
            cosine_sim = np.dot(original_presence, presence) / (
                np.linalg.norm(original_presence) * np.linalg.norm(presence) + 1e-8
            )
            distance = 1 - cosine_sim  # Convert similarity to distance
            distances.append(distance)
        
        distances = np.array(distances)
        
        # Convert distances to weights using exponential kernel
        sigma = 0.25  # Kernel width
        weights = np.exp(-distances**2 / sigma**2)
        
        # Fit weighted linear regression to explain local behavior
        try:
            from sklearn.linear_model import Ridge
            
            # Add small regularization for stability
            linear_model = Ridge(alpha=0.01)
            
            # Fit model with sample weights
            linear_model.fit(presence_matrix, predictions, sample_weight=weights)
            
            # Extract feature importance (coefficients)
            importance = np.abs(linear_model.coef_)
            
            # Ensure we only return importance for actual words
            if len(importance) > len(words):
                importance = importance[:len(words)]
            
            # Pad to max_words if needed
            if len(importance) < max_words:
                padded_importance = np.zeros(max_words)
                padded_importance[:len(importance)] = importance
                importance = padded_importance
            
        except Exception as e:
            self.logger.warning(f"Linear model fitting failed: {e}, using fallback method")
            # Fallback: use correlation between word presence and prediction
            importance = np.zeros(max_words)
            for i in range(min(len(words), max_words)):
                word_presence = presence_matrix[:, i]
                if np.std(word_presence) > 0:  # Only if there's variation
                    correlation = np.corrcoef(word_presence, predictions)[0, 1]
                    importance[i] = abs(correlation) if not np.isnan(correlation) else 0.0
                else:
                    importance[i] = 0.0
        
        # Normalize importance scores
        if np.sum(importance) > 0:
            importance = importance / np.sum(importance)
        else:
            # Fallback to position-based importance
            importance = np.zeros(max_words)
            for i in range(min(len(words), max_words)):
                importance[i] = 1.0 - (i / len(words))  # Earlier words more important
            
            if np.sum(importance) > 0:
                importance = importance / np.sum(importance)
            else:
                importance = np.ones(max_words) / max_words
        
        return importance


class IntegratedGradientsExplainer(BaseExplainer):
    """Integrated Gradients explainer (simplified implementation)"""
    
    supported_data_types = ['tabular', 'text']
    supported_model_types = ['mlp', 'cnn', 'vit']
    
    def explain(self, dataset) -> Dict[str, Any]:
        """Generate Integrated Gradients explanations (only for differentiable models)."""
        import warnings
        import time
        start_time = time.time()
        X_train, X_test, y_train, y_test = dataset.get_data()
        model = self.model
        # Check for scikit-learn MLP or non-differentiable model
        sklearn_mlp = False
        try:
            from sklearn.neural_network import MLPClassifier, MLPRegressor
            if isinstance(model, (MLPClassifier, MLPRegressor)):
                sklearn_mlp = True
        except ImportError:
            pass
        if sklearn_mlp:
            warnings.warn("Integrated Gradients is not supported for scikit-learn MLP. Skipping explanation.")
            self.logger.warning("Integrated Gradients is not supported for scikit-learn MLP. Skipping explanation.")
            return {
                'explanations': [],
                'generation_time': 0.0,
                'method': 'integrated_gradients',
                'info': {
                    'n_explanations': 0,
                    'feature_names': dataset.feature_names if hasattr(dataset, 'feature_names') else []
                }
            }
        # Handle text data
        if isinstance(X_test, list):
            # Text data - use text-specific Integrated Gradients implementation
            return self._explain_text_integrated_gradients(X_test, X_train, y_test, y_train, start_time)
        
        # Optionally: check for PyTorch or TensorFlow model here and implement real IG
        # For now, warn if not implemented
        # If you want a real implementation, use Captum (PyTorch) or tf-explain (TensorFlow)
        explanations = []
        feature_names = dataset.feature_names if hasattr(dataset, 'feature_names') else [f'feature_{i}' for i in range(X_test.shape[1])]
        n_explanations = min(30, len(X_test))
        test_subset = X_test[:n_explanations]
        for i, instance in enumerate(test_subset):
            # Calculate integrated gradients (finite-diff fallback, not recommended)
            importance = self._calculate_integrated_gradients(instance, X_train)
            explanation = {
                'instance_id': i,
                'feature_importance': importance,
                'feature_names': feature_names,
                'prediction': model.predict([instance])[0],
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
    
    def _explain_text_integrated_gradients(self, X_test, X_train, y_test, y_train, start_time) -> Dict[str, Any]:
        """Text-specific Integrated Gradients implementation using word-level gradients"""
        explanations = []
        
        # Generate explanations for a subset of test instances
        n_explanations = min(30, len(X_test))  # Limit for performance
        test_subset = X_test[:n_explanations]
        
        for i, text_instance in enumerate(test_subset):
            # Calculate word-level integrated gradients for text
            word_importance, feature_names = self._calculate_text_integrated_gradients(
                text_instance, X_train, y_train
            )
            
            # Get model prediction
            prediction = self.model.predict([text_instance])[0]
            
            # Calculate baseline prediction (average of training data predictions)
            if len(X_train) > 100:
                # Sample for efficiency
                train_sample = X_train[:100]
            else:
                train_sample = X_train
            
            train_predictions = [self.model.predict([text])[0] for text in train_sample]
            baseline_prediction = np.mean(train_predictions)
            
            explanation = {
                'instance_id': i,
                'feature_importance': word_importance,
                'feature_names': feature_names,
                'prediction': prediction,
                'true_label': y_test[i] if i < len(y_test) else None,
                'input': list(range(len(word_importance))),  # Numerical indices for evaluation
                'text_content': text_instance,  # Store original text separately
                'baseline_prediction': baseline_prediction
            }
            explanations.append(explanation)
        
        self.generation_time = time.time() - start_time
        
        return {
            'explanations': explanations,
            'generation_time': self.generation_time,
            'method': 'integrated_gradients',
            'info': {
                'n_explanations': len(explanations),
                'feature_names': explanations[0]['feature_names'] if explanations else []
            }
        }
    
    def _calculate_text_integrated_gradients(self, text_instance, X_train, y_train):
        """Calculate Integrated Gradients for words in text using path integration"""
        # Ensure text_instance is a string
        if not isinstance(text_instance, str):
            # Handle numpy arrays or other data types
            if hasattr(text_instance, 'item'):
                text_instance = str(text_instance.item())
            elif hasattr(text_instance, 'tolist'):
                text_instance = ' '.join(str(x) for x in text_instance.tolist() if x)
            else:
                text_instance = str(text_instance)
        
        words = text_instance.split()
        n_words = len(words)
        
        if n_words == 0:
            return np.array([]), []
        
        # Limit to first 50 words for computational efficiency
        max_words = 50
        if n_words > max_words:
            words = words[:max_words]
            n_words = max_words
        
        # Get baseline prediction (empty text represented by training set average)
        if len(X_train) > 50:
            train_sample = X_train[:50]
        else:
            train_sample = X_train
        
        baseline_predictions = [self.model.predict([text])[0] for text in train_sample]
        baseline_prediction = np.mean(baseline_predictions)
        
        # Get full text prediction
        full_prediction = self.model.predict([text_instance])[0]
        
        # Calculate integrated gradients using path integration
        integrated_gradients = np.zeros(n_words)
        n_steps = 20  # Number of integration steps
        
        for word_idx in range(n_words):
            gradient_sum = 0
            
            for step in range(1, n_steps + 1):
                alpha = step / n_steps
                
                # Create interpolated text by including words with probability alpha
                interpolated_words = []
                for i, word in enumerate(words):
                    if i == word_idx:
                        # Current word: include with probability alpha
                        if alpha >= 0.5:  # Threshold-based inclusion
                            interpolated_words.append(word)
                    else:
                        # Other words: include all (since we're focusing on one word at a time)
                        interpolated_words.append(word)
                
                interpolated_text = ' '.join(interpolated_words)
                
                # Calculate gradient using finite differences
                epsilon = 0.1  # Step size for perturbation
                
                # Create two versions: with and without the word at current alpha level
                text_with_word = interpolated_text
                text_without_word = ' '.join([word for i, word in enumerate(interpolated_words) if i != word_idx])
                
                if not text_without_word.strip():
                    pred_without = baseline_prediction
                else:
                    pred_without = self.model.predict([text_without_word])[0]
                
                pred_with = self.model.predict([text_with_word])[0]
                
                # Gradient approximation
                gradient = pred_with - pred_without
                gradient_sum += gradient
            
            # Integrated gradient for this word
            # Scale by the difference from baseline
            word_contribution = (full_prediction - baseline_prediction) * (gradient_sum / n_steps)
            integrated_gradients[word_idx] = abs(word_contribution)
        
        # Normalize to ensure completeness property approximately holds
        total_effect = full_prediction - baseline_prediction
        current_sum = np.sum(integrated_gradients)
        if abs(current_sum) > 1e-8 and abs(total_effect) > 1e-8:
            integrated_gradients = integrated_gradients * (abs(total_effect) / current_sum)
        
        # Create feature names
        feature_names = [f"word_{i}_{word}" for i, word in enumerate(words)]
        
        return integrated_gradients, feature_names 