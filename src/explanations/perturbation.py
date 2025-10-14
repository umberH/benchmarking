"""
Perturbation-based explanation methods
"""

import numpy as np
import logging
import time
from typing import Dict, Any, List

from .base_explainer import BaseExplainer


class OcclusionExplainer(BaseExplainer):
    """
    Occlusion-based explainer for images
    Generates saliency maps by occluding patches and measuring prediction changes
    """

    supported_data_types = ['image']
    supported_model_types = ['cnn', 'vit']

    def explain(self, dataset) -> dict:
        """Generate occlusion explanations for image data: occlude each patch and measure prediction change."""
        import numpy as np
        import time
        start_time = time.time()

        X_train, X_test, y_train, y_test = dataset.get_data()
        model = self.model
        explanations = []

        # Use config to determine number of explanations
        max_test_samples = self.config.get('experiment', {}).get('explanation', {}).get('max_test_samples', None)
        n_explanations = len(X_test) if max_test_samples is None else min(max_test_samples, len(X_test))
        n_explanations = min(n_explanations, 50)  # Limit for computational efficiency

        test_subset = X_test[:n_explanations]
        self.logger.info(f"OcclusionExplainer: Generating for {n_explanations} instances")

        for i, instance in enumerate(test_subset):
            try:
                # Ensure instance is a numpy array
                if not isinstance(instance, np.ndarray):
                    instance = np.array(instance)

                # Handle flattened images - reshape to proper dimensions
                if instance.ndim == 1:
                    instance = self._reshape_flattened_image(instance)
                    if instance is None:
                        self.logger.warning(f"Cannot reshape instance {i}, skipping")
                        continue

                # Store original shape for reshaping back
                original_shape = instance.shape

                # Get base prediction
                base_pred = model.predict([instance])[0]

                # Generate occlusion-based importance map
                importance_map = self._generate_occlusion_map(instance, model, base_pred)

                # Flatten importance map for storage
                importance_flat = importance_map.flatten().tolist()

                explanation = {
                    'instance_id': i,
                    'importance_map': importance_map.tolist(),  # 2D or 3D map
                    'feature_importance': importance_flat,  # Flattened for consistency
                    'prediction': float(base_pred),
                    'true_label': float(y_test[i]) if i < len(y_test) else None,
                    'image_shape': list(original_shape)
                }
                explanations.append(explanation)

            except Exception as e:
                self.logger.warning(f"Error processing instance {i}: {e}")
                continue

        generation_time = time.time() - start_time
        self.logger.info(f"OcclusionExplainer: Generated {len(explanations)} explanations in {generation_time:.2f}s")

        return {
            'explanations': explanations,
            'generation_time': generation_time,
            'method': 'occlusion',
            'info': {
                'n_explanations': len(explanations),
                'method_description': 'Occlusion-based saliency map generation'
            }
        }

    def _reshape_flattened_image(self, instance: np.ndarray) -> np.ndarray:
        """
        Reshape flattened image to proper dimensions

        Args:
            instance: Flattened image array

        Returns:
            Reshaped image or None if cannot reshape
        """
        length = len(instance)

        # Try square grayscale
        side_len = int(np.sqrt(length))
        if side_len * side_len == length:
            return instance.reshape(side_len, side_len)

        # Try common image sizes
        if length == 784:  # 28x28 (MNIST, Fashion-MNIST)
            return instance.reshape(28, 28)
        elif length == 3072:  # 32x32x3 (CIFAR-10)
            return instance.reshape(32, 32, 3)
        elif length == 12288:  # 64x64x3
            return instance.reshape(64, 64, 3)
        elif length == 150528:  # 224x224x3 (ImageNet)
            return instance.reshape(224, 224, 3)

        return None

    def _generate_occlusion_map(self, instance: np.ndarray, model, base_pred: float,
                                 patch_size: int = None) -> np.ndarray:
        """
        Generate importance map by occluding patches

        Args:
            instance: Image instance (2D or 3D)
            model: Model to use for predictions
            base_pred: Baseline prediction
            patch_size: Size of occlusion patches (auto-determined if None)

        Returns:
            Importance map same shape as instance (or 2D if instance is 3D)
        """
        # Determine image dimensions
        if instance.ndim == 2:
            h, w = instance.shape
            c = 1
        elif instance.ndim == 3:
            h, w, c = instance.shape
        else:
            raise ValueError(f"Unsupported image shape: {instance.shape}")

        # Auto-determine patch size if not provided
        if patch_size is None:
            patch_size = max(2, min(h, w) // 8)

        # Initialize importance map
        importance_map = np.zeros((h, w))

        # Occlude patches and measure prediction change
        for y in range(0, h, patch_size):
            for x in range(0, w, patch_size):
                try:
                    # Create occluded version
                    occluded = instance.copy()
                    y_end = min(y + patch_size, h)
                    x_end = min(x + patch_size, w)

                    # Occlude patch (set to zero or mean)
                    if instance.ndim == 3:
                        occluded[y:y_end, x:x_end, :] = 0
                    else:
                        occluded[y:y_end, x:x_end] = 0

                    # Get prediction on occluded image
                    occluded_pred = model.predict([occluded])[0]

                    # Calculate importance as prediction change
                    importance = abs(base_pred - occluded_pred)

                    # Assign importance to patch
                    importance_map[y:y_end, x:x_end] = importance

                except Exception as e:
                    self.logger.warning(f"Error occluding patch at ({y}, {x}): {e}")
                    continue

        return importance_map


class FeatureAblationExplainer(BaseExplainer):
    """Feature ablation explainer (placeholder)"""
    
    supported_data_types = ['tabular']
    supported_model_types = ['all']
    
    def explain(self, dataset) -> dict:
        """Generate feature ablation explanations for tabular data: ablate each feature and measure prediction change."""
        import numpy as np
        import time
        start_time = time.time()
        X_train, X_test, y_train, y_test = dataset.get_data()
        model = self.model
        explanations = []
        # Get max test samples from config (None means use full test set)
        max_test_samples = self.config.get('experiment', {}).get('explanation', {}).get('max_test_samples', None)
        n_explanations = len(X_test) if max_test_samples is None else min(max_test_samples, len(X_test))
        test_subset = X_test[:n_explanations]
        print(f"[DEBUG] FeatureAblationExplainer: Generating for {n_explanations} instances.")
        for i, instance in enumerate(test_subset):
            # Ensure instance is a numpy array
            if not isinstance(instance, np.ndarray):
                instance = np.array(instance)
            
            try:
                base_pred = model.predict([instance])[0]
            except Exception as e:
                print(f"[DEBUG] FeatureAblationExplainer: Error predicting instance {i}: {e}")
                continue
                
            feature_importance = []
            for j in range(instance.shape[0]):
                ablated = instance.copy()
                ablated[j] = 0  # Zero out feature j
                try:
                    ablated_pred = model.predict([ablated])[0]
                    importance = abs(base_pred - ablated_pred)
                    feature_importance.append(importance)
                except Exception:
                    feature_importance.append(0.0)  # Default if prediction fails
            explanations.append({
                'instance_id': i,
                'feature_importance': feature_importance,
                'prediction': float(base_pred),
                'true_label': float(y_test[i]) if i < len(y_test) else None
            })
        generation_time = time.time() - start_time
        print(f"[DEBUG] FeatureAblationExplainer: Done. Generated {len(explanations)} explanations in {generation_time:.2f}s.")
        return {
            'explanations': explanations,
            'generation_time': generation_time,
            'method': 'feature_ablation',
            'info': {'n_explanations': len(explanations)}
        }


class TextOcclusionExplainer(BaseExplainer):
    """Text Occlusion explainer - occlude words/phrases and measure prediction change"""
    
    supported_data_types = ['text']
    supported_model_types = ['all']
    
    def explain(self, dataset) -> dict:
        """Generate text occlusion explanations by masking words and measuring prediction change"""
        start_time = time.time()
        X_train, X_test, y_train, y_test = dataset.get_data()
        model = self.model
        explanations = []
        # Get max test samples from config (None means use full test set)
        max_test_samples = self.config.get('experiment', {}).get('explanation', {}).get('max_test_samples', None)
        n_explanations = len(X_test) if max_test_samples is None else min(max_test_samples, len(X_test))
        test_subset = X_test[:n_explanations]
        
        self.logger.info(f"TextOcclusionExplainer: Generating for {n_explanations} instances.")
        
        for i, text_instance in enumerate(test_subset):
            if not isinstance(text_instance, str):
                continue
                
            try:
                # Get baseline prediction
                base_pred = model.predict([text_instance])[0]
                
                # Calculate word-level importance using occlusion
                word_importance, feature_names = self._calculate_word_occlusion_importance(
                    text_instance, model
                )
                
                explanation = {
                    'instance_id': i,
                    'feature_importance': word_importance,
                    'feature_names': feature_names,
                    'prediction': float(base_pred),
                    'true_label': float(y_test[i]) if i < len(y_test) else None,
                    'input': list(range(len(word_importance))),  # Numerical indices for evaluation
                    'text_content': text_instance  # Store original text separately
                }
                explanations.append(explanation)
                
            except Exception as e:
                self.logger.error(f"Error processing text instance {i}: {e}")
                continue
        
        generation_time = time.time() - start_time
        self.logger.info(f"TextOcclusionExplainer: Generated {len(explanations)} explanations in {generation_time:.2f}s")
        
        return {
            'explanations': explanations,
            'generation_time': generation_time,
            'method': 'text_occlusion',
            'info': {'n_explanations': len(explanations)}
        }
    
    def _calculate_word_occlusion_importance(self, text_instance, model):
        """Calculate importance of each word by occluding it and measuring prediction change"""
        words = text_instance.split()
        n_words = len(words)
        
        if n_words == 0:
            return np.array([]), []
        
        # Limit to first 100 words for computational efficiency
        max_words = 100
        if n_words > max_words:
            words = words[:max_words]
            n_words = max_words
        
        # Get baseline prediction
        try:
            baseline_prediction = model.predict([text_instance])[0]
        except Exception as e:
            self.logger.error(f"Error getting baseline prediction: {e}")
            return np.zeros(n_words), [f"word_{i}" for i in range(n_words)]
        
        # Calculate importance by occluding each word
        word_importance = []
        
        for i in range(n_words):
            # Create occluded text by removing word i
            occluded_words = words[:i] + words[i+1:]
            occluded_text = ' '.join(occluded_words)
            
            try:
                # Handle empty text case
                if not occluded_text.strip():
                    # For empty text, use a neutral placeholder or average prediction
                    occluded_prediction = 0.5  # Neutral prediction
                else:
                    occluded_prediction = model.predict([occluded_text])[0]
                
                # Importance is the change in prediction when word is removed
                importance = abs(baseline_prediction - occluded_prediction)
                word_importance.append(importance)
                
            except Exception as e:
                self.logger.warning(f"Error occluding word {i} ('{words[i]}'): {e}")
                word_importance.append(0.0)  # Default importance if prediction fails
        
        # Create feature names
        feature_names = [f"word_{i}_{word}" for i, word in enumerate(words)]
        
        return np.array(word_importance), feature_names