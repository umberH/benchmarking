"""
Concept-based explanation methods
"""

import numpy as np
import time
import logging
from typing import Dict, Any, List, Tuple
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans

from .base_explainer import BaseExplainer


class TCAVExplainer(BaseExplainer):
    """
    TCAV (Testing with Concept Activation Vectors) explainer for images
    Generates concept-based explanations by identifying directional derivatives in activation space
    """

    supported_data_types = ['image']
    supported_model_types = ['cnn', 'vit']

    def explain(self, dataset) -> dict:
        """Generate TCAV explanations for image data"""
        start_time = time.time()

        # Get data
        X_train, X_test, y_train, y_test = dataset.get_data()

        # Use config to determine number of explanations
        max_test_samples = self.config.get('experiment', {}).get('explanation', {}).get('max_test_samples', None)
        n_explanations = len(X_test) if max_test_samples is None else min(max_test_samples, len(X_test))
        # Limit for computational efficiency to prevent hanging
        max_tcav_limit = 200
        n_explanations = min(n_explanations, max_tcav_limit)

        test_subset = X_test[:n_explanations]

        self.logger.info(f"TCAV: Generating explanations for {n_explanations} instances (limited to {max_tcav_limit} for efficiency)")

        # Generate concept vectors by clustering training data
        concept_vectors = self._generate_concept_vectors(X_train, y_train)

        explanations = []

        for i, instance in enumerate(test_subset):
            try:
                # Ensure instance is numpy array
                if not isinstance(instance, np.ndarray):
                    instance = np.array(instance)

                # Flatten for processing
                instance_flat = instance.flatten()

                # Calculate TCAV scores for each concept
                tcav_scores = self._calculate_tcav_scores(instance_flat, concept_vectors)

                # Get model prediction
                prediction = self.model.predict([instance])[0]

                # Create concept names
                concept_names = [f"concept_{j}" for j in range(len(concept_vectors))]

                # Calculate feature importance as weighted sum of concept contributions
                # Each concept vector's contribution weighted by its TCAV score
                feature_importance = np.zeros_like(instance_flat)
                for j, (score, concept_vec) in enumerate(zip(tcav_scores, concept_vectors)):
                    # Weight each concept vector by its TCAV score
                    feature_importance += abs(score) * np.abs(concept_vec)

                # Normalize to avoid extremely large values
                if np.max(feature_importance) > 0:
                    feature_importance = feature_importance / np.max(feature_importance)

                explanation = {
                    'instance_id': i,
                    'tcav_scores': tcav_scores.tolist(),
                    'concept_names': concept_names,
                    'feature_importance': feature_importance.tolist(),
                    'prediction': float(prediction),
                    'true_label': float(y_test[i]) if i < len(y_test) else None,
                    'n_concepts': len(concept_vectors)
                }
                explanations.append(explanation)

            except Exception as e:
                self.logger.warning(f"Error processing instance {i}: {e}")
                continue

        generation_time = time.time() - start_time
        self.logger.info(f"TCAV: Generated {len(explanations)} explanations in {generation_time:.2f}s")

        return {
            'explanations': explanations,
            'generation_time': generation_time,
            'method': 'tcav',
            'info': {
                'n_explanations': len(explanations),
                'n_concepts': len(concept_vectors),
                'concept_names': [f"concept_{j}" for j in range(len(concept_vectors))]
            }
        }

    def _generate_concept_vectors(self, X_train: np.ndarray, y_train: np.ndarray, n_concepts: int = 5) -> List[np.ndarray]:
        """
        Generate concept vectors from training data using clustering

        Args:
            X_train: Training images
            y_train: Training labels
            n_concepts: Number of concepts to generate

        Returns:
            List of concept vectors
        """
        # Sample training data for efficiency
        n_samples = min(1000, len(X_train))
        indices = np.random.choice(len(X_train), n_samples, replace=False)
        X_sample = X_train[indices]

        # Flatten images
        X_flat = X_sample.reshape(len(X_sample), -1)

        # Reduce dimensionality if needed
        if X_flat.shape[1] > 100:
            pca = PCA(n_components=100)
            X_reduced = pca.fit_transform(X_flat)
        else:
            X_reduced = X_flat

        # Cluster to find concepts
        kmeans = KMeans(n_clusters=n_concepts, random_state=42, n_init=10)
        kmeans.fit(X_reduced)

        # Concept vectors are cluster centers
        if X_flat.shape[1] > 100:
            # Transform back to original space
            concept_vectors = pca.inverse_transform(kmeans.cluster_centers_)
        else:
            concept_vectors = kmeans.cluster_centers_

        return [vec for vec in concept_vectors]

    def _calculate_tcav_scores(self, instance: np.ndarray, concept_vectors: List[np.ndarray]) -> np.ndarray:
        """
        Calculate TCAV scores (directional derivatives) for an instance

        Args:
            instance: Flattened image instance
            concept_vectors: List of concept vectors

        Returns:
            Array of TCAV scores
        """
        scores = []

        for concept_vec in concept_vectors:
            # Normalize vectors
            instance_norm = instance / (np.linalg.norm(instance) + 1e-10)
            concept_norm = concept_vec / (np.linalg.norm(concept_vec) + 1e-10)

            # Calculate directional derivative (cosine similarity)
            score = np.dot(instance_norm, concept_norm)
            scores.append(score)

        return np.array(scores)


class ConceptBottleneckExplainer(BaseExplainer):
    """
    Concept Bottleneck Model explainer for images
    Explains predictions through interpretable concept activations
    """

    supported_data_types = ['image']
    supported_model_types = ['cnn', 'vit']

    def explain(self, dataset) -> dict:
        """Generate concept bottleneck explanations for image data"""
        start_time = time.time()

        # Get data
        X_train, X_test, y_train, y_test = dataset.get_data()

        # Use config to determine number of explanations
        max_test_samples = self.config.get('experiment', {}).get('explanation', {}).get('max_test_samples', None)
        n_explanations = len(X_test) if max_test_samples is None else min(max_test_samples, len(X_test))
        # Limit for computational efficiency to prevent hanging
        max_cb_limit = 200
        n_explanations = min(n_explanations, max_cb_limit)

        test_subset = X_test[:n_explanations]

        self.logger.info(f"ConceptBottleneck: Generating explanations for {n_explanations} instances (limited to {max_cb_limit} for efficiency)")

        # Learn concept representations
        concept_model, concept_names = self._learn_concept_bottleneck(X_train, y_train)

        explanations = []

        for i, instance in enumerate(test_subset):
            try:
                # Ensure instance is numpy array
                if not isinstance(instance, np.ndarray):
                    instance = np.array(instance)

                # Flatten for processing
                instance_flat = instance.flatten().reshape(1, -1)

                # Apply PCA transformation if it was used during training
                if hasattr(self, 'pca') and self.pca is not None:
                    instance_transformed = self.pca.transform(instance_flat)
                else:
                    instance_transformed = instance_flat

                # Get concept activations
                concept_activations = concept_model.predict_proba(instance_transformed)[0]

                # Get model prediction
                prediction = self.model.predict([instance])[0]

                # Calculate feature importance from concept model coefficients
                # Use the logistic regression coefficients weighted by concept activations
                if hasattr(concept_model, 'coef_'):
                    # Get coefficients for the predicted class
                    pred_class_idx = int(prediction) if prediction < len(concept_model.classes_) else 0
                    if concept_model.coef_.ndim > 1:
                        class_coef = concept_model.coef_[pred_class_idx]
                    else:
                        class_coef = concept_model.coef_

                    # Map back to original feature space
                    if hasattr(self, 'pca') and self.pca is not None:
                        # Transform PCA components back to original space
                        feature_importance_full = np.dot(class_coef, self.pca.components_)
                    else:
                        feature_importance_full = class_coef

                    # Take absolute values and normalize
                    feature_importance = np.abs(feature_importance_full)
                    if np.max(feature_importance) > 0:
                        feature_importance = feature_importance / np.max(feature_importance)
                else:
                    # Fallback: use concept activations as proxy
                    feature_importance = concept_activations

                explanation = {
                    'instance_id': i,
                    'concept_activations': concept_activations.tolist(),
                    'concept_names': concept_names,
                    'feature_importance': feature_importance.tolist() if isinstance(feature_importance, np.ndarray) else list(feature_importance),
                    'prediction': float(prediction),
                    'true_label': float(y_test[i]) if i < len(y_test) else None,
                    'n_concepts': len(concept_names)
                }
                explanations.append(explanation)

            except Exception as e:
                self.logger.warning(f"Error processing instance {i}: {e}")
                continue

        generation_time = time.time() - start_time
        self.logger.info(f"ConceptBottleneck: Generated {len(explanations)} explanations in {generation_time:.2f}s")

        return {
            'explanations': explanations,
            'generation_time': generation_time,
            'method': 'concept_bottleneck',
            'info': {
                'n_explanations': len(explanations),
                'n_concepts': len(concept_names),
                'concept_names': concept_names
            }
        }

    def _learn_concept_bottleneck(self, X_train: np.ndarray, y_train: np.ndarray, n_concepts: int = 5) -> Tuple[Any, List[str]]:
        """
        Learn concept bottleneck model from training data

        Args:
            X_train: Training images
            y_train: Training labels
            n_concepts: Number of concepts

        Returns:
            Tuple of (concept_model, concept_names)
        """
        # Sample training data for efficiency
        n_samples = min(1000, len(X_train))
        indices = np.random.choice(len(X_train), n_samples, replace=False)
        X_sample = X_train[indices]
        y_sample = y_train[indices]

        # Flatten images
        X_flat = X_sample.reshape(len(X_sample), -1)

        # Reduce dimensionality if needed
        if X_flat.shape[1] > 100:
            self.pca = PCA(n_components=100)
            X_reduced = self.pca.fit_transform(X_flat)
        else:
            self.pca = None
            X_reduced = X_flat

        # Train a simple logistic regression as concept predictor
        concept_model = LogisticRegression(max_iter=1000, random_state=42)
        concept_model.fit(X_reduced, y_sample)

        # Generate concept names based on classes
        unique_labels = np.unique(y_sample)
        concept_names = [f"concept_class_{int(label)}" for label in unique_labels]

        # Store for later use
        self.concept_model_trained = concept_model

        return concept_model, concept_names 