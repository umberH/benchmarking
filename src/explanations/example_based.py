"""
Example-based explanation methods
"""

from .base_explainer import BaseExplainer


class PrototypeExplainer(BaseExplainer):
    """Prototype-based explainer (placeholder)"""
    
    supported_data_types = ['tabular', 'image']
    supported_model_types = ['all']
    
    def explain(self, dataset) -> dict:
        """Generate prototype explanations for tabular data: nearest neighbor of same class."""
        import numpy as np
        import time
        start_time = time.time()
        X_train, X_test, y_train, y_test = dataset.get_data()
        
        # Ensure data is in numpy array format
        if not isinstance(X_train, np.ndarray):
            X_train = np.array(X_train)
        if not isinstance(X_test, np.ndarray):
            X_test = np.array(X_test)
        if not isinstance(y_train, np.ndarray):
            y_train = np.array(y_train)
        if not isinstance(y_test, np.ndarray):
            y_test = np.array(y_test)
        
        # For high-dimensional data (images), flatten for distance calculations
        if X_train.ndim > 2:
            original_train_shape = X_train.shape
            original_test_shape = X_test.shape
            X_train = X_train.reshape(X_train.shape[0], -1)
            X_test = X_test.reshape(X_test.shape[0], -1)
        
        model = self.model
        explanations = []
        # Use more comprehensive sampling strategy
        # Get max_test_samples from config (None means use full test set)
        max_test_samples = self.config.get('experiment', {}).get('explanation', {}).get('max_test_samples', None)
        n_explanations = len(X_test) if max_test_samples is None else min(max_test_samples, len(X_test))
        # Stratified sampling to ensure all classes represented
        unique_classes = np.unique(y_test)
        samples_per_class = n_explanations // len(unique_classes)
        
        selected_indices = []
        for cls in unique_classes:
            cls_indices = np.where(y_test == cls)[0]
            if len(cls_indices) > 0:
                n_samples = min(samples_per_class, len(cls_indices))
                selected_indices.extend(np.random.choice(cls_indices, n_samples, replace=False))
        
        # Fill remaining slots if needed
        remaining_slots = n_explanations - len(selected_indices)
        if remaining_slots > 0:
            all_remaining = [i for i in range(len(X_test)) if i not in selected_indices]
            if len(all_remaining) > 0:
                additional = np.random.choice(all_remaining, 
                                           min(remaining_slots, len(all_remaining)), 
                                           replace=False)
                selected_indices.extend(additional)
        
        n_explanations = len(selected_indices)
        test_subset = X_test[selected_indices]
        
        # Predict on test subset - handle different model input formats
        try:
            y_pred = model.predict(test_subset)
        except Exception as e:
            print(f"[DEBUG] Error in model prediction: {e}")
            # Fallback: try predicting one by one
            y_pred = []
            for instance in test_subset:
                try:
                    # Ensure instance is properly shaped for prediction
                    if isinstance(instance, np.ndarray):
                        instance_input = instance.reshape(1, *instance.shape)
                    else:
                        instance_input = np.array([instance])
                    pred = model.predict(instance_input)[0]
                    y_pred.append(pred)
                except Exception:
                    y_pred.append(0)  # Default prediction
            y_pred = np.array(y_pred)
        
        print(f"[DEBUG] PrototypeExplainer: Generating for {n_explanations} instances (stratified sampling).")
        for i, (instance, pred) in enumerate(zip(test_subset, y_pred)):
            original_idx = selected_indices[i]  # Get original index for proper labeling
            # Find nearest instance in train set with same class
            same_idx = np.where(y_train == pred)[0]
            if len(same_idx) == 0:
                print(f"[DEBUG] No same class found for instance {original_idx}.")
                proto_instance = None
                proto_distance = None
            else:
                diffs = X_train[same_idx] - instance
                dists = np.linalg.norm(diffs, axis=1)
                min_idx = np.argmin(dists)
                proto_instance = X_train[same_idx][min_idx]
                proto_distance = dists[min_idx]
            # Robust conversion of proto_distance to float
            if proto_distance is not None:
                if isinstance(proto_distance, (np.ndarray, list)):
                    # If it's a length-1 array, extract the scalar
                    if hasattr(proto_distance, 'shape') and proto_distance.shape == ():
                        pdist = float(proto_distance)
                    elif hasattr(proto_distance, '__len__') and len(proto_distance) == 1:
                        pdist = float(proto_distance[0])
                    else:
                        pdist = float(np.asarray(proto_distance).item()) if np.asarray(proto_distance).size == 1 else float(np.asarray(proto_distance).flatten()[0])
                else:
                    pdist = float(proto_distance)
            else:
                pdist = None
            # Calculate feature importance as absolute difference from prototype
            if proto_instance is not None:
                feature_importance = np.abs(instance - proto_instance).tolist()
            else:
                # If no prototype found, use zeros
                feature_importance = np.zeros_like(instance).tolist()

            explanations.append({
                'instance_id': original_idx,
                'original': instance.tolist() if hasattr(instance, 'tolist') else list(instance),
                'prediction': int(pred),
                'prototype': proto_instance.tolist() if proto_instance is not None and hasattr(proto_instance, 'tolist') else (list(proto_instance) if proto_instance is not None else None),
                'proto_distance': pdist,
                'feature_importance': feature_importance,
                'true_label': int(y_test[original_idx]) if original_idx < len(y_test) else None,
                'confidence': model.predict_proba(instance.reshape(1, *instance.shape))[0].max() if hasattr(model, 'predict_proba') else None
            })
        generation_time = time.time() - start_time
        print(f"[DEBUG] PrototypeExplainer: Done. Generated {len(explanations)} explanations in {generation_time:.2f}s.")
        
        # Add dataset-level summary statistics
        proto_distances = [exp['proto_distance'] for exp in explanations if exp['proto_distance'] is not None]

        # Debug: Check for None true_labels
        none_labels = sum(1 for exp in explanations if exp['true_label'] is None)
        if none_labels > 0:
            print(f"[DEBUG] WARNING: {none_labels}/{len(explanations)} explanations have None true_label")
            print(f"[DEBUG] y_test length: {len(y_test)}, max selected_indices: {max(selected_indices) if selected_indices else 'N/A'}")

        accuracy = sum(1 for exp in explanations if exp['true_label'] is not None and exp['prediction'] == exp['true_label']) / len(explanations) if explanations else 0.0
        class_coverage = len(set(exp['prediction'] for exp in explanations))
        
        return {
            'explanations': explanations,
            'generation_time': generation_time,
            'method': 'prototype',
            'info': {
                'n_explanations': len(explanations),
                'accuracy': accuracy,
                'class_coverage': class_coverage,
                'avg_proto_distance': np.mean(proto_distances) if proto_distances else 0.0,
                'std_proto_distance': np.std(proto_distances) if proto_distances else 0.0,
                'total_classes': len(np.unique(y_test)),
                'sampling_strategy': 'stratified'
            }
        }


class CounterfactualExplainer(BaseExplainer):
    """Counterfactual explainer for tabular, image, and text data"""

    supported_data_types = ['tabular', 'image', 'text']
    supported_model_types = ['all']
    
    def explain(self, dataset) -> dict:
        """Generate counterfactual explanations using nearest neighbor from opposite class."""
        import numpy as np
        import time
        start_time = time.time()
        X_train, X_test, y_train, y_test = dataset.get_data()

        # Detect if this is text data
        is_text = isinstance(X_test, list) and (len(X_test) == 0 or isinstance(X_test[0], str))

        if is_text:
            # For text data, use TF-IDF vectorization for distance calculations
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_distances

            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            X_train_vec = vectorizer.fit_transform(X_train).toarray()
            X_test_vec = vectorizer.transform(X_test).toarray()
        else:
            # Ensure data is in numpy array format
            if not isinstance(X_train, np.ndarray):
                X_train = np.array(X_train)
            if not isinstance(X_test, np.ndarray):
                X_test = np.array(X_test)

            # For high-dimensional data (images), flatten for distance calculations
            if X_train.ndim > 2:
                X_train = X_train.reshape(X_train.shape[0], -1)
                X_test = X_test.reshape(X_test.shape[0], -1)

            X_train_vec = X_train
            X_test_vec = X_test

        # Ensure labels are numpy arrays
        if not isinstance(y_train, np.ndarray):
            y_train = np.array(y_train)
        if not isinstance(y_test, np.ndarray):
            y_test = np.array(y_test)
        
        model = self.model
        explanations = []
        # Use more comprehensive sampling strategy
        # Get max_test_samples from config (None means use full test set)
        max_test_samples = self.config.get('experiment', {}).get('explanation', {}).get('max_test_samples', None)
        n_explanations = len(X_test) if max_test_samples is None else min(max_test_samples, len(X_test))
        # Stratified sampling to ensure all classes represented
        unique_classes = np.unique(y_test)
        samples_per_class = n_explanations // len(unique_classes)
        
        selected_indices = []
        for cls in unique_classes:
            cls_indices = np.where(y_test == cls)[0]
            if len(cls_indices) > 0:
                n_samples = min(samples_per_class, len(cls_indices))
                selected_indices.extend(np.random.choice(cls_indices, n_samples, replace=False))
        
        # Fill remaining slots if needed
        remaining_slots = n_explanations - len(selected_indices)
        if remaining_slots > 0:
            all_remaining = [i for i in range(len(X_test)) if i not in selected_indices]
            if len(all_remaining) > 0:
                additional = np.random.choice(all_remaining, 
                                           min(remaining_slots, len(all_remaining)), 
                                           replace=False)
                selected_indices.extend(additional)
        
        n_explanations = len(selected_indices)
        test_subset = X_test[selected_indices]
        
        # Predict on test subset - handle different model input formats
        try:
            y_pred = model.predict(test_subset)
        except Exception as e:
            print(f"[DEBUG] Error in model prediction: {e}")
            # Fallback: try predicting one by one
            y_pred = []
            for instance in test_subset:
                try:
                    # Ensure instance is properly shaped for prediction
                    if isinstance(instance, np.ndarray):
                        instance_input = instance.reshape(1, *instance.shape)
                    else:
                        instance_input = np.array([instance])
                    pred = model.predict(instance_input)[0]
                    y_pred.append(pred)
                except Exception:
                    y_pred.append(0)  # Default prediction
            y_pred = np.array(y_pred)
        
        # Get test subset - handle list (text) vs array indexing
        if is_text:
            test_subset = [X_test[i] for i in selected_indices]
            test_subset_vec = X_test_vec[selected_indices]
        else:
            test_subset = X_test[selected_indices]
            test_subset_vec = X_test_vec[selected_indices]

        print(f"[DEBUG] CounterfactualExplainer: Generating for {n_explanations} instances (stratified sampling).")
        for i, (instance, instance_vec, pred) in enumerate(zip(test_subset, test_subset_vec, y_pred)):
            original_idx = selected_indices[i]  # Get original index for proper labeling
            # Find nearest instance in train set with opposite class
            opposite_idx = np.where(y_train != pred)[0]
            if len(opposite_idx) == 0:
                print(f"[DEBUG] No opposite class found for instance {original_idx}.")
                cf_instance = None
                cf_distance = None
            else:
                # Use vectorized representations for distance calculation
                diffs = X_train_vec[opposite_idx] - instance_vec
                dists = np.linalg.norm(diffs, axis=1)
                min_idx = np.argmin(dists)
                # Get the original instance (text or array)
                cf_instance = X_train[opposite_idx[min_idx]] if is_text else X_train[opposite_idx][min_idx]
                cf_distance = dists[min_idx]

            # Calculate feature importance as absolute difference from counterfactual
            if cf_instance is not None:
                if is_text:
                    # For text, use zeros (text has separate handling in evaluator)
                    feature_importance = []
                else:
                    # For tabular/image data
                    cf_vec = X_train_vec[opposite_idx][min_idx] if 'min_idx' in locals() else cf_instance
                    feature_importance = np.abs(instance_vec - cf_vec).tolist()
            else:
                # If no counterfactual found, use zeros
                if is_text:
                    feature_importance = []
                else:
                    feature_importance = np.zeros_like(instance_vec).tolist()

            # Store the original format (text string or array)
            explanations.append({
                'instance_id': original_idx,
                'original': instance if is_text else (instance.tolist() if hasattr(instance, 'tolist') else list(instance)),
                'prediction': int(pred),
                'counterfactual': cf_instance if is_text else (cf_instance.tolist() if cf_instance is not None and hasattr(cf_instance, 'tolist') else (list(cf_instance) if cf_instance is not None else None)),
                'cf_distance': float(cf_distance) if cf_distance is not None else None,
                'feature_importance': feature_importance,
                'true_label': int(y_test[original_idx]) if original_idx < len(y_test) else None,
                'confidence': model.predict_proba([instance])[0].max() if is_text and hasattr(model, 'predict_proba') else (model.predict_proba(instance.reshape(1, *instance.shape))[0].max() if hasattr(model, 'predict_proba') else None)
            })
        generation_time = time.time() - start_time
        print(f"[DEBUG] CounterfactualExplainer: Done. Generated {len(explanations)} explanations in {generation_time:.2f}s.")
        
        # Add dataset-level summary statistics
        cf_distances = [exp['cf_distance'] for exp in explanations if exp['cf_distance'] is not None]

        # Debug: Check for None true_labels
        none_labels = sum(1 for exp in explanations if exp['true_label'] is None)
        if none_labels > 0:
            print(f"[DEBUG] WARNING: {none_labels}/{len(explanations)} explanations have None true_label")
            print(f"[DEBUG] y_test length: {len(y_test)}, max selected_indices: {max(selected_indices) if selected_indices else 'N/A'}")

        accuracy = sum(1 for exp in explanations if exp['true_label'] is not None and exp['prediction'] == exp['true_label']) / len(explanations) if explanations else 0.0
        class_coverage = len(set(exp['prediction'] for exp in explanations))
        
        return {
            'explanations': explanations,
            'generation_time': generation_time,
            'method': 'counterfactual',
            'info': {
                'n_explanations': len(explanations),
                'accuracy': accuracy,
                'class_coverage': class_coverage,
                'avg_cf_distance': np.mean(cf_distances) if cf_distances else 0.0,
                'std_cf_distance': np.std(cf_distances) if cf_distances else 0.0,
                'total_classes': len(np.unique(y_test)),
                'sampling_strategy': 'stratified'
            }
        }