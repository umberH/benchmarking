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
        model = self.model
        explanations = []
        n_explanations = min(50, len(X_test))
        test_subset = X_test[:n_explanations]
        y_pred = model.predict(test_subset)
        print(f"[DEBUG] PrototypeExplainer: Generating for {n_explanations} instances.")
        for i, (instance, pred) in enumerate(zip(test_subset, y_pred)):
            # Find nearest instance in train set with same class
            same_idx = np.where(y_train == pred)[0]
            if len(same_idx) == 0:
                print(f"[DEBUG] No same class found for instance {i}.")
                proto_instance = None
                proto_distance = None
            else:
                diffs = X_train[same_idx] - instance
                dists = np.linalg.norm(diffs, axis=1)
                min_idx = np.argmin(dists)
                proto_instance = X_train[same_idx][min_idx]
                proto_distance = dists[min_idx]
            explanations.append({
                'instance_id': i,
                'original': instance.tolist() if hasattr(instance, 'tolist') else list(instance),
                'prediction': int(pred),
                'prototype': proto_instance.tolist() if proto_instance is not None and hasattr(proto_instance, 'tolist') else (list(proto_instance) if proto_instance is not None else None),
                'proto_distance': float(proto_distance) if proto_distance is not None else None,
                'true_label': int(y_test[i]) if i < len(y_test) else None
            })
        generation_time = time.time() - start_time
        print(f"[DEBUG] PrototypeExplainer: Done. Generated {len(explanations)} explanations in {generation_time:.2f}s.")
        return {
            'explanations': explanations,
            'generation_time': generation_time,
            'method': 'prototype',
            'info': {'n_explanations': len(explanations)}
        }


class CounterfactualExplainer(BaseExplainer):
    """Counterfactual explainer (placeholder)"""
    
    supported_data_types = ['tabular']
    supported_model_types = ['all']
    
    def explain(self, dataset) -> dict:
        """Generate counterfactual explanations for tabular data using nearest neighbor from opposite class."""
        import numpy as np
        import time
        start_time = time.time()
        X_train, X_test, y_train, y_test = dataset.get_data()
        model = self.model
        explanations = []
        n_explanations = min(50, len(X_test))
        test_subset = X_test[:n_explanations]
        y_pred = model.predict(test_subset)
        print(f"[DEBUG] CounterfactualExplainer: Generating for {n_explanations} instances.")
        for i, (instance, pred) in enumerate(zip(test_subset, y_pred)):
            # Find nearest instance in train set with opposite class
            opposite_idx = np.where(y_train != pred)[0]
            if len(opposite_idx) == 0:
                print(f"[DEBUG] No opposite class found for instance {i}.")
                cf_instance = None
                cf_distance = None
            else:
                diffs = X_train[opposite_idx] - instance
                dists = np.linalg.norm(diffs, axis=1)
                min_idx = np.argmin(dists)
                cf_instance = X_train[opposite_idx][min_idx]
                cf_distance = dists[min_idx]
            explanations.append({
                'instance_id': i,
                'original': instance.tolist() if hasattr(instance, 'tolist') else list(instance),
                'prediction': int(pred),
                'counterfactual': cf_instance.tolist() if cf_instance is not None and hasattr(cf_instance, 'tolist') else (list(cf_instance) if cf_instance is not None else None),
                'cf_distance': float(cf_distance) if cf_distance is not None else None,
                'true_label': int(y_test[i]) if i < len(y_test) else None
            })
        generation_time = time.time() - start_time
        print(f"[DEBUG] CounterfactualExplainer: Done. Generated {len(explanations)} explanations in {generation_time:.2f}s.")
        return {
            'explanations': explanations,
            'generation_time': generation_time,
            'method': 'counterfactual',
            'info': {'n_explanations': len(explanations)}
        }