"""
Perturbation-based explanation methods
"""

from .base_explainer import BaseExplainer


class OcclusionExplainer(BaseExplainer):
    """Occlusion explainer (placeholder)"""
    
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
        n_explanations = min(10, len(X_test))  # Images are expensive
        test_subset = X_test[:n_explanations]
        print(f"[DEBUG] OcclusionExplainer: Generating for {n_explanations} instances.")
        for i, instance in enumerate(test_subset):
            base_pred = model.predict([instance])[0]
            # Assume instance is 2D or 3D image (HWC)
            if instance.ndim == 2:
                h, w = instance.shape
                c = 1
            elif instance.ndim == 3:
                h, w, c = instance.shape
            else:
                print(f"[DEBUG] OcclusionExplainer: Unsupported shape {instance.shape}")
                continue
            patch_size = max(1, h // 8)
            importance_map = np.zeros((h, w))
            for y in range(0, h, patch_size):
                for x in range(0, w, patch_size):
                    occluded = instance.copy()
                    occluded[y:y+patch_size, x:x+patch_size, ...] = 0
                    occluded_pred = model.predict([occluded])[0]
                    importance = abs(base_pred - occluded_pred)
                    importance_map[y:y+patch_size, x:x+patch_size] = importance
            explanations.append({
                'instance_id': i,
                'importance_map': importance_map.tolist(),
                'prediction': float(base_pred),
                'true_label': float(y_test[i]) if i < len(y_test) else None
            })
        generation_time = time.time() - start_time
        print(f"[DEBUG] OcclusionExplainer: Done. Generated {len(explanations)} explanations in {generation_time:.2f}s.")
        return {
            'explanations': explanations,
            'generation_time': generation_time,
            'method': 'occlusion',
            'info': {'n_explanations': len(explanations)}
        }


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
        n_explanations = min(50, len(X_test))
        test_subset = X_test[:n_explanations]
        print(f"[DEBUG] FeatureAblationExplainer: Generating for {n_explanations} instances.")
        for i, instance in enumerate(test_subset):
            base_pred = model.predict([instance])[0]
            feature_importance = []
            for j in range(instance.shape[0]):
                ablated = instance.copy()
                ablated[j] = 0  # Zero out feature j
                ablated_pred = model.predict([ablated])[0]
                importance = abs(base_pred - ablated_pred)
                feature_importance.append(importance)
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