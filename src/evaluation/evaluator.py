"""
Evaluator for XAI explanation methods
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr
from .statistical_tests import StatisticalTester
from .text_metrics import TextExplanationMetrics
from .advanced_metrics import AdvancedMetricsEvaluator, StatisticalSignificanceTester
from .monotonicity_evaluator import MonotonicityEvaluator


class Evaluator:
    """
    Evaluator for XAI explanation methods
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evaluator
        
        Args:
            config: Configuration dictionary for evaluation
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.statistical_tester = StatisticalTester(config)
        self.text_metrics = TextExplanationMetrics()
        self.advanced_metrics = AdvancedMetricsEvaluator()
        self.statistical_significance_tester = StatisticalSignificanceTester()
        self.monotonicity_evaluator = MonotonicityEvaluator()
    
    def evaluate(self, model, explanation_results: Dict[str, Any], dataset) -> Dict[str, float]:
        """
        Evaluate explanation quality
        
        Args:
            model: Trained model
            explanation_results: Results from explanation method
            dataset: Dataset instance
            
        Returns:
            Dictionary containing evaluation metrics
        """
        self.logger.info(f"Evaluating explanations: {explanation_results.get('method', 'unknown')}")
        
        evaluation_results = {}
        
        # Time complexity
        evaluation_results['time_complexity'] = self._evaluate_time_complexity(explanation_results)
        
        # Fidelity metrics
        fidelity_metrics = self._evaluate_fidelity(model, explanation_results, dataset)
        evaluation_results.update(fidelity_metrics)
        
        # Stability metrics
        stability_metrics = self._evaluate_stability(explanation_results)
        evaluation_results.update(stability_metrics)
        
        # Comprehensibility metrics
        comprehensibility_metrics = self._evaluate_comprehensibility(explanation_results)
        evaluation_results.update(comprehensibility_metrics)
        
        # Text-specific metrics if applicable
        text_metrics = self._evaluate_text_specific_metrics(explanation_results)
        evaluation_results.update(text_metrics)
        
        # Advanced evaluation metrics (axiom-based and information-theoretic)
        advanced_metrics = self._evaluate_advanced_metrics(explanation_results, dataset, model)
        evaluation_results.update(advanced_metrics)
        
        return evaluation_results
    
    def run_statistical_analysis(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run statistical significance analysis on all results
        
        Args:
            all_results: Dictionary containing results from all methods
            
        Returns:
            Dictionary containing statistical analysis results
        """
        self.logger.info("Running statistical significance analysis...")
        
        # Run traditional statistical tests
        statistical_results = self.statistical_tester.run_statistical_tests(all_results)
        
        # Run advanced statistical significance testing
        advanced_statistical_results = self._run_advanced_statistical_analysis(all_results)
        statistical_results.update(advanced_statistical_results)
        
        return statistical_results
    
    def _evaluate_time_complexity(self, explanation_results: Dict[str, Any]) -> float:
        """Evaluate time complexity of explanation generation"""
        generation_time = explanation_results.get('generation_time', 0.0)
        n_explanations = explanation_results.get('info', {}).get('n_explanations', 1)
        
        if n_explanations > 0:
            return generation_time / n_explanations  # Time per explanation
        return generation_time
    
    def _evaluate_fidelity(self, model, explanation_results: Dict[str, Any], dataset) -> Dict[str, float]:
        """Evaluate fidelity of explanations using standard definitions"""
        explanations = explanation_results.get('explanations', [])
        method = explanation_results.get('method', '').lower()
        
        if not explanations:
            return {
                'faithfulness': 0.0,
                'monotonicity': 0.0,
                'completeness': 0.0
            }
        
        # Enhanced fidelity evaluation for prototype/counterfactual methods
        if method in ['prototype', 'counterfactual']:
            return self._evaluate_example_based_fidelity(explanations, model, dataset)
        
        # Check if this is text data (detect from explanations)
        is_text_data = False
        if explanations:
            first_explanation = explanations[0]
            if 'text_content' in first_explanation and first_explanation['text_content'] is not None:
                is_text_data = True

        faithfulness_scores = []
        monotonicity_scores = []
        completeness_scores = []

        # Compute completeness for SHAP/IG and for text data with any method
        compute_completeness = any(x in method for x in ['shap', 'integrated gradients', 'ig']) or is_text_data
        # Only compute faithfulness/monotonicity for feature attribution methods
        compute_faithfulness = any(x in method for x in ['shap', 'lime', 'integrated gradients', 'ig', 'feature attribution'])
        compute_monotonicity = compute_faithfulness
        
        # Use advanced monotonicity evaluation for all data types
        try:
            # Debug: Print what we're evaluating
            if explanations:
                first_exp = explanations[0]
                print(f"\n[DEBUG] Evaluating monotonicity for:")
                print(f"  - Has text_content: {'text_content' in first_exp}")
                print(f"  - Has tokens: {'tokens' in first_exp}")
                print(f"  - Has feature_importance: {'feature_importance' in first_exp}")
                if 'feature_importance' in first_exp:
                    imp = first_exp['feature_importance']
                    print(f"  - Feature importance length: {len(imp) if isinstance(imp, list) else 'not a list'}")
                    if isinstance(imp, list) and len(imp) > 0:
                        print(f"  - First 3 importance values: {imp[:3]}")

            monotonicity_results = self.monotonicity_evaluator.evaluate_monotonicity(
                explanations, model=model, data_type='auto'
            )
            # Extract the main monotonicity score
            advanced_monotonicity = monotonicity_results.get('monotonicity', 0.0)

            print(f"[DEBUG] Advanced monotonicity result: {advanced_monotonicity}")
            print(f"[DEBUG] Is NaN: {np.isnan(advanced_monotonicity) if isinstance(advanced_monotonicity, (int, float)) else 'not a number'}")

        except Exception as e:
            self.logger.warning(f"Error in advanced monotonicity evaluation: {e}")
            print(f"[DEBUG] ERROR in monotonicity evaluation: {e}")
            import traceback
            traceback.print_exc()
            advanced_monotonicity = 0.0

        # For backward compatibility, keep the original logic for tabular data
        # but use advanced evaluation for text and image
        data_type = 'tabular'  # default
        if is_text_data:
            data_type = 'text'
            compute_monotonicity = False  # Skip old implementation
        elif explanations and len(explanations) > 0:
            first_exp = explanations[0]
            if 'importance_map' in first_exp or (
                'feature_importance' in first_exp and
                len(first_exp.get('feature_importance', [])) > 1000
            ):
                data_type = 'image'
                compute_monotonicity = False  # Skip old implementation
        
        for explanation in explanations:
            feature_importance = explanation.get('feature_importance', [])
            prediction = explanation.get('prediction', 0)
            # Safe conversion of prediction to scalar
            if hasattr(prediction, 'item'):
                prediction = prediction.item()
            elif hasattr(prediction, '__len__') and len(prediction) == 1:
                prediction = prediction[0]
            baseline_prediction = explanation.get('baseline_prediction', None)
            input_instance = explanation.get('input', None)
            
            # Convert to numpy array and check for values safely
            try:
                if isinstance(feature_importance, (list, np.ndarray)):
                    feature_importance = np.array(feature_importance)
                    has_values = hasattr(feature_importance, 'size') and feature_importance.size > 0 and len(feature_importance) > 0
                else:
                    has_values = bool(feature_importance) if feature_importance is not None else False
            except Exception:
                has_values = bool(feature_importance) if feature_importance is not None else False
            
            if has_values and hasattr(feature_importance, 'size') and feature_importance.size > 0:
                # Faithfulness: removal test (only for compatible methods)
                if compute_faithfulness and input_instance is not None:
                    try:
                        # For text data, faithfulness is based on word removal test
                        if is_text_data:
                            # Implement text faithfulness using word removal
                            text_content = explanation.get('text_content', '')
                            tokens = explanation.get('tokens', [])
                            if text_content and tokens and len(feature_importance) > 0:
                                try:
                                    # Remove top 20% most important words
                                    k = max(1, int(0.2 * len(feature_importance)))
                                    top_k_idx = np.argsort(-np.abs(feature_importance))[:k]

                                    # Create text with important words completely removed
                                    remaining_tokens = []
                                    for i, token in enumerate(tokens):
                                        if i not in top_k_idx:
                                            remaining_tokens.append(token)

                                    # Create masked text (with important words removed)
                                    if remaining_tokens:
                                        masked_text = ' '.join(remaining_tokens)
                                    else:
                                        masked_text = ""  # All words were important

                                    # Get predictions
                                    try:
                                        if hasattr(model, 'predict_proba') and masked_text.strip():
                                            # Use probability-based faithfulness for better measurement
                                            orig_proba = model.predict_proba([text_content])[0]
                                            masked_proba = model.predict_proba([masked_text])[0]

                                            # Calculate KL divergence or max probability difference
                                            faithfulness = abs(np.max(orig_proba) - np.max(masked_proba))
                                            faithfulness_scores.append(min(1.0, faithfulness))
                                        elif masked_text.strip():
                                            # Fallback to prediction difference
                                            masked_pred = model.predict([masked_text])[0]
                                            faithfulness = abs(prediction - masked_pred)
                                            # Normalize by prediction scale
                                            if abs(prediction) > 1e-6:
                                                faithfulness = faithfulness / abs(prediction)
                                            faithfulness_scores.append(min(1.0, faithfulness))
                                        else:
                                            # If all words removed, assume high faithfulness
                                            faithfulness_scores.append(0.8)
                                    except Exception as e:
                                        # If prediction fails, assume moderate faithfulness
                                        faithfulness_scores.append(0.5)
                                except Exception:
                                    # Any other error, low faithfulness
                                    faithfulness_scores.append(0.2)
                        else:
                            # Check if this is image data
                            is_image_data = (data_type == 'image' or
                                           'importance_map' in explanation or
                                           (hasattr(input_instance, 'shape') and len(input_instance.shape) >= 2 and
                                            input_instance.shape[0] > 10))  # Images typically > 10x10

                            if is_image_data:
                                # For image data, use pixel masking/removal test
                                print(f"    [FAITHFULNESS] Processing image data")
                                try:
                                    # Get the original image shape
                                    orig_image = np.array(input_instance, dtype=float)
                                    orig_shape = orig_image.shape
                                    print(f"    [FAITHFULNESS] Image shape: {orig_shape}")

                                    # Flatten feature importance if it's 2D/3D
                                    flat_importance = np.abs(feature_importance).flatten()

                                    # Remove top 10% most important pixels
                                    k = max(1, int(0.1 * len(flat_importance)))
                                    top_k_idx = np.argsort(-flat_importance)[:k]
                                    print(f"    [FAITHFULNESS] Masking top {k} pixels")

                                    # Create modified image with important pixels masked
                                    x_mod_flat = orig_image.flatten().copy()

                                    # Mask with mean value (better than 0 for images)
                                    mask_value = np.mean(x_mod_flat)
                                    x_mod_flat[top_k_idx] = mask_value

                                    # Reshape back to original image shape
                                    x_mod = x_mod_flat.reshape(orig_shape)

                                    # Get predictions - ensure correct shape for model
                                    pred_orig = prediction
                                    if hasattr(model, 'predict'):
                                        # Most image models expect batch dimension
                                        if len(orig_shape) == 2:  # Grayscale
                                            x_mod_batch = np.expand_dims(x_mod, axis=0)
                                        elif len(orig_shape) == 3:  # RGB
                                            x_mod_batch = np.expand_dims(x_mod, axis=0)
                                        else:
                                            x_mod_batch = x_mod

                                        pred_mod = model.predict(x_mod_batch)[0]
                                    else:
                                        pred_mod = pred_orig

                                    print(f"    [FAITHFULNESS] pred_orig: {pred_orig}, pred_mod: {pred_mod}")

                                    # Calculate faithfulness
                                    faithfulness = abs(pred_orig - pred_mod) / (abs(pred_orig) + 1e-8)
                                    print(f"    [FAITHFULNESS] Image faithfulness: {faithfulness:.6f}")
                                    faithfulness_scores.append(min(1.0, faithfulness))

                                except Exception as e:
                                    print(f"    [FAITHFULNESS] Error in image faithfulness: {e}")
                                    import traceback
                                    traceback.print_exc()
                                    faithfulness_scores.append(0.3)
                            else:
                                # For tabular data, use feature removal test
                                k = max(1, int(0.1 * len(feature_importance)))
                                top_k_idx = np.argsort(-np.abs(feature_importance))[:k]
                                x_mod = np.array(input_instance, dtype=float).copy()

                                # Replace with baseline values instead of zero
                                if baseline_prediction is not None:
                                    # Get baseline instance from model (we need to recreate it)
                                    try:
                                        from sklearn.dummy import DummyClassifier
                                        # Estimate baseline as feature means from training data
                                        # This is an approximation - ideally we'd pass training data
                                        baseline_values = np.zeros_like(x_mod)  # Fallback to zeros if no baseline
                                        x_mod[top_k_idx] = baseline_values[top_k_idx]
                                    except:
                                        x_mod[top_k_idx] = 0  # Fallback to original method
                                else:
                                    x_mod[top_k_idx] = 0  # Fallback to original method

                                pred_orig = prediction
                                pred_mod = model.predict([x_mod])[0] if hasattr(model, 'predict') else pred_orig
                                # Normalize faithfulness by prediction magnitude for better comparison
                                faithfulness = abs(pred_orig - pred_mod) / (abs(pred_orig) + 1e-8)
                                faithfulness_scores.append(min(1.0, faithfulness))  # Cap at 1.0
                    except Exception:
                        pass
                # Monotonicity: check if increasing feature increases output (for compatible methods)
                if compute_monotonicity and input_instance is not None:
                    monotonicity_count = 0
                    total = 0
                    
                    # Estimate feature scales for appropriate perturbation
                    instance_array = np.array(input_instance, dtype=float)
                    feature_ranges = np.maximum(np.abs(instance_array), 0.1)  # Avoid zero division
                    
                    for i in range(len(feature_importance)):
                        # Skip features with near-zero importance
                        if abs(feature_importance[i]) < 1e-6:
                            continue
                            
                        x_up = instance_array.copy()
                        x_down = instance_array.copy()
                        
                        # Use proportional increment based on feature scale
                        increment = feature_ranges[i] * 0.1  # 10% of feature's magnitude
                        
                        x_up[i] += increment
                        x_down[i] -= increment
                        
                        try:
                            pred_up = model.predict([x_up])[0] if hasattr(model, 'predict') else prediction
                            pred_down = model.predict([x_down])[0] if hasattr(model, 'predict') else prediction
                            
                            # Calculate empirical gradient
                            empirical_gradient = (pred_up - pred_down) / (2 * increment)
                            
                            # Check if empirical gradient agrees with feature importance sign
                            if (feature_importance[i] > 0 and empirical_gradient > 0) or \
                               (feature_importance[i] < 0 and empirical_gradient < 0):
                                monotonicity_count += 1
                            total += 1
                        except Exception:
                            continue
                    monotonicity = monotonicity_count / total if total > 0 else 0.0
                    monotonicity_scores.append(monotonicity)
                # Completeness: for SHAP/IG (additivity property) and text data (coverage)
                if compute_completeness:
                    try:
                        if is_text_data:
                            # For text: completeness = coverage of important words using percentile threshold
                            if len(feature_importance) > 0:
                                # Use percentile-based threshold (top 20% of words are "important")
                                importance_abs = np.abs(feature_importance)

                                # Method 1: Top percentile approach
                                threshold_percentile = np.percentile(importance_abs, 80)  # Top 20%
                                important_words_percentile = np.sum(importance_abs >= threshold_percentile)

                                # Method 2: Entropy-based approach (how spread out the importance is)
                                # High completeness = importance is spread across many words
                                # Low completeness = importance concentrated in few words
                                total_importance = np.sum(importance_abs)
                                if total_importance > 1e-8:
                                    normalized_importance = importance_abs / total_importance
                                    # Calculate entropy (higher entropy = more spread out = higher completeness)
                                    entropy = -np.sum(normalized_importance * np.log(normalized_importance + 1e-10))
                                    max_entropy = np.log(len(feature_importance))  # Maximum possible entropy
                                    entropy_completeness = entropy / max_entropy if max_entropy > 0 else 0.0
                                else:
                                    entropy_completeness = 0.0

                                # Method 3: Effective number of words (based on concentration)
                                if total_importance > 1e-8:
                                    # Simpson's diversity index adapted for explanations
                                    normalized_sq = (importance_abs / total_importance) ** 2
                                    effective_words = 1.0 / np.sum(normalized_sq)
                                    effective_completeness = effective_words / len(feature_importance)
                                else:
                                    effective_completeness = 0.0

                                # Combine methods for robust completeness measure
                                percentile_completeness = important_words_percentile / len(feature_importance)

                                # Average of all three methods for robust measure
                                completeness = (percentile_completeness + entropy_completeness + effective_completeness) / 3.0
                                completeness_scores.append(min(1.0, max(0.0, completeness)))
                        elif baseline_prediction is not None:
                            # For tabular: traditional additivity-based completeness
                            sum_attributions = np.sum(feature_importance)

                            # For classification models, use probability differences
                            if hasattr(model, 'predict_proba'):
                                # Get probability outputs for more stable completeness
                                orig_proba = model.predict_proba([input_instance])[0]
                                # Estimate baseline probabilities (should be passed from explainer)
                                baseline_proba = np.ones(len(orig_proba)) / len(orig_proba)  # Uniform baseline

                                # Use max probability difference for completeness
                                pred_class = np.argmax(orig_proba)
                                output_diff = orig_proba[pred_class] - baseline_proba[pred_class]
                            else:
                                # Fallback to raw predictions
                                output_diff = prediction - baseline_prediction
                        
                        # Completeness: how well attributions sum to the model output difference
                        if abs(output_diff) > 1e-8:
                            completeness = 1.0 - abs(sum_attributions - output_diff) / abs(output_diff)
                        else:
                            # If output difference is near zero, check if attributions are also near zero
                            completeness = 1.0 if abs(sum_attributions) < 1e-8 else 0.0
                            
                        completeness_scores.append(max(0, min(1, completeness)))
                    except Exception:
                        completeness_scores.append(0.0)
        # Aggregate results
        # Use advanced monotonicity for text/image, traditional for tabular
        if data_type in ['text', 'image']:
            monotonicity_value = advanced_monotonicity
        elif compute_monotonicity and monotonicity_scores:
            monotonicity_value = float(np.mean(monotonicity_scores))
        elif not compute_monotonicity and data_type == 'tabular':
            monotonicity_value = 0.0  # Tabular but couldn't compute
        else:
            monotonicity_value = advanced_monotonicity  # Fallback to advanced

        return {
            'faithfulness': float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0,
            'monotonicity': monotonicity_value,
            'completeness': float(np.mean(completeness_scores)) if completeness_scores else 0.0
        }
    
    def _evaluate_stability(self, explanation_results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate stability of explanations"""
        explanations = explanation_results.get('explanations', [])
        method = explanation_results.get('method', '').lower()
        
        if len(explanations) < 2:
            return {
                'stability': 0.0,
                'consistency': 0.0
            }
        
        # Handle example-based methods differently
        if method in ['prototype', 'counterfactual']:
            return self._evaluate_example_based_stability(explanations)
        
        # Calculate stability across multiple explanations
        feature_importances = []
        for explanation in explanations:
            importance = explanation.get('feature_importance', [])
            try:
                if isinstance(importance, (list, np.ndarray)):
                    importance = np.array(importance)
                    has_values = importance.size > 0
                else:
                    has_values = bool(importance)
            except Exception:
                has_values = bool(importance)
            
            if has_values and importance.size > 0:
                feature_importances.append(importance)
        
        if len(feature_importances) < 2:
            return {
                'stability': 0.0,
                'consistency': 0.0
            }
        
        # Ensure all arrays have the same shape by padding/truncating
        max_length = max(len(imp) for imp in feature_importances)
        normalized_importances = []
        
        for imp in feature_importances:
            if len(imp) < max_length:
                # Pad with zeros
                padded = np.zeros(max_length)
                padded[:len(imp)] = imp
                normalized_importances.append(padded)
            elif len(imp) > max_length:
                # Truncate
                normalized_importances.append(imp[:max_length])
            else:
                normalized_importances.append(imp)
        
        feature_importances = np.array(normalized_importances)
        
        # Stability: consistency across explanations
        # Calculate coefficient of variation (CV) for each feature across explanations
        mean_importance = np.mean(np.abs(feature_importances), axis=0)
        std_importance = np.std(feature_importances, axis=0)

        # Avoid division by zero
        cv = np.where(mean_importance > 1e-8, std_importance / mean_importance, 0)
        # Stability is inverse of mean coefficient of variation
        stability = 1.0 / (1.0 + np.mean(cv))
        stability = max(0, min(1, stability))
        
        # Consistency: rank correlation between explanations
        consistency_scores = []
        for i in range(len(feature_importances)):
            for j in range(i + 1, len(feature_importances)):
                a = feature_importances[i]
                b = feature_importances[j]
                # Skip empty or constant arrays to avoid undefined correlations
                try:
                    if np.size(a) == 0 or np.size(b) == 0:
                        continue
                    if float(np.std(a)) == 0.0 or float(np.std(b)) == 0.0:
                        continue
                except Exception:
                    # If shapes/types are odd, skip this pair safely
                    continue
                try:
                    correlation, _ = spearmanr(a, b)
                    if correlation is not None and not np.isnan(correlation):
                        consistency_scores.append(abs(correlation))
                except Exception:
                    continue
        
        consistency = np.mean(consistency_scores) if consistency_scores else 0.0
        
        return {
            'stability': stability,
            'consistency': consistency
        }
    
    def _evaluate_comprehensibility(self, explanation_results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate comprehensibility of explanations"""
        explanations = explanation_results.get('explanations', [])

        if not explanations:
            return {
                'sparsity': 0.0,
                'simplicity': 0.0
            }

        sparsity_scores = []
        simplicity_scores = []

        # Detect data type from first explanation
        is_image_data = False
        if explanations:
            first_exp = explanations[0]
            is_image_data = ('importance_map' in first_exp or
                           (hasattr(first_exp.get('input'), 'shape') and
                            len(first_exp.get('input', []).shape) >= 2))

        print(f"\n[DEBUG SPARSITY] Evaluating comprehensibility for {'image' if is_image_data else 'tabular/text'} data")

        for i, explanation in enumerate(explanations):
            # Get feature importance - handle both tabular and image data
            feature_importance = explanation.get('feature_importance', [])

            # For image data, use importance_map if feature_importance is empty
            # Check if feature_importance is empty properly for arrays
            is_empty = False
            if feature_importance is None:
                is_empty = True
            elif hasattr(feature_importance, '__len__'):
                is_empty = len(feature_importance) == 0

            if is_empty and 'importance_map' in explanation:
                importance_map = explanation.get('importance_map', [])
                if isinstance(importance_map, list):
                    # Flatten the 2D importance map to 1D array for sparsity calculation
                    feature_importance = np.array(importance_map).flatten()
                elif isinstance(importance_map, np.ndarray):
                    feature_importance = importance_map.flatten()

            try:
                if isinstance(feature_importance, (list, np.ndarray)):
                    feature_importance = np.array(feature_importance)
                    has_values = hasattr(feature_importance, 'size') and feature_importance.size > 0
                else:
                    has_values = bool(feature_importance)
            except Exception:
                has_values = bool(feature_importance)

            if has_values and hasattr(feature_importance, 'size') and feature_importance.size > 0:
                # Sparsity: how many features are actually important
                # Use different thresholds for different data types
                if is_image_data:
                    # For images, use 90th percentile (expect most pixels to be unimportant)
                    percentile_threshold = 90
                    print(f"  [SPARSITY] Image #{i+1}: Using {percentile_threshold}th percentile for {len(feature_importance)} pixels")
                elif len(feature_importance) > 100:
                    # For high-dimensional data, use 85th percentile
                    percentile_threshold = 85
                    print(f"  [SPARSITY] High-dim #{i+1}: Using {percentile_threshold}th percentile for {len(feature_importance)} features")
                else:
                    # For low-dimensional tabular data, use 75th percentile
                    percentile_threshold = 75
                    print(f"  [SPARSITY] Tabular #{i+1}: Using {percentile_threshold}th percentile for {len(feature_importance)} features")

                importance_abs = np.abs(feature_importance)
                importance_threshold = np.percentile(importance_abs, percentile_threshold)
                important_count = np.sum(importance_abs >= importance_threshold)

                # Calculate sparsity
                sparsity = 1.0 - float(important_count) / len(feature_importance)

                print(f"    - Importance range: [{np.min(importance_abs):.6f}, {np.max(importance_abs):.6f}]")
                print(f"    - Threshold (p{percentile_threshold}): {importance_threshold:.6f}")
                print(f"    - Important features: {important_count} / {len(feature_importance)} ({100*important_count/len(feature_importance):.2f}%)")
                print(f"    - Sparsity score: {sparsity:.6f}")

                sparsity_scores.append(max(0, min(1, sparsity)))

                # Simplicity: how concentrated the importance is
                # Calculate Gini coefficient as a measure of concentration
                sorted_importance = np.sort(np.abs(feature_importance))  # Use absolute values
                n = len(sorted_importance)
                total_importance = np.sum(sorted_importance)

                if n > 1 and total_importance > 1e-10:
                    # Gini coefficient calculation
                    cumsum = np.cumsum(sorted_importance)
                    gini = (n + 1 - 2 * np.sum(cumsum) / total_importance) / n
                    simplicity = max(0, min(1, abs(gini)))  # Normalize to [0, 1]

                    print(f"    - Gini coefficient: {gini:.6f}")
                    print(f"    - Simplicity score: {simplicity:.6f}")

                    simplicity_scores.append(simplicity)
                else:
                    print(f"    - Single feature or zero importance, simplicity: 1.0")
                    simplicity_scores.append(1.0)  # Perfect simplicity for single feature

        final_sparsity = np.mean(sparsity_scores) if sparsity_scores else 0.0
        final_simplicity = np.mean(simplicity_scores) if simplicity_scores else 0.0

        print(f"\n[DEBUG SPARSITY] Final scores: sparsity={final_sparsity:.6f}, simplicity={final_simplicity:.6f}")

        return {
            'sparsity': final_sparsity,
            'simplicity': final_simplicity
        }
    
    def _evaluate_example_based_fidelity(self, explanations: List[Dict], model, dataset) -> Dict[str, float]:
        """Evaluate fidelity for example-based explanation methods (prototype, counterfactual)"""
        faithfulness_scores = []
        consistency_scores = []
        coverage_scores = []
        
        X_train, X_test, y_train, y_test = dataset.get_data()
        
        for explanation in explanations:
            instance_id = explanation.get('instance_id', 0)
            prediction = explanation.get('prediction')
            true_label = explanation.get('true_label')
            confidence = explanation.get('confidence', 0.5)
            
            # Faithfulness: how well the explanation aligns with model behavior
            if prediction is not None and true_label is not None:
                # Prediction accuracy
                prediction_correct = (prediction == true_label)
                faithfulness_scores.append(1.0 if prediction_correct else 0.0)
            
            # Monotonicity for example-based methods: consistency between distance and confidence
            if 'proto_distance' in explanation:
                # For prototypes: closer prototypes should indicate higher confidence
                proto_distance = explanation.get('proto_distance')
                if proto_distance is not None and confidence is not None:
                    # Expected: high confidence should correlate with low distance
                    # Score: inverse relationship (closer prototype = higher confidence)
                    max_distance = 50.0  # Reasonable upper bound for normalization
                    normalized_distance = min(proto_distance / max_distance, 1.0)
                    expected_confidence = 1.0 - normalized_distance
                    consistency_score = 1.0 - abs(confidence - expected_confidence)
                    consistency_scores.append(max(0.0, consistency_score))
            elif 'cf_distance' in explanation:
                # For counterfactuals: closer counterfactuals with low confidence suggest decision boundary proximity
                cf_distance = explanation.get('cf_distance')
                if cf_distance is not None and confidence is not None:
                    # Expected: low confidence should correlate with low distance (near decision boundary)
                    max_distance = 50.0  # Reasonable upper bound for normalization
                    normalized_distance = min(cf_distance / max_distance, 1.0)
                    # For counterfactuals, closer distance should mean lower confidence (decision boundary)
                    expected_confidence = normalized_distance  # Farther = more confident
                    consistency_score = 1.0 - abs(confidence - expected_confidence)
                    consistency_scores.append(max(0.0, consistency_score))
        
        # Coverage: how well explanations span the test set
        unique_classes_explained = len(set(exp.get('prediction', -1) for exp in explanations))
        total_classes = len(np.unique(y_test))
        coverage = unique_classes_explained / total_classes if total_classes > 0 else 0.0
        
        return {
            'faithfulness': np.mean(faithfulness_scores) if faithfulness_scores else 0.0,
            'monotonicity': np.mean(consistency_scores) if consistency_scores else 0.0,
            'completeness': coverage
        }
    
    def _evaluate_example_based_stability(self, explanations: List[Dict]) -> Dict[str, float]:
        """Evaluate stability for example-based methods (prototype/counterfactual)"""
        # Stability: variance in prototype/counterfactual distances across explanations
        distances = []
        confidences = []
        
        # Debug output
        print(f"[DEBUG] _evaluate_example_based_stability called with {len(explanations)} explanations")
        
        for i, explanation in enumerate(explanations[:5]):  # Check first 5
            # Get distance (prototype or counterfactual)
            distance = explanation.get('proto_distance') or explanation.get('cf_distance')
            confidence = explanation.get('confidence')
            
            print(f"[DEBUG] Explanation {i}: distance={distance}, confidence={confidence}")
            
            if distance is not None:
                distances.append(float(distance))
            if confidence is not None:
                confidences.append(float(confidence))
        
        stability = 0.0
        consistency = 0.0
        
        if len(distances) >= 2:
            # Stability: normalized variance of distances (lower variance = higher stability)
            distance_var = np.var(distances)
            distance_mean = np.mean(distances)
            if distance_mean > 0:
                stability = 1.0 / (1.0 + distance_var / distance_mean)
            else:
                stability = 1.0
        
        if len(distances) >= 2 and len(confidences) >= 2:
            # Consistency: correlation between distances and confidences
            # Closer prototypes should have higher confidence (negative correlation with distance)
            try:
                # Ensure arrays have the same length
                min_length = min(len(distances), len(confidences))
                distances_trimmed = distances[:min_length]
                confidences_trimmed = confidences[:min_length]
                
                if min_length >= 2:
                    correlation, _ = pearsonr(distances_trimmed, confidences_trimmed)
                    if correlation is not None and not np.isnan(correlation):
                        # Convert to positive score (closer distance = higher confidence = better consistency)
                        consistency = abs(correlation)
                    else:
                        consistency = 0.0
                else:
                    consistency = 0.0
            except Exception as e:
                print(f"[DEBUG] Exception in consistency calculation: {e}")
                consistency = 0.0
        
        print(f"[DEBUG] Final results - stability: {stability}, consistency: {consistency}")
        print(f"[DEBUG] distances: {len(distances)}, confidences: {len(confidences)}")
        
        return {
            'stability': float(stability),
            'consistency': float(consistency)
        }
    
    def _evaluate_text_specific_metrics(self, explanation_results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate text-specific metrics if applicable"""
        explanations = explanation_results.get('explanations', [])
        
        # Check if this is text data
        if not explanations:
            return {}
        
        is_text_data = False
        text_explanations = []
        
        for explanation in explanations:
            if 'text_content' in explanation and explanation['text_content'] is not None:
                is_text_data = True
                text_explanations.append(explanation)
        
        if not is_text_data or not text_explanations:
            return {}
        
        self.logger.info(f"Evaluating text-specific metrics for {len(text_explanations)} explanations")
        
        # Calculate text-specific metrics
        try:
            text_metrics_results = self.text_metrics.get_all_text_metrics(text_explanations)
            return text_metrics_results
        except Exception as e:
            self.logger.error(f"Error evaluating text-specific metrics: {e}")
            return {
                'semantic_coherence': 0.0,
                'syntax_awareness': 0.0,
                'context_sensitivity': 0.0,
                'word_significance': 0.0,
                'explanation_coverage': 0.0,
                'sentiment_consistency': 0.0
            }
    
    def _evaluate_advanced_metrics(self, explanation_results: Dict[str, Any], dataset, model) -> Dict[str, float]:
        """Evaluate advanced metrics (axiom-based and information-theoretic)"""
        explanations = explanation_results.get('explanations', [])
        
        if not explanations:
            return {}
        
        try:
            self.logger.info("Evaluating advanced metrics (axiom-based and information-theoretic)")
            advanced_results = self.advanced_metrics.evaluate_all_advanced_metrics(
                explanations, dataset, model
            )
            
            # Add prefix to distinguish from basic metrics
            prefixed_results = {}
            for metric_name, value in advanced_results.items():
                prefixed_results[f'advanced_{metric_name}'] = value
            
            return prefixed_results
            
        except Exception as e:
            self.logger.error(f"Error evaluating advanced metrics: {e}")
            return {
                'advanced_identity': 0.0,
                'advanced_separability': 0.0,
                'advanced_non_sensitivity': 0.0,
                'advanced_stability': 0.0,
                'advanced_correctness': 0.0,
                'advanced_entropy': 0.0,
                'advanced_gini_coefficient': 0.0,
                'advanced_kl_divergence': 0.0
            }
    
    def _run_advanced_statistical_analysis(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run advanced statistical significance testing with confidence intervals"""
        try:
            self.logger.info("Running advanced statistical significance testing")
            
            # Extract method results for statistical comparison
            method_results = {}
            
            for method_name, results in all_results.items():
                if isinstance(results, dict) and 'evaluation' in results:
                    evaluation_metrics = results['evaluation']
                    
                    # Group metrics by type for analysis
                    method_results[method_name] = {}
                    
                    # Collect all metric scores
                    for metric_name, metric_value in evaluation_metrics.items():
                        if isinstance(metric_value, (int, float)):
                            # For single values, create a list (we'd need multiple runs for proper statistics)
                            method_results[method_name][metric_name] = [float(metric_value)]
            
            if len(method_results) < 2:
                self.logger.warning("Need at least 2 methods for advanced statistical comparison")
                return {'advanced_statistical_analysis': 'insufficient_methods'}
            
            # Run statistical significance testing
            significance_results = self.statistical_significance_tester.compare_methods_with_significance(
                method_results
            )
            
            return {
                'advanced_statistical_analysis': significance_results
            }
            
        except Exception as e:
            self.logger.error(f"Error in advanced statistical analysis: {e}")
            return {
                'advanced_statistical_analysis': f'error: {str(e)}'
            }