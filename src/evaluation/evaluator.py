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
        
        faithfulness_scores = []
        monotonicity_scores = []
        completeness_scores = []
        
        # Only compute completeness for SHAP/IG
        compute_completeness = any(x in method for x in ['shap', 'integrated gradients', 'ig'])
        # Only compute faithfulness/monotonicity for feature attribution methods
        compute_faithfulness = any(x in method for x in ['shap', 'lime', 'integrated gradients', 'ig', 'feature attribution'])
        compute_monotonicity = compute_faithfulness
        
        # Check if this is text data (detect from explanations)
        is_text_data = False
        if explanations:
            first_explanation = explanations[0]
            if 'text_content' in first_explanation and first_explanation['text_content'] is not None:
                is_text_data = True
        
        # For text data, skip monotonicity (doesn't make sense for word importance)
        if is_text_data:
            compute_monotonicity = False
        
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
                    has_values = feature_importance.size > 0 and len(feature_importance) > 0
                else:
                    has_values = bool(feature_importance) if feature_importance is not None else False
            except Exception:
                has_values = bool(feature_importance) if feature_importance is not None else False
            
            if has_values and feature_importance.size > 0:
                # Faithfulness: removal test (only for compatible methods)
                if compute_faithfulness and input_instance is not None:
                    try:
                        # For text data, faithfulness is based on prediction accuracy
                        if is_text_data:
                            # For text, faithfulness is simply prediction accuracy
                            true_label = explanation.get('true_label')
                            if true_label is not None:
                                faithfulness = 1.0 if prediction == true_label else 0.0
                                faithfulness_scores.append(faithfulness)
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
                # Completeness: only for SHAP/IG (additivity property)
                if compute_completeness and baseline_prediction is not None:
                    try:
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
        return {
            'faithfulness': float(np.mean(faithfulness_scores)) if faithfulness_scores else 0.0,
            'monotonicity': float(np.mean(monotonicity_scores)) if monotonicity_scores else 0.0,
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
        stability = 1.0 - np.mean(np.std(feature_importances, axis=0)) / (np.mean(feature_importances) + 1e-8)
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
        
        for explanation in explanations:
            feature_importance = explanation.get('feature_importance', [])
            
            try:
                if isinstance(feature_importance, (list, np.ndarray)):
                    feature_importance = np.array(feature_importance)
                    has_values = feature_importance.size > 0
                else:
                    has_values = bool(feature_importance)
            except Exception:
                has_values = bool(feature_importance)
            
            if has_values and feature_importance.size > 0:
                # Sparsity: how many features are actually important
                # Use 75th percentile as threshold for important features
                importance_threshold = np.percentile(np.abs(feature_importance), 75)
                important_count = np.sum(np.abs(feature_importance) >= importance_threshold)
                sparsity = 1.0 - float(important_count) / len(feature_importance)
                sparsity_scores.append(max(0, min(1, sparsity)))
                
                # Simplicity: how concentrated the importance is
                # Calculate Gini coefficient as a measure of concentration
                sorted_importance = np.sort(feature_importance)
                n = len(sorted_importance)
                if n > 1 and np.sum(sorted_importance) > 0:
                    # Gini coefficient calculation
                    cumsum = np.cumsum(sorted_importance)
                    gini = (n + 1 - 2 * np.sum(cumsum) / np.sum(sorted_importance)) / n
                    simplicity = max(0, min(1, gini))  # Normalize to [0, 1]
                    simplicity_scores.append(simplicity)
                else:
                    simplicity_scores.append(1.0)  # Perfect simplicity for single feature
        
        return {
            'sparsity': np.mean(sparsity_scores) if sparsity_scores else 0.0,
            'simplicity': np.mean(simplicity_scores) if simplicity_scores else 0.0
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
            'monotonicity': np.mean(consistency_scores) if consistency_scores else 0.0,  # Reuse for consistency
            'completeness': coverage  # Reuse for coverage
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
                'advanced_compactness': 0.0,
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