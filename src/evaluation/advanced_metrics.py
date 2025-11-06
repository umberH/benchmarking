"""
Advanced Evaluation Metrics
Axiom-based and information-theoretic metrics for comprehensive explanation evaluation
"""

import logging
import numpy as np
import scipy.stats as stats
from typing import Dict, List, Any, Tuple, Optional
from collections import Counter
from sklearn.metrics import mutual_info_score
import warnings
warnings.filterwarnings('ignore')


class CompactnessEvaluator:
    """
    Evaluates the brevity and simplicity of explanations.
    Compactness is essential for improving human cognition, as shorter and simpler
    explanations are easier for users to process.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def evaluate_compactness(self, explanations: List[Dict[str, Any]],
                            method_type: str = 'feature_attribution') -> Dict[str, float]:
        """
        Evaluate compactness metrics based on explanation method type

        Args:
            explanations: List of explanation dictionaries
            method_type: Type of explanation method ('feature_attribution', 'rule_based', 'text')

        Returns:
            Dictionary of compactness metrics
        """
        results = {}

        if method_type == 'feature_attribution':
            results.update(self._evaluate_feature_compactness(explanations))
        elif method_type == 'rule_based':
            results.update(self._evaluate_rule_compactness(explanations))
        elif method_type == 'text':
            results.update(self._evaluate_text_compactness(explanations))
        else:
            # Default to feature attribution metrics
            results.update(self._evaluate_feature_compactness(explanations))

        return results

    def _evaluate_feature_compactness(self, explanations: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate compactness for feature attribution methods
        Measures: sparsity (L0 norm), top-k coverage, effective features
        """
        sparsity_scores = []
        top5_coverage = []
        top10_coverage = []
        effective_features_scores = []

        for explanation in explanations:
            feature_importance = explanation.get('feature_importance', [])

            # Handle both list and numpy array cases
            importance_empty = (isinstance(feature_importance, np.ndarray) and feature_importance.size == 0) or \
                               (not isinstance(feature_importance, np.ndarray) and len(feature_importance) == 0)
            if importance_empty:
                continue

            importance_array = np.abs(np.array(feature_importance))

            # Skip if contains NaN or inf
            if np.any(np.isnan(importance_array)) or np.any(np.isinf(importance_array)):
                continue

            # 1. Sparsity (L0 norm) - ratio of non-zero features
            non_zero_count = np.count_nonzero(importance_array > 1e-8)
            sparsity = 1.0 - (non_zero_count / len(importance_array)) if len(importance_array) > 0 else 0.0
            sparsity_scores.append(sparsity)

            # 2. Top-k coverage - cumulative importance of top-k features
            if np.sum(importance_array) > 0:
                sorted_importance = np.sort(importance_array)[::-1]
                normalized_importance = sorted_importance / np.sum(importance_array)

                # Top-5 coverage
                top5_sum = np.sum(normalized_importance[:min(5, len(normalized_importance))])
                top5_coverage.append(top5_sum)

                # Top-10 coverage
                top10_sum = np.sum(normalized_importance[:min(10, len(normalized_importance))])
                top10_coverage.append(top10_sum)

                # 3. Effective features (inverse participation ratio)
                prob_dist = importance_array / np.sum(importance_array)
                effective_features = 1.0 / np.sum(prob_dist ** 2) if np.sum(prob_dist ** 2) > 0 else len(prob_dist)

                # Normalize: fewer effective features = higher compactness
                compactness_score = 1.0 - (effective_features - 1) / (len(prob_dist) - 1) if len(prob_dist) > 1 else 1.0
                effective_features_scores.append(max(0.0, compactness_score))
            else:
                top5_coverage.append(0.0)
                top10_coverage.append(0.0)
                effective_features_scores.append(0.0)

        return {
            'compactness_sparsity': float(np.mean(sparsity_scores)) if sparsity_scores else 0.0,
            'compactness_top5_coverage': float(np.mean(top5_coverage)) if top5_coverage else 0.0,
            'compactness_top10_coverage': float(np.mean(top10_coverage)) if top10_coverage else 0.0,
            'compactness_effective_features': float(np.mean(effective_features_scores)) if effective_features_scores else 0.0
        }

    def _evaluate_rule_compactness(self, explanations: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate compactness for rule-based explanations
        Measures: average rule length, total conditions, rule set size
        """
        rule_lengths = []
        total_conditions = []
        rule_set_sizes = []

        for explanation in explanations:
            rules = explanation.get('rules', [])

            if isinstance(rules, list) and len(rules) > 0:
                # Rule set size
                rule_set_sizes.append(len(rules))

                # Average rule length and total conditions
                rule_length = 0
                conditions_count = 0

                for rule in rules:
                    if isinstance(rule, dict):
                        # Rule might have 'conditions' or 'antecedents'
                        conditions = rule.get('conditions', rule.get('antecedents', []))
                        if isinstance(conditions, list):
                            rule_length += len(conditions)
                            conditions_count += len(conditions)
                    elif isinstance(rule, str):
                        # Simple string rule - count AND/OR operators as proxy for conditions
                        conditions_count += rule.count(' AND ') + rule.count(' OR ') + 1
                        rule_length += conditions_count

                avg_rule_length = rule_length / len(rules) if len(rules) > 0 else 0
                rule_lengths.append(avg_rule_length)
                total_conditions.append(conditions_count)
            else:
                # Check for text-based rules in 'explanation_text'
                rule_text = explanation.get('explanation_text', '')
                if 'if' in rule_text.lower() or 'when' in rule_text.lower():
                    # Simple heuristic for text-based rules
                    rule_count = rule_text.count('if') + rule_text.count('when')
                    rule_set_sizes.append(max(1, rule_count))
                    # Estimate complexity by word count
                    word_count = len(rule_text.split())
                    rule_lengths.append(word_count / max(1, rule_count))
                    total_conditions.append(rule_count * 2)  # Rough estimate

        # Normalize scores - lower values = more compact
        max_rule_length = 10.0  # Assume 10 conditions is very complex
        max_conditions = 50.0   # Assume 50 total conditions is very complex
        max_set_size = 20.0     # Assume 20 rules is a large set

        return {
            'compactness_avg_rule_length': 1.0 - min(np.mean(rule_lengths) / max_rule_length, 1.0) if rule_lengths else 1.0,
            'compactness_total_conditions': 1.0 - min(np.mean(total_conditions) / max_conditions, 1.0) if total_conditions else 1.0,
            'compactness_rule_set_size': 1.0 - min(np.mean(rule_set_sizes) / max_set_size, 1.0) if rule_set_sizes else 1.0
        }

    def _evaluate_text_compactness(self, explanations: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate compactness for text explanations
        Measures: token ratio, span compactness, explanation brevity
        """
        token_ratios = []
        span_lengths = []
        brevity_scores = []

        for explanation in explanations:
            text_content = explanation.get('text_content', '')
            feature_importance = explanation.get('feature_importance', [])
            highlighted_words = explanation.get('highlighted_words', [])

            if text_content:
                words = text_content.split()
                total_words = len(words)

                if total_words > 0:
                    # Token ratio: highlighted tokens / total tokens
                    if highlighted_words:
                        highlighted_count = len(highlighted_words)
                    elif self._has_content(feature_importance) and len(feature_importance) > 0:
                        # Count non-zero importance as highlighted
                        importance_array = np.array(feature_importance[:total_words])
                        highlighted_count = np.count_nonzero(np.abs(importance_array) > 1e-8)
                    else:
                        highlighted_count = 0

                    token_ratio = highlighted_count / total_words
                    token_ratios.append(1.0 - token_ratio)  # Lower ratio = more compact

                    # Span compactness: measure continuity of highlighted regions
                    if self._has_content(feature_importance) and len(feature_importance) >= total_words:
                        importance_array = np.abs(np.array(feature_importance[:total_words]))
                        binary_importance = (importance_array > 1e-8).astype(int)

                        # Count number of continuous spans
                        spans = 0
                        in_span = False
                        span_length = 0
                        current_span_lengths = []

                        for val in binary_importance:
                            if val == 1:
                                if not in_span:
                                    spans += 1
                                    in_span = True
                                    span_length = 1
                                else:
                                    span_length += 1
                            else:
                                if in_span:
                                    current_span_lengths.append(span_length)
                                    in_span = False
                                    span_length = 0

                        if in_span:
                            current_span_lengths.append(span_length)

                        if current_span_lengths:
                            avg_span_length = np.mean(current_span_lengths)
                            span_lengths.append(avg_span_length)

                    # Brevity score: inverse of explanation length
                    max_acceptable_words = 100  # Consider 100 words as maximum acceptable
                    brevity = 1.0 - min(total_words / max_acceptable_words, 1.0)
                    brevity_scores.append(brevity)

        return {
            'compactness_token_ratio': float(np.mean(token_ratios)) if token_ratios else 0.0,
            'compactness_avg_span_length': 1.0 - min(np.mean(span_lengths) / 10.0, 1.0) if span_lengths else 1.0,
            'compactness_brevity': float(np.mean(brevity_scores)) if brevity_scores else 0.0
        }

    def _has_content(self, feature_importance) -> bool:
        """
        Check if feature_importance has content (works for both lists and numpy arrays)

        Args:
            feature_importance: Feature importance (list or numpy array)

        Returns:
            True if has content, False otherwise
        """
        if feature_importance is None:
            return False
        if isinstance(feature_importance, np.ndarray):
            return feature_importance.size > 0
        return len(feature_importance) > 0 if hasattr(feature_importance, '__len__') else False


class AdvancedMetricsEvaluator:
    """
    Advanced evaluation metrics including axiom-based properties and information-theoretic measures
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.compactness_evaluator = CompactnessEvaluator()

    def _has_content(self, feature_importance) -> bool:
        """
        Check if feature_importance has content (works for both lists and numpy arrays)

        Args:
            feature_importance: Feature importance (list or numpy array)

        Returns:
            True if has content, False otherwise
        """
        if feature_importance is None:
            return False
        if isinstance(feature_importance, np.ndarray):
            return feature_importance.size > 0
        return len(feature_importance) > 0 if hasattr(feature_importance, '__len__') else False
    
    def evaluate_all_advanced_metrics(self, explanations: List[Dict[str, Any]],
                                    dataset=None, model=None) -> Dict[str, float]:
        """Evaluate all advanced metrics for explanations"""
        results = {}

        # Axiom-based metrics
        axiom_metrics = self.evaluate_axiom_based_metrics(explanations, dataset, model)
        results.update(axiom_metrics)

        # Information-theoretic metrics
        info_metrics = self.evaluate_information_theoretic_metrics(explanations, dataset)
        results.update(info_metrics)

        # Compactness metrics
        compactness_metrics = self.evaluate_compactness_metrics(explanations)
        results.update(compactness_metrics)

        return results

    def evaluate_compactness_metrics(self, explanations: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate compactness metrics for explanations
        Automatically detects the method type based on explanation structure
        """
        if not explanations:
            return {}

        # Detect method type based on explanation structure
        method_type = self._detect_explanation_method_type(explanations)

        # Evaluate compactness metrics
        compactness_results = self.compactness_evaluator.evaluate_compactness(
            explanations, method_type=method_type
        )

        return compactness_results

    def _detect_explanation_method_type(self, explanations: List[Dict[str, Any]]) -> str:
        """Detect the type of explanation method from the explanation structure"""
        if not explanations:
            return 'feature_attribution'

        # Check first non-empty explanation
        for explanation in explanations:
            if explanation:
                # Check for rule-based explanations
                if 'rules' in explanation or (
                    'explanation_text' in explanation and
                    ('if' in explanation.get('explanation_text', '').lower() or
                     'when' in explanation.get('explanation_text', '').lower())
                ):
                    return 'rule_based'

                # Check for text explanations
                if 'text_content' in explanation or 'highlighted_words' in explanation:
                    return 'text'

                # Default to feature attribution
                if 'feature_importance' in explanation:
                    return 'feature_attribution'

        return 'feature_attribution'

    def evaluate_axiom_based_metrics(self, explanations: List[Dict[str, Any]],
                                   dataset=None, model=None) -> Dict[str, float]:
        """
        Evaluate axiom-based properties of explanations:
        - Identity: Baseline explanation should be zero
        - Separability: Different features should have distinguishable contributions  
        - Non-sensitivity: Small input changes should yield similar explanations
        - Compactness: Explanations should be sparse and focused
        """
        results = {}
        
        try:
            results['identity'] = self._evaluate_identity_axiom(explanations)
        except Exception as e:
            self.logger.warning(f"Error evaluating identity axiom: {e}")
            results['identity'] = 0.0
        
        try:
            results['separability'] = self._evaluate_separability_axiom(explanations)
        except Exception as e:
            self.logger.warning(f"Error evaluating separability axiom: {e}")
            results['separability'] = 0.0
        
        try:
            results['non_sensitivity'] = self._evaluate_non_sensitivity_axiom(explanations, dataset, model)
        except Exception as e:
            self.logger.warning(f"Error evaluating non-sensitivity axiom: {e}")
            results['non_sensitivity'] = 0.0
        
        try:
            results['stability'] = self._evaluate_stability_axiom(explanations)
        except Exception as e:
            self.logger.warning(f"Error evaluating stability axiom: {e}")
            results['stability'] = 0.0
        
        return results
    
    def evaluate_information_theoretic_metrics(self, explanations: List[Dict[str, Any]], 
                                             dataset=None) -> Dict[str, float]:
        """
        Evaluate information-theoretic properties:
        - Correctness: Information gain from explanation
        - Entropy: Uncertainty/randomness in feature importance
        - Gini: Concentration/inequality of importance distribution
        - KL Divergence: Distance from uniform distribution
        """
        results = {}
        
        try:
            results['correctness'] = self._evaluate_correctness(explanations, dataset)
        except Exception as e:
            self.logger.warning(f"Error evaluating correctness: {e}")
            results['correctness'] = 0.0
        
        try:
            results['entropy'] = self._evaluate_entropy(explanations)
        except Exception as e:
            self.logger.warning(f"Error evaluating entropy: {e}")
            results['entropy'] = 0.0
        
        try:
            results['gini_coefficient'] = self._evaluate_gini_coefficient(explanations)
        except Exception as e:
            self.logger.warning(f"Error evaluating gini coefficient: {e}")
            results['gini_coefficient'] = 0.0
        
        try:
            results['kl_divergence'] = self._evaluate_kl_divergence(explanations)
        except Exception as e:
            self.logger.warning(f"Error evaluating KL divergence: {e}")
            results['kl_divergence'] = 0.0
        
        return results
    
    def _evaluate_identity_axiom(self, explanations: List[Dict[str, Any]]) -> float:
        """
        Identity Axiom: When input equals baseline, explanation should be zero
        Measures how well explanations satisfy the identity property
        """
        identity_violations = []
        
        for explanation in explanations:
            feature_importance = explanation.get('feature_importance', [])
            baseline_prediction = explanation.get('baseline_prediction')
            prediction = explanation.get('prediction')
            
            # Handle both list and numpy array cases
            importance_empty = (isinstance(feature_importance, np.ndarray) and feature_importance.size == 0) or \
                               (not isinstance(feature_importance, np.ndarray) and len(feature_importance) == 0)
            if importance_empty or baseline_prediction is None or prediction is None:
                continue
            
            # Check if prediction is close to baseline (identity condition)
            if abs(prediction - baseline_prediction) < 0.01:  # Close to baseline
                # Importance should be close to zero
                importance_magnitude = np.sum(np.abs(feature_importance))
                identity_violations.append(importance_magnitude)
        
        if not identity_violations:
            return 1.0  # Perfect identity if no violations found
        
        # Identity score: lower violation = higher score
        avg_violation = np.mean(identity_violations)
        identity_score = 1.0 / (1.0 + avg_violation)  # Normalize to [0,1]
        
        return float(identity_score)
    
    def _evaluate_separability_axiom(self, explanations: List[Dict[str, Any]]) -> float:
        """
        Separability Axiom: Different features should have distinguishable importance values
        Measures the ability to distinguish between feature contributions
        """
        separability_scores = []
        
        for explanation in explanations:
            feature_importance = explanation.get('feature_importance', [])
            
            # Handle both list and numpy array cases
            importance_len = feature_importance.size if isinstance(feature_importance, np.ndarray) else len(feature_importance)
            if importance_len == 0 or importance_len < 2:
                continue
            
            importance_array = np.array(feature_importance)

            # Skip if contains NaN or inf
            if np.any(np.isnan(importance_array)) or np.any(np.isinf(importance_array)):
                continue

            # Calculate coefficient of variation (std/mean) as separability measure
            if np.mean(np.abs(importance_array)) > 1e-8:
                cv = np.std(importance_array) / np.mean(np.abs(importance_array))
                separability_score = min(cv, 2.0) / 2.0  # Normalize to [0,1]
            else:
                separability_score = 0.0
            
            separability_scores.append(separability_score)
        
        return float(np.mean(separability_scores)) if separability_scores else 0.0
    
    def _evaluate_non_sensitivity_axiom(self, explanations: List[Dict[str, Any]], 
                                       dataset=None, model=None) -> float:
        """
        Non-sensitivity Axiom: Small input changes should yield similar explanations
        Measures explanation stability under minor perturbations
        """
        if not dataset or not model or len(explanations) < 2:
            # Fallback: measure explanation consistency across instances
            return self._measure_explanation_consistency(explanations)
        
        try:
            # Get data for perturbation analysis
            X_train, X_test, y_train, y_test = dataset.get_data()

            if isinstance(X_test, list):  # Text data
                return self._measure_text_non_sensitivity(explanations)

            # For tabular/image data, measure sensitivity to small perturbations
            sensitivity_scores = []
            n_samples = min(5, len(explanations))  # Limited sampling for efficiency

            for i in range(n_samples):
                explanation = explanations[i]
                instance_id = explanation.get('instance_id', i)

                if instance_id < len(X_test):
                    original_instance = X_test[instance_id]
                    original_importance = np.array(explanation.get('feature_importance', []))

                    if original_importance.size == 0:
                        continue

                    # Flatten instance for consistency (images may be multi-dimensional)
                    original_instance_flat = original_instance.flatten() if hasattr(original_instance, 'flatten') else original_instance

                    # Create small perturbation
                    # For images, calculate std on flattened data
                    if X_train.ndim > 2:  # Image data
                        # Flatten training data to calculate std
                        X_train_flat = X_train.reshape(X_train.shape[0], -1)
                        feature_std = np.std(X_train_flat, axis=0)
                    else:  # Tabular data
                        feature_std = np.std(X_train, axis=0)

                    # Ensure noise_scale matches the flattened instance shape
                    if np.isscalar(feature_std):
                        noise_scale = 0.01 * feature_std
                    else:
                        noise_scale = 0.01 * feature_std
                        # Ensure noise_scale matches the shape of flattened original_instance
                        if len(noise_scale) != len(original_instance_flat):
                            noise_scale = np.resize(noise_scale, len(original_instance_flat))

                    perturbed_instance = original_instance_flat + np.random.normal(0, noise_scale, size=len(original_instance_flat))
                    
                    # Get explanation for perturbed instance (simplified)
                    try:
                        perturbed_importance = self._get_perturbation_importance(
                            perturbed_instance, original_instance_flat, original_importance, model
                        )
                        
                        # Calculate similarity between original and perturbed explanations
                        if len(perturbed_importance) == len(original_importance):
                            correlation = np.corrcoef(original_importance, perturbed_importance)[0, 1]
                            if not np.isnan(correlation):
                                sensitivity_scores.append(abs(correlation))
                    except Exception:
                        continue
            
            return float(np.mean(sensitivity_scores)) if sensitivity_scores else 0.5
            
        except Exception as e:
            self.logger.warning(f"Error in non-sensitivity evaluation: {e}")
            return self._measure_explanation_consistency(explanations)
    
    def _measure_explanation_consistency(self, explanations: List[Dict[str, Any]]) -> float:
        """Fallback: measure consistency across explanations"""
        if len(explanations) < 2:
            return 1.0
        
        importance_vectors = []
        for explanation in explanations:
            importance = explanation.get('feature_importance', [])
            if importance:
                importance_vectors.append(np.array(importance))
        
        if len(importance_vectors) < 2:
            return 1.0
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(importance_vectors)):
            for j in range(i + 1, len(importance_vectors)):
                if len(importance_vectors[i]) == len(importance_vectors[j]):
                    corr = np.corrcoef(importance_vectors[i], importance_vectors[j])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
        
        return float(np.mean(correlations)) if correlations else 0.5
    
    def _measure_text_non_sensitivity(self, explanations: List[Dict[str, Any]]) -> float:
        """Measure non-sensitivity for text explanations"""
        # For text, measure consistency of word importance patterns
        word_importance_patterns = []
        
        for explanation in explanations:
            text_content = explanation.get('text_content', '')
            feature_importance = explanation.get('feature_importance', [])

            # Properly check if feature_importance has content (works for lists and arrays)
            has_importance = False
            if isinstance(feature_importance, np.ndarray):
                has_importance = feature_importance.size > 0
            elif feature_importance:
                has_importance = len(feature_importance) > 0

            if text_content and has_importance:
                words = text_content.split()
                if len(words) > 0 and len(feature_importance) >= len(words):
                    # Normalize importance by word frequency as stability measure
                    word_freq = Counter(words)
                    normalized_importance = []
                    
                    for i, word in enumerate(words):
                        if i < len(feature_importance):
                            freq = word_freq[word]
                            normalized_imp = feature_importance[i] / (freq + 1)  # +1 to avoid division by 0
                            normalized_importance.append(normalized_imp)
                    
                    if normalized_importance:
                        word_importance_patterns.append(np.array(normalized_importance))
        
        if len(word_importance_patterns) < 2:
            return 1.0
        
        # Measure consistency of normalized patterns
        pattern_correlations = []
        for i in range(len(word_importance_patterns)):
            for j in range(i + 1, len(word_importance_patterns)):
                # Compare patterns of similar length
                pattern_i = word_importance_patterns[i]
                pattern_j = word_importance_patterns[j]
                
                min_len = min(len(pattern_i), len(pattern_j))
                if min_len > 1:
                    corr = np.corrcoef(pattern_i[:min_len], pattern_j[:min_len])[0, 1]
                    if not np.isnan(corr):
                        pattern_correlations.append(abs(corr))
        
        return float(np.mean(pattern_correlations)) if pattern_correlations else 0.5
    
    def _get_perturbation_importance(self, perturbed_instance, original_instance, 
                                   original_importance, model) -> np.ndarray:
        """Get importance for perturbed instance (simplified approximation)"""
        # Simplified: assume importance scales with input change
        input_change = perturbed_instance - original_instance
        
        # Scale original importance by input change magnitude
        change_magnitude = np.abs(input_change) / (np.abs(original_instance) + 1e-8)
        perturbed_importance = original_importance * (1 + 0.1 * change_magnitude)  # Small scaling
        
        return perturbed_importance
    
    def _evaluate_stability_axiom(self, explanations: List[Dict[str, Any]]) -> float:
        """
        Compactness Axiom: Explanations should be sparse and focused on important features
        Measures the concentration of explanation mass on few features
        """
        stability_scores = []
        
        for explanation in explanations:
            feature_importance = explanation.get('feature_importance', [])

            # Handle both list and numpy array cases
            if isinstance(feature_importance, np.ndarray):
                if feature_importance.size == 0:
                    continue
            elif len(feature_importance) == 0:
                continue
            
            importance_array = np.abs(np.array(feature_importance))

            # Skip if contains NaN or inf
            if np.any(np.isnan(importance_array)) or np.any(np.isinf(importance_array)):
                continue

            if np.sum(importance_array) == 0:
                stability_scores.append(0.0)
                continue
            
            # Normalize to probability distribution
            prob_dist = importance_array / np.sum(importance_array)
            
            # Calculate effective number of features (inverse participation ratio)
            effective_features = 1.0 / np.sum(prob_dist ** 2) if np.sum(prob_dist ** 2) > 0 else len(prob_dist)
            
            # Compactness: fewer effective features = higher stability
            stability = 1.0 - (effective_features - 1) / (len(prob_dist) - 1) if len(prob_dist) > 1 else 1.0
            stability_scores.append(max(0.0, stability))
        
        return float(np.mean(stability_scores)) if stability_scores else 0.0
    
    def _evaluate_correctness(self, explanations: List[Dict[str, Any]], dataset=None) -> float:
        """
        Correctness: Information gain provided by the explanation
        Measures how much the explanation reduces uncertainty about the prediction
        """
        correctness_scores = []
        
        for explanation in explanations:
            prediction = explanation.get('prediction')
            true_label = explanation.get('true_label')
            feature_importance = explanation.get('feature_importance', [])
            
            # Handle both list and numpy array cases
            importance_empty = (isinstance(feature_importance, np.ndarray) and feature_importance.size == 0) or \
                               (not isinstance(feature_importance, np.ndarray) and len(feature_importance) == 0)
            if prediction is None or true_label is None or importance_empty:
                continue
            
            # Prediction correctness
            pred_correctness = 1.0 if abs(prediction - true_label) < 0.5 else 0.0
            
            # Explanation informativeness: higher variance in importance = more informative
            importance_array = np.array(feature_importance)

            # Skip if contains NaN or inf
            if np.any(np.isnan(importance_array)) or np.any(np.isinf(importance_array)):
                continue

            if len(importance_array) > 1:
                importance_variance = np.var(importance_array)
                max_possible_variance = np.var([1.0] + [0.0] * (len(importance_array) - 1))  # Max possible
                informativeness = min(importance_variance / max_possible_variance, 1.0) if max_possible_variance > 0 else 0.0
            else:
                informativeness = 0.0
            
            # Combined correctness: prediction accuracy + explanation informativeness
            correctness = 0.7 * pred_correctness + 0.3 * informativeness
            correctness_scores.append(correctness)
        
        return float(np.mean(correctness_scores)) if correctness_scores else 0.0
    
    def _evaluate_entropy(self, explanations: List[Dict[str, Any]]) -> float:
        """
        Entropy: Measure uncertainty/randomness in feature importance distribution
        Lower entropy = more focused explanation, Higher entropy = more distributed
        """
        entropy_scores = []
        
        for explanation in explanations:
            feature_importance = explanation.get('feature_importance', [])

            # Handle both list and numpy array cases
            if isinstance(feature_importance, np.ndarray):
                if feature_importance.size == 0:
                    continue
            elif len(feature_importance) == 0:
                continue
            
            # Convert to non-negative values and normalize
            importance_array = np.abs(np.array(feature_importance))

            # Skip if contains NaN or inf
            if np.any(np.isnan(importance_array)) or np.any(np.isinf(importance_array)):
                continue

            if np.sum(importance_array) == 0:
                entropy_scores.append(0.0)  # No uncertainty if no importance
                continue
            
            # Normalize to probability distribution
            prob_dist = importance_array / np.sum(importance_array)
            
            # Calculate Shannon entropy
            entropy = -np.sum(prob_dist * np.log2(prob_dist + 1e-10))  # Add small constant to avoid log(0)
            
            # Normalize by maximum possible entropy (uniform distribution)
            max_entropy = np.log2(len(prob_dist)) if len(prob_dist) > 1 else 0.0
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
            
            entropy_scores.append(normalized_entropy)
        
        return float(np.mean(entropy_scores)) if entropy_scores else 0.0
    
    def _evaluate_gini_coefficient(self, explanations: List[Dict[str, Any]]) -> float:
        """
        Gini Coefficient: Measure inequality/concentration in importance distribution
        0 = perfect equality, 1 = perfect inequality (concentrated on one feature)
        """
        gini_scores = []
        
        for explanation in explanations:
            feature_importance = explanation.get('feature_importance', [])

            # Handle both list and numpy array cases
            if isinstance(feature_importance, np.ndarray):
                if feature_importance.size == 0:
                    continue
            elif len(feature_importance) == 0:
                continue
            
            # Use absolute values and sort
            importance_array = np.sort(np.abs(np.array(feature_importance)))
            n = len(importance_array)
            
            if n <= 1 or np.sum(importance_array) == 0:
                gini_scores.append(0.0)
                continue
            
            # Calculate Gini coefficient
            index = np.arange(1, n + 1)
            gini = (2 * np.sum(index * importance_array)) / (n * np.sum(importance_array)) - (n + 1) / n
            
            gini_scores.append(max(0.0, min(1.0, gini)))  # Ensure [0,1] range
        
        return float(np.mean(gini_scores)) if gini_scores else 0.0
    
    def _evaluate_kl_divergence(self, explanations: List[Dict[str, Any]]) -> float:
        """
        KL Divergence: Measure distance from uniform distribution
        Lower KL = closer to uniform, Higher KL = more concentrated
        """
        kl_scores = []
        
        for explanation in explanations:
            feature_importance = explanation.get('feature_importance', [])

            # Handle both list and numpy array cases
            if isinstance(feature_importance, np.ndarray):
                if feature_importance.size == 0:
                    continue
            elif len(feature_importance) == 0:
                continue
            
            # Convert to probability distribution
            importance_array = np.abs(np.array(feature_importance))

            # Skip if contains NaN or inf
            if np.any(np.isnan(importance_array)) or np.any(np.isinf(importance_array)):
                continue

            if np.sum(importance_array) == 0:
                kl_scores.append(0.0)
                continue
            
            prob_dist = importance_array / np.sum(importance_array)
            
            # Uniform distribution
            uniform_dist = np.ones(len(prob_dist)) / len(prob_dist)
            
            # Calculate KL divergence from uniform
            kl_div = np.sum(prob_dist * np.log2((prob_dist + 1e-10) / (uniform_dist + 1e-10)))
            
            # Normalize by maximum possible KL (all mass on one feature)
            max_kl = np.log2(len(prob_dist)) if len(prob_dist) > 1 else 0.0
            normalized_kl = kl_div / max_kl if max_kl > 0 else 0.0
            
            kl_scores.append(max(0.0, min(1.0, normalized_kl)))
        
        return float(np.mean(kl_scores)) if kl_scores else 0.0


class StatisticalSignificanceTester:
    """
    Statistical significance testing and confidence interval computation for explanation evaluation
    """
    
    def __init__(self, confidence_level: float = 0.95):
        self.confidence_level = confidence_level
        self.alpha = 1 - confidence_level
        self.logger = logging.getLogger(__name__)
    
    def compare_methods_with_significance(self, method_results: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
        """
        Compare multiple explanation methods with statistical significance testing
        
        Args:
            method_results: {method_name: {metric_name: [scores]}}
            
        Returns:
            Statistical comparison results with significance tests
        """
        comparison_results = {
            'pairwise_comparisons': {},
            'confidence_intervals': {},
            'effect_sizes': {},
            'significance_summary': {}
        }
        
        methods = list(method_results.keys())
        
        if len(methods) < 2:
            self.logger.warning("Need at least 2 methods for statistical comparison")
            return comparison_results
        
        # Get common metrics across all methods
        common_metrics = set.intersection(*[set(results.keys()) for results in method_results.values()])
        
        for metric in common_metrics:
            comparison_results['pairwise_comparisons'][metric] = {}
            comparison_results['confidence_intervals'][metric] = {}
            comparison_results['effect_sizes'][metric] = {}
            
            # Calculate confidence intervals for each method
            for method in methods:
                scores = method_results[method][metric]
                ci = self._calculate_confidence_interval(scores)
                comparison_results['confidence_intervals'][metric][method] = ci
            
            # Pairwise statistical tests
            for i, method1 in enumerate(methods):
                for j, method2 in enumerate(methods[i+1:], i+1):
                    scores1 = method_results[method1][metric]
                    scores2 = method_results[method2][metric]
                    
                    if len(scores1) > 1 and len(scores2) > 1:
                        # Perform statistical test
                        test_result = self._statistical_test(scores1, scores2, method1, method2)
                        comparison_key = f"{method1}_vs_{method2}"
                        comparison_results['pairwise_comparisons'][metric][comparison_key] = test_result
                        
                        # Effect size
                        effect_size = self._calculate_effect_size(scores1, scores2)
                        comparison_results['effect_sizes'][metric][comparison_key] = effect_size
        
        # Summary of significant differences
        comparison_results['significance_summary'] = self._create_significance_summary(
            comparison_results['pairwise_comparisons']
        )
        
        return comparison_results
    
    def _calculate_confidence_interval(self, scores: List[float]) -> Dict[str, float]:
        """Calculate confidence interval for a set of scores"""
        if not scores or len(scores) < 2:
            return {'mean': 0.0, 'lower': 0.0, 'upper': 0.0, 'std_error': 0.0}
        
        scores_array = np.array(scores)
        mean = np.mean(scores_array)
        std_error = stats.sem(scores_array)  # Standard error of mean
        
        # t-distribution for small samples, normal for large samples
        if len(scores) < 30:
            t_value = stats.t.ppf(1 - self.alpha/2, len(scores) - 1)
            margin_error = t_value * std_error
        else:
            z_value = stats.norm.ppf(1 - self.alpha/2)
            margin_error = z_value * std_error
        
        return {
            'mean': float(mean),
            'lower': float(mean - margin_error),
            'upper': float(mean + margin_error),
            'std_error': float(std_error)
        }
    
    def _statistical_test(self, scores1: List[float], scores2: List[float], 
                         method1: str, method2: str) -> Dict[str, Any]:
        """Perform appropriate statistical test for comparing two groups"""
        scores1_array = np.array(scores1)
        scores2_array = np.array(scores2)
        
        # Remove NaN values
        scores1_clean = scores1_array[~np.isnan(scores1_array)]
        scores2_clean = scores2_array[~np.isnan(scores2_array)]
        
        if len(scores1_clean) < 2 or len(scores2_clean) < 2:
            return {
                'test_type': 'insufficient_data',
                'p_value': 1.0,
                'statistic': 0.0,
                'significant': False,
                'better_method': 'inconclusive'
            }
        
        # Choose appropriate test
        if len(scores1_clean) < 30 or len(scores2_clean) < 30:
            # Small sample: use t-test (assuming approximately normal)
            if len(scores1_clean) == len(scores2_clean):
                # Paired t-test if same number of observations
                try:
                    statistic, p_value = stats.ttest_rel(scores1_clean, scores2_clean)
                    test_type = 'paired_t_test'
                except:
                    statistic, p_value = stats.ttest_ind(scores1_clean, scores2_clean)
                    test_type = 'independent_t_test'
            else:
                # Independent t-test
                statistic, p_value = stats.ttest_ind(scores1_clean, scores2_clean)
                test_type = 'independent_t_test'
        else:
            # Large sample: use Mann-Whitney U test (non-parametric)
            statistic, p_value = stats.mannwhitneyu(scores1_clean, scores2_clean, alternative='two-sided')
            test_type = 'mann_whitney_u'
        
        # Determine which method is better
        mean1 = np.mean(scores1_clean)
        mean2 = np.mean(scores2_clean)
        
        if mean1 > mean2:
            better_method = method1
        elif mean2 > mean1:
            better_method = method2
        else:
            better_method = 'tie'
        
        return {
            'test_type': test_type,
            'p_value': float(p_value),
            'statistic': float(statistic),
            'significant': p_value < self.alpha,
            'better_method': better_method,
            'mean_1': float(mean1),
            'mean_2': float(mean2)
        }
    
    def _calculate_effect_size(self, scores1: List[float], scores2: List[float]) -> Dict[str, float]:
        """Calculate effect size (Cohen's d) between two groups"""
        scores1_array = np.array(scores1)
        scores2_array = np.array(scores2)
        
        # Remove NaN values
        scores1_clean = scores1_array[~np.isnan(scores1_array)]
        scores2_clean = scores2_array[~np.isnan(scores2_array)]
        
        if len(scores1_clean) < 2 or len(scores2_clean) < 2:
            return {'cohens_d': 0.0, 'interpretation': 'no_data'}
        
        mean1 = np.mean(scores1_clean)
        mean2 = np.mean(scores2_clean)
        std1 = np.std(scores1_clean, ddof=1)
        std2 = np.std(scores2_clean, ddof=1)
        
        # Pooled standard deviation
        n1, n2 = len(scores1_clean), len(scores2_clean)
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
        
        # Cohen's d
        if pooled_std > 0:
            cohens_d = (mean1 - mean2) / pooled_std
        else:
            cohens_d = 0.0
        
        # Interpretation
        abs_d = abs(cohens_d)
        if abs_d < 0.2:
            interpretation = 'negligible'
        elif abs_d < 0.5:
            interpretation = 'small'
        elif abs_d < 0.8:
            interpretation = 'medium'
        else:
            interpretation = 'large'
        
        return {
            'cohens_d': float(cohens_d),
            'interpretation': interpretation
        }
    
    def _create_significance_summary(self, pairwise_comparisons: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
        """Create summary of significant differences"""
        summary = {
            'total_comparisons': 0,
            'significant_comparisons': 0,
            'by_metric': {}
        }
        
        for metric, comparisons in pairwise_comparisons.items():
            metric_summary = {
                'total': len(comparisons),
                'significant': 0,
                'significant_pairs': []
            }
            
            for pair, result in comparisons.items():
                summary['total_comparisons'] += 1
                
                if result.get('significant', False):
                    summary['significant_comparisons'] += 1
                    metric_summary['significant'] += 1
                    metric_summary['significant_pairs'].append({
                        'pair': pair,
                        'p_value': result['p_value'],
                        'better_method': result['better_method']
                    })
            
            summary['by_metric'][metric] = metric_summary
        
        # Overall significance rate
        if summary['total_comparisons'] > 0:
            summary['significance_rate'] = summary['significant_comparisons'] / summary['total_comparisons']
        else:
            summary['significance_rate'] = 0.0
        
        return summary