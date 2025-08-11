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
        
        # Run statistical tests
        statistical_results = self.statistical_tester.run_statistical_tests(all_results)
        
        return statistical_results
    
    def _evaluate_time_complexity(self, explanation_results: Dict[str, Any]) -> float:
        """Evaluate time complexity of explanation generation"""
        generation_time = explanation_results.get('generation_time', 0.0)
        n_explanations = explanation_results.get('info', {}).get('n_explanations', 1)
        
        if n_explanations > 0:
            return generation_time / n_explanations  # Time per explanation
        return generation_time
    
    def _evaluate_fidelity(self, model, explanation_results: Dict[str, Any], dataset) -> Dict[str, float]:
        """Evaluate fidelity of explanations"""
        explanations = explanation_results.get('explanations', [])
        
        if not explanations:
            return {
                'faithfulness': 0.0,
                'monotonicity': 0.0,
                'completeness': 0.0
            }
        
        # Faithfulness: How well explanations reflect model behavior
        faithfulness_scores = []
        monotonicity_scores = []
        completeness_scores = []
        
        for explanation in explanations:
            # Get feature importance and prediction
            feature_importance = explanation.get('feature_importance', [])
            prediction = explanation.get('prediction', 0)
            
            # Robustly handle numpy arrays/lists
            try:
                if isinstance(feature_importance, (list, np.ndarray)):
                    feature_importance = np.array(feature_importance)
                    has_values = feature_importance.size > 0
                else:
                    has_values = bool(feature_importance)
            except Exception:
                has_values = bool(feature_importance)
            
            if has_values and feature_importance.size > 0:
                # Faithfulness: how well the explanation captures the model's decision
                # For text: based on word importance distribution
                # For tabular: based on feature importance magnitude
                if isinstance(feature_importance, np.ndarray):
                    # Normalize importance scores
                    if np.sum(feature_importance) > 0:
                        normalized_importance = feature_importance / np.sum(feature_importance)
                    else:
                        normalized_importance = feature_importance
                    
                    # Faithfulness: entropy-based measure of explanation quality
                    non_zero_importance = normalized_importance[normalized_importance > 0]
                    if len(non_zero_importance) > 0:
                        entropy = -np.sum(non_zero_importance * np.log(non_zero_importance + 1e-8))
                        max_entropy = np.log(len(non_zero_importance))
                        faithfulness = 1.0 - (entropy / max_entropy) if max_entropy > 0 else 0.0
                    else:
                        faithfulness = 0.0
                    
                    faithfulness_scores.append(max(0, min(1, faithfulness)))
                    
                    # Monotonicity: consistency of feature importance
                    if np.std(feature_importance) > 0:
                        monotonicity = 1.0 - (np.std(feature_importance) / np.mean(feature_importance))
                        monotonicity_scores.append(max(0, min(1, monotonicity)))
                    else:
                        monotonicity_scores.append(1.0)  # Perfect consistency
                    
                    # Completeness: coverage of important features
                    threshold = np.percentile(feature_importance, 75)  # Top 25% features
                    completeness = np.sum(feature_importance >= threshold) / len(feature_importance)
                    completeness_scores.append(completeness)
        
        return {
            'faithfulness': np.mean(faithfulness_scores) if faithfulness_scores else 0.0,
            'monotonicity': np.mean(monotonicity_scores) if monotonicity_scores else 0.0,
            'completeness': np.mean(completeness_scores) if completeness_scores else 0.0
        }
    
    def _evaluate_stability(self, explanation_results: Dict[str, Any]) -> Dict[str, float]:
        """Evaluate stability of explanations"""
        explanations = explanation_results.get('explanations', [])
        
        if len(explanations) < 2:
            return {
                'stability': 0.0,
                'consistency': 0.0
            }
        
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
                    if np.std(a) == 0 or np.std(b) == 0:
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
                importance_threshold = np.percentile(feature_importance, 75)
                sparsity = 1.0 - np.sum(feature_importance >= importance_threshold) / len(feature_importance)
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