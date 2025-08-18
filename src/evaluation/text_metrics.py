"""
Text-Specific Evaluation Metrics
Specialized evaluation metrics for text explanation methods
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import re


class TextExplanationMetrics:
    """
    Text-specific evaluation metrics for explanation quality assessment
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        # Common English stop words (basic set)
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
            'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'this', 'that', 'these', 'those'
        }
    
    def evaluate_semantic_coherence(self, explanations: List[Dict[str, Any]]) -> float:
        """
        Evaluate if important words make semantic sense together
        
        Args:
            explanations: List of explanation dictionaries
            
        Returns:
            Semantic coherence score (0-1, higher is better)
        """
        coherence_scores = []
        
        for explanation in explanations:
            text_content = explanation.get('text_content', '')
            feature_importance = explanation.get('feature_importance', [])
            feature_names = explanation.get('feature_names', [])
            
            # Ensure feature_importance is a numpy array
            if not isinstance(feature_importance, np.ndarray):
                feature_importance = np.array(feature_importance) if feature_importance else np.array([])
            
            if not text_content or len(feature_importance) == 0:
                continue
            
            # Get top important words (top 20% or at least 3)
            top_k = max(3, int(0.2 * len(feature_importance)))
            top_indices = np.argsort(-np.abs(feature_importance))[:top_k]
            
            # Extract important words
            words = text_content.split()
            important_words = []
            for idx in top_indices:
                if idx < len(words):
                    word = words[idx].lower().strip('.,!?":;')
                    important_words.append(word)
            
            # Calculate coherence score
            coherence = self._calculate_word_coherence(important_words, text_content)
            coherence_scores.append(coherence)
        
        return float(np.mean(coherence_scores)) if coherence_scores else 0.0
    
    def evaluate_syntax_awareness(self, explanations: List[Dict[str, Any]]) -> float:
        """
        Evaluate if method respects linguistic structure (POS tags, syntax)
        
        Args:
            explanations: List of explanation dictionaries
            
        Returns:
            Syntax awareness score (0-1, higher is better)
        """
        syntax_scores = []
        
        for explanation in explanations:
            text_content = explanation.get('text_content', '')
            feature_importance = explanation.get('feature_importance', [])
            
            # Ensure feature_importance is a numpy array
            if not isinstance(feature_importance, np.ndarray):
                feature_importance = np.array(feature_importance) if feature_importance else np.array([])
            
            if not text_content or len(feature_importance) == 0:
                continue
            
            # Simple syntax awareness: content words should be more important than function words
            words = text_content.split()
            content_word_importance = 0
            function_word_importance = 0
            content_count = 0
            function_count = 0
            
            for i, word in enumerate(words):
                if i >= len(feature_importance):
                    break
                
                clean_word = word.lower().strip('.,!?":;')
                importance = float(abs(feature_importance[i]))
                
                if clean_word in self.stop_words or len(clean_word) <= 2:
                    # Function word
                    function_word_importance += importance
                    function_count += 1
                else:
                    # Content word
                    content_word_importance += importance
                    content_count += 1
            
            # Calculate ratio: content words should have higher average importance
            if content_count > 0 and function_count > 0:
                avg_content_importance = content_word_importance / content_count
                avg_function_importance = function_word_importance / function_count
                
                # Syntax score: higher when content words are more important
                if avg_content_importance + avg_function_importance > 0:
                    syntax_score = avg_content_importance / (avg_content_importance + avg_function_importance)
                else:
                    syntax_score = 0.5
                
                syntax_scores.append(syntax_score)
        
        return float(np.mean(syntax_scores)) if syntax_scores else 0.5
    
    def evaluate_context_sensitivity(self, explanations: List[Dict[str, Any]]) -> float:
        """
        Evaluate if the same words get different importance in different contexts
        
        Args:
            explanations: List of explanation dictionaries
            
        Returns:
            Context sensitivity score (0-1, higher is better)
        """
        if len(explanations) < 2:
            return 0.0
        
        # Build word-importance mapping across different texts
        word_importances = {}  # word -> list of importance scores
        
        for explanation in explanations:
            text_content = explanation.get('text_content', '')
            feature_importance = explanation.get('feature_importance', [])
            
            # Ensure feature_importance is a numpy array
            if not isinstance(feature_importance, np.ndarray):
                feature_importance = np.array(feature_importance) if feature_importance else np.array([])
            
            if not text_content or len(feature_importance) == 0:
                continue
            
            words = text_content.split()
            for i, word in enumerate(words):
                if i >= len(feature_importance):
                    break
                
                clean_word = word.lower().strip('.,!?":;')
                if len(clean_word) > 2 and clean_word not in self.stop_words:
                    if clean_word not in word_importances:
                        word_importances[clean_word] = []
                    word_importances[clean_word].append(abs(feature_importance[i]))
        
        # Calculate variance in importance for words that appear in multiple texts
        context_scores = []
        for word, importances in word_importances.items():
            if len(importances) >= 2:  # Word appears in multiple texts
                # Higher variance indicates better context sensitivity
                variance = np.var(importances)
                mean_importance = np.mean(importances)
                
                # Normalize variance by mean to get coefficient of variation
                if mean_importance > 0:
                    cv = variance / (mean_importance ** 2)
                    # Convert to 0-1 score (higher CV = more context sensitive)
                    context_score = min(1.0, cv)
                    context_scores.append(context_score)
        
        return float(np.mean(context_scores)) if context_scores else 0.0
    
    def evaluate_word_significance(self, explanations: List[Dict[str, Any]]) -> float:
        """
        Evaluate if important words are actually significant (not just frequent)
        
        Args:
            explanations: List of explanation dictionaries
            
        Returns:
            Word significance score (0-1, higher is better)
        """
        significance_scores = []
        
        for explanation in explanations:
            text_content = explanation.get('text_content', '')
            feature_importance = explanation.get('feature_importance', [])
            
            # Ensure feature_importance is a numpy array
            if not isinstance(feature_importance, np.ndarray):
                feature_importance = np.array(feature_importance) if feature_importance else np.array([])
            
            if not text_content or len(feature_importance) == 0:
                continue
            
            words = text_content.split()
            word_freq = Counter(word.lower().strip('.,!?":;') for word in words)
            total_words = len(words)
            
            # Calculate significance score
            significance_sum = 0
            valid_words = 0
            
            for i, word in enumerate(words):
                if i >= len(feature_importance):
                    break
                
                clean_word = word.lower().strip('.,!?":;')
                if len(clean_word) > 2 and clean_word not in self.stop_words:
                    importance = float(abs(feature_importance[i]))
                    frequency = word_freq[clean_word] / total_words
                    
                    # Significant words: high importance, not just high frequency
                    # Penalize words that are important only because they're frequent
                    if frequency > 0:
                        significance = importance / (1 + frequency)  # Normalize by frequency
                        significance_sum += significance
                        valid_words += 1
            
            if valid_words > 0:
                avg_significance = significance_sum / valid_words
                significance_scores.append(avg_significance)
        
        return float(np.mean(significance_scores)) if significance_scores else 0.0
    
    def evaluate_explanation_coverage(self, explanations: List[Dict[str, Any]]) -> float:
        """
        Evaluate what fraction of the text is covered by important words
        
        Args:
            explanations: List of explanation dictionaries
            
        Returns:
            Coverage score (0-1, moderate coverage is better than too high or too low)
        """
        coverage_scores = []
        
        for explanation in explanations:
            text_content = explanation.get('text_content', '')
            feature_importance = explanation.get('feature_importance', [])
            
            # Ensure feature_importance is a numpy array
            if not isinstance(feature_importance, np.ndarray):
                feature_importance = np.array(feature_importance) if feature_importance else np.array([])
            
            if not text_content or len(feature_importance) == 0:
                continue
            
            words = text_content.split()
            if len(words) == 0:
                continue
            
            # Count words with above-average importance
            threshold = np.mean(np.abs(feature_importance))
            important_word_count = 0
            
            for i, importance in enumerate(feature_importance):
                if i >= len(words):
                    break
                if float(abs(importance)) > float(threshold):
                    important_word_count += 1
            
            coverage = important_word_count / len(words)
            
            # Optimal coverage is around 20-40% (not too sparse, not too dense)
            optimal_coverage = 0.3
            distance_from_optimal = abs(coverage - optimal_coverage)
            coverage_score = max(0, 1 - (distance_from_optimal / optimal_coverage))
            
            coverage_scores.append(coverage_score)
        
        return float(np.mean(coverage_scores)) if coverage_scores else 0.0
    
    def evaluate_sentiment_consistency(self, explanations: List[Dict[str, Any]]) -> float:
        """
        Evaluate if important words align with the sentiment/prediction
        
        Args:
            explanations: List of explanation dictionaries
            
        Returns:
            Sentiment consistency score (0-1, higher is better)
        """
        # Simple sentiment word lists
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'love', 'like',
            'best', 'perfect', 'awesome', 'brilliant', 'outstanding', 'superb', 'magnificent'
        }
        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'hate', 'worst', 'disgusting', 'pathetic',
            'useless', 'disappointing', 'annoying', 'stupid', 'ridiculous', 'boring'
        }
        
        consistency_scores = []
        
        for explanation in explanations:
            text_content = explanation.get('text_content', '')
            feature_importance = explanation.get('feature_importance', [])
            prediction = explanation.get('prediction', 0)
            
            # Ensure feature_importance is a numpy array
            if not isinstance(feature_importance, np.ndarray):
                feature_importance = np.array(feature_importance) if feature_importance else np.array([])
            
            if not text_content or len(feature_importance) == 0:
                continue
            
            words = text_content.split()
            
            # Get top important words
            top_k = max(5, int(0.3 * len(feature_importance)))
            top_indices = np.argsort(-np.abs(feature_importance))[:top_k]
            
            positive_importance = 0
            negative_importance = 0
            
            for idx in top_indices:
                if idx >= len(words):
                    continue
                
                word = words[idx].lower().strip('.,!?":;')
                importance = abs(feature_importance[idx])
                
                if word in positive_words:
                    positive_importance += importance
                elif word in negative_words:
                    negative_importance += importance
            
            # Calculate consistency based on prediction
            total_sentiment_importance = positive_importance + negative_importance
            if total_sentiment_importance > 0:
                if prediction == 1:  # Assuming 1 = positive, 0 = negative
                    consistency = positive_importance / total_sentiment_importance
                else:
                    consistency = negative_importance / total_sentiment_importance
                
                consistency_scores.append(consistency)
        
        return float(np.mean(consistency_scores)) if consistency_scores else 0.5
    
    def _calculate_word_coherence(self, important_words: List[str], full_text: str) -> float:
        """
        Calculate coherence score for a set of important words
        
        Args:
            important_words: List of important words
            full_text: Full text content
            
        Returns:
            Coherence score (0-1)
        """
        if len(important_words) < 2:
            return 1.0
        
        # Simple coherence: important words should appear close to each other
        words = full_text.lower().split()
        word_positions = {}
        
        for i, word in enumerate(words):
            clean_word = word.strip('.,!?":;')
            if clean_word in important_words:
                if clean_word not in word_positions:
                    word_positions[clean_word] = []
                word_positions[clean_word].append(i)
        
        if len(word_positions) < 2:
            return 1.0
        
        # Calculate average distance between important words
        distances = []
        positions = []
        for word, pos_list in word_positions.items():
            positions.extend(pos_list)
        
        positions.sort()
        
        for i in range(len(positions) - 1):
            distances.append(positions[i + 1] - positions[i])
        
        if not distances:
            return 1.0
        
        avg_distance = np.mean(distances)
        text_length = len(words)
        
        # Normalize: shorter average distance = higher coherence
        coherence = max(0, 1 - (avg_distance / (text_length / len(important_words))))
        return min(1.0, coherence)
    
    def get_all_text_metrics(self, explanations: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate all text-specific metrics
        
        Args:
            explanations: List of explanation dictionaries
            
        Returns:
            Dictionary of metric_name -> score
        """
        metrics = {}
        
        try:
            metrics['semantic_coherence'] = self.evaluate_semantic_coherence(explanations)
        except Exception as e:
            self.logger.warning(f"Error calculating semantic coherence: {e}")
            metrics['semantic_coherence'] = 0.0
        
        try:
            metrics['syntax_awareness'] = self.evaluate_syntax_awareness(explanations)
        except Exception as e:
            self.logger.warning(f"Error calculating syntax awareness: {e}")
            metrics['syntax_awareness'] = 0.0
        
        try:
            metrics['context_sensitivity'] = self.evaluate_context_sensitivity(explanations)
        except Exception as e:
            self.logger.warning(f"Error calculating context sensitivity: {e}")
            metrics['context_sensitivity'] = 0.0
        
        try:
            metrics['word_significance'] = self.evaluate_word_significance(explanations)
        except Exception as e:
            self.logger.warning(f"Error calculating word significance: {e}")
            metrics['word_significance'] = 0.0
        
        try:
            metrics['explanation_coverage'] = self.evaluate_explanation_coverage(explanations)
        except Exception as e:
            self.logger.warning(f"Error calculating explanation coverage: {e}")
            metrics['explanation_coverage'] = 0.0
        
        try:
            metrics['sentiment_consistency'] = self.evaluate_sentiment_consistency(explanations)
        except Exception as e:
            self.logger.warning(f"Error calculating sentiment consistency: {e}")
            metrics['sentiment_consistency'] = 0.0
        
        return metrics