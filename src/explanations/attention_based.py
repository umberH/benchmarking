"""
Attention-based explanation methods for text
"""

import logging
import time
import numpy as np
from typing import Dict, Any, List

from .base_explainer import BaseExplainer


class AttentionVisualizationExplainer(BaseExplainer):
    """Attention Visualization explainer for text models with attention mechanisms"""
    
    supported_data_types = ['text']
    supported_model_types = ['all']  # Works with any model, simulates attention patterns
    
    def explain(self, dataset) -> Dict[str, Any]:
        """Generate attention-based explanations for text"""
        start_time = time.time()
        
        # Get test data
        X_train, X_test, y_train, y_test = dataset.get_data()
        
        # Handle only text data
        if not isinstance(X_test, list):
            self.logger.warning("AttentionVisualizationExplainer only supports text data, skipping...")
            return {
                'explanations': [],
                'generation_time': 0.0,
                'method': 'attention_visualization',
                'info': {
                    'n_explanations': 0,
                    'feature_names': []
                }
            }
        
        explanations = []
        
        # Generate explanations for a subset of test instances
        # Get max test samples from config (None means use full test set)
        max_test_samples = self.config.get('experiment', {}).get('explanation', {}).get('max_test_samples', None)
        n_explanations = len(X_test) if max_test_samples is None else min(max_test_samples, len(X_test))
        test_subset = X_test[:n_explanations]
        
        for i, text_instance in enumerate(test_subset):
            # Calculate attention-based word importance
            word_importance, feature_names = self._calculate_attention_importance(
                text_instance, X_train, y_train
            )
            
            # Get model prediction
            prediction = self.model.predict([text_instance])[0]
            
            # Calculate baseline prediction (average of training data predictions)
            if len(X_train) > 100:
                train_sample = X_train[:100]
            else:
                train_sample = X_train
            
            train_predictions = [self.model.predict([text])[0] for text in train_sample]
            baseline_prediction = np.mean(train_predictions)
            
            explanation = {
                'instance_id': i,
                'feature_importance': word_importance,
                'feature_names': feature_names,
                'prediction': prediction,
                'true_label': y_test[i] if i < len(y_test) else None,
                'input': list(range(len(word_importance))),  # Numerical indices for evaluation
                'text_content': text_instance,  # Store original text separately
                'baseline_prediction': baseline_prediction
            }
            explanations.append(explanation)
        
        self.generation_time = time.time() - start_time
        
        return {
            'explanations': explanations,
            'generation_time': self.generation_time,
            'method': 'attention_visualization',
            'info': {
                'n_explanations': len(explanations),
                'feature_names': explanations[0]['feature_names'] if explanations else []
            }
        }
    
    def _calculate_attention_importance(self, text_instance, X_train, y_train):
        """
        Calculate attention-based importance scores for words
        
        Since we don't have access to actual attention weights from the model,
        this is a proxy implementation that simulates attention patterns.
        In a real implementation, you would extract attention weights from the model.
        """
        # Ensure text_instance is a string
        if not isinstance(text_instance, str):
            # Handle numpy arrays or other data types
            if hasattr(text_instance, 'item'):
                text_instance = str(text_instance.item())
            elif hasattr(text_instance, 'tolist'):
                text_instance = ' '.join(str(x) for x in text_instance.tolist() if x)
            else:
                text_instance = str(text_instance)
        
        words = text_instance.split()
        n_words = len(words)
        
        if n_words == 0:
            return np.array([]), []
        
        # Limit to first 100 words for computational efficiency
        max_words = 100
        if n_words > max_words:
            words = words[:max_words]
            n_words = max_words
        
        # Simulate attention patterns based on:
        # 1. Word position (attention often focuses on certain positions)
        # 2. Word length (longer words often get more attention)
        # 3. Word frequency (rare words often get more attention)
        # 4. Contextual influence (words that change prediction when masked)
        
        attention_scores = np.zeros(n_words)
        
        # 1. Position-based attention (beginning and end often important)
        for i in range(n_words):
            position_weight = 1.0
            if i < 3:  # Beginning words
                position_weight = 1.5
            elif i >= n_words - 3:  # End words
                position_weight = 1.3
            elif i == n_words // 2:  # Middle word
                position_weight = 1.2
            
            attention_scores[i] += position_weight
        
        # 2. Length-based attention (longer words often more informative)
        for i, word in enumerate(words):
            length_weight = min(len(word) / 8.0, 2.0)  # Cap at 2.0
            attention_scores[i] += length_weight
        
        # 3. Frequency-based attention (estimate rarity from training data)
        word_frequencies = {}
        if isinstance(X_train, list):
            total_words = 0
            for text in X_train[:100]:  # Sample for efficiency
                if isinstance(text, str):
                    train_words = text.lower().split()
                    total_words += len(train_words)
                    for word in train_words:
                        word_frequencies[word] = word_frequencies.get(word, 0) + 1
        
        for i, word in enumerate(words):
            word_lower = word.lower()
            frequency = word_frequencies.get(word_lower, 1)
            # Rare words get higher attention (inverse frequency)
            rarity_weight = np.log(max(total_words / frequency, 1)) if total_words > 0 else 1.0
            attention_scores[i] += min(rarity_weight, 3.0)  # Cap at 3.0
        
        # 4. Contextual influence (proxy: masking impact)
        try:
            base_prediction = self.model.predict([text_instance])[0]
            
            for i in range(n_words):
                # Create masked version by replacing word with [MASK] or removing it
                masked_words = words.copy()
                masked_words[i] = '[MASK]'  # or could remove entirely
                masked_text = ' '.join(masked_words)
                
                try:
                    masked_prediction = self.model.predict([masked_text])[0]
                    influence = abs(base_prediction - masked_prediction)
                    attention_scores[i] += influence * 2.0  # Scale contextual influence
                except Exception:
                    # If masking fails, use position-based fallback
                    pass
        except Exception as e:
            self.logger.warning(f"Error calculating contextual influence: {e}")
        
        # Normalize attention scores to sum to 1 (like real attention)
        if np.sum(attention_scores) > 0:
            attention_scores = attention_scores / np.sum(attention_scores)
        else:
            attention_scores = np.ones(n_words) / n_words
        
        # Convert to importance scores (multiply by prediction confidence)
        try:
            prediction_confidence = abs(self.model.predict([text_instance])[0] - 0.5) * 2  # Scale to 0-1
            importance_scores = attention_scores * prediction_confidence
        except Exception:
            importance_scores = attention_scores
        
        # Create feature names
        feature_names = [f"word_{i}_{word}" for i, word in enumerate(words)]
        
        return importance_scores, feature_names
    
    def _extract_model_attention(self, text_instance):
        """
        Extract actual attention weights from the model if available.
        This is a placeholder for models that expose attention weights.
        
        Returns:
            attention_weights: numpy array of attention scores per token
        """
        # This is where you would extract attention weights from models like:
        # - Transformers with attention mechanisms
        # - BERT, GPT, etc.
        # - Custom models that expose attention
        
        # Example for transformers library (if available):
        # try:
        #     from transformers import AutoTokenizer, AutoModel
        #     tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        #     model = AutoModel.from_pretrained(self.model_name, output_attentions=True)
        #     
        #     inputs = tokenizer(text_instance, return_tensors="pt")
        #     outputs = model(**inputs)
        #     attentions = outputs.attentions  # List of attention tensors
        #     
        #     # Aggregate attention across layers and heads
        #     # Return aggregated attention weights
        # except ImportError:
        #     pass
        
        # For now, return None to indicate no model attention available
        return None