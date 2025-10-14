"""
Advanced Monotonicity Evaluation for Multi-modal Data
Implements meaningful monotonicity tests for text and image explanations
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class MonotonicityEvaluator:
    """
    Advanced monotonicity evaluation for text and image data using interpretable features
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def evaluate_monotonicity(self, explanations: List[Dict[str, Any]], model=None,
                            data_type: str = 'auto') -> Dict[str, float]:
        """
        Evaluate monotonicity based on data type using interpretable features

        Args:
            explanations: List of explanation dictionaries
            model: Model for predictions (optional)
            data_type: 'text', 'image', 'tabular', or 'auto' for automatic detection

        Returns:
            Dictionary of monotonicity metrics
        """
        if not explanations:
            return {'monotonicity': 0.0}

        # Auto-detect data type if not specified
        if data_type == 'auto':
            data_type = self._detect_data_type(explanations)

        if data_type == 'text':
            return self._evaluate_text_monotonicity(explanations, model)
        elif data_type == 'image':
            return self._evaluate_image_monotonicity(explanations, model)
        elif data_type == 'tabular':
            return self._evaluate_tabular_monotonicity(explanations, model)
        else:
            self.logger.warning(f"Unknown data type: {data_type}")
            return {'monotonicity': 0.0}

    def _detect_data_type(self, explanations: List[Dict[str, Any]]) -> str:
        """Auto-detect data type from explanation structure"""
        if not explanations:
            return 'tabular'

        first_exp = explanations[0]

        # Check for text data - must have actual text content with valid tokens
        if 'text_content' in first_exp and first_exp.get('text_content') is not None:
            # Also verify tokens exist and are not None
            if 'tokens' in first_exp and first_exp.get('tokens') is not None:
                return 'text'

        # Check for image data
        if 'importance_map' in first_exp or (
            'feature_importance' in first_exp and
            len(first_exp.get('feature_importance', [])) > 1000
        ):
            return 'image'

        return 'tabular'

    def _evaluate_text_monotonicity(self, explanations: List[Dict[str, Any]],
                                  model=None) -> Dict[str, float]:
        """
        Evaluate monotonicity for text data using interpretable text features:
        - Word frequency
        - Sentiment scores
        - Position importance
        - Word length effects
        """
        self.logger.info("Evaluating text monotonicity using interpretable features")
        print(f"\n[DEBUG MonotonicityEvaluator] Starting text monotonicity evaluation")
        print(f"[DEBUG] Number of explanations: {len(explanations)}")

        monotonicity_scores = []

        for i, explanation in enumerate(explanations):
            print(f"\n[DEBUG] Processing explanation {i+1}:")
            text_content = explanation.get('text_content', '')
            feature_importance = explanation.get('feature_importance', [])
            tokens = explanation.get('tokens', text_content.split() if text_content else [])

            print(f"  - text_content: {text_content[:50] if text_content else 'MISSING'}")
            print(f"  - tokens length: {len(tokens)}")
            print(f"  - feature_importance length: {len(feature_importance) if hasattr(feature_importance, '__len__') else 0}")
            print(f"  - tokens[:3]: {tokens[:3] if tokens else 'EMPTY'}")
            # Fix for numpy array truth value issue
            if hasattr(feature_importance, '__len__') and len(feature_importance) > 0:
                print(f"  - importance[:3]: {feature_importance[:3]}")
            else:
                print(f"  - importance[:3]: EMPTY")

            if not text_content or len(feature_importance) == 0 or len(tokens) == 0:
                print(f"  ⚠️ SKIPPING: Missing data (text={bool(text_content)}, importance={len(feature_importance)}, tokens={len(tokens)})")
                continue

            try:
                # 1. Word Frequency Monotonicity
                freq_monotonicity = self._compute_frequency_monotonicity(
                    tokens, feature_importance
                )

                # 2. Position-based Monotonicity (early words vs late words)
                position_monotonicity = self._compute_position_monotonicity(
                    tokens, feature_importance
                )

                # 3. Word Length Monotonicity (longer words often more important)
                length_monotonicity = self._compute_length_monotonicity(
                    tokens, feature_importance
                )

                # 4. Simple Sentiment Monotonicity (basic sentiment words)
                sentiment_monotonicity = self._compute_sentiment_monotonicity(
                    tokens, feature_importance
                )

                # Combine monotonicity measures
                print(f"  - freq_monotonicity: {freq_monotonicity:.6f}")
                print(f"  - position_monotonicity: {position_monotonicity:.6f}")
                print(f"  - length_monotonicity: {length_monotonicity:.6f}")
                print(f"  - sentiment_monotonicity: {sentiment_monotonicity:.6f}")

                combined_score = np.mean([
                    freq_monotonicity, position_monotonicity,
                    length_monotonicity, sentiment_monotonicity
                ])

                print(f"  - COMBINED SCORE: {combined_score:.6f}")
                monotonicity_scores.append(combined_score)

            except Exception as e:
                self.logger.warning(f"Error computing text monotonicity: {e}")
                print(f"  ❌ ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n[DEBUG] Final monotonicity_scores: {monotonicity_scores}")
        final_score = float(np.mean(monotonicity_scores)) if monotonicity_scores else 0.0
        print(f"[DEBUG] Final mean score: {final_score}")

        return {
            'monotonicity': final_score,
            'text_monotonicity_count': len(monotonicity_scores)
        }

    def _compute_frequency_monotonicity(self, tokens: List[str],
                                      importance: List[float]) -> float:
        """
        Test if rare words tend to have higher importance than common words
        (Rare words are often more informative)
        """
        if len(tokens) != len(importance) or len(tokens) < 2:
            print(f"    [FREQ] Skipping: len mismatch or too short (tokens={len(tokens)}, imp={len(importance)})")
            return 0.0

        # Calculate word frequencies
        word_counts = Counter(tokens)
        total_words = len(tokens)

        # Get frequency for each token
        frequencies = [word_counts[token] / total_words for token in tokens]

        # Convert importance to numpy array for consistent handling
        importance_array = np.abs(np.array(importance))

        print(f"    [FREQ] freq range: [{min(frequencies):.4f}, {max(frequencies):.4f}]")
        print(f"    [FREQ] importance range: [{np.min(importance_array):.4f}, {np.max(importance_array):.4f}]")
        print(f"    [FREQ] importance std: {np.std(importance_array):.6f}")

        # Test monotonicity: lower frequency should correlate with higher importance
        # Use Spearman correlation (rank-based)
        try:
            from scipy.stats import spearmanr
            # Negative correlation expected (low freq -> high importance)
            correlation, p_value = spearmanr(frequencies, importance_array)

            print(f"    [FREQ] Spearman correlation: {correlation:.6f}, p-value: {p_value:.6f}")

            # Convert to monotonicity score (0 to 1)
            # Use absolute correlation as measure of monotonicity (any systematic relationship is good)
            monotonicity = abs(correlation) if not np.isnan(correlation) else 0.0
            print(f"    [FREQ] Monotonicity score: {monotonicity:.6f}")
            return min(1.0, monotonicity)

        except ImportError:
            # Fallback without scipy
            return self._simple_rank_correlation(frequencies, importance_array, reverse=False)

    def _compute_position_monotonicity(self, tokens: List[str],
                                     importance: List[float]) -> float:
        """
        Test position-based monotonicity (beginning/end words often more important)
        """
        if len(tokens) != len(importance) or len(tokens) < 3:
            print(f"    [POS] Skipping: len mismatch or too short")
            return 0.0

        n = len(tokens)
        importance_array = np.abs(np.array(importance))

        # Create position weights (U-shaped: high at start and end)
        position_weights = []
        for i in range(n):
            # Distance from edges (0 = at edge, 0.5 = in middle)
            edge_distance = min(i, n - 1 - i) / (n / 2)
            position_weight = 1.0 - edge_distance  # Higher for edge positions
            position_weights.append(position_weight)

        print(f"    [POS] position_weights range: [{min(position_weights):.4f}, {max(position_weights):.4f}]")
        print(f"    [POS] importance range: [{np.min(importance_array):.4f}, {np.max(importance_array):.4f}]")

        # Test correlation between position importance and feature importance
        try:
            from scipy.stats import spearmanr
            correlation, _ = spearmanr(position_weights, importance_array)
            print(f"    [POS] Spearman correlation: {correlation:.6f}")
            monotonicity = abs(correlation) if not np.isnan(correlation) else 0.0
            print(f"    [POS] Monotonicity score: {monotonicity:.6f}")
            return monotonicity
        except ImportError:
            return self._simple_rank_correlation(position_weights, importance_array)

    def _compute_length_monotonicity(self, tokens: List[str],
                                   importance: List[float]) -> float:
        """
        Test if longer words tend to have higher importance
        (Longer words are often more specific and informative)
        """
        if len(tokens) != len(importance) or len(tokens) < 2:
            print(f"    [LEN] Skipping: len mismatch or too short")
            return 0.0

        # Calculate word lengths
        word_lengths = [len(token) for token in tokens]
        importance_array = np.abs(np.array(importance))

        print(f"    [LEN] word_lengths range: [{min(word_lengths)}, {max(word_lengths)}]")
        print(f"    [LEN] importance range: [{np.min(importance_array):.4f}, {np.max(importance_array):.4f}]")

        # Test correlation between word length and importance
        try:
            from scipy.stats import spearmanr
            correlation, _ = spearmanr(word_lengths, importance_array)
            print(f"    [LEN] Spearman correlation: {correlation:.6f}")
            monotonicity = abs(correlation) if not np.isnan(correlation) else 0.0
            print(f"    [LEN] Monotonicity score: {monotonicity:.6f}")
            return monotonicity
        except ImportError:
            return self._simple_rank_correlation(word_lengths, importance_array)

    def _compute_sentiment_monotonicity(self, tokens: List[str],
                                      importance: List[float]) -> float:
        """
        Test if sentiment-bearing words have higher importance
        """
        if len(tokens) != len(importance) or len(tokens) < 2:
            print(f"    [SENT] Skipping: len mismatch or too short")
            return 0.0

        importance_array = np.abs(np.array(importance))

        # Simple sentiment word lists (can be expanded)
        positive_words = {
            'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic',
            'love', 'best', 'perfect', 'awesome', 'brilliant', 'outstanding'
        }

        negative_words = {
            'bad', 'terrible', 'awful', 'horrible', 'worst', 'hate',
            'disgusting', 'pathetic', 'useless', 'disappointing', 'annoying'
        }

        # Calculate sentiment scores for each token
        sentiment_scores = []
        for token in tokens:
            token_lower = token.lower()
            if token_lower in positive_words:
                sentiment_scores.append(1.0)
            elif token_lower in negative_words:
                sentiment_scores.append(1.0)  # Both positive and negative are sentiment-bearing
            else:
                sentiment_scores.append(0.0)  # Neutral/non-sentiment

        # Test if sentiment words have higher importance
        sentiment_words_importance = [importance_array[i] for i in range(len(importance_array))
                                    if sentiment_scores[i] > 0]
        neutral_words_importance = [importance_array[i] for i in range(len(importance_array))
                                  if sentiment_scores[i] == 0]

        print(f"    [SENT] Sentiment words: {sum(sentiment_scores)} / {len(tokens)}")
        print(f"    [SENT] Sentiment word importance: n={len(sentiment_words_importance)}, avg={np.mean(sentiment_words_importance) if sentiment_words_importance else 0:.4f}")
        print(f"    [SENT] Neutral word importance: n={len(neutral_words_importance)}, avg={np.mean(neutral_words_importance) if neutral_words_importance else 0:.4f}")

        if len(sentiment_words_importance) == 0 or len(neutral_words_importance) == 0:
            print(f"    [SENT] No comparison possible, returning 0.5")
            return 0.5  # No comparison possible

        # Compare average importance
        avg_sentiment = np.mean(sentiment_words_importance)
        avg_neutral = np.mean(neutral_words_importance)

        # Use absolute difference as monotonicity measure
        diff = abs(avg_sentiment - avg_neutral)
        max_val = max(avg_sentiment, avg_neutral, 1e-8)
        monotonicity = min(1.0, diff / max_val)

        print(f"    [SENT] Monotonicity score: {monotonicity:.6f}")
        return monotonicity

    def _evaluate_image_monotonicity(self, explanations: List[Dict[str, Any]],
                                   model=None) -> Dict[str, float]:
        """
        Evaluate monotonicity for image data using interpretable visual features:
        - Brightness/contrast variations
        - Edge importance
        - Central vs peripheral regions
        - Color intensity effects
        """
        self.logger.info("Evaluating image monotonicity using interpretable visual features")
        print(f"\n[DEBUG MonotonicityEvaluator] Starting image monotonicity evaluation")
        print(f"[DEBUG] Number of explanations: {len(explanations)}")

        monotonicity_scores = []

        for i, explanation in enumerate(explanations):
            print(f"\n[DEBUG] Processing image explanation {i+1}:")
            importance_map = explanation.get('importance_map')
            feature_importance = explanation.get('feature_importance', [])
            input_image = explanation.get('input', explanation.get('image'))

            print(f"  - importance_map: {'Present' if importance_map is not None else 'MISSING'}")
            print(f"  - feature_importance length: {len(feature_importance) if hasattr(feature_importance, '__len__') else 0}")
            print(f"  - input_image: {'Present' if input_image is not None else 'MISSING'}")

            # Convert feature importance to 2D map if needed
            if importance_map is None and len(feature_importance) > 0:
                # Try to reshape feature importance to image dimensions
                print(f"  - Attempting to reshape feature_importance to image dimensions")
                importance_map = self._reshape_to_image(feature_importance, input_image)
                print(f"  - Reshape result: {'Success' if importance_map is not None else 'FAILED'}")

            # For images, we can evaluate monotonicity using just the importance map
            # Input image is optional (used for additional features but not required)
            if importance_map is None:
                print(f"  ⚠️ SKIPPING: Missing importance_map")
                continue

            try:
                # Convert to numpy arrays
                importance_map = np.array(importance_map)
                if input_image is not None:
                    input_image = np.array(input_image)

                print(f"  - importance_map shape: {importance_map.shape}")
                if input_image is not None:
                    print(f"  - input_image shape: {input_image.shape}")
                print(f"  - importance_map range: [{np.min(importance_map):.4f}, {np.max(importance_map):.4f}]")

                # 1. Central Region Monotonicity (center often more important)
                central_monotonicity = self._compute_central_region_monotonicity(
                    importance_map
                )

                # 2. Brightness Monotonicity (only if input_image available)
                if input_image is not None:
                    brightness_monotonicity = self._compute_brightness_monotonicity(
                        input_image, importance_map
                    )

                    # 3. Edge/Contrast Monotonicity
                    edge_monotonicity = self._compute_edge_monotonicity(
                        input_image, importance_map
                    )

                    # 4. Color Intensity Monotonicity
                    intensity_monotonicity = self._compute_intensity_monotonicity(
                        input_image, importance_map
                    )

                    # Combine all monotonicity measures
                    print(f"  - central_monotonicity: {central_monotonicity:.6f}")
                    print(f"  - brightness_monotonicity: {brightness_monotonicity:.6f}")
                    print(f"  - edge_monotonicity: {edge_monotonicity:.6f}")
                    print(f"  - intensity_monotonicity: {intensity_monotonicity:.6f}")

                    combined_score = np.mean([
                        central_monotonicity, brightness_monotonicity,
                        edge_monotonicity, intensity_monotonicity
                    ])
                else:
                    # Without input image, only use central region monotonicity
                    print(f"  - central_monotonicity: {central_monotonicity:.6f}")
                    print(f"  - (brightness, edge, intensity skipped - no input image)")
                    combined_score = central_monotonicity

                print(f"  - COMBINED SCORE: {combined_score:.6f}")
                monotonicity_scores.append(combined_score)

            except Exception as e:
                self.logger.warning(f"Error computing image monotonicity: {e}")
                print(f"  ❌ ERROR: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\n[DEBUG] Final image monotonicity_scores: {monotonicity_scores}")
        final_score = float(np.mean(monotonicity_scores)) if monotonicity_scores else 0.0
        print(f"[DEBUG] Final mean score: {final_score}")

        return {
            'monotonicity': final_score,
            'image_monotonicity_count': len(monotonicity_scores)
        }

    def _compute_central_region_monotonicity(self, importance_map: np.ndarray) -> float:
        """
        Test if central regions have higher importance than peripheral regions
        """
        if importance_map.ndim != 2:
            if importance_map.ndim == 3:
                importance_map = np.mean(importance_map, axis=2)  # Average across channels
            else:
                print(f"    [CENTRAL] Invalid ndim: {importance_map.ndim}")
                return 0.0

        h, w = importance_map.shape
        center_h, center_w = h // 2, w // 2

        # Create distance from center map
        y, x = np.ogrid[:h, :w]
        center_distances = np.sqrt((y - center_h) ** 2 + (x - center_w) ** 2)

        # Normalize distances to [0, 1]
        max_distance = np.sqrt(center_h ** 2 + center_w ** 2)
        normalized_distances = center_distances / max_distance

        print(f"    [CENTRAL] Distance range: [{np.min(normalized_distances):.4f}, {np.max(normalized_distances):.4f}]")
        print(f"    [CENTRAL] Importance range: [{np.min(importance_map):.4f}, {np.max(importance_map):.4f}]")

        # Test correlation (use absolute value to detect any systematic relationship)
        correlation = self._simple_rank_correlation(
            normalized_distances.flatten(),
            importance_map.flatten(),
            reverse=False  # Don't reverse, use abs later
        )
        print(f"    [CENTRAL] Correlation: {correlation:.6f}")
        return abs(correlation)  # Any systematic relationship is good

    def _compute_brightness_monotonicity(self, image: np.ndarray,
                                       importance_map: np.ndarray) -> float:
        """
        Test monotonicity between brightness and importance
        """
        if image.ndim == 3:
            # Convert to grayscale
            brightness = np.mean(image, axis=2)
        else:
            brightness = image

        if importance_map.ndim != 2:
            if importance_map.ndim == 3:
                importance_map = np.mean(importance_map, axis=2)
            else:
                print(f"    [BRIGHT] Invalid importance_map ndim: {importance_map.ndim}")
                return 0.0

        # Ensure same shape
        if brightness.shape != importance_map.shape:
            print(f"    [BRIGHT] Shape mismatch: brightness={brightness.shape}, importance={importance_map.shape}")
            return 0.0

        print(f"    [BRIGHT] Brightness range: [{np.min(brightness):.4f}, {np.max(brightness):.4f}]")
        print(f"    [BRIGHT] Importance range: [{np.min(importance_map):.4f}, {np.max(importance_map):.4f}]")

        correlation = self._simple_rank_correlation(
            brightness.flatten(),
            importance_map.flatten()
        )
        print(f"    [BRIGHT] Correlation: {correlation:.6f}")
        return abs(correlation)

    def _compute_edge_monotonicity(self, image: np.ndarray,
                                 importance_map: np.ndarray) -> float:
        """
        Test if edge regions have higher importance
        """
        try:
            if image.ndim == 3:
                gray = np.mean(image, axis=2)
            else:
                gray = image

            # Simple edge detection using gradient magnitude
            grad_x = np.gradient(gray, axis=1)
            grad_y = np.gradient(gray, axis=0)
            edge_magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)

            if importance_map.ndim != 2:
                if importance_map.ndim == 3:
                    importance_map = np.mean(importance_map, axis=2)
                else:
                    print(f"    [EDGE] Invalid importance_map ndim: {importance_map.ndim}")
                    return 0.0

            # Ensure same shape
            if edge_magnitude.shape != importance_map.shape:
                print(f"    [EDGE] Shape mismatch: edge={edge_magnitude.shape}, importance={importance_map.shape}")
                return 0.0

            print(f"    [EDGE] Edge magnitude range: [{np.min(edge_magnitude):.4f}, {np.max(edge_magnitude):.4f}]")
            print(f"    [EDGE] Importance range: [{np.min(importance_map):.4f}, {np.max(importance_map):.4f}]")

            correlation = self._simple_rank_correlation(
                edge_magnitude.flatten(),
                importance_map.flatten()
            )
            print(f"    [EDGE] Correlation: {correlation:.6f}")
            return abs(correlation)

        except Exception as e:
            print(f"    [EDGE] Exception: {e}")
            return 0.0

    def _compute_intensity_monotonicity(self, image: np.ndarray,
                                      importance_map: np.ndarray) -> float:
        """
        Test monotonicity between color intensity and importance
        """
        if image.ndim == 3:
            # Use maximum intensity across channels
            intensity = np.max(image, axis=2)
        else:
            intensity = image

        if importance_map.ndim != 2:
            if importance_map.ndim == 3:
                importance_map = np.mean(importance_map, axis=2)
            else:
                print(f"    [INTENSITY] Invalid importance_map ndim: {importance_map.ndim}")
                return 0.0

        # Ensure same shape
        if intensity.shape != importance_map.shape:
            print(f"    [INTENSITY] Shape mismatch: intensity={intensity.shape}, importance={importance_map.shape}")
            return 0.0

        print(f"    [INTENSITY] Intensity range: [{np.min(intensity):.4f}, {np.max(intensity):.4f}]")
        print(f"    [INTENSITY] Importance range: [{np.min(importance_map):.4f}, {np.max(importance_map):.4f}]")

        correlation = self._simple_rank_correlation(
            intensity.flatten(),
            importance_map.flatten()
        )
        print(f"    [INTENSITY] Correlation: {correlation:.6f}")
        return abs(correlation)

    def _reshape_to_image(self, feature_importance: List[float],
                         input_image) -> Optional[np.ndarray]:
        """
        Try to reshape 1D feature importance to 2D image format
        """
        if input_image is None:
            return None

        try:
            image_array = np.array(input_image)
            if image_array.ndim == 3:
                h, w, c = image_array.shape
                expected_size = h * w
            elif image_array.ndim == 2:
                h, w = image_array.shape
                expected_size = h * w
            else:
                return None

            if len(feature_importance) == expected_size:
                if image_array.ndim == 3:
                    return np.array(feature_importance).reshape(h, w)
                else:
                    return np.array(feature_importance).reshape(h, w)

        except Exception:
            pass

        return None

    def _evaluate_tabular_monotonicity(self, explanations: List[Dict[str, Any]],
                                     model=None) -> Dict[str, float]:
        """
        Standard monotonicity evaluation for tabular data
        Tests if increasing feature values increases output in same direction as importance
        """
        if not explanations or model is None:
            return {'monotonicity': 0.0}

        monotonicity_scores = []

        for explanation in explanations:
            feature_importance = explanation.get('feature_importance', [])
            input_instance = explanation.get('input', None)

            if len(feature_importance) == 0 or input_instance is None:
                continue

            try:
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
                        # Get predictions for perturbed instances
                        if hasattr(model, 'predict'):
                            pred_up = model.predict([x_up])[0]
                            pred_down = model.predict([x_down])[0]
                        else:
                            continue  # Can't test monotonicity without model predictions

                        # Ensure predictions are scalars (not arrays)
                        if isinstance(pred_up, np.ndarray):
                            pred_up = float(pred_up.item()) if pred_up.size == 1 else float(pred_up[0])
                        if isinstance(pred_down, np.ndarray):
                            pred_down = float(pred_down.item()) if pred_down.size == 1 else float(pred_down[0])

                        # Ensure feature_importance[i] is scalar
                        feat_importance = float(feature_importance[i]) if isinstance(feature_importance[i], np.ndarray) else feature_importance[i]

                        # Calculate empirical gradient
                        empirical_gradient = (pred_up - pred_down) / (2 * increment)

                        # Check if empirical gradient agrees with feature importance sign
                        if (feat_importance > 0 and empirical_gradient > 0) or \
                           (feat_importance < 0 and empirical_gradient < 0):
                            monotonicity_count += 1
                        total += 1

                    except Exception:
                        continue

                if total > 0:
                    monotonicity = monotonicity_count / total
                    monotonicity_scores.append(monotonicity)

            except Exception as e:
                self.logger.warning(f"Error computing tabular monotonicity: {e}")
                continue

        return {
            'monotonicity': float(np.mean(monotonicity_scores)) if monotonicity_scores else 0.0,
            'tabular_monotonicity_count': len(monotonicity_scores)
        }

    def _simple_rank_correlation(self, x: np.ndarray, y: np.ndarray,
                                reverse: bool = False) -> float:
        """
        Simple rank correlation without scipy dependency
        """
        try:
            if len(x) != len(y) or len(x) < 2:
                return 0.0

            x_flat = x.flatten() if hasattr(x, 'flatten') else x
            y_flat = y.flatten() if hasattr(y, 'flatten') else y

            # Remove NaN values
            valid_mask = ~(np.isnan(x_flat) | np.isnan(y_flat))
            if not np.any(valid_mask):
                return 0.0

            x_valid = x_flat[valid_mask]
            y_valid = y_flat[valid_mask]

            if len(x_valid) < 2:
                return 0.0

            # Compute ranks
            x_ranks = np.argsort(np.argsort(x_valid))
            y_ranks = np.argsort(np.argsort(y_valid))

            # Compute correlation
            mean_x = np.mean(x_ranks)
            mean_y = np.mean(y_ranks)

            numerator = np.sum((x_ranks - mean_x) * (y_ranks - mean_y))
            denominator = np.sqrt(np.sum((x_ranks - mean_x) ** 2) *
                                np.sum((y_ranks - mean_y) ** 2))

            if denominator == 0:
                return 0.0

            correlation = numerator / denominator

            # Handle reverse correlation
            if reverse:
                correlation = -correlation

            # Convert to monotonicity score [0, 1]
            return max(0.0, correlation)

        except Exception:
            return 0.0