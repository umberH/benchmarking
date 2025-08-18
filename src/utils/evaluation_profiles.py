"""
Evaluation Metrics Profiles
Defines which evaluation metrics are appropriate for different types of explanation methods
"""

from typing import Dict, List, Set, Optional
from dataclasses import dataclass
from enum import Enum
import logging


class MetricCategory(Enum):
    """Categories of evaluation metrics"""
    FIDELITY = "fidelity"           # How faithful explanations are to model behavior
    STABILITY = "stability"         # How consistent explanations are
    COMPREHENSIBILITY = "comprehensibility"  # How easy to understand
    EFFICIENCY = "efficiency"       # Computational aspects


@dataclass
class MetricInfo:
    """Information about an evaluation metric"""
    name: str
    category: MetricCategory
    description: str
    applicable_to: Set[str]  # Method categories this applies to
    requires_feature_importance: bool = True
    requires_baseline: bool = False
    requires_instance_data: bool = False
    output_range: tuple = (0.0, 1.0)
    higher_is_better: bool = True


class EvaluationProfileManager:
    """
    Manager for evaluation metric profiles and compatibility
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = self._initialize_metrics_registry()
        self.profiles = self._initialize_method_profiles()
    
    def _initialize_metrics_registry(self) -> Dict[str, MetricInfo]:
        """Initialize the registry of available evaluation metrics"""
        metrics = {}
        
        # Fidelity Metrics
        metrics['faithfulness'] = MetricInfo(
            name='faithfulness',
            category=MetricCategory.FIDELITY,
            description='Measures how much predictions change when important features are removed',
            applicable_to={'feature_attribution', 'perturbation_based', 'example_based'},
            requires_feature_importance=False,  # Different implementations for different methods
            requires_instance_data=True,
            output_range=(0.0, 1.0),
            higher_is_better=True
        )
        
        metrics['monotonicity'] = MetricInfo(
            name='monotonicity',
            category=MetricCategory.FIDELITY,
            description='Checks if feature importance direction matches actual model gradients',
            applicable_to={'feature_attribution', 'perturbation_based'},
            requires_feature_importance=True,
            requires_instance_data=True,
            output_range=(0.0, 1.0),
            higher_is_better=True
        )
        
        metrics['completeness'] = MetricInfo(
            name='completeness',
            category=MetricCategory.FIDELITY,
            description='For SHAP/IG: checks if attributions sum to prediction difference',
            applicable_to={'feature_attribution'},
            requires_feature_importance=True,
            requires_baseline=True,
            output_range=(0.0, 1.0),
            higher_is_better=True
        )
        
        # Stability Metrics
        metrics['stability'] = MetricInfo(
            name='stability',
            category=MetricCategory.STABILITY,
            description='Measures consistency of explanations across similar inputs',
            applicable_to={'feature_attribution', 'example_based', 'perturbation_based', 'concept_based'},
            requires_feature_importance=False,  # Method-dependent
            output_range=(0.0, 1.0),
            higher_is_better=True
        )
        
        metrics['consistency'] = MetricInfo(
            name='consistency',
            category=MetricCategory.STABILITY,
            description='Measures rank correlation between different explanations',
            applicable_to={'feature_attribution', 'example_based', 'perturbation_based'},
            requires_feature_importance=False,  # Method-dependent
            output_range=(0.0, 1.0),
            higher_is_better=True
        )
        
        # Comprehensibility Metrics
        metrics['sparsity'] = MetricInfo(
            name='sparsity',
            category=MetricCategory.COMPREHENSIBILITY,
            description='Measures how many features are actually important (fewer is better)',
            applicable_to={'feature_attribution', 'perturbation_based'},
            requires_feature_importance=True,
            output_range=(0.0, 1.0),
            higher_is_better=True  # Higher sparsity = fewer important features = more interpretable
        )
        
        metrics['simplicity'] = MetricInfo(
            name='simplicity',
            category=MetricCategory.COMPREHENSIBILITY,
            description='Measures concentration of importance using Gini coefficient',
            applicable_to={'feature_attribution', 'perturbation_based'},
            requires_feature_importance=True,
            output_range=(0.0, 1.0),
            higher_is_better=True
        )
        
        # Efficiency Metrics
        metrics['time_complexity'] = MetricInfo(
            name='time_complexity',
            category=MetricCategory.EFFICIENCY,
            description='Time taken per explanation (seconds)',
            applicable_to={'feature_attribution', 'example_based', 'perturbation_based', 'concept_based'},
            requires_feature_importance=False,
            output_range=(0.0, float('inf')),
            higher_is_better=False  # Lower time is better
        )
        
        # Text-Specific Metrics
        metrics['semantic_coherence'] = MetricInfo(
            name='semantic_coherence',
            category=MetricCategory.COMPREHENSIBILITY,
            description='Measures if important words make semantic sense together',
            applicable_to={'feature_attribution', 'perturbation_based'},
            requires_feature_importance=True,
            output_range=(0.0, 1.0),
            higher_is_better=True
        )
        
        metrics['syntax_awareness'] = MetricInfo(
            name='syntax_awareness',
            category=MetricCategory.FIDELITY,
            description='Evaluates if method respects linguistic structure (content vs function words)',
            applicable_to={'feature_attribution', 'perturbation_based'},
            requires_feature_importance=True,
            output_range=(0.0, 1.0),
            higher_is_better=True
        )
        
        metrics['context_sensitivity'] = MetricInfo(
            name='context_sensitivity',
            category=MetricCategory.FIDELITY,
            description='Measures if same words get different importance in different contexts',
            applicable_to={'feature_attribution', 'perturbation_based'},
            requires_feature_importance=True,
            output_range=(0.0, 1.0),
            higher_is_better=True
        )
        
        metrics['word_significance'] = MetricInfo(
            name='word_significance',
            category=MetricCategory.FIDELITY,
            description='Evaluates if important words are actually significant (not just frequent)',
            applicable_to={'feature_attribution', 'perturbation_based'},
            requires_feature_importance=True,
            output_range=(0.0, 1.0),
            higher_is_better=True
        )
        
        metrics['explanation_coverage'] = MetricInfo(
            name='explanation_coverage',
            category=MetricCategory.COMPREHENSIBILITY,
            description='Measures what fraction of text is covered by important words',
            applicable_to={'feature_attribution', 'perturbation_based'},
            requires_feature_importance=True,
            output_range=(0.0, 1.0),
            higher_is_better=True
        )
        
        metrics['sentiment_consistency'] = MetricInfo(
            name='sentiment_consistency',
            category=MetricCategory.FIDELITY,
            description='Evaluates if important words align with the sentiment/prediction',
            applicable_to={'feature_attribution', 'perturbation_based'},
            requires_feature_importance=True,
            output_range=(0.0, 1.0),
            higher_is_better=True
        )
        
        # Advanced Axiom-Based Metrics
        metrics['advanced_identity'] = MetricInfo(
            name='advanced_identity',
            category=MetricCategory.FIDELITY,
            description='Identity Axiom: explanations should be zero when input equals baseline',
            applicable_to={'feature_attribution', 'perturbation_based', 'example_based', 'concept_based'},
            requires_feature_importance=True,
            requires_baseline=True,
            output_range=(0.0, 1.0),
            higher_is_better=True
        )
        
        metrics['advanced_separability'] = MetricInfo(
            name='advanced_separability',
            category=MetricCategory.FIDELITY,
            description='Separability Axiom: different features should have distinguishable importance',
            applicable_to={'feature_attribution', 'perturbation_based'},
            requires_feature_importance=True,
            output_range=(0.0, 1.0),
            higher_is_better=True
        )
        
        metrics['advanced_non_sensitivity'] = MetricInfo(
            name='advanced_non_sensitivity',
            category=MetricCategory.STABILITY,
            description='Non-sensitivity Axiom: small input changes should yield similar explanations',
            applicable_to={'feature_attribution', 'perturbation_based', 'example_based'},
            requires_feature_importance=True,
            requires_instance_data=True,
            output_range=(0.0, 1.0),
            higher_is_better=True
        )
        
        metrics['advanced_compactness'] = MetricInfo(
            name='advanced_compactness',
            category=MetricCategory.COMPREHENSIBILITY,
            description='Compactness Axiom: explanations should be sparse and focused on important features',
            applicable_to={'feature_attribution', 'perturbation_based'},
            requires_feature_importance=True,
            output_range=(0.0, 1.0),
            higher_is_better=True
        )
        
        # Advanced Information-Theoretic Metrics
        metrics['advanced_correctness'] = MetricInfo(
            name='advanced_correctness',
            category=MetricCategory.FIDELITY,
            description='Information gain provided by the explanation about the prediction',
            applicable_to={'feature_attribution', 'perturbation_based', 'example_based', 'concept_based'},
            requires_feature_importance=True,
            output_range=(0.0, 1.0),
            higher_is_better=True
        )
        
        metrics['advanced_entropy'] = MetricInfo(
            name='advanced_entropy',
            category=MetricCategory.COMPREHENSIBILITY,
            description='Shannon entropy of feature importance distribution (lower = more focused)',
            applicable_to={'feature_attribution', 'perturbation_based'},
            requires_feature_importance=True,
            output_range=(0.0, 1.0),
            higher_is_better=False  # Lower entropy = more focused = better
        )
        
        metrics['advanced_gini_coefficient'] = MetricInfo(
            name='advanced_gini_coefficient',
            category=MetricCategory.COMPREHENSIBILITY,
            description='Gini coefficient of importance distribution (higher = more concentrated)',
            applicable_to={'feature_attribution', 'perturbation_based'},
            requires_feature_importance=True,
            output_range=(0.0, 1.0),
            higher_is_better=True  # Higher Gini = more concentrated = better interpretability
        )
        
        metrics['advanced_kl_divergence'] = MetricInfo(
            name='advanced_kl_divergence',
            category=MetricCategory.COMPREHENSIBILITY,
            description='KL divergence from uniform distribution (higher = more concentrated)',
            applicable_to={'feature_attribution', 'perturbation_based'},
            requires_feature_importance=True,
            output_range=(0.0, 1.0),
            higher_is_better=True  # Higher KL = farther from uniform = more focused
        )
        
        return metrics
    
    def _initialize_method_profiles(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize evaluation profiles for different method types"""
        profiles = {}
        
        # Feature Attribution Methods Profile
        profiles['feature_attribution'] = {
            'core_metrics': ['faithfulness', 'time_complexity'],
            'fidelity_metrics': ['faithfulness', 'monotonicity'],
            'stability_metrics': ['stability', 'consistency'],
            'comprehensibility_metrics': ['sparsity', 'simplicity'],
            'special_metrics': {
                'shap': ['completeness'],
                'integrated_gradients': ['completeness']
            }
        }
        
        # Example-Based Methods Profile
        profiles['example_based'] = {
            'core_metrics': ['faithfulness', 'time_complexity'],
            'fidelity_metrics': ['faithfulness'],
            'stability_metrics': ['stability', 'consistency'],
            'comprehensibility_metrics': [],  # Not applicable
            'distance_based_metrics': ['stability', 'consistency']  # Custom for this category
        }
        
        # Perturbation-Based Methods Profile
        profiles['perturbation_based'] = {
            'core_metrics': ['faithfulness', 'time_complexity'],
            'fidelity_metrics': ['faithfulness'],
            'stability_metrics': ['stability'],
            'comprehensibility_metrics': ['sparsity', 'simplicity'],  # For spatial importance maps
            'special_metrics': {
                'feature_ablation': ['monotonicity', 'consistency'],  # Has feature importance
                'occlusion': [],  # Spatial method, different metrics needed
                'text_occlusion': ['semantic_coherence', 'syntax_awareness', 'context_sensitivity',
                                  'word_significance', 'explanation_coverage', 'sentiment_consistency']
            }
        }
        
        # Text-Specific Methods Profile
        profiles['text_based'] = {
            'core_metrics': ['faithfulness', 'time_complexity'],
            'fidelity_metrics': ['faithfulness', 'syntax_awareness', 'context_sensitivity',
                               'word_significance', 'sentiment_consistency'],
            'stability_metrics': ['stability', 'consistency'],
            'comprehensibility_metrics': ['semantic_coherence', 'explanation_coverage'],
            'text_specific_metrics': ['semantic_coherence', 'syntax_awareness', 'context_sensitivity',
                                   'word_significance', 'explanation_coverage', 'sentiment_consistency']
        }
        
        # Concept-Based Methods Profile
        profiles['concept_based'] = {
            'core_metrics': ['faithfulness', 'time_complexity'],
            'fidelity_metrics': ['faithfulness'],
            'stability_metrics': ['stability'],
            'comprehensibility_metrics': [],  # Concept-level interpretability
            'concept_metrics': []  # Would need custom concept-level metrics
        }
        
        return profiles
    
    def get_metrics_for_method(self, method_name: str, method_category: str = None, data_type: str = None) -> List[str]:
        """
        Get appropriate evaluation metrics for a specific method
        
        Args:
            method_name: Name of the explanation method
            method_category: Category of the method (if known)
            data_type: Type of data (tabular, text, image) to select appropriate metrics
            
        Returns:
            List of metric names suitable for this method
        """
        # Method-specific overrides
        method_specific_metrics = {
            'shap': ['time_complexity', 'faithfulness', 'monotonicity', 'completeness', 
                    'stability', 'consistency', 'sparsity', 'simplicity'],
            'shap_text': ['time_complexity', 'faithfulness', 'completeness', 'stability', 'consistency',
                         'semantic_coherence', 'syntax_awareness', 'context_sensitivity', 
                         'word_significance', 'explanation_coverage', 'sentiment_consistency'],
            'lime': ['time_complexity', 'faithfulness', 'monotonicity', 
                    'stability', 'consistency', 'sparsity', 'simplicity'],
            'lime_text': ['time_complexity', 'faithfulness', 'stability', 'consistency',
                         'semantic_coherence', 'syntax_awareness', 'context_sensitivity',
                         'word_significance', 'explanation_coverage', 'sentiment_consistency'],
            'integrated_gradients': ['time_complexity', 'faithfulness', 'monotonicity', 'completeness',
                                   'stability', 'consistency', 'sparsity', 'simplicity'],
            'integrated_gradients_text': ['time_complexity', 'faithfulness', 'completeness',
                                        'stability', 'consistency', 'semantic_coherence',
                                        'syntax_awareness', 'context_sensitivity',
                                        'word_significance', 'explanation_coverage', 'sentiment_consistency'],
            'text_occlusion': ['time_complexity', 'faithfulness', 'stability',
                             'semantic_coherence', 'syntax_awareness', 'context_sensitivity',
                             'word_significance', 'explanation_coverage', 'sentiment_consistency'],
            'attention_visualization': ['time_complexity', 'faithfulness', 'stability',
                                      'semantic_coherence', 'syntax_awareness', 'context_sensitivity',
                                      'word_significance', 'explanation_coverage', 'sentiment_consistency'],
            'feature_ablation': ['time_complexity', 'faithfulness', 'monotonicity',
                               'stability', 'consistency', 'sparsity', 'simplicity'],
            'prototype': ['time_complexity', 'faithfulness', 'stability', 'consistency'],
            'counterfactual': ['time_complexity', 'faithfulness', 'stability', 'consistency'],
            'occlusion': ['time_complexity', 'faithfulness', 'stability', 'sparsity', 'simplicity'],
            'tcav': ['time_complexity', 'faithfulness', 'stability'],
            'concept_bottleneck': ['time_complexity', 'faithfulness', 'stability'],
            
            # Advanced Methods (new)
            'causal_shap': ['time_complexity', 'faithfulness', 'monotonicity', 'completeness',
                           'stability', 'consistency', 'sparsity', 'simplicity',
                           'advanced_identity', 'advanced_separability', 'advanced_non_sensitivity',
                           'advanced_compactness', 'advanced_correctness', 'advanced_entropy',
                           'advanced_gini_coefficient', 'advanced_kl_divergence'],
            'shapley_flow': ['time_complexity', 'faithfulness', 'monotonicity', 'stability', 'consistency',
                           'sparsity', 'simplicity', 'advanced_identity', 'advanced_separability',
                           'advanced_non_sensitivity', 'advanced_compactness', 'advanced_correctness',
                           'advanced_entropy', 'advanced_gini_coefficient', 'advanced_kl_divergence'],
            'bayesian_rule_list': ['time_complexity', 'faithfulness', 'stability', 'consistency',
                                 'advanced_identity', 'advanced_correctness', 'advanced_compactness'],
            'influence_functions': ['time_complexity', 'faithfulness', 'stability', 'consistency',
                                  'advanced_identity', 'advanced_non_sensitivity', 'advanced_correctness'],
            'shap_interactive': ['time_complexity', 'faithfulness', 'monotonicity', 'completeness',
                               'stability', 'consistency', 'sparsity', 'simplicity',
                               'advanced_identity', 'advanced_separability', 'advanced_non_sensitivity',
                               'advanced_compactness', 'advanced_correctness', 'advanced_entropy',
                               'advanced_gini_coefficient', 'advanced_kl_divergence'],
            'corels': ['time_complexity', 'faithfulness', 'stability', 'consistency',
                     'advanced_identity', 'advanced_correctness', 'advanced_compactness']
        }
        
        # Check for text-specific variant first if data type is text
        if data_type == 'text':
            text_variant_name = f"{method_name}_text"
            if text_variant_name in method_specific_metrics:
                return method_specific_metrics[text_variant_name]
        
        if method_name in method_specific_metrics:
            metrics = method_specific_metrics[method_name].copy()
            
            # Add text-specific metrics if data type is text and method supports them
            if data_type == 'text' and method_name in ['lime', 'shap', 'integrated_gradients', 'text_occlusion', 'attention_visualization']:
                text_metrics = ['semantic_coherence', 'syntax_awareness', 'context_sensitivity',
                              'word_significance', 'explanation_coverage', 'sentiment_consistency']
                # Remove monotonicity for text (doesn't make sense)
                if 'monotonicity' in metrics:
                    metrics.remove('monotonicity')
                # Add text-specific metrics
                for metric in text_metrics:
                    if metric not in metrics:
                        metrics.append(metric)
            
            return metrics
        
        # Fallback to category-based selection
        if method_category and method_category in self.profiles:
            profile = self.profiles[method_category]
            metrics = profile.get('core_metrics', [])
            metrics.extend(profile.get('fidelity_metrics', []))
            metrics.extend(profile.get('stability_metrics', []))
            metrics.extend(profile.get('comprehensibility_metrics', []))
            return list(set(metrics))  # Remove duplicates
        
        # Ultimate fallback
        return ['time_complexity', 'faithfulness']
    
    def get_metrics_by_category(self, category: MetricCategory) -> List[str]:
        """Get all metrics in a specific category"""
        return [name for name, info in self.metrics.items() if info.category == category]
    
    def validate_metric_applicability(self, metric_name: str, method_category: str) -> bool:
        """Check if a metric is applicable to a method category"""
        if metric_name not in self.metrics:
            return False
        
        metric_info = self.metrics[metric_name]
        return method_category in metric_info.applicable_to
    
    def get_metric_info(self, metric_name: str) -> Optional[MetricInfo]:
        """Get detailed information about a metric"""
        return self.metrics.get(metric_name)
    
    def create_evaluation_plan(self, selected_methods: List[str], 
                             method_categories: Dict[str, str] = None) -> Dict[str, List[str]]:
        """
        Create a comprehensive evaluation plan for selected methods
        
        Args:
            selected_methods: List of method names
            method_categories: Optional mapping of method -> category
            
        Returns:
            Dictionary mapping method -> list of metrics
        """
        evaluation_plan = {}
        
        for method in selected_methods:
            # Determine method category
            category = None
            if method_categories:
                category = method_categories.get(method)
            
            # Get metrics for this method
            metrics = self.get_metrics_for_method(method, category)
            evaluation_plan[method] = metrics
        
        return evaluation_plan
    
    def get_comparative_metrics(self, methods: List[str]) -> List[str]:
        """
        Get metrics that are common across multiple methods for fair comparison
        
        Args:
            methods: List of method names to compare
            
        Returns:
            List of metrics suitable for cross-method comparison
        """
        if not methods:
            return []
        
        # Get metrics for each method
        method_metrics = {}
        for method in methods:
            method_metrics[method] = set(self.get_metrics_for_method(method))
        
        # Find intersection (common metrics)
        common_metrics = set.intersection(*method_metrics.values())
        
        # Ensure we have at least basic metrics for comparison
        essential_metrics = {'time_complexity', 'faithfulness'}
        for metric in essential_metrics:
            if all(metric in method_metrics[method] for method in methods):
                common_metrics.add(metric)
        
        return list(common_metrics)
    
    def get_evaluation_summary_template(self, method: str) -> Dict[str, str]:
        """
        Get a template for evaluation summary reporting
        
        Args:
            method: Method name
            
        Returns:
            Dictionary with metric categories and their descriptions
        """
        metrics = self.get_metrics_for_method(method)
        
        summary = {
            'method': method,
            'efficiency': [],
            'fidelity': [],
            'stability': [],
            'comprehensibility': []
        }
        
        for metric_name in metrics:
            metric_info = self.get_metric_info(metric_name)
            if metric_info:
                category_name = metric_info.category.value
                if category_name == 'efficiency':
                    summary['efficiency'].append(metric_name)
                elif category_name == 'fidelity':
                    summary['fidelity'].append(metric_name)
                elif category_name == 'stability':
                    summary['stability'].append(metric_name)
                elif category_name == 'comprehensibility':
                    summary['comprehensibility'].append(metric_name)
        
        return summary
    
    def filter_metrics_by_requirements(self, metrics: List[str], 
                                     has_feature_importance: bool = True,
                                     has_baseline: bool = True,
                                     has_instance_data: bool = True) -> List[str]:
        """
        Filter metrics based on available data requirements
        
        Args:
            metrics: List of metric names to filter
            has_feature_importance: Whether feature importance data is available
            has_baseline: Whether baseline predictions are available
            has_instance_data: Whether instance data is available
            
        Returns:
            Filtered list of applicable metrics
        """
        filtered = []
        
        for metric_name in metrics:
            metric_info = self.get_metric_info(metric_name)
            if not metric_info:
                continue
            
            # Check requirements
            if metric_info.requires_feature_importance and not has_feature_importance:
                continue
            if metric_info.requires_baseline and not has_baseline:
                continue
            if metric_info.requires_instance_data and not has_instance_data:
                continue
            
            filtered.append(metric_name)
        
        return filtered