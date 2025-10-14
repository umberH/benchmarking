#!/usr/bin/env python3
"""
Interactive Explanation Comparison Component
A powerful component for side-by-side comparison of different XAI methods with real-time visualization
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import altair as alt
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon, kruskal, mannwhitneyu, chi2_contingency, ranksums
import seaborn as sns
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.contingency_tables import mcnemar

class ExplanationComparator:
    """Interactive component for comparing explanation methods with real-time visualizations"""
    
    def __init__(self, experiment_data: Dict[str, Any]):
        """Initialize the comparator with experiment data"""
        self.experiment_data = experiment_data
        self.comparison_cache = {}
        
    def render_comparison_widget(self):
        """Render the main comparison interface"""
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            padding: 2rem;
            border-radius: 20px;
            margin: 1rem 0;
            text-align: center;
            box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        ">
            <h2 style="color: white; margin: 0; font-size: 2.5rem; font-weight: 700;">
                🔬 Real-time Explanation Comparator
            </h2>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.2rem;">
                Compare XAI methods side-by-side with interactive visualizations
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for different comparison modes
        comp_tab1, comp_tab2, comp_tab3, comp_tab4 = st.tabs([
            "Method vs Method", 
            "Performance Matrix", 
            "Live Benchmark", 
            "Custom Analysis"
        ])
        
        with comp_tab1:
            self._render_method_comparison()
            
        with comp_tab2:
            self._render_performance_matrix()
            
        with comp_tab3:
            self._render_live_benchmark()
            
        with comp_tab4:
            self._render_custom_analysis()
    
    def _render_method_comparison(self):
        """Render side-by-side method comparison"""
        st.markdown("### 🔄 Side-by-Side Method Comparison")
        
        # Get available methods
        available_methods = self._extract_available_methods()
        
        if len(available_methods) < 2:
            st.warning("Need at least 2 methods for comparison. Found: " + str(len(available_methods)))
            return
        
        # Method selection
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🅰️ Method A")
            method_a = st.selectbox(
                "Select first method:",
                available_methods,
                key="method_a_selector",
                help="Choose the first method to compare"
            )
            
        with col2:
            st.markdown("#### 🅱️ Method B")
            method_b = st.selectbox(
                "Select second method:",
                [m for m in available_methods if m != method_a],
                key="method_b_selector",
                help="Choose the second method to compare"
            )
        
        if method_a and method_b and method_a != method_b:
            # Get comparison data
            comparison_data = self._prepare_comparison_data(method_a, method_b)
            
            if comparison_data:
                # Create comparison visualizations
                self._create_comparison_visualizations(method_a, method_b, comparison_data)
                
                # Performance metrics comparison
                self._create_metrics_comparison(method_a, method_b, comparison_data)
                
                # Statistical significance test
                self._perform_statistical_tests(method_a, method_b, comparison_data)
            else:
                st.error("No comparable data found between selected methods.")
    
    def _render_performance_matrix(self):
        """Render comprehensive performance matrix"""
        st.markdown("### 📊 Performance Matrix Analysis")
        
        # Extract metrics data
        metrics_data = self._extract_metrics_data()
        
        if not metrics_data:
            st.warning("No metrics data available for matrix analysis.")
            return
        
        # Create performance matrix
        df = pd.DataFrame(metrics_data)
        
        # Matrix configuration
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown("#### ⚙️ Matrix Configuration")
            
            # Metric selection
            available_metrics = [col for col in df.columns if col not in ['Dataset', 'Model', 'Method']]
            selected_metrics = st.multiselect(
                "Select metrics to display:",
                available_metrics,
                default=available_metrics[:5] if len(available_metrics) >= 5 else available_metrics,
                help="Choose which metrics to include in the matrix"
            )
            
            # Aggregation method
            agg_method = st.selectbox(
                "Aggregation method:",
                ['mean', 'median', 'max', 'min'],
                help="How to aggregate multiple values"
            )
            
            # Matrix type
            matrix_type = st.selectbox(
                "Matrix type:",
                ['Method vs Dataset', 'Method vs Model', 'Dataset vs Model'],
                help="Choose the comparison axes"
            )
        
        with col2:
            if selected_metrics:
                self._create_performance_matrix(df, selected_metrics, agg_method, matrix_type)
    
    def _render_live_benchmark(self):
        """Render live benchmarking interface"""
        st.markdown("### ⚡ Live Performance Benchmark")
        
        # Real-time metrics display
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate live metrics
        live_metrics = self._calculate_live_metrics()
        
        with col1:
            st.metric(
                label="🎯 Best Faithfulness",
                value=f"{live_metrics.get('best_faithfulness', 0):.3f}",
                delta=f"{live_metrics.get('faithfulness_improvement', 0):.3f}",
                help="Highest faithfulness score across all methods"
            )
        
        with col2:
            st.metric(
                label="⚡ Fastest Method",
                value=live_metrics.get('fastest_method', 'N/A'),
                delta=f"{live_metrics.get('speed_improvement', 0):.2f}s",
                help="Method with lowest generation time"
            )
        
        with col3:
            st.metric(
                label="🎖️ Most Stable",
                value=live_metrics.get('most_stable', 'N/A'),
                delta=f"{live_metrics.get('stability_score', 0):.3f}",
                help="Method with highest stability score"
            )
        
        with col4:
            st.metric(
                label="🏆 Overall Winner",
                value=live_metrics.get('overall_winner', 'N/A'),
                delta=f"Score: {live_metrics.get('winner_score', 0):.3f}",
                help="Best method based on weighted performance"
            )
        
        # Live performance radar chart
        st.markdown("#### 📡 Live Performance Radar")
        self._create_live_radar_chart()
        
        # Performance evolution timeline
        st.markdown("#### 📈 Performance Evolution")
        self._create_performance_timeline()
    
    def _render_custom_analysis(self):
        """Render custom analysis interface"""
        st.markdown("### 🎯 Custom Analysis Builder")
        
        # Analysis configuration
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### 🔧 Analysis Configuration")
            
            # Analysis type
            analysis_type = st.selectbox(
                "Analysis type:",
                [
                    "Feature Importance Comparison",
                    "Time vs Accuracy Trade-off", 
                    "Method Clustering",
                    "Performance Correlation",
                    "Dataset Sensitivity Analysis"
                ],
                help="Choose the type of custom analysis"
            )
            
            # Additional parameters based on analysis type
            if analysis_type == "Feature Importance Comparison":
                top_n_features = st.slider("Top N features", 5, 50, 15)
                importance_threshold = st.slider("Importance threshold", 0.0, 1.0, 0.1)
                
            elif analysis_type == "Time vs Accuracy Trade-off":
                time_metric = st.selectbox("Time metric:", ["generation_time", "total_time"])
                accuracy_metric = st.selectbox("Accuracy metric:", ["faithfulness", "completeness"])
                
            elif analysis_type == "Method Clustering":
                n_clusters = st.slider("Number of clusters", 2, 10, 3)
                clustering_method = st.selectbox("Clustering method:", ["K-means", "Hierarchical"])
        
        with col2:
            # Execute custom analysis
            if st.button("🚀 Run Analysis", type="primary"):
                with st.spinner("Running custom analysis..."):
                    result = self._execute_custom_analysis(analysis_type, locals())
                    if result:
                        st.success("✅ Analysis completed!")
                        self._display_custom_results(analysis_type, result)
                    else:
                        st.error("❌ Analysis failed. Please check your parameters.")
    
    def _extract_available_methods(self) -> List[str]:
        """Extract available explanation methods from experiment data"""
        methods = set()
        
        comprehensive_results = self.experiment_data.get('comprehensive_results', [])
        for result in comprehensive_results:
            if isinstance(result, dict):
                method = result.get('explanation_method', '')
                if method:
                    methods.add(method)
        
        return sorted(list(methods))
    
    def _extract_metrics_data(self) -> List[Dict[str, Any]]:
        """Extract metrics data for matrix analysis"""
        metrics_data = []
        
        comprehensive_results = self.experiment_data.get('comprehensive_results', [])
        for result in comprehensive_results:
            if isinstance(result, dict):
                evaluation = result.get('evaluations', {})
                
                row = {
                    'Dataset': result.get('dataset', 'unknown'),
                    'Model': result.get('model', 'unknown'),
                    'Method': result.get('explanation_method', 'unknown'),
                    'faithfulness': float(evaluation.get('faithfulness', 0.0)),
                    'stability': float(evaluation.get('stability', 0.0)),
                    'completeness': float(evaluation.get('completeness', 0.0)),
                    'sparsity': float(evaluation.get('sparsity', 0.0)),
                    'monotonicity': float(evaluation.get('monotonicity', 0.0)),
                    'consistency': float(evaluation.get('consistency', 0.0)),
                    'time_complexity': float(evaluation.get('time_complexity', 0.0)),
                    'simplicity': float(evaluation.get('simplicity', 0.0))
                }
                metrics_data.append(row)
        
        return metrics_data
    
    def _prepare_comparison_data(self, method_a: str, method_b: str) -> Optional[Dict]:
        """Prepare data for comparing two methods"""
        metrics_data = self._extract_metrics_data()
        df = pd.DataFrame(metrics_data)
        
        method_a_data = df[df['Method'] == method_a]
        method_b_data = df[df['Method'] == method_b]
        
        if method_a_data.empty or method_b_data.empty:
            return None
        
        return {
            'method_a': method_a_data,
            'method_b': method_b_data,
            'combined': df[df['Method'].isin([method_a, method_b])]
        }
    
    def _create_comparison_visualizations(self, method_a: str, method_b: str, data: Dict):
        """Create side-by-side comparison visualizations"""
        st.markdown("#### 📊 Visual Comparison")
        
        # Create subplots for side-by-side comparison
        metrics = ['faithfulness', 'stability', 'completeness', 'sparsity']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Method A performance
            method_a_means = data['method_a'][metrics].mean()
            
            fig_a = go.Figure(data=[
                go.Bar(
                    x=metrics,
                    y=method_a_means.values,
                    marker_color='rgba(102, 126, 234, 0.8)',
                    text=[f"{v:.3f}" for v in method_a_means.values],
                    textposition='auto'
                )
            ])
            
            fig_a.update_layout(
                title=f"🅰️ {method_a} Performance",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1]),
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_a, use_container_width=True)
        
        with col2:
            # Method B performance
            method_b_means = data['method_b'][metrics].mean()
            
            fig_b = go.Figure(data=[
                go.Bar(
                    x=metrics,
                    y=method_b_means.values,
                    marker_color='rgba(245, 158, 11, 0.8)',
                    text=[f"{v:.3f}" for v in method_b_means.values],
                    textposition='auto'
                )
            ])
            
            fig_b.update_layout(
                title=f"🅱️ {method_b} Performance",
                yaxis_title="Score",
                yaxis=dict(range=[0, 1]),
                height=400,
                showlegend=False
            )
            st.plotly_chart(fig_b, use_container_width=True)
        
        # Combined radar chart
        st.markdown("#### 🕸️ Performance Radar Comparison")
        
        fig_radar = go.Figure()
        
        # Method A
        values_a = method_a_means.values.tolist()
        values_a += values_a[:1]  # Close the radar
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values_a,
            theta=metrics + [metrics[0]],
            fill='toself',
            name=f"🅰️ {method_a}",
            line_color='rgba(102, 126, 234, 1)'
        ))
        
        # Method B
        values_b = method_b_means.values.tolist()
        values_b += values_b[:1]  # Close the radar
        
        fig_radar.add_trace(go.Scatterpolar(
            r=values_b,
            theta=metrics + [metrics[0]],
            fill='toself',
            name=f"🅱️ {method_b}",
            line_color='rgba(245, 158, 11, 1)'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            height=500,
            title="Method Performance Radar Comparison"
        )
        
        st.plotly_chart(fig_radar, use_container_width=True)
    
    def _create_metrics_comparison(self, method_a: str, method_b: str, data: Dict):
        """Create detailed metrics comparison table"""
        st.markdown("#### 📋 Detailed Metrics Comparison")
        
        metrics = ['faithfulness', 'stability', 'completeness', 'sparsity', 'monotonicity', 'consistency']
        
        # Calculate statistics
        comparison_stats = []
        
        for metric in metrics:
            a_values = data['method_a'][metric].values
            b_values = data['method_b'][metric].values
            
            a_mean = np.mean(a_values)
            b_mean = np.mean(b_values)
            a_std = np.std(a_values)
            b_std = np.std(b_values)
            
            # Determine winner
            winner = "🅰️" if a_mean > b_mean else "🅱️"
            difference = abs(a_mean - b_mean)
            
            comparison_stats.append({
                'Metric': metric.title(),
                f'{method_a} (🅰️)': f"{a_mean:.3f} ± {a_std:.3f}",
                f'{method_b} (🅱️)': f"{b_mean:.3f} ± {b_std:.3f}",
                'Winner': winner,
                'Difference': f"{difference:.3f}",
                'Improvement': f"{(difference/max(a_mean, b_mean, 1e-10)*100):.1f}%" if max(a_mean, b_mean) > 1e-10 else "N/A (zero baseline)"
            })
        
        comparison_df = pd.DataFrame(comparison_stats)
        
        # Style the dataframe
        def highlight_winner(row):
            winner_color = 'background-color: rgba(102, 126, 234, 0.3)' if row['Winner'] == '🅰️' else 'background-color: rgba(245, 158, 11, 0.3)'
            return [''] * len(row)
        
        styled_df = comparison_df.style.apply(highlight_winner, axis=1)
        st.dataframe(styled_df, use_container_width=True)

    def _debug_monotonicity_scores(self, method_a: str, method_b: str, data: Dict):
        """Debug monotonicity scores to identify division by zero issues"""

        st.markdown("#### 🐛 Monotonicity Debug Information")

        debug_info = []

        for dataset_name, dataset_data in data.items():
            method_a_data = dataset_data.get(method_a, {})
            method_b_data = dataset_data.get(method_b, {})

            # Extract monotonicity scores
            a_monotonicity = method_a_data.get('monotonicity', [])
            b_monotonicity = method_b_data.get('monotonicity', [])

            if isinstance(a_monotonicity, (int, float)):
                a_monotonicity = [a_monotonicity]
            if isinstance(b_monotonicity, (int, float)):
                b_monotonicity = [b_monotonicity]

            # Convert to numpy arrays and handle NaN/inf
            a_values = np.array(a_monotonicity, dtype=float)
            b_values = np.array(b_monotonicity, dtype=float)

            # Filter out NaN and inf values
            a_valid = a_values[np.isfinite(a_values)]
            b_valid = b_values[np.isfinite(b_values)]

            debug_entry = {
                'Dataset': dataset_name,
                f'{method_a} Raw': str(a_monotonicity),
                f'{method_b} Raw': str(b_monotonicity),
                f'{method_a} Valid Count': len(a_valid),
                f'{method_b} Valid Count': len(b_valid),
                f'{method_a} Mean': f"{np.mean(a_valid):.6f}" if len(a_valid) > 0 else "N/A",
                f'{method_b} Mean': f"{np.mean(b_valid):.6f}" if len(b_valid) > 0 else "N/A",
                f'{method_a} Std': f"{np.std(a_valid):.6f}" if len(a_valid) > 0 else "N/A",
                f'{method_b} Std': f"{np.std(b_valid):.6f}" if len(b_valid) > 0 else "N/A",
                f'{method_a} Unique': len(np.unique(a_valid)) if len(a_valid) > 0 else 0,
                f'{method_b} Unique': len(np.unique(b_valid)) if len(b_valid) > 0 else 0,
                'Issue': self._identify_monotonicity_issue(a_valid, b_valid)
            }

            debug_info.append(debug_entry)

        debug_df = pd.DataFrame(debug_info)
        st.dataframe(debug_df, use_container_width=True)

        # Add summary warnings
        self._display_monotonicity_warnings(debug_info)

    def _identify_monotonicity_issue(self, a_values: np.ndarray, b_values: np.ndarray) -> str:
        """Identify specific issues with monotonicity data"""

        issues = []

        # Check for empty data
        if len(a_values) == 0 or len(b_values) == 0:
            issues.append("Empty data")

        # Check for identical values (zero variance)
        if len(a_values) > 0 and np.std(a_values) < 1e-10:
            issues.append("Method A: Zero variance")

        if len(b_values) > 0 and np.std(b_values) < 1e-10:
            issues.append("Method B: Zero variance")

        # Check for all zeros
        if len(a_values) > 0 and np.all(a_values == 0):
            issues.append("Method A: All zeros")

        if len(b_values) > 0 and np.all(b_values == 0):
            issues.append("Method B: All zeros")

        # Check for single unique value
        if len(a_values) > 0 and len(np.unique(a_values)) == 1:
            issues.append(f"Method A: Single value ({a_values[0]:.3f})")

        if len(b_values) > 0 and len(np.unique(b_values)) == 1:
            issues.append(f"Method B: Single value ({b_values[0]:.3f})")

        # Check for problematic ranges
        if len(a_values) > 0 and (np.max(a_values) - np.min(a_values)) < 1e-10:
            issues.append("Method A: Tiny range")

        if len(b_values) > 0 and (np.max(b_values) - np.min(b_values)) < 1e-10:
            issues.append("Method B: Tiny range")

        return "; ".join(issues) if issues else "No issues detected"

    def _display_monotonicity_warnings(self, debug_info: List[Dict]):
        """Display warnings about monotonicity issues"""

        warnings = []

        for entry in debug_info:
            if "Zero variance" in entry['Issue'] or "All zeros" in entry['Issue']:
                warnings.append(f"⚠️ **{entry['Dataset']}**: {entry['Issue']}")

            if "Single value" in entry['Issue']:
                warnings.append(f"🔍 **{entry['Dataset']}**: {entry['Issue']}")

        if warnings:
            st.markdown("#### 🚨 Detected Issues:")
            for warning in warnings:
                st.markdown(warning)

            st.markdown("""
            **Common Causes:**
            - All explanation methods return identical monotonicity scores
            - Text/image data still returning NaN despite improvements
            - Models producing identical behavior across different explanation methods
            - Dataset characteristics leading to uniform explanation quality

            **Solutions:**
            - Check if new monotonicity implementation is working correctly
            - Verify that different explanation methods are actually producing different results
            - Consider filtering out datasets/models that produce degenerate results
            """)
        else:
            st.markdown("✅ No critical monotonicity issues detected")
    
    def _perform_statistical_tests(self, method_a: str, method_b: str, data: Dict):
        """Perform comprehensive statistical significance tests for different data types"""

        # Add debugging for monotonicity issues
        self._debug_monotonicity_scores(method_a, method_b, data)
        st.markdown("#### 📈 Statistical Significance Tests")
        
        # Detect data types present in the comparison
        data_types = self._detect_data_types(data)
        
        # Create tabs for different test categories
        test_tab1, test_tab2, test_tab3, test_tab4, test_tab5 = st.tabs([
            "🧮 Parametric Tests", 
            "🔀 Non-parametric Tests", 
            "🎯 Wilcoxon Tests",
            "📊 Multi-method Comparison", 
            "📈 Data Type Specific"
        ])
        
        with test_tab1:
            self._perform_parametric_tests(method_a, method_b, data)
        
        with test_tab2:
            self._perform_nonparametric_tests(method_a, method_b, data)
        
        with test_tab3:
            self._perform_wilcoxon_tests(method_a, method_b, data)
        
        with test_tab4:
            self._perform_multimethod_tests(data)
        
        with test_tab5:
            self._perform_datatype_specific_tests(method_a, method_b, data, data_types)
    
    def _detect_data_types(self, data: Dict) -> Dict[str, bool]:
        """Detect which data types are present in the comparison"""
        combined_data = data['combined']
        
        data_types = {
            'tabular': False,
            'image': False,
            'text': False
        }
        
        # Check dataset names and model types to infer data types
        for _, row in combined_data.iterrows():
            dataset = row['Dataset'].lower()
            model = row['Model'].lower()
            
            # Tabular data indicators
            if any(keyword in dataset for keyword in ['adult', 'income', 'census', 'bank', 'credit']):
                data_types['tabular'] = True
            
            # Image data indicators
            if any(keyword in dataset for keyword in ['mnist', 'cifar', 'imagenet']) or \
               any(keyword in model for keyword in ['cnn', 'vit', 'resnet', 'vision']):
                data_types['image'] = True
            
            # Text data indicators  
            if any(keyword in dataset for keyword in ['imdb', 'sentiment', 'text', 'nlp']) or \
               any(keyword in model for keyword in ['bert', 'lstm', 'transformer', 'nlp']):
                data_types['text'] = True
        
        return data_types
    
    def _perform_parametric_tests(self, method_a: str, method_b: str, data: Dict):
        """Perform parametric statistical tests"""
        st.markdown("##### 🧮 Parametric Tests (Assuming Normal Distribution)")
        
        metrics = ['faithfulness', 'stability', 'completeness', 'sparsity', 'monotonicity', 'consistency']
        test_results = []
        
        for metric in metrics:
            a_values = data['method_a'][metric].values
            b_values = data['method_b'][metric].values
            
            if len(a_values) > 1 and len(b_values) > 1:
                # Filter out identical values to avoid warnings
                a_clean = a_values[np.isfinite(a_values)]
                b_clean = b_values[np.isfinite(b_values)]

                # Skip if all values are identical (zero variance)
                if np.std(a_clean) < 1e-10 or np.std(b_clean) < 1e-10:
                    continue

                # Test for normality first (with error handling)
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=UserWarning)
                        shapiro_a = stats.shapiro(a_clean) if len(a_clean) >= 3 else (None, 1.0)
                        shapiro_b = stats.shapiro(b_clean) if len(b_clean) >= 3 else (None, 1.0)
                except:
                    shapiro_a = (None, 1.0)
                    shapiro_b = (None, 1.0)
                
                # Independent t-test (with error handling)
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=RuntimeWarning)
                        t_stat, t_p_value = stats.ttest_ind(a_clean, b_clean)
                except:
                    t_stat, t_p_value = 0.0, 1.0

                # Paired t-test (if same length)
                paired_stat, paired_p = None, None
                if len(a_clean) == len(b_clean):
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=RuntimeWarning)
                            paired_stat, paired_p = stats.ttest_rel(a_clean, b_clean)
                    except:
                        paired_stat, paired_p = 0.0, 1.0
                
                # Effect size (Cohen's d) - use cleaned data
                pooled_std = np.sqrt(((len(a_clean) - 1) * np.var(a_clean) +
                                    (len(b_clean) - 1) * np.var(b_clean)) /
                                   (len(a_clean) + len(b_clean) - 2)) if (len(a_clean) + len(b_clean)) > 2 else 0

                # Avoid division by zero for Cohen's d
                if pooled_std > 1e-10:
                    cohens_d = (np.mean(a_clean) - np.mean(b_clean)) / pooled_std
                else:
                    cohens_d = 0.0  # Identical distributions have Cohen's d = 0

                # Levene's test for equal variances (with error handling)
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=RuntimeWarning)
                        levene_stat, levene_p = stats.levene(a_clean, b_clean)
                except:
                    levene_stat, levene_p = 0.0, 1.0
                
                test_results.append({
                    'Metric': metric.title(),
                    'Normality A (p)': f"{shapiro_a.pvalue:.4f}",
                    'Normality B (p)': f"{shapiro_b.pvalue:.4f}",
                    'Equal Var (p)': f"{levene_p:.4f}",
                    'T-test (p)': f"{t_p_value:.4f}",
                    'Paired T (p)': f"{paired_p:.4f}" if paired_p else "N/A",
                    'Effect Size (d)': f"{abs(cohens_d):.3f}",
                    'Interpretation': self._interpret_effect_size(abs(cohens_d)),
                    'Significance': self._get_significance_level(t_p_value)
                })
        
        if test_results:
            test_df = pd.DataFrame(test_results)
            st.dataframe(test_df, use_container_width=True)
            
            st.info("""
            **Interpretation Guide:**
            - Normality: p > 0.05 suggests normal distribution
            - Equal Var: p > 0.05 suggests equal variances (Levene's test)
            - Effect Size: Small (0.2), Medium (0.5), Large (0.8)
            """)
    
    def _perform_nonparametric_tests(self, method_a: str, method_b: str, data: Dict):
        """Perform non-parametric statistical tests"""
        st.markdown("##### 🔀 Non-parametric Tests (Distribution-free)")
        
        metrics = ['faithfulness', 'stability', 'completeness', 'sparsity', 'monotonicity', 'consistency']
        test_results = []
        
        for metric in metrics:
            a_values = data['method_a'][metric].values
            b_values = data['method_b'][metric].values
            
            if len(a_values) > 1 and len(b_values) > 1:
                # Filter out identical values to avoid warnings
                a_clean = a_values[np.isfinite(a_values)]
                b_clean = b_values[np.isfinite(b_values)]

                # Skip if all values are identical (zero variance)
                if np.std(a_clean) < 1e-10 or np.std(b_clean) < 1e-10:
                    continue

                # Mann-Whitney U test (independent samples)
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=RuntimeWarning)
                        mw_stat, mw_p = mannwhitneyu(a_clean, b_clean, alternative='two-sided')
                except:
                    mw_stat, mw_p = 0.0, 1.0

                # Wilcoxon signed-rank test (paired samples, if same length)
                wilcoxon_stat, wilcoxon_p = None, None
                if len(a_clean) == len(b_clean):
                    try:
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=RuntimeWarning)
                            wilcoxon_stat, wilcoxon_p = wilcoxon(a_clean, b_clean)
                    except:
                        wilcoxon_stat, wilcoxon_p = 0.0, 1.0
                
                # Effect size for Mann-Whitney (r = Z / sqrt(N))
                z_score = stats.norm.ppf(1 - mw_p/2)  # Approximate Z from p-value
                n_total = len(a_values) + len(b_values)
                effect_size_r = abs(z_score) / np.sqrt(n_total)
                
                # Median difference
                median_diff = np.median(a_values) - np.median(b_values)
                
                test_results.append({
                    'Metric': metric.title(),
                    'Mann-Whitney U': f"{mw_stat:.1f}",
                    'MW p-value': f"{mw_p:.4f}",
                    'Wilcoxon p': f"{wilcoxon_p:.4f}" if wilcoxon_p else "N/A",
                    'Effect Size (r)': f"{effect_size_r:.3f}",
                    'Median Diff': f"{median_diff:.3f}",
                    'MW Significance': self._get_significance_level(mw_p),
                    'W Significance': self._get_significance_level(wilcoxon_p) if wilcoxon_p else "N/A"
                })
        
        if test_results:
            test_df = pd.DataFrame(test_results)
            st.dataframe(test_df, use_container_width=True)
            
            st.info("""
            **Non-parametric Test Guide:**
            - Mann-Whitney U: Tests if one group tends to have larger values
            - Wilcoxon: Tests paired differences (requires same sample size)
            - Effect Size (r): Small (0.1), Medium (0.3), Large (0.5)
            """)
    
    def _perform_wilcoxon_tests(self, method_a: str, method_b: str, data: Dict):
        """Perform comprehensive Wilcoxon rank tests"""
        st.markdown("##### 🎯 Wilcoxon Rank Tests (Specialized Non-parametric)")
        
        st.info("""
        **Wilcoxon Tests** are powerful non-parametric tests for comparing two groups:
        - **Wilcoxon Signed-Rank**: For paired samples (same instances)
        - **Wilcoxon Rank-Sum**: Alternative to Mann-Whitney U
        - **Robust to outliers** and **distribution-free**
        """)
        
        metrics = ['faithfulness', 'stability', 'completeness', 'sparsity', 'monotonicity', 'consistency']
        wilcoxon_results = []
        
        for metric in metrics:
            a_values = data['method_a'][metric].values
            b_values = data['method_b'][metric].values
            
            if len(a_values) > 1 and len(b_values) > 1:
                # Filter out identical values to avoid warnings
                a_clean = a_values[np.isfinite(a_values)]
                b_clean = b_values[np.isfinite(b_values)]

                # Skip if all values are identical (zero variance)
                if np.std(a_clean) < 1e-10 or np.std(b_clean) < 1e-10:
                    continue

                # Wilcoxon signed-rank test (paired samples)
                wilcoxon_stat_signed, wilcoxon_p_signed = None, None
                if len(a_clean) == len(b_clean):
                    try:
                        # Paired differences
                        differences = a_clean - b_clean
                        non_zero_diffs = differences[differences != 0]

                        if len(non_zero_diffs) > 0:
                            with warnings.catch_warnings():
                                warnings.filterwarnings('ignore', category=RuntimeWarning)
                                wilcoxon_stat_signed, wilcoxon_p_signed = wilcoxon(a_clean, b_clean, alternative='two-sided')
                            
                            # Effect size for Wilcoxon signed-rank (r = Z / sqrt(N))
                            n_pairs = len(non_zero_diffs)
                            z_score = stats.norm.ppf(1 - wilcoxon_p_signed/2)
                            effect_size_signed = abs(z_score) / np.sqrt(n_pairs)
                        else:
                            wilcoxon_stat_signed, wilcoxon_p_signed = 0, 1.0
                            effect_size_signed = 0
                    except ValueError as e:
                        wilcoxon_stat_signed, wilcoxon_p_signed = None, None
                        effect_size_signed = None
                else:
                    effect_size_signed = None
                
                # Wilcoxon rank-sum test (independent samples) - alternative to Mann-Whitney
                try:
                    ranksum_stat, ranksum_p = ranksums(a_values, b_values)
                    
                    # Effect size for rank-sum test
                    n_total = len(a_values) + len(b_values)
                    z_score_rs = ranksum_stat  # ranksums returns Z-statistic directly
                    effect_size_ranksum = abs(z_score_rs) / np.sqrt(n_total)
                except Exception:
                    ranksum_stat, ranksum_p = None, None
                    effect_size_ranksum = None
                
                # Additional Wilcoxon statistics
                # Median of differences (for signed-rank)
                if len(a_values) == len(b_values):
                    median_diff = np.median(a_values - b_values)
                    
                    # Probability of superiority (PS) - P(X > Y)
                    comparisons = []
                    for a_val in a_values:
                        for b_val in b_values:
                            if a_val > b_val:
                                comparisons.append(1)
                            elif a_val < b_val:
                                comparisons.append(0)
                            else:
                                comparisons.append(0.5)
                    
                    prob_superiority = np.mean(comparisons) if comparisons else 0.5
                else:
                    median_diff = np.median(a_values) - np.median(b_values)
                    prob_superiority = None
                
                # Confidence interval for median difference (Hodges-Lehmann estimator)
                if len(a_values) == len(b_values) and wilcoxon_p_signed is not None:
                    # Walsh averages for Hodges-Lehmann estimator
                    differences = a_values - b_values
                    walsh_averages = []
                    
                    for i in range(len(differences)):
                        for j in range(i, len(differences)):
                            walsh_averages.append((differences[i] + differences[j]) / 2)
                    
                    walsh_averages = np.array(walsh_averages)
                    
                    # Confidence interval (approximate)
                    alpha = 0.05  # 95% CI
                    n_walsh = len(walsh_averages)
                    
                    if n_walsh > 0:
                        # Critical value for Wilcoxon (approximate)
                        z_crit = stats.norm.ppf(1 - alpha/2)
                        se_walsh = np.sqrt(n_walsh * (n_walsh + 1) * (2*n_walsh + 1) / 24) / n_walsh
                        
                        # Sort Walsh averages
                        sorted_walsh = np.sort(walsh_averages)
                        
                        # Calculate CI bounds (simplified approach)
                        lower_idx = max(0, int(n_walsh/2 - z_crit * se_walsh * n_walsh))
                        upper_idx = min(n_walsh-1, int(n_walsh/2 + z_crit * se_walsh * n_walsh))
                        
                        ci_lower = sorted_walsh[lower_idx]
                        ci_upper = sorted_walsh[upper_idx]
                        hodges_lehmann = np.median(sorted_walsh)
                    else:
                        ci_lower, ci_upper, hodges_lehmann = None, None, None
                else:
                    ci_lower, ci_upper, hodges_lehmann = None, None, None
                
                wilcoxon_results.append({
                    'Metric': metric.title(),
                    'W Signed-Rank': f"{wilcoxon_stat_signed:.1f}" if wilcoxon_stat_signed is not None else "N/A",
                    'Signed p-value': f"{wilcoxon_p_signed:.4f}" if wilcoxon_p_signed is not None else "N/A", 
                    'W Rank-Sum': f"{ranksum_stat:.3f}" if ranksum_stat is not None else "N/A",
                    'Rank-Sum p': f"{ranksum_p:.4f}" if ranksum_p is not None else "N/A",
                    'Effect Size (Signed)': f"{effect_size_signed:.3f}" if effect_size_signed is not None else "N/A",
                    'Effect Size (Rank-Sum)': f"{effect_size_ranksum:.3f}" if effect_size_ranksum is not None else "N/A",
                    'Median Diff': f"{median_diff:.3f}",
                    'P(A > B)': f"{prob_superiority:.3f}" if prob_superiority is not None else "N/A",
                    'Hodges-Lehmann': f"{hodges_lehmann:.3f}" if hodges_lehmann is not None else "N/A",
                    'CI Lower': f"{ci_lower:.3f}" if ci_lower is not None else "N/A",
                    'CI Upper': f"{ci_upper:.3f}" if ci_upper is not None else "N/A",
                    'Signed Sig': self._get_significance_level(wilcoxon_p_signed) if wilcoxon_p_signed else "N/A",
                    'Rank-Sum Sig': self._get_significance_level(ranksum_p) if ranksum_p else "N/A"
                })
        
        if wilcoxon_results:
            wilcoxon_df = pd.DataFrame(wilcoxon_results)
            st.dataframe(wilcoxon_df, use_container_width=True)
            
            # Detailed interpretation
            st.markdown("#### 📊 Wilcoxon Test Interpretation")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Wilcoxon Signed-Rank Test:**
                - Tests if median difference = 0
                - Requires paired data (same sample size)
                - More powerful than sign test
                - Considers magnitude of differences
                
                **Test Statistics:**
                - **W**: Sum of positive/negative ranks
                - **Effect Size**: r = Z/√N (like correlation)
                - **Hodges-Lehmann**: Robust estimate of median difference
                """)
            
            with col2:
                st.markdown("""
                **Wilcoxon Rank-Sum Test:**
                - Alternative to Mann-Whitney U
                - Tests if distributions differ
                - Independent samples
                - Ranks all observations together
                
                **Interpretation:**
                - **P(A > B)**: Probability of superiority
                - **Median Diff**: Difference in medians
                - **CI**: Confidence interval for median difference
                """)
            
            # Visualization of significant results
            significant_results = [
                row for row in wilcoxon_results 
                if (row['Signed p-value'] != "N/A" and float(row['Signed p-value']) < 0.05) or
                   (row['Rank-Sum p'] != "N/A" and float(row['Rank-Sum p']) < 0.05)
            ]
            
            if significant_results:
                st.markdown("#### 🎯 Significant Results Visualization")
                
                # Extract data for significant results
                sig_metrics = [row['Metric'] for row in significant_results]
                effect_sizes_signed = []
                effect_sizes_ranksum = []
                
                for row in significant_results:
                    try:
                        es_signed = float(row['Effect Size (Signed)']) if row['Effect Size (Signed)'] != "N/A" else 0
                        es_ranksum = float(row['Effect Size (Rank-Sum)']) if row['Effect Size (Rank-Sum)'] != "N/A" else 0
                    except:
                        es_signed, es_ranksum = 0, 0
                    
                    effect_sizes_signed.append(es_signed)
                    effect_sizes_ranksum.append(es_ranksum)
                
                # Create comparison chart
                comparison_data = pd.DataFrame({
                    'Metric': sig_metrics + sig_metrics,
                    'Effect Size': effect_sizes_signed + effect_sizes_ranksum,
                    'Test Type': ['Signed-Rank'] * len(sig_metrics) + ['Rank-Sum'] * len(sig_metrics)
                })
                
                fig = px.bar(
                    comparison_data,
                    x='Metric',
                    y='Effect Size',
                    color='Test Type',
                    title="Effect Sizes for Significant Wilcoxon Tests",
                    barmode='group',
                    color_discrete_map={'Signed-Rank': '#1f77b4', 'Rank-Sum': '#ff7f0e'}
                )
                
                # Add effect size interpretation lines
                fig.add_hline(y=0.1, line_dash="dash", line_color="green", 
                             annotation_text="Small Effect (0.1)")
                fig.add_hline(y=0.3, line_dash="dash", line_color="orange", 
                             annotation_text="Medium Effect (0.3)")
                fig.add_hline(y=0.5, line_dash="dash", line_color="red", 
                             annotation_text="Large Effect (0.5)")
                
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Summary recommendations
            st.markdown("#### 💡 Test Selection Recommendations")
            
            recommendations = []
            
            for row in wilcoxon_results:
                metric = row['Metric']
                signed_available = row['Signed p-value'] != "N/A"
                ranksum_available = row['Rank-Sum p'] != "N/A"
                
                if signed_available and ranksum_available:
                    signed_p = float(row['Signed p-value'])
                    ranksum_p = float(row['Rank-Sum p'])
                    
                    if signed_p < 0.05 and ranksum_p < 0.05:
                        recommendation = "Both tests significant - strong evidence of difference"
                    elif signed_p < 0.05:
                        recommendation = "Signed-rank significant - paired analysis preferred"
                    elif ranksum_p < 0.05:
                        recommendation = "Rank-sum significant - independent analysis"
                    else:
                        recommendation = "No significant difference detected"
                elif signed_available:
                    recommendation = "Use signed-rank (paired data available)"
                elif ranksum_available:
                    recommendation = "Use rank-sum (independent samples)"
                else:
                    recommendation = "Insufficient data for Wilcoxon tests"
                
                recommendations.append({
                    'Metric': metric,
                    'Recommendation': recommendation,
                    'Preferred Test': 'Signed-Rank' if signed_available else 'Rank-Sum' if ranksum_available else 'Neither'
                })
            
            rec_df = pd.DataFrame(recommendations)
            st.dataframe(rec_df, use_container_width=True)
            
            st.info("""
            **When to use Wilcoxon Tests:**
            - **Signed-Rank**: When you have paired observations (same instances evaluated by both methods)
            - **Rank-Sum**: When you have independent samples from two groups
            - **Both are robust** to outliers and don't assume normal distributions
            - **More powerful** than simple non-parametric tests like the median test
            """)
    
    def _perform_multimethod_tests(self, data: Dict):
        """Perform tests comparing multiple methods (Friedman test etc.)"""
        st.markdown("##### 📊 Multi-method Comparison Tests")
        
        combined_data = data['combined']
        methods = combined_data['Method'].unique()
        
        if len(methods) < 3:
            st.warning("Multi-method tests require at least 3 methods. Add more methods to comparison.")
            return
        
        metrics = ['faithfulness', 'stability', 'completeness', 'sparsity']
        friedman_results = []
        
        # Prepare data for Friedman test
        for metric in metrics:
            # Group data by dataset/model combinations to create "blocks"
            combined_data['combo'] = combined_data['Dataset'] + "_" + combined_data['Model']
            pivot_data = combined_data.pivot_table(
                values=metric, 
                index='combo', 
                columns='Method', 
                aggfunc='mean'
            ).dropna()
            
            if len(pivot_data) > 1 and len(pivot_data.columns) >= 3:
                # Friedman test (non-parametric ANOVA for repeated measures)
                try:
                    friedman_stat, friedman_p = friedmanchisquare(*[pivot_data[col].values for col in pivot_data.columns])
                    
                    # Kruskal-Wallis test (non-parametric ANOVA for independent samples)
                    kruskal_stat, kruskal_p = kruskal(*[combined_data[combined_data['Method'] == method][metric].values 
                                                       for method in methods])
                    
                    # Effect size for Friedman (Kendall's W)
                    n_blocks = len(pivot_data)
                    k_methods = len(pivot_data.columns)
                    kendalls_w = friedman_stat / (n_blocks * (k_methods - 1))
                    
                    friedman_results.append({
                        'Metric': metric.title(),
                        'Friedman χ²': f"{friedman_stat:.3f}",
                        'Friedman p': f"{friedman_p:.4f}",
                        'Kruskal-Wallis': f"{kruskal_stat:.3f}",
                        'KW p-value': f"{kruskal_p:.4f}",
                        "Kendall's W": f"{kendalls_w:.3f}",
                        'Blocks (n)': n_blocks,
                        'Methods (k)': k_methods,
                        'Friedman Sig': self._get_significance_level(friedman_p),
                        'KW Sig': self._get_significance_level(kruskal_p)
                    })
                except Exception as e:
                    st.warning(f"Could not perform Friedman test for {metric}: {e}")
        
        if friedman_results:
            friedman_df = pd.DataFrame(friedman_results)
            st.dataframe(friedman_df, use_container_width=True)
            
            # Post-hoc analysis if significant
            significant_metrics = [row for row in friedman_results if float(row['Friedman p']) < 0.05]
            
            if significant_metrics:
                st.markdown("##### 🎯 Post-hoc Analysis (Significant Results)")
                
                for metric_result in significant_metrics:
                    metric = metric_result['Metric'].lower()
                    st.markdown(f"**{metric_result['Metric']} Post-hoc Comparisons:**")
                    
                    # Pairwise comparisons with Bonferroni correction
                    method_data = []
                    method_labels = []
                    
                    for method in methods:
                        method_values = combined_data[combined_data['Method'] == method][metric].values
                        method_data.extend(method_values)
                        method_labels.extend([method] * len(method_values))
                    
                    # Create comparison matrix
                    pairwise_results = []
                    for i, method1 in enumerate(methods):
                        for j, method2 in enumerate(methods[i+1:], i+1):
                            data1 = combined_data[combined_data['Method'] == method1][metric].values
                            data2 = combined_data[combined_data['Method'] == method2][metric].values
                            
                            if len(data1) > 0 and len(data2) > 0:
                                _, p_val = mannwhitneyu(data1, data2, alternative='two-sided')
                                # Bonferroni correction
                                n_comparisons = len(methods) * (len(methods) - 1) // 2
                                corrected_p = min(p_val * n_comparisons, 1.0)
                                
                                pairwise_results.append({
                                    'Comparison': f"{method1} vs {method2}",
                                    'Raw p-value': f"{p_val:.4f}",
                                    'Corrected p': f"{corrected_p:.4f}",
                                    'Significant': "Yes" if corrected_p < 0.05 else "No"
                                })
                    
                    if pairwise_results:
                        pairwise_df = pd.DataFrame(pairwise_results)
                        st.dataframe(pairwise_df, use_container_width=True)
            
            st.info("""
            **Multi-method Test Guide:**
            - Friedman Test: Non-parametric repeated measures ANOVA
            - Kruskal-Wallis: Non-parametric independent groups ANOVA  
            - Kendall's W: Effect size for Friedman (0-1, higher = more agreement)
            - Post-hoc: Pairwise comparisons with Bonferroni correction
            """)
    
    def _perform_datatype_specific_tests(self, method_a: str, method_b: str, data: Dict, data_types: Dict[str, bool]):
        """Perform data type specific statistical tests"""
        st.markdown("##### 📈 Data Type Specific Tests")
        
        detected_types = [dtype for dtype, present in data_types.items() if present]
        
        if not detected_types:
            st.warning("Could not detect specific data types. Using general tests.")
            return
        
        st.info(f"**Detected Data Types:** {', '.join(detected_types).title()}")
        
        # Tabular data specific tests
        if data_types['tabular']:
            st.markdown("**📊 Tabular Data Tests**")
            self._tabular_specific_tests(method_a, method_b, data)
        
        # Image data specific tests
        if data_types['image']:
            st.markdown("**🖼️ Image Data Tests**")
            self._image_specific_tests(method_a, method_b, data)
        
        # Text data specific tests
        if data_types['text']:
            st.markdown("**📝 Text Data Tests**")
            self._text_specific_tests(method_a, method_b, data)
    
    def _tabular_specific_tests(self, method_a: str, method_b: str, data: Dict):
        """Perform tabular data specific tests"""
        tabular_results = []
        
        # Focus on tabular-relevant metrics
        tabular_metrics = ['faithfulness', 'completeness', 'consistency', 'sparsity']
        
        for metric in tabular_metrics:
            a_values = data['method_a'][metric].values
            b_values = data['method_b'][metric].values
            
            if len(a_values) > 1 and len(b_values) > 1:
                # Permutation test (exact p-value for small samples)
                def permutation_test(x, y, n_permutations=10000):
                    observed_diff = np.mean(x) - np.mean(y)
                    combined = np.concatenate([x, y])
                    
                    perm_diffs = []
                    for _ in range(n_permutations):
                        np.random.shuffle(combined)
                        perm_x = combined[:len(x)]
                        perm_y = combined[len(x):]
                        perm_diffs.append(np.mean(perm_x) - np.mean(perm_y))
                    
                    p_value = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff))
                    return observed_diff, p_value
                
                perm_diff, perm_p = permutation_test(a_values, b_values)
                
                # Bootstrap confidence interval for difference
                def bootstrap_ci(x, y, n_bootstrap=1000, confidence=0.95):
                    bootstrap_diffs = []
                    for _ in range(n_bootstrap):
                        boot_x = np.random.choice(x, size=len(x), replace=True)
                        boot_y = np.random.choice(y, size=len(y), replace=True)
                        bootstrap_diffs.append(np.mean(boot_x) - np.mean(boot_y))
                    
                    alpha = 1 - confidence
                    lower = np.percentile(bootstrap_diffs, 100 * alpha/2)
                    upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha/2))
                    return lower, upper
                
                ci_lower, ci_upper = bootstrap_ci(a_values, b_values)
                
                tabular_results.append({
                    'Metric': metric.title(),
                    'Permutation p': f"{perm_p:.4f}",
                    'Mean Difference': f"{perm_diff:.3f}",
                    '95% CI Lower': f"{ci_lower:.3f}",
                    '95% CI Upper': f"{ci_upper:.3f}",
                    'CI Contains 0': "Yes" if ci_lower <= 0 <= ci_upper else "No",
                    'Significance': self._get_significance_level(perm_p)
                })
        
        if tabular_results:
            tabular_df = pd.DataFrame(tabular_results)
            st.dataframe(tabular_df, use_container_width=True)
    
    def _image_specific_tests(self, method_a: str, method_b: str, data: Dict):
        """Perform image data specific tests"""
        st.info("**Image Data Analysis:** Focusing on spatial consistency and visual interpretation metrics")
        
        # Image-specific metrics might include spatial coherence, visual saliency measures
        image_metrics = ['stability', 'consistency', 'completeness']
        image_results = []
        
        for metric in image_metrics:
            a_values = data['method_a'][metric].values
            b_values = data['method_b'][metric].values
            
            if len(a_values) > 1 and len(b_values) > 1:
                # Sign test (good for image data where direction of change matters)
                differences = a_values[:min(len(a_values), len(b_values))] - b_values[:min(len(a_values), len(b_values))]
                positive_diffs = np.sum(differences > 0)
                total_diffs = len(differences[differences != 0])
                
                if total_diffs > 0:
                    sign_p = 2 * min(stats.binom.cdf(positive_diffs, total_diffs, 0.5),
                                   1 - stats.binom.cdf(positive_diffs - 1, total_diffs, 0.5))
                else:
                    sign_p = 1.0
                
                # Median test
                median_combined = np.median(np.concatenate([a_values, b_values]))
                a_above_median = np.sum(a_values > median_combined)
                b_above_median = np.sum(b_values > median_combined)
                a_below_median = len(a_values) - a_above_median
                b_below_median = len(b_values) - b_above_median
                
                contingency_table = np.array([[a_above_median, a_below_median],
                                            [b_above_median, b_below_median]])
                
                try:
                    chi2_stat, chi2_p, _, _ = chi2_contingency(contingency_table)
                except:
                    chi2_stat, chi2_p = np.nan, np.nan
                
                image_results.append({
                    'Metric': metric.title(),
                    'Sign Test p': f"{sign_p:.4f}",
                    'Positive Diffs': f"{positive_diffs}/{total_diffs}",
                    'Median Test χ²': f"{chi2_stat:.3f}" if not np.isnan(chi2_stat) else "N/A",
                    'Median Test p': f"{chi2_p:.4f}" if not np.isnan(chi2_p) else "N/A",
                    'Sign Significance': self._get_significance_level(sign_p),
                    'Median Significance': self._get_significance_level(chi2_p) if not np.isnan(chi2_p) else "N/A"
                })
        
        if image_results:
            image_df = pd.DataFrame(image_results)
            st.dataframe(image_df, use_container_width=True)
    
    def _text_specific_tests(self, method_a: str, method_b: str, data: Dict):
        """Perform text data specific tests"""
        st.info("**Text Data Analysis:** Focusing on semantic consistency and linguistic interpretation")
        
        # Text-specific considerations
        text_metrics = ['faithfulness', 'consistency', 'simplicity']
        text_results = []
        
        for metric in text_metrics:
            a_values = data['method_a'][metric].values
            b_values = data['method_b'][metric].values
            
            if len(a_values) > 1 and len(b_values) > 1:
                # McNemar's test (for paired categorical outcomes in text analysis)
                # Convert continuous scores to binary (above/below median)
                combined_median = np.median(np.concatenate([a_values, b_values]))
                
                if len(a_values) == len(b_values):
                    a_binary = (a_values > combined_median).astype(int)
                    b_binary = (b_values > combined_median).astype(int)
                    
                    # Create contingency table for McNemar's test
                    both_high = np.sum((a_binary == 1) & (b_binary == 1))
                    a_high_b_low = np.sum((a_binary == 1) & (b_binary == 0))
                    a_low_b_high = np.sum((a_binary == 0) & (b_binary == 1))
                    both_low = np.sum((a_binary == 0) & (b_binary == 0))
                    
                    # McNemar's test focuses on discordant pairs
                    if a_high_b_low + a_low_b_high > 0:
                        mcnemar_stat = (abs(a_high_b_low - a_low_b_high) - 1)**2 / (a_high_b_low + a_low_b_high)
                        mcnemar_p = 1 - stats.chi2.cdf(mcnemar_stat, 1)
                    else:
                        mcnemar_stat, mcnemar_p = 0, 1.0
                else:
                    mcnemar_stat, mcnemar_p = np.nan, np.nan
                
                # Kolmogorov-Smirnov test (for distribution differences)
                ks_stat, ks_p = stats.ks_2samp(a_values, b_values)
                
                # Anderson-Darling test (more sensitive to tails)
                try:
                    ad_stat, ad_critical, ad_p = stats.anderson_ksamp([a_values, b_values])
                except:
                    ad_stat, ad_p = np.nan, np.nan
                
                text_results.append({
                    'Metric': metric.title(),
                    'KS Statistic': f"{ks_stat:.3f}",
                    'KS p-value': f"{ks_p:.4f}",
                    'McNemar χ²': f"{mcnemar_stat:.3f}" if not np.isnan(mcnemar_stat) else "N/A",
                    'McNemar p': f"{mcnemar_p:.4f}" if not np.isnan(mcnemar_p) else "N/A",
                    'AD Statistic': f"{ad_stat:.3f}" if not np.isnan(ad_stat) else "N/A",
                    'KS Significance': self._get_significance_level(ks_p),
                    'McNemar Sig': self._get_significance_level(mcnemar_p) if not np.isnan(mcnemar_p) else "N/A"
                })
        
        if text_results:
            text_df = pd.DataFrame(text_results)
            st.dataframe(text_df, use_container_width=True)
    
    def _get_significance_level(self, p_value: float) -> str:
        """Get significance level indicator"""
        if pd.isna(p_value):
            return "N/A"
        if p_value < 0.001:
            return "***"
        elif p_value < 0.01:
            return "**"
        elif p_value < 0.05:
            return "*"
        else:
            return "ns"
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        if d < 0.2:
            return "Negligible"
        elif d < 0.5:
            return "Small"
        elif d < 0.8:
            return "Medium"
        else:
            return "Large"
    
    def _create_performance_matrix(self, df: pd.DataFrame, metrics: List[str], agg_method: str, matrix_type: str):
        """Create interactive performance matrix"""
        st.markdown("#### 🔥 Interactive Performance Matrix")
        
        # Parse matrix type
        if matrix_type == 'Method vs Dataset':
            index_col, column_col = 'Method', 'Dataset'
        elif matrix_type == 'Method vs Model':
            index_col, column_col = 'Method', 'Model'
        else:  # Dataset vs Model
            index_col, column_col = 'Dataset', 'Model'
        
        # Create matrix for each metric
        for metric in metrics:
            pivot_data = df.pivot_table(
                values=metric,
                index=index_col,
                columns=column_col,
                aggfunc=agg_method,
                fill_value=0
            ).round(3)
            
            # Create heatmap
            fig = px.imshow(
                pivot_data.values,
                x=pivot_data.columns,
                y=pivot_data.index,
                color_continuous_scale='RdYlBu',
                aspect='auto',
                text_auto=True
            )
            
            fig.update_layout(
                title=f"{metric.title()} - {matrix_type} ({agg_method.title()})",
                height=400,
                font=dict(size=12)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _calculate_live_metrics(self) -> Dict[str, Any]:
        """Calculate live performance metrics"""
        metrics_data = self._extract_metrics_data()
        
        if not metrics_data:
            return {}
        
        df = pd.DataFrame(metrics_data)
        
        # Best faithfulness
        best_faithfulness_idx = df['faithfulness'].idxmax()
        best_faithfulness = df.loc[best_faithfulness_idx, 'faithfulness']
        
        # Fastest method (lowest time complexity)
        fastest_method_idx = df['time_complexity'].idxmin()
        fastest_method = df.loc[fastest_method_idx, 'Method']
        
        # Most stable
        most_stable_idx = df['stability'].idxmax()
        most_stable = df.loc[most_stable_idx, 'Method']
        stability_score = df.loc[most_stable_idx, 'stability']
        
        # Overall winner (weighted score)
        weights = {'faithfulness': 0.3, 'stability': 0.2, 'completeness': 0.2, 'consistency': 0.15, 'simplicity': 0.15}
        
        df['weighted_score'] = sum(df[metric] * weight for metric, weight in weights.items())
        winner_idx = df['weighted_score'].idxmax()
        overall_winner = df.loc[winner_idx, 'Method']
        winner_score = df.loc[winner_idx, 'weighted_score']
        
        return {
            'best_faithfulness': best_faithfulness,
            'fastest_method': fastest_method,
            'most_stable': most_stable,
            'stability_score': stability_score,
            'overall_winner': overall_winner,
            'winner_score': winner_score,
            'faithfulness_improvement': best_faithfulness - df['faithfulness'].median(),
            'speed_improvement': df['time_complexity'].median() - df.loc[fastest_method_idx, 'time_complexity']
        }
    
    def _create_live_radar_chart(self):
        """Create live updating radar chart"""
        metrics_data = self._extract_metrics_data()
        
        if not metrics_data:
            st.warning("No data available for live radar chart.")
            return
        
        df = pd.DataFrame(metrics_data)
        methods = df['Method'].unique()[:5]  # Top 5 methods
        
        fig = go.Figure()
        
        radar_metrics = ['faithfulness', 'stability', 'completeness', 'consistency', 'simplicity']
        colors = px.colors.qualitative.Set3
        
        for i, method in enumerate(methods):
            method_data = df[df['Method'] == method][radar_metrics].mean()
            values = method_data.values.tolist()
            values += values[:1]  # Close the radar
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=radar_metrics + [radar_metrics[0]],
                fill='toself',
                name=method,
                line_color=colors[i % len(colors)]
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            showlegend=True,
            height=500,
            title="🔴 Live Method Performance Radar"
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _create_performance_timeline(self):
        """Create performance evolution timeline"""
        st.info("📈 Performance timeline would show historical performance data across different experiment runs.")
        
        # Placeholder for timeline visualization
        # In a real implementation, this would show performance trends over time
        fake_timeline_data = {
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='D'),
            'faithfulness': np.random.uniform(0.6, 0.9, 10),
            'stability': np.random.uniform(0.5, 0.8, 10),
            'method': ['SHAP'] * 5 + ['LIME'] * 5
        }
        
        timeline_df = pd.DataFrame(fake_timeline_data)
        
        fig = px.line(
            timeline_df,
            x='timestamp',
            y='faithfulness',
            color='method',
            title="Performance Evolution Over Time (Sample Data)",
            markers=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def _execute_custom_analysis(self, analysis_type: str, params: Dict) -> Optional[Dict]:
        """Execute custom analysis based on selected type"""
        metrics_data = self._extract_metrics_data()
        
        if not metrics_data:
            return None
        
        df = pd.DataFrame(metrics_data)
        
        if analysis_type == "Feature Importance Comparison":
            # Simulate feature importance analysis
            return {"status": "completed", "type": "feature_importance", "data": df}
        
        elif analysis_type == "Time vs Accuracy Trade-off":
            # Create time vs accuracy analysis
            return {"status": "completed", "type": "time_accuracy", "data": df}
        
        elif analysis_type == "Method Clustering":
            # Perform method clustering
            from sklearn.cluster import KMeans
            metrics = ['faithfulness', 'stability', 'completeness', 'consistency']
            
            try:
                method_means = df.groupby('Method')[metrics].mean()
                
                n_clusters = params.get('n_clusters', 3)
                kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(method_means.values)
                
                result_data = method_means.copy()
                result_data['Cluster'] = clusters
                
                return {"status": "completed", "type": "clustering", "data": result_data, "n_clusters": n_clusters}
            except Exception as e:
                st.error(f"Clustering failed: {e}")
                return None
        
        return {"status": "completed", "type": "generic", "data": df}
    
    def _display_custom_results(self, analysis_type: str, result: Dict):
        """Display results of custom analysis"""
        st.markdown("#### 🎉 Analysis Results")
        
        if result['type'] == 'clustering':
            st.markdown("##### 📊 Method Clustering Results")
            
            cluster_data = result['data']
            
            # Display cluster assignments
            st.dataframe(cluster_data, use_container_width=True)
            
            # Create cluster visualization
            fig = px.scatter(
                cluster_data.reset_index(),
                x='faithfulness',
                y='stability',
                color='Cluster',
                hover_name='Method',
                title=f"Method Clustering ({result['n_clusters']} clusters)",
                color_continuous_scale='Viridis'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        elif result['type'] == 'time_accuracy':
            st.markdown("##### ⚡ Time vs Accuracy Trade-off")
            
            df = result['data']
            
            fig = px.scatter(
                df,
                x='time_complexity',
                y='faithfulness',
                color='Method',
                size='completeness',
                hover_data=['Dataset', 'Model'],
                title="Time Complexity vs Faithfulness Trade-off",
                labels={'time_complexity': 'Time Complexity', 'faithfulness': 'Faithfulness'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.markdown("##### 📋 Generic Analysis Results")
            st.dataframe(result['data'], use_container_width=True)


def create_explanation_comparator(experiment_data: Dict[str, Any]) -> ExplanationComparator:
    """Factory function to create an ExplanationComparator instance"""
    return ExplanationComparator(experiment_data)