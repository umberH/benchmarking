#!/usr/bin/env python3
"""
Statistical Experiment Planner for XAI Method Comparison
A component for planning rigorous statistical experiments to compare explanation methods
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
import itertools
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import scikit_posthocs as sp

@dataclass
class ExperimentConfig:
    """Configuration for a statistical experiment"""
    name: str
    description: str
    datasets: List[str]
    models: List[str]
    explanation_methods: List[str]
    metrics: List[str]
    sample_size: int
    alpha_level: float
    power: float
    effect_size: float
    test_type: str
    correction_method: str
    randomization_scheme: str
    blocking_factor: Optional[str] = None

class ExperimentPlanner:
    """Interactive experiment planner for statistical significance testing"""
    
    def __init__(self, available_data: Dict[str, Any]):
        """Initialize with available datasets, models, and methods"""
        self.available_data = available_data
        self.experiment_configs = []
        
    def render_planner(self):
        """Render the main experiment planning interface"""
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 50%, #fecfef 100%);
            padding: 2rem;
            border-radius: 20px;
            margin: 1rem 0;
            text-align: center;
            box-shadow: 0 10px 30px rgba(255, 154, 158, 0.3);
        ">
            <h2 style="color: white; margin: 0; font-size: 2.5rem; font-weight: 700;">
                🧪 Statistical Experiment Planner
            </h2>
            <p style="color: rgba(255,255,255,0.9); margin: 0.5rem 0 0 0; font-size: 1.2rem;">
                Design rigorous experiments for comparing XAI methods
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create tabs for different planning aspects
        plan_tab1, plan_tab2, plan_tab3, plan_tab4, plan_tab5, plan_tab6, plan_tab7 = st.tabs([
            "🎯 Experiment Design",
            "📊 Power Analysis", 
            "🔬 Sample Size Calc",
            "📋 Comparison Matrix",
            "🚀 Execution Plan",
            "🧮 Specialized Tests",
            "📊 Critical Difference"
        ])
        
        with plan_tab1:
            self._render_experiment_design()
        
        with plan_tab2:
            self._render_power_analysis()
        
        with plan_tab3:
            self._render_sample_size_calculator()
        
        with plan_tab4:
            self._render_comparison_matrix()
        
        with plan_tab5:
            self._render_execution_plan()
        
        with plan_tab6:
            self._render_specialized_tests()
        
        with plan_tab7:
            self._render_critical_difference_analysis()
    
    def _render_experiment_design(self):
        """Render experiment design interface"""
        st.markdown("### 🎯 Experiment Design Configuration")
        
        # Basic experiment info
        col1, col2 = st.columns(2)
        
        with col1:
            experiment_name = st.text_input(
                "Experiment Name:",
                value="XAI Method Comparison Study",
                help="Give your experiment a descriptive name"
            )
            
            experiment_description = st.text_area(
                "Description:",
                value="Statistical comparison of explanation methods across multiple datasets and models",
                help="Describe the purpose and scope of your experiment"
            )
        
        with col2:
            # Research questions
            st.markdown("#### 🤔 Research Questions")
            research_questions = st.multiselect(
                "Select research questions:",
                [
                    "Which explanation method is most faithful?",
                    "Are there significant differences in stability?",
                    "How does performance vary across datasets?",
                    "Which method is most consistent across models?",
                    "What's the speed vs accuracy trade-off?",
                    "Do methods perform differently on tabular vs image data?"
                ],
                default=["Which explanation method is most faithful?"],
                help="Select the research questions your experiment will address"
            )
        
        # Extract available options from data
        available_datasets, available_models, available_methods = self._extract_available_options()
        
        # Experimental factors
        st.markdown("#### 🧪 Experimental Factors")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Datasets (Treatment)**")
            selected_datasets = st.multiselect(
                "Select datasets:",
                available_datasets,
                default=available_datasets[:3] if len(available_datasets) >= 3 else available_datasets,
                help="Choose datasets for comparison"
            )
        
        with col2:
            st.markdown("**Models (Blocking Factor)**")
            selected_models = st.multiselect(
                "Select models:",
                available_models,
                default=available_models[:3] if len(available_models) >= 3 else available_models,
                help="Choose models to test explanation methods on"
            )
        
        with col3:
            st.markdown("**Explanation Methods (Main Factor)**")
            selected_methods = st.multiselect(
                "Select explanation methods:",
                available_methods,
                default=available_methods[:4] if len(available_methods) >= 4 else available_methods,
                help="Choose explanation methods to compare"
            )
        
        # Dependent variables (metrics)
        st.markdown("#### 📊 Dependent Variables (Metrics)")
        
        available_metrics = [
            'faithfulness', 'stability', 'completeness', 'sparsity', 
            'monotonicity', 'consistency', 'time_complexity', 'simplicity'
        ]
        
        col1, col2 = st.columns(2)
        
        with col1:
            primary_metric = st.selectbox(
                "Primary metric (main analysis):",
                available_metrics,
                index=0,
                help="The main metric for your primary analysis"
            )
        
        with col2:
            secondary_metrics = st.multiselect(
                "Secondary metrics (exploratory):",
                [m for m in available_metrics if m != primary_metric],
                default=['stability', 'completeness'],
                help="Additional metrics for secondary analyses"
            )
        
        all_metrics = [primary_metric] + secondary_metrics
        
        # Experimental design type
        st.markdown("#### 🏗️ Experimental Design")
        
        col1, col2 = st.columns(2)
        
        with col1:
            design_type = st.selectbox(
                "Experimental design:",
                [
                    "Completely Randomized Design (CRD)",
                    "Randomized Complete Block Design (RCBD)", 
                    "Latin Square Design",
                    "Factorial Design",
                    "Repeated Measures Design"
                ],
                index=1,
                help="Choose the experimental design structure"
            )
        
        with col2:
            blocking_factor = st.selectbox(
                "Blocking factor:",
                ["None", "Dataset", "Model", "Data Type"],
                index=2,
                help="Factor to use for blocking (reduces noise)"
            )
        
        # Statistical test configuration
        st.markdown("#### 📈 Statistical Analysis Plan")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_type = st.selectbox(
                "Primary statistical test:",
                [
                    "Friedman Test (non-parametric)",
                    "Kruskal-Wallis Test", 
                    "Wilcoxon Signed-Rank Test",
                    "McNemar Test (paired proportions)",
                    "One-way ANOVA",
                    "Two-way ANOVA",
                    "Mixed-effects ANOVA",
                    "Paired t-tests with correction"
                ],
                index=0,
                help="Main statistical test for your analysis",
                key="experiment_design_test_type"
            )
        
        with col2:
            alpha_level = st.selectbox(
                "Significance level (α):",
                [0.001, 0.01, 0.05, 0.10],
                index=2,
                help="Type I error rate"
            )
        
        with col3:
            correction_method = st.selectbox(
                "Multiple comparison correction:",
                [
                    "Bonferroni",
                    "Holm-Bonferroni", 
                    "Benjamini-Hochberg (FDR)",
                    "Tukey HSD",
                    "None"
                ],
                index=1,
                help="Method to control family-wise error rate"
            )
        
        # Save experiment configuration
        if st.button("💾 Save Experiment Configuration", type="primary"):
            config = ExperimentConfig(
                name=experiment_name,
                description=experiment_description,
                datasets=selected_datasets,
                models=selected_models,
                explanation_methods=selected_methods,
                metrics=all_metrics,
                sample_size=0,  # Will be calculated
                alpha_level=alpha_level,
                power=0.8,  # Default
                effect_size=0.5,  # Default medium effect
                test_type=test_type,
                correction_method=correction_method,
                randomization_scheme=design_type,
                blocking_factor=blocking_factor if blocking_factor != "None" else None
            )
            
            self.experiment_configs.append(config)
            st.success(f"✅ Experiment configuration '{experiment_name}' saved!")
            
            # Display summary
            self._display_experiment_summary(config)
    
    def _render_power_analysis(self):
        """Render power analysis interface"""
        st.markdown("### 📊 Statistical Power Analysis")
        
        st.info("""
        **Power Analysis** helps determine if your experiment can detect meaningful differences.
        - **Power (1-β)**: Probability of detecting a true effect
        - **Effect Size**: Magnitude of difference you want to detect
        - **Alpha (α)**: Type I error rate (false positive)
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### ⚙️ Power Analysis Parameters")
            
            # Effect size
            effect_size_type = st.selectbox(
                "Effect size type:",
                ["Cohen's d (standardized)", "Raw difference", "Percentage difference"],
                help="How to specify the effect size"
            )
            
            if effect_size_type == "Cohen's d (standardized)":
                effect_size = st.selectbox(
                    "Expected effect size:",
                    [0.2, 0.5, 0.8, 1.0, 1.2],
                    index=1,
                    format_func=lambda x: f"{x} ({'Small' if x==0.2 else 'Medium' if x==0.5 else 'Large' if x==0.8 else 'Very Large'})",
                    help="Cohen's d: Small (0.2), Medium (0.5), Large (0.8)"
                )
            else:
                effect_size = st.number_input(
                    "Effect size:",
                    min_value=0.01,
                    max_value=2.0,
                    value=0.5,
                    step=0.1,
                    help="Minimum meaningful difference"
                )
            
            # Power parameters
            desired_power = st.slider(
                "Desired statistical power:",
                min_value=0.50,
                max_value=0.99,
                value=0.80,
                step=0.05,
                help="Probability of detecting the effect if it exists"
            )
            
            alpha = st.selectbox(
                "Alpha level:",
                [0.001, 0.01, 0.05, 0.10],
                index=2,
                help="Type I error rate"
            )
            
            # Number of groups
            n_methods = st.number_input(
                "Number of explanation methods:",
                min_value=2,
                max_value=10,
                value=4,
                help="Number of groups in comparison"
            )
        
        with col2:
            st.markdown("#### 📈 Power Calculation Results")
            
            # Calculate required sample size
            required_n = self._calculate_sample_size(
                effect_size=effect_size,
                alpha=alpha,
                power=desired_power,
                n_groups=n_methods
            )
            
            # Display results
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.metric(
                    "Required Sample Size",
                    f"{required_n} per group",
                    help="Minimum number of observations per explanation method"
                )
                
                total_observations = required_n * n_methods
                st.metric(
                    "Total Observations",
                    f"{total_observations}",
                    help="Total number of experiments needed"
                )
            
            with result_col2:
                # Calculate power for different sample sizes
                sample_sizes = range(5, 50, 5)
                powers = [self._calculate_power(n, effect_size, alpha, n_methods) for n in sample_sizes]
                
                # Power curve
                fig = px.line(
                    x=sample_sizes,
                    y=powers,
                    title="Power Curve",
                    labels={'x': 'Sample Size per Group', 'y': 'Statistical Power'},
                    markers=True
                )
                
                # Add target power line
                fig.add_hline(
                    y=desired_power,
                    line_dash="dash",
                    line_color="red",
                    annotation_text=f"Target Power ({desired_power})"
                )
                
                # Add required sample size line
                fig.add_vline(
                    x=required_n,
                    line_dash="dash", 
                    line_color="green",
                    annotation_text=f"Required N ({required_n})"
                )
                
                fig.update_layout(height=300)
                st.plotly_chart(fig, use_container_width=True)
        
        # Sensitivity analysis
        st.markdown("#### 🔍 Sensitivity Analysis")
        
        # Create interactive sensitivity table
        sensitivity_data = []
        
        for es in [0.2, 0.3, 0.5, 0.8]:
            for pwr in [0.70, 0.80, 0.90]:
                n_req = self._calculate_sample_size(es, alpha, pwr, n_methods)
                sensitivity_data.append({
                    'Effect Size': es,
                    'Power': pwr,
                    'Required N per Group': n_req,
                    'Total Observations': n_req * n_methods,
                    'Interpretation': self._interpret_effect_size(es)
                })
        
        sensitivity_df = pd.DataFrame(sensitivity_data)
        
        # Style the dataframe
        styled_df = sensitivity_df.style.background_gradient(
            subset=['Required N per Group'], 
            cmap='RdYlGn_r'
        )
        
        st.dataframe(styled_df, use_container_width=True)
    
    def _render_sample_size_calculator(self):
        """Render sample size calculator"""
        st.markdown("### 🔬 Sample Size Calculator")
        
        # Calculator type
        calc_type = st.selectbox(
            "Calculator type:",
            [
                "General Comparison (ANOVA/Kruskal-Wallis)",
                "Pairwise Comparisons (t-tests)",
                "Equivalence Testing",
                "Non-inferiority Testing"
            ],
            help="Choose the type of analysis for sample size calculation"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Input Parameters")
            
            if calc_type == "General Comparison (ANOVA/Kruskal-Wallis)":
                # ANOVA sample size
                between_var = st.number_input(
                    "Between-group variance:",
                    min_value=0.01,
                    max_value=5.0,
                    value=0.5,
                    step=0.1,
                    help="Expected variance between explanation methods"
                )
                
                within_var = st.number_input(
                    "Within-group variance:",
                    min_value=0.01,
                    max_value=5.0,
                    value=0.3,
                    step=0.1,
                    help="Expected variance within each method"
                )
                
                f_effect_size = between_var / within_var
                
            elif calc_type == "Pairwise Comparisons (t-tests)":
                # t-test sample size
                mean_diff = st.number_input(
                    "Expected mean difference:",
                    min_value=0.01,
                    max_value=2.0,
                    value=0.3,
                    step=0.05,
                    help="Expected difference between methods"
                )
                
                pooled_sd = st.number_input(
                    "Pooled standard deviation:",
                    min_value=0.01,
                    max_value=2.0,
                    value=0.5,
                    step=0.05,
                    help="Expected standard deviation"
                )
                
                cohens_d = mean_diff / pooled_sd
            
            # Common parameters
            alpha = st.selectbox("Alpha level:", [0.001, 0.01, 0.05], index=2)
            power = st.slider("Desired power:", 0.7, 0.95, 0.8, 0.05)
            
            # Multiple comparisons
            n_comparisons = st.number_input(
                "Number of comparisons:",
                min_value=1,
                max_value=50,
                value=6,
                help="Total number of pairwise comparisons"
            )
            
            correction = st.selectbox(
                "Correction method:",
                ["None", "Bonferroni", "Holm", "FDR"],
                index=1
            )
            
            # Adjust alpha for multiple comparisons
            if correction == "Bonferroni":
                adjusted_alpha = alpha / n_comparisons
            else:
                adjusted_alpha = alpha  # Simplified for other methods
        
        with col2:
            st.markdown("#### 📊 Results")
            
            if calc_type == "General Comparison (ANOVA/Kruskal-Wallis)":
                # Use F-test sample size calculation
                # Simplified calculation
                n_per_group = max(3, int(
                    2 * ((stats.norm.ppf(1-adjusted_alpha/2) + stats.norm.ppf(power))**2) / (f_effect_size**2)
                ))
                
                st.metric("F Effect Size", f"{f_effect_size:.3f}")
                
            else:  # t-test based
                # Calculate sample size for t-test
                delta = cohens_d
                n_per_group = max(2, int(
                    2 * ((stats.norm.ppf(1-adjusted_alpha/2) + stats.norm.ppf(power))**2) / (delta**2)
                ))
                
                st.metric("Cohen's d", f"{cohens_d:.3f}")
            
            # Display results
            st.metric("Sample Size per Group", f"{n_per_group}")
            st.metric("Total Sample Size", f"{n_per_group * 4}")  # Assuming 4 methods
            st.metric("Adjusted Alpha", f"{adjusted_alpha:.6f}")
            
            # Cost estimation
            st.markdown("#### 💰 Resource Estimation")
            
            time_per_experiment = st.number_input(
                "Time per experiment (minutes):",
                min_value=0.1,
                max_value=60.0,
                value=2.0,
                step=0.5
            )
            
            total_experiments = n_per_group * 4 * 3  # methods * datasets
            total_time_hours = (total_experiments * time_per_experiment) / 60
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric("Total Experiments", f"{total_experiments}")
                st.metric("Estimated Time", f"{total_time_hours:.1f} hours")
            
            with col_b:
                # Visualization of sample allocation
                methods = ['SHAP', 'LIME', 'IG', 'Counterfactual']
                allocation_data = pd.DataFrame({
                    'Method': methods,
                    'Sample Size': [n_per_group] * len(methods)
                })
                
                fig = px.bar(
                    allocation_data,
                    x='Method',
                    y='Sample Size',
                    title="Sample Size Allocation",
                    color='Sample Size',
                    color_continuous_scale='Blues'
                )
                fig.update_layout(height=250)
                st.plotly_chart(fig, use_container_width=True)
    
    def _render_comparison_matrix(self):
        """Render comparison matrix planner"""
        st.markdown("### 📋 Comparison Matrix Planner")
        
        st.info("""
        Plan all possible comparisons in your experiment. This helps ensure you have 
        adequate power for all comparisons of interest.
        """)
        
        # Load available options
        available_datasets, available_models, available_methods = self._extract_available_options()
        
        # Selection for matrix
        col1, col2 = st.columns(2)
        
        with col1:
            selected_methods = st.multiselect(
                "Select methods for comparison matrix:",
                available_methods,
                default=available_methods[:4] if len(available_methods) >= 4 else available_methods
            )
            
            selected_datasets = st.multiselect(
                "Select datasets:",
                available_datasets,
                default=available_datasets[:3] if len(available_datasets) >= 3 else available_datasets
            )
        
        with col2:
            comparison_type = st.selectbox(
                "Comparison type:",
                [
                    "All pairwise method comparisons",
                    "Methods vs control (best performer)",
                    "Methods vs baseline",
                    "Custom comparisons"
                ]
            )
            
            analysis_level = st.selectbox(
                "Analysis level:",
                [
                    "Overall (pooled across datasets)",
                    "Per dataset (separate analyses)",
                    "Stratified by data type"
                ]
            )
        
        if selected_methods and len(selected_methods) >= 2:
            # Generate comparison matrix
            st.markdown("#### 🔄 Planned Comparisons")
            
            if comparison_type == "All pairwise method comparisons":
                comparisons = list(itertools.combinations(selected_methods, 2))
            elif comparison_type == "Methods vs control (best performer)":
                control_method = st.selectbox("Select control method:", selected_methods)
                comparisons = [(control_method, method) for method in selected_methods if method != control_method]
            else:
                comparisons = list(itertools.combinations(selected_methods, 2))  # Default
            
            # Create comparison matrix dataframe
            comparison_data = []
            
            for i, (method1, method2) in enumerate(comparisons):
                for dataset in selected_datasets:
                    comparison_data.append({
                        'Comparison ID': f"C{i+1:02d}",
                        'Method A': method1,
                        'Method B': method2,
                        'Dataset': dataset,
                        'Priority': 'High' if i < 3 else 'Medium',
                        'Expected Effect': 'Medium',
                        'Sample Size Needed': 15,  # Placeholder
                        'Status': 'Planned'
                    })
            
            comparison_df = pd.DataFrame(comparison_data)
            
            # Interactive table with editing capabilities
            st.dataframe(comparison_df, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Comparisons", len(comparison_df))
            
            with col2:
                high_priority = len(comparison_df[comparison_df['Priority'] == 'High'])
                st.metric("High Priority", high_priority)
            
            with col3:
                total_experiments = comparison_df['Sample Size Needed'].sum()
                st.metric("Total Experiments", total_experiments)
            
            # Comparison network visualization
            st.markdown("#### 🕸️ Comparison Network")
            
            # Create network graph
            import networkx as nx
            
            G = nx.Graph()
            
            # Add nodes (methods)
            for method in selected_methods:
                G.add_node(method)
            
            # Add edges (comparisons)
            for method1, method2 in comparisons:
                G.add_edge(method1, method2)
            
            # Get positions
            pos = nx.spring_layout(G)
            
            # Create plotly network graph
            edge_x = []
            edge_y = []
            
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            edge_trace = go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=2, color='#888'),
                hoverinfo='none',
                mode='lines'
            )
            
            node_x = []
            node_y = []
            node_text = []
            
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
            
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                hoverinfo='text',
                text=node_text,
                textposition="middle center",
                marker=dict(
                    size=50,
                    color='lightblue',
                    line=dict(width=2, color='darkblue')
                )
            )
            
            fig = go.Figure(data=[edge_trace, node_trace],
                          layout=go.Layout(
                              title=dict(text='Method Comparison Network', font_size=16),
                              showlegend=False,
                              hovermode='closest',
                              margin=dict(b=20,l=5,r=5,t=40),
                              annotations=[ dict(
                                  text="Each line represents a planned comparison",
                                  showarrow=False,
                                  xref="paper", yref="paper",
                                  x=0.005, y=-0.002,
                                  xanchor="left", yanchor="bottom",
                                  font=dict(color="#888", size=12)
                              )],
                              xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                              yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
                          )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_execution_plan(self):
        """Render execution plan interface"""
        st.markdown("### 🚀 Experiment Execution Plan")
        
        if not self.experiment_configs:
            st.warning("No experiment configurations saved. Please design an experiment first.")
            return
        
        # Select configuration
        config_names = [config.name for config in self.experiment_configs]
        selected_config_name = st.selectbox(
            "Select experiment configuration:",
            config_names
        )
        
        selected_config = next(config for config in self.experiment_configs if config.name == selected_config_name)
        
        # Generate execution timeline
        st.markdown("#### 📅 Execution Timeline")
        
        # Calculate execution phases
        phases = self._generate_execution_phases(selected_config)
        
        # Display timeline
        timeline_data = []
        start_day = 0
        
        for i, phase in enumerate(phases):
            timeline_data.append({
                'Phase': phase['name'],
                'Duration (days)': phase['duration'],
                'Start Day': start_day,
                'End Day': start_day + phase['duration'],
                'Description': phase['description'],
                'Resources': phase['resources']
            })
            start_day += phase['duration']
        
        timeline_df = pd.DataFrame(timeline_data)
        st.dataframe(timeline_df, use_container_width=True)
        
        # Gantt chart
        fig = px.timeline(
            timeline_df,
            x_start='Start Day',
            x_end='End Day', 
            y='Phase',
            title='Experiment Execution Timeline',
            color='Duration (days)',
            hover_data=['Description']
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Resource requirements
        st.markdown("#### 💻 Resource Requirements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            total_experiments = len(selected_config.datasets) * len(selected_config.models) * len(selected_config.explanation_methods)
            total_days = timeline_df['Duration (days)'].sum()
            
            st.metric("Total Experiments", total_experiments)
            st.metric("Estimated Duration", f"{total_days} days")
            st.metric("Computational Hours", f"{total_experiments * 0.5:.1f}")
        
        with col2:
            # Generate checklist
            st.markdown("**Pre-execution Checklist:**")
            
            checklist_items = [
                "Data preprocessing pipelines ready",
                "Model training scripts tested",
                "Explanation method implementations verified", 
                "Evaluation metrics implemented",
                "Result storage system configured",
                "Statistical analysis scripts prepared",
                "Progress monitoring dashboard set up",
                "Backup and recovery procedures in place"
            ]
            
            for item in checklist_items:
                st.checkbox(item, key=f"checklist_{item}")
        
        # Export experiment plan
        if st.button("📄 Export Experiment Plan", type="primary"):
            plan_dict = {
                'experiment_config': selected_config.__dict__,
                'timeline': timeline_df.to_dict('records'),
                'total_experiments': total_experiments,
                'estimated_duration': total_days,
                'checklist': checklist_items,
                'created_date': datetime.now().isoformat()
            }
            
            plan_json = json.dumps(plan_dict, indent=2)
            st.download_button(
                label="💾 Download Plan (JSON)",
                data=plan_json,
                file_name=f"experiment_plan_{selected_config.name.replace(' ', '_')}.json",
                mime="application/json"
            )
            
            st.success("✅ Experiment plan exported successfully!")
    
    def _extract_available_options(self) -> Tuple[List[str], List[str], List[str]]:
        """Extract available datasets, models, and methods from data"""
        datasets = set()
        models = set()
        methods = set()
        
        comprehensive_results = self.available_data.get('comprehensive_results', [])
        
        for result in comprehensive_results:
            if isinstance(result, dict):
                datasets.add(result.get('dataset', 'unknown'))
                models.add(result.get('model', 'unknown'))
                methods.add(result.get('explanation_method', 'unknown'))
        
        return sorted(list(datasets)), sorted(list(models)), sorted(list(methods))
    
    def _calculate_sample_size(self, effect_size: float, alpha: float, power: float, n_groups: int) -> int:
        """Calculate required sample size for given parameters"""
        # Simplified calculation for ANOVA F-test
        # More sophisticated calculation would use specific statistical libraries
        
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        # For F-test with multiple groups
        df_num = n_groups - 1
        df_denom_factor = 2 * ((z_alpha + z_beta) ** 2) / (effect_size ** 2)
        
        n_per_group = max(3, int(df_denom_factor / df_num) + 2)
        
        return n_per_group
    
    def _calculate_power(self, n: int, effect_size: float, alpha: float, n_groups: int) -> float:
        """Calculate statistical power for given sample size"""
        # Simplified power calculation
        z_alpha = stats.norm.ppf(1 - alpha/2)
        
        # Effect size adjusted for sample size
        ncp = n * (effect_size ** 2) * (n_groups - 1) / 2
        
        # Approximate power calculation
        z_beta = np.sqrt(ncp) - z_alpha
        power = stats.norm.cdf(z_beta)
        
        return min(0.99, max(0.01, power))
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret effect size magnitude"""
        if d < 0.2:
            return "Negligible"
        elif d < 0.5:
            return "Small"
        elif d < 0.8:
            return "Medium"
        else:
            return "Large"
    
    def _display_experiment_summary(self, config: ExperimentConfig):
        """Display experiment configuration summary"""
        st.markdown("#### 📋 Experiment Summary")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Name:** {config.name}")
            st.write(f"**Datasets:** {', '.join(config.datasets)}")
            st.write(f"**Models:** {', '.join(config.models)}")
            st.write(f"**Methods:** {', '.join(config.explanation_methods)}")
        
        with col2:
            st.write(f"**Primary Metric:** {config.metrics[0]}")
            st.write(f"**Test Type:** {config.test_type}")
            st.write(f"**Alpha Level:** {config.alpha_level}")
            st.write(f"**Correction:** {config.correction_method}")
        
        total_combinations = len(config.datasets) * len(config.models) * len(config.explanation_methods)
        st.metric("Total Experimental Conditions", total_combinations)
    
    def _generate_execution_phases(self, config: ExperimentConfig) -> List[Dict]:
        """Generate execution phases for the experiment"""
        phases = [
            {
                'name': 'Setup & Preparation',
                'duration': 2,
                'description': 'Environment setup, data preparation, code verification',
                'resources': 'Dev team'
            },
            {
                'name': 'Pilot Testing',
                'duration': 1,
                'description': 'Small-scale test runs to validate setup',
                'resources': 'Dev team'
            },
            {
                'name': 'Main Experiments',
                'duration': 5,
                'description': 'Execute all planned experimental conditions',
                'resources': 'Compute cluster'
            },
            {
                'name': 'Data Collection',
                'duration': 1,
                'description': 'Gather and validate all experimental results',
                'resources': 'Storage systems'
            },
            {
                'name': 'Statistical Analysis',
                'duration': 3,
                'description': 'Perform planned statistical tests and analysis',
                'resources': 'Analysis team'
            },
            {
                'name': 'Report Generation',
                'duration': 2,
                'description': 'Create final reports and visualizations',
                'resources': 'Analysis team'
            }
        ]
        
        return phases
    
    def _render_specialized_tests(self):
        """Render specialized statistical tests interface for Wilcoxon and McNemar"""
        st.markdown("### 🧮 Specialized Statistical Tests")
        
        st.info("""
        **Specialized tests** for specific experimental scenarios in XAI method comparison:
        - **Wilcoxon Signed-Rank**: For paired, non-parametric comparisons
        - **McNemar Test**: For paired binary outcomes (success/failure rates)
        """)
        
        test_choice = st.selectbox(
            "Select specialized test:",
            ["Wilcoxon Signed-Rank Test", "McNemar Test"],
            help="Choose the appropriate test for your data type",
            key="specialized_test_choice"
        )
        
        if test_choice == "Wilcoxon Signed-Rank Test":
            self._render_wilcoxon_setup()
        else:
            self._render_mcnemar_setup()
    
    def _render_wilcoxon_setup(self):
        """Render Wilcoxon signed-rank test setup"""
        st.markdown("#### 🔄 Wilcoxon Signed-Rank Test Setup")
        
        st.markdown("""
        **Use Case**: Comparing two explanation methods on the same instances (paired data)
        - Non-parametric alternative to paired t-test
        - Robust to outliers and non-normal distributions
        - Tests if median difference = 0
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📊 Test Configuration")
            
            # Available methods for comparison
            available_datasets, available_models, available_methods = self._extract_available_options()
            
            if len(available_methods) >= 2:
                method_a = st.selectbox(
                    "Method A (baseline):",
                    available_methods,
                    help="First explanation method for comparison",
                    key="wilcoxon_method_a"
                )
                
                method_b = st.selectbox(
                    "Method B (comparison):",
                    [m for m in available_methods if m != method_a],
                    help="Second explanation method for comparison",
                    key="wilcoxon_method_b"
                )
                
                # Metric to compare
                metrics = ['faithfulness', 'stability', 'completeness', 'sparsity', 'consistency']
                comparison_metric = st.selectbox(
                    "Metric to compare:",
                    metrics,
                    help="Which metric to use for comparison",
                    key="wilcoxon_metric"
                )
                
                # Test parameters
                alpha = st.selectbox(
                    "Significance level:",
                    [0.01, 0.05, 0.10],
                    index=1,
                    help="Type I error rate",
                    key="wilcoxon_alpha"
                )
                
                alternative = st.selectbox(
                    "Alternative hypothesis:",
                    ["two-sided", "greater", "less"],
                    help="Direction of the test",
                    key="wilcoxon_alternative"
                )
            else:
                st.warning("Need at least 2 explanation methods for comparison")
                return
        
        with col2:
            st.markdown("##### 📈 Power Analysis & Sample Size")
            
            # Effect size specification
            effect_size_type = st.selectbox(
                "Effect size specification:",
                ["Probability of superiority", "Median difference", "Cohen's d equivalent"],
                help="How to specify the expected effect",
                key="wilcoxon_effect_type"
            )
            
            if effect_size_type == "Probability of superiority":
                prob_superiority = st.slider(
                    "P(Method B > Method A):",
                    0.5, 1.0, 0.7, 0.05,
                    help="Probability that Method B performs better",
                    key="wilcoxon_prob_sup"
                )
                effect_size = prob_superiority
            elif effect_size_type == "Median difference":
                median_diff = st.number_input(
                    "Expected median difference:",
                    0.0, 1.0, 0.1, 0.05,
                    help="Expected difference in medians",
                    key="wilcoxon_median_diff"
                )
                effect_size = median_diff
            else:  # Cohen's d equivalent
                cohens_d = st.selectbox(
                    "Effect size (Cohen's d):",
                    [0.2, 0.5, 0.8],
                    index=1,
                    format_func=lambda x: f"{x} ({'Small' if x==0.2 else 'Medium' if x==0.5 else 'Large'})",
                    key="wilcoxon_cohens_d"
                )
                effect_size = cohens_d
            
            # Power calculation
            desired_power = st.slider(
                "Desired power:",
                0.70, 0.95, 0.80, 0.05,
                help="Probability of detecting the effect",
                key="wilcoxon_power"
            )
            
            # Calculate sample size for Wilcoxon
            n_required = self._calculate_wilcoxon_sample_size(effect_size, alpha, desired_power)
            
            st.metric("Required Sample Size", f"{n_required} pairs")
            st.metric("Total Experiments", f"{n_required * 2}")
        
        # Dataset generation for Wilcoxon
        st.markdown("##### 📋 Dataset Generation Plan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_datasets = st.multiselect(
                "Select datasets for testing:",
                available_datasets,
                default=available_datasets[:3] if len(available_datasets) >= 3 else available_datasets,
                key="wilcoxon_datasets"
            )
            
            instances_per_dataset = st.number_input(
                "Instances per dataset:",
                min_value=10,
                max_value=500,
                value=max(30, n_required),
                help="Number of instances to test on each dataset",
                key="wilcoxon_instances"
            )
        
        with col2:
            st.markdown("**Experimental Design:**")
            st.write(f"- **Method A**: {method_a}")
            st.write(f"- **Method B**: {method_b}")
            st.write(f"- **Metric**: {comparison_metric}")
            st.write(f"- **Test**: Wilcoxon signed-rank")
            st.write(f"- **Alpha**: {alpha}")
            st.write(f"- **Power**: {desired_power}")
            
            total_pairs = len(selected_datasets) * instances_per_dataset
            st.metric("Total Instance Pairs", total_pairs)
        
        # Generate experiment configuration
        if st.button("📝 Generate Wilcoxon Test Plan", type="primary", key="generate_wilcoxon"):
            wilcoxon_config = {
                'test_type': 'Wilcoxon Signed-Rank Test',
                'method_a': method_a,
                'method_b': method_b,
                'comparison_metric': comparison_metric,
                'datasets': selected_datasets,
                'instances_per_dataset': instances_per_dataset,
                'total_pairs': total_pairs,
                'alpha': alpha,
                'power': desired_power,
                'effect_size': effect_size,
                'alternative': alternative,
                'required_sample_size': n_required
            }
            
            st.success("✅ Wilcoxon test configuration generated!")
            
            # Display configuration as JSON for download
            config_json = json.dumps(wilcoxon_config, indent=2)
            st.download_button(
                "📄 Download Configuration",
                config_json,
                f"wilcoxon_test_plan_{method_a}_{method_b}.json",
                "application/json",
                key="download_wilcoxon"
            )
    
    def _render_mcnemar_setup(self):
        """Render McNemar test setup"""
        st.markdown("#### 🎲 McNemar Test Setup")
        
        st.markdown("""
        **Use Case**: Comparing success rates of two explanation methods on the same instances
        - Tests for marginal homogeneity in 2×2 contingency tables
        - Suitable for binary outcomes (success/failure, correct/incorrect)
        - Accounts for paired nature of comparisons
        """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📊 Test Configuration")
            
            # Available methods for comparison
            available_datasets, available_models, available_methods = self._extract_available_options()
            
            if len(available_methods) >= 2:
                method_a = st.selectbox(
                    "Method A (baseline):",
                    available_methods,
                    help="First explanation method",
                    key="mcnemar_method_a"
                )
                
                method_b = st.selectbox(
                    "Method B (comparison):",
                    [m for m in available_methods if m != method_a],
                    help="Second explanation method",
                    key="mcnemar_method_b"
                )
                
                # Success criterion
                success_criterion = st.selectbox(
                    "Success criterion:",
                    [
                        "Faithfulness > threshold",
                        "Stability > threshold", 
                        "User satisfaction > threshold",
                        "Explanation quality > threshold",
                        "Custom binary outcome"
                    ],
                    help="How to define 'success' for each explanation",
                    key="mcnemar_criterion"
                )
                
                if "threshold" in success_criterion:
                    threshold = st.slider(
                        "Success threshold:",
                        0.0, 1.0, 0.7, 0.05,
                        help="Minimum value to consider successful",
                        key="mcnemar_threshold"
                    )
                
                # Test parameters
                alpha = st.selectbox(
                    "Significance level:",
                    [0.01, 0.05, 0.10],
                    index=1,
                    key="mcnemar_alpha"
                )
            else:
                st.warning("Need at least 2 explanation methods for comparison")
                return
        
        with col2:
            st.markdown("##### 📈 Power Analysis & Sample Size")
            
            # Expected proportions
            st.markdown("**Expected Success Rates:**")
            
            p_a = st.slider(
                f"Expected success rate for {method_a}:",
                0.1, 0.9, 0.6, 0.05,
                help="Proportion of successful explanations for Method A",
                key="mcnemar_p_a"
            )
            
            p_b = st.slider(
                f"Expected success rate for {method_b}:",
                0.1, 0.9, 0.8, 0.05,
                help="Proportion of successful explanations for Method B",
                key="mcnemar_p_b"
            )
            
            # Discordant pairs
            st.markdown("**Discordant Pairs (key for power):**")
            
            p_01 = st.slider(
                "P(A fails, B succeeds):",
                0.01, 0.5, 0.15, 0.01,
                help="Proportion where A fails but B succeeds",
                key="mcnemar_p_01"
            )
            
            p_10 = st.slider(
                "P(A succeeds, B fails):",
                0.01, 0.5, 0.05, 0.01,
                help="Proportion where A succeeds but B fails",
                key="mcnemar_p_10"
            )
            
            # Power calculation
            desired_power = st.slider(
                "Desired power:",
                0.70, 0.95, 0.80, 0.05,
                key="mcnemar_power"
            )
            
            # Calculate sample size for McNemar
            n_required = self._calculate_mcnemar_sample_size(p_01, p_10, alpha, desired_power)
            
            st.metric("Required Sample Size", f"{n_required} pairs")
            
            # Show expected contingency table
            n_11 = int(n_required * (p_a - p_10))  # Both succeed
            n_00 = int(n_required * (1 - p_a - p_01))  # Both fail
            n_01 = int(n_required * p_01)  # A fails, B succeeds
            n_10 = int(n_required * p_10)  # A succeeds, B fails
            
            st.markdown("**Expected 2×2 Table:**")
            contingency_df = pd.DataFrame({
                'Method B Success': [n_11, n_01, n_11 + n_01],
                'Method B Failure': [n_10, n_00, n_10 + n_00],
                'Total': [n_11 + n_10, n_01 + n_00, n_required]
            }, index=['Method A Success', 'Method A Failure', 'Total'])
            
            st.dataframe(contingency_df)
        
        # Dataset generation for McNemar
        st.markdown("##### 📋 Dataset Generation Plan")
        
        col1, col2 = st.columns(2)
        
        with col1:
            selected_datasets = st.multiselect(
                "Select datasets for testing:",
                available_datasets,
                default=available_datasets[:2] if len(available_datasets) >= 2 else available_datasets,
                key="mcnemar_datasets"
            )
            
            instances_per_dataset = st.number_input(
                "Instances per dataset:",
                min_value=20,
                max_value=1000,
                value=max(50, n_required),
                help="Number of instances to evaluate on each dataset",
                key="mcnemar_instances"
            )
        
        with col2:
            st.markdown("**Experimental Design:**")
            st.write(f"- **Method A**: {method_a}")
            st.write(f"- **Method B**: {method_b}")
            st.write(f"- **Success criterion**: {success_criterion}")
            if "threshold" in success_criterion:
                st.write(f"- **Threshold**: {threshold}")
            st.write(f"- **Test**: McNemar test")
            st.write(f"- **Alpha**: {alpha}")
            st.write(f"- **Power**: {desired_power}")
            
            total_instances = len(selected_datasets) * instances_per_dataset
            st.metric("Total Instances", total_instances)
        
        # Generate experiment configuration
        if st.button("📝 Generate McNemar Test Plan", type="primary", key="generate_mcnemar"):
            mcnemar_config = {
                'test_type': 'McNemar Test',
                'method_a': method_a,
                'method_b': method_b,
                'success_criterion': success_criterion,
                'threshold': threshold if "threshold" in success_criterion else None,
                'datasets': selected_datasets,
                'instances_per_dataset': instances_per_dataset,
                'total_instances': total_instances,
                'alpha': alpha,
                'power': desired_power,
                'expected_success_rates': {'method_a': p_a, 'method_b': p_b},
                'expected_discordant': {'p_01': p_01, 'p_10': p_10},
                'required_sample_size': n_required,
                'expected_contingency_table': contingency_df.to_dict()
            }
            
            st.success("✅ McNemar test configuration generated!")
            
            # Display configuration as JSON for download
            config_json = json.dumps(mcnemar_config, indent=2, default=str)
            st.download_button(
                "📄 Download Configuration",
                config_json,
                f"mcnemar_test_plan_{method_a}_{method_b}.json",
                "application/json",
                key="download_mcnemar"
            )
    
    def _calculate_wilcoxon_sample_size(self, effect_size: float, alpha: float, power: float) -> int:
        """Calculate required sample size for Wilcoxon signed-rank test"""
        # Simplified calculation based on normal approximation
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        # For Wilcoxon, the effect size is related to P(X > Y)
        # Convert effect size to standard normal equivalent
        if effect_size > 0.5:  # Probability of superiority
            # Convert to Z-score equivalent
            z_effect = stats.norm.ppf(effect_size)
            n = max(10, int(((z_alpha + z_beta) / z_effect) ** 2))
        else:  # Cohen's d or median difference
            n = max(10, int(2 * ((z_alpha + z_beta) ** 2) / (effect_size ** 2)))
        
        return min(n, 200)  # Cap at reasonable maximum
    
    def _calculate_mcnemar_sample_size(self, p_01: float, p_10: float, alpha: float, power: float) -> int:
        """Calculate required sample size for McNemar test"""
        # McNemar test focuses on discordant pairs
        p_discordant = p_01 + p_10
        
        if p_discordant < 0.01:
            return 500  # Large sample needed if very few discordant pairs
        
        # Simplified calculation based on chi-square approximation
        z_alpha = stats.norm.ppf(1 - alpha/2)
        z_beta = stats.norm.ppf(power)
        
        # Effect size for McNemar is the difference in discordant proportions
        effect = abs(p_01 - p_10)
        
        if effect < 0.01:
            return 1000  # Large sample needed for very small effects
        
        # Sample size calculation
        n = ((z_alpha + z_beta) ** 2) * p_discordant / (effect ** 2)
        
        return max(20, min(int(n), 1000))
    
    def _render_critical_difference_analysis(self):
        """Render critical difference plot analysis with data type categorization"""
        st.markdown("### 📊 Critical Difference Plot Analysis")
        
        st.info("""
        **Critical Difference Plots** visualize statistical significance across multiple methods.
        Analyze results by data types: binary classification, multiclass, image data, and text data.
        """)
        
        # Data type categorization
        st.markdown("#### 📋 Data Type Categorization")
        
        available_datasets, available_models, available_methods = self._extract_available_options()
        
        # Create data type categories
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("##### 📊 Tabular Data Categories")
            
            # Binary classification datasets (tabular only)
            binary_tabular_default = [
                "adult_income", "compas", "breast_cancer", 
                "heart_disease", "german_credit"
            ]
            binary_datasets = st.multiselect(
                "Binary Classification (Tabular):",
                available_datasets,
                default=[d for d in binary_tabular_default if d in available_datasets],
                help="Tabular datasets with binary target variables",
                key="binary_datasets"
            )
            
            # Multiclass datasets (tabular only)
            multiclass_tabular_default = [
                "iris", "wine_quality", "diabetes", 
                "wine_classification", "digits", "synthetic_tabular"
            ]
            multiclass_datasets = st.multiselect(
                "Multiclass Classification (Tabular):",
                [d for d in available_datasets if d not in binary_datasets],
                default=[d for d in multiclass_tabular_default if d in available_datasets and d not in binary_datasets],
                help="Tabular datasets with multiple classes",
                key="multiclass_datasets"
            )
        
        with col2:
            st.markdown("##### 🗂️ Other Data Modalities")
            
            # Image datasets (all image data regardless of binary/multiclass)
            image_default = [
                "mnist", "cifar10", "fashion_mnist", 
                "cifar100", "imagenette", "chest_xray"
            ]
            image_datasets = st.multiselect(
                "Image Datasets:",
                available_datasets,
                default=[d for d in image_default if d in available_datasets],
                help="Visual/image-based datasets (all classification types)",
                key="image_datasets"
            )
            
            # Text datasets (all text data regardless of binary/multiclass)
            text_default = [
                "imdb", "20newsgroups", "ag_news", 
                "yelp_reviews", "reuters21578"
            ]
            text_datasets = st.multiselect(
                "Text Datasets:",
                available_datasets,
                default=[d for d in text_default if d in available_datasets],
                help="Natural language/text datasets (all classification types)",
                key="text_datasets"
            )
        
        # Create 4 categories as requested
        categories = {
            "Binary Classification (Tabular)": binary_datasets,
            "Multiclass Classification (Tabular)": multiclass_datasets,
            "Image Data": image_datasets,  
            "Text Data": text_datasets
        }
        
        # Statistical test setup for each category
        st.markdown("#### 🧮 Statistical Testing by Category")
        
        test_type = st.selectbox(
            "Select statistical test:",
            ["Wilcoxon Signed-Rank Test", "McNemar Test", "Both Tests"],
            help="Test to apply across categories",
            key="cd_test_type"
        )
        
        metrics_to_analyze = st.multiselect(
            "Select metrics for analysis:",
            ['faithfulness', 'stability', 'completeness', 'sparsity', 'consistency', 'monotonicity', 'simplicity'],
            default=['faithfulness', 'stability', 'completeness'],
            help="Metrics to compare across methods",
            key="cd_metrics"
        )
        
        # Analysis configuration for each category
        for category_name, datasets in categories.items():
            if datasets:  # Only show categories with datasets
                st.markdown(f"##### {category_name}")
                
                with st.expander(f"Configure {category_name} Analysis"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Datasets:** {', '.join(datasets)}")
                        
                        selected_methods = st.multiselect(
                            f"Methods for {category_name}:",
                            available_methods,
                            default=available_methods[:4] if len(available_methods) >= 4 else available_methods,
                            key=f"{category_name.lower().replace(' ', '_')}_methods"
                        )
                        
                        if test_type in ["Wilcoxon Signed-Rank Test", "Both Tests"]:
                            alpha_wilcoxon = st.selectbox(
                                "Wilcoxon α level:",
                                [0.01, 0.05, 0.10],
                                index=1,
                                key=f"wilcoxon_alpha_{category_name.lower().replace(' ', '_')}"
                            )
                    
                    with col2:
                        if test_type in ["McNemar Test", "Both Tests"]:
                            success_threshold = st.slider(
                                "McNemar success threshold:",
                                0.0, 1.0, 0.7, 0.05,
                                help="Threshold for binary success classification",
                                key=f"mcnemar_threshold_{category_name.lower().replace(' ', '_')}"
                            )
                            
                            alpha_mcnemar = st.selectbox(
                                "McNemar α level:",
                                [0.01, 0.05, 0.10],
                                index=1,
                                key=f"mcnemar_alpha_{category_name.lower().replace(' ', '_')}"
                            )
                        
                        # Critical difference parameters
                        cd_correction = st.selectbox(
                            "Multiple comparison correction:",
                            ["Bonferroni", "Holm", "Nemenyi", "None"],
                            index=2,
                            help="Correction method for critical difference",
                            key=f"cd_correction_{category_name.lower().replace(' ', '_')}"
                        )
        
        # Generate critical difference plot configuration
        if st.button("🎨 Generate Critical Difference Analysis Plan", type="primary", key="generate_cd_plan"):
            cd_config = {
                'analysis_type': 'Critical Difference Analysis',
                'test_types': test_type,
                'metrics': metrics_to_analyze,
                'categories': {},
                'created_date': datetime.now().isoformat()
            }
            
            # Add configuration for each category
            for category_name, datasets in categories.items():
                if datasets:
                    category_key = category_name.lower().replace(' ', '_')
                    
                    category_config = {
                        'datasets': datasets,
                        'methods': st.session_state.get(f"{category_key}_methods", []),
                        'correction_method': st.session_state.get(f"cd_correction_{category_key}", "Nemenyi")
                    }
                    
                    if test_type in ["Wilcoxon Signed-Rank Test", "Both Tests"]:
                        category_config['wilcoxon'] = {
                            'alpha': st.session_state.get(f"wilcoxon_alpha_{category_key}", 0.05)
                        }
                    
                    if test_type in ["McNemar Test", "Both Tests"]:
                        category_config['mcnemar'] = {
                            'threshold': st.session_state.get(f"mcnemar_threshold_{category_key}", 0.7),
                            'alpha': st.session_state.get(f"mcnemar_alpha_{category_key}", 0.05)
                        }
                    
                    cd_config['categories'][category_name] = category_config
            
            st.success("✅ Critical difference analysis plan generated!")
            
            # Display summary
            st.markdown("#### 📋 Analysis Summary")
            
            for category_name, config in cd_config['categories'].items():
                with st.expander(f"{category_name} Configuration"):
                    st.json(config)
            
            # Download configuration
            config_json = json.dumps(cd_config, indent=2, default=str)
            st.download_button(
                "📄 Download CD Analysis Plan",
                config_json,
                f"critical_difference_analysis_plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                key="download_cd_config"
            )
        
        # Interactive Critical Difference Plot Generator
        st.markdown("#### 🎯 Interactive Critical Difference Plot Generator")
        st.info("Generate CD plots for specific modality-metric combinations using Wilcoxon signed-rank test")
        
        # Interactive selection
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Select modality
            available_modalities = []
            if binary_datasets:
                available_modalities.append("Binary Classification (Tabular)")
            if multiclass_datasets:
                available_modalities.append("Multiclass Classification (Tabular)")
            if image_datasets:
                available_modalities.append("Image Data")
            if text_datasets:
                available_modalities.append("Text Data")
            
            if available_modalities:
                selected_modality = st.selectbox(
                    "Select Data Modality:",
                    available_modalities,
                    help="Choose the data type for CD plot analysis",
                    key="cd_modality_selector"
                )
            else:
                st.warning("Please configure dataset categories first.")
                return
        
        with col2:
            # Select metric
            selected_metric = st.selectbox(
                "Select Metric:",
                metrics_to_analyze if metrics_to_analyze else ['faithfulness', 'stability', 'completeness', 'sparsity', 'consistency'],
                help="Choose the metric to analyze",
                key="cd_metric_selector"
            )
        
        with col3:
            # Get datasets for selected modality
            if selected_modality == "Binary Classification (Tabular)":
                modality_datasets = binary_datasets
            elif selected_modality == "Multiclass Classification (Tabular)":
                modality_datasets = multiclass_datasets
            elif selected_modality == "Image Data":
                modality_datasets = image_datasets
            else:  # Text Data
                modality_datasets = text_datasets
            
            st.metric("Datasets in Modality", len(modality_datasets))
            st.metric("Available Methods", len(available_methods))
        
        # Configuration for CD plot
        col1, col2, col3 = st.columns(3)
        
        with col1:
            cd_alpha = st.selectbox(
                "Significance Level (α):",
                [0.01, 0.05, 0.10],
                index=1,
                help="Alpha level for statistical tests",
                key="cd_alpha_dynamic"
            )
        
        with col2:
            n_datasets_sim = st.number_input(
                "Number of datasets (simulation):",
                min_value=len(modality_datasets) if modality_datasets else 5,
                max_value=20,
                value=max(10, len(modality_datasets)) if modality_datasets else 10,
                help="Number of datasets to simulate",
                key="n_datasets_dynamic"
            )
        
        with col3:
            statistical_test = st.selectbox(
                "Statistical Test:",
                ["Wilcoxon Signed-Rank", "Friedman + Nemenyi"],
                index=0,
                help="Choose the statistical test approach",
                key="cd_test_method"
            )
        
        # Display selected configuration
        st.markdown("##### 🎯 Current Configuration")
        config_col1, config_col2 = st.columns(2)
        
        with config_col1:
            st.write(f"**Modality:** {selected_modality}")
            st.write(f"**Metric:** {selected_metric}")
            st.write(f"**Datasets:** {', '.join(modality_datasets) if modality_datasets else 'None'}")
        
        with config_col2:
            st.write(f"**Statistical Test:** {statistical_test}")
            st.write(f"**Alpha Level:** {cd_alpha}")
            st.write(f"**Methods:** {len(available_methods)} explanation methods")
        
        # Generate CD Plot button
        if st.button("📊 Generate Critical Difference Plot", type="primary", key="generate_dynamic_cd"):
            if modality_datasets and available_methods:
                self._generate_modality_specific_cd_plot(
                    modality=selected_modality,
                    datasets=modality_datasets,
                    metric=selected_metric,
                    methods=available_methods,
                    n_datasets=n_datasets_sim,
                    alpha=cd_alpha,
                    test_method=statistical_test
                )
            else:
                st.warning("Please ensure you have configured datasets and methods for the selected modality.")
    
    def _generate_scikit_posthocs_cd_plot(self, methods: List[str], metric: str, n_datasets: int, alpha: float):
        """Generate a critical difference plot using scikit-posthocs"""
        
        # Simulate performance data for methods across datasets
        np.random.seed(42)
        n_methods = len(methods)
        
        # Generate realistic performance data with different performance levels
        performance_data = []
        
        for dataset_idx in range(n_datasets):
            dataset_results = {}
            for i, method in enumerate(methods):
                # Create different performance levels with some noise
                base_performance = 0.6 + (i * 0.04) + np.random.normal(0, 0.02)
                # Add dataset-specific variation
                dataset_variation = np.random.normal(0, 0.05)
                score = max(0.3, min(1.0, base_performance + dataset_variation))
                dataset_results[method] = score
            
            performance_data.append(dataset_results)
        
        # Convert to DataFrame format expected by scikit-posthocs
        data_for_ranking = []
        for dataset_idx, results in enumerate(performance_data):
            for method, score in results.items():
                data_for_ranking.append({
                    'dataset': f'Dataset_{dataset_idx}',
                    'method': method,
                    'score': score
                })
        
        df = pd.DataFrame(data_for_ranking)
        
        # Calculate rankings (1 = best for each dataset)
        df['ranking'] = df.groupby('dataset')['score'].rank(method='dense', ascending=False)
        
        # Check for and handle duplicates
        df_clean = df.drop_duplicates(subset=['dataset', 'method'])
        
        # Prepare data for critical difference plot
        # scikit-posthocs expects a matrix where rows are datasets and columns are methods
        ranking_matrix = df_clean.pivot(index='dataset', columns='method', values='ranking')
        
        # Perform Friedman test
        try:
            friedman_stat, friedman_p = stats.friedmanchisquare(
                *[ranking_matrix[method].values for method in methods]
            )
            
            st.markdown("#### 📊 Statistical Test Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Friedman Statistic", f"{friedman_stat:.4f}")
                st.metric("p-value", f"{friedman_p:.6f}")
            
            with col2:
                significance = "Significant" if friedman_p < alpha else "Not Significant"
                st.metric("Result", significance)
                st.metric("Alpha Level", f"{alpha}")
            
            if friedman_p < alpha:
                st.success("📈 Significant differences found between methods - proceeding with post-hoc analysis")
                
                # Perform Nemenyi post-hoc test
                try:
                    # Create the critical difference plot using scikit-posthocs
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # Calculate average rankings
                    avg_ranks = ranking_matrix.mean().sort_values()
                    
                    # Use scikit-posthocs for critical difference calculation
                    posthoc_results = sp.posthoc_nemenyi_friedman(ranking_matrix)
                    
                    # Create critical difference plot
                    sp.critical_difference_diagram(
                        avg_ranks, 
                        posthoc_results, 
                        ax=ax
                    )
                    
                    ax.set_title(f'Critical Difference Plot - {metric.title()}\n'
                               f'Friedman p-value: {friedman_p:.6f}, α = {alpha}', 
                               fontsize=14, pad=20)
                    
                    # Improve plot styling
                    ax.grid(True, alpha=0.3)
                    ax.set_xlabel('Average Ranking (1 = Best)', fontsize=12)
                    
                    st.pyplot(fig)
                    plt.close()
                    
                    # Display detailed pairwise comparisons
                    st.markdown("#### 🔍 Pairwise Comparison Results")
                    
                    # Create a more readable comparison table
                    comparison_results = []
                    for i, method1 in enumerate(methods):
                        for j, method2 in enumerate(methods):
                            if i < j:  # Avoid duplicate comparisons
                                p_val = posthoc_results.loc[method1, method2]
                                avg_rank1 = avg_ranks[method1]
                                avg_rank2 = avg_ranks[method2]
                                rank_diff = abs(avg_rank1 - avg_rank2)
                                
                                significance = "Significant" if p_val < alpha else "Not Significant"
                                better_method = method1 if avg_rank1 < avg_rank2 else method2
                                
                                comparison_results.append({
                                    'Method A': method1,
                                    'Method B': method2,
                                    'Avg Rank A': f"{avg_rank1:.3f}",
                                    'Avg Rank B': f"{avg_rank2:.3f}",
                                    'Rank Difference': f"{rank_diff:.3f}",
                                    'p-value': f"{p_val:.6f}",
                                    'Significance': significance,
                                    'Better Method': better_method
                                })
                    
                    comparison_df = pd.DataFrame(comparison_results)
                    
                    # Style the dataframe
                    def highlight_significant(row):
                        if row['Significance'] == 'Significant':
                            return ['background-color: #90EE90'] * len(row)
                        else:
                            return ['background-color: #FFE4E1'] * len(row)
                    
                    styled_df = comparison_df.style.apply(highlight_significant, axis=1)
                    st.dataframe(styled_df, use_container_width=True)
                    
                    # Summary statistics
                    st.markdown("#### 📈 Summary Statistics")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        significant_pairs = len([r for r in comparison_results if r['Significance'] == 'Significant'])
                        total_pairs = len(comparison_results)
                        st.metric("Significant Pairs", f"{significant_pairs}/{total_pairs}")
                    
                    with col2:
                        best_method = avg_ranks.index[0]
                        st.metric("Best Method", best_method)
                    
                    with col3:
                        max_rank_diff = max([float(r['Rank Difference']) for r in comparison_results])
                        st.metric("Max Rank Difference", f"{max_rank_diff:.3f}")
                    
                except Exception as e:
                    st.error(f"Error generating critical difference plot: {str(e)}")
                    st.info("Falling back to basic ranking display...")
                    
                    # Fallback: simple ranking table
                    avg_ranks = ranking_matrix.mean().sort_values()
                    ranking_df = pd.DataFrame({
                        'Method': avg_ranks.index,
                        'Average Rank': avg_ranks.values,
                        'Rank Position': range(1, len(avg_ranks) + 1)
                    })
                    
                    st.dataframe(ranking_df, use_container_width=True)
            
            else:
                st.info("🔄 No significant differences found between methods. All methods perform similarly.")
                
                # Still show the ranking
                avg_ranks = ranking_matrix.mean().sort_values()
                ranking_df = pd.DataFrame({
                    'Method': avg_ranks.index,
                    'Average Rank': avg_ranks.values
                })
                
                st.dataframe(ranking_df, use_container_width=True)
        
        except Exception as e:
            st.error(f"Error performing statistical analysis: {str(e)}")
            st.info("This is a simulation with synthetic data. In practice, you would use your actual experimental results.")
    
    def _generate_modality_specific_cd_plot(self, modality: str, datasets: List[str], metric: str, 
                                           methods: List[str], n_datasets: int, alpha: float, test_method: str):
        """Generate a critical difference plot for a specific modality and metric"""
        
        st.markdown(f"### 📊 Critical Difference Analysis")
        st.markdown(f"**Modality:** {modality} | **Metric:** {metric}")
        
        # Simulate realistic performance data based on modality characteristics
        np.random.seed(42)
        performance_data = []
        
        # Define modality-specific performance characteristics
        modality_characteristics = {
            "Binary Classification (Tabular)": {
                "base_range": (0.65, 0.95),  # Higher performance for binary
                "method_preferences": {
                    "shap": 0.05, "lime": 0.02, "integrated_gradients": -0.02,
                    "causal_shap": 0.08, "prototype": -0.05, "counterfactual": 0.01
                }
            },
            "Multiclass Classification (Tabular)": {
                "base_range": (0.55, 0.85),  # Lower performance for multiclass
                "method_preferences": {
                    "shap": 0.04, "lime": 0.01, "integrated_gradients": 0.02,
                    "causal_shap": 0.06, "prototype": -0.03, "counterfactual": -0.01
                }
            },
            "Image Data": {
                "base_range": (0.60, 0.90),  # Good performance for images
                "method_preferences": {
                    "integrated_gradients": 0.08, "lime": -0.02, "shap": 0.03,
                    "occlusion": 0.05, "feature_ablation": 0.04, "prototype": 0.02
                }
            },
            "Text Data": {
                "base_range": (0.50, 0.80),  # Variable performance for text
                "method_preferences": {
                    "lime": 0.06, "shap": 0.04, "integrated_gradients": 0.02,
                    "prototype": -0.04, "counterfactual": 0.01, "influence_functions": 0.03
                }
            }
        }
        
        characteristics = modality_characteristics.get(modality, modality_characteristics["Binary Classification (Tabular)"])
        base_min, base_max = characteristics["base_range"]
        method_prefs = characteristics["method_preferences"]
        
        # Generate performance data for each dataset
        for dataset_idx in range(n_datasets):
            dataset_results = {}
            # Ensure unique dataset names
            if datasets:
                base_name = datasets[dataset_idx % len(datasets)]
                dataset_name = f"{base_name}_{dataset_idx // len(datasets) + 1}" if dataset_idx >= len(datasets) else base_name
            else:
                dataset_name = f"Dataset_{dataset_idx + 1}"
            
            for method in methods:
                # Base performance for this modality
                base_performance = np.random.uniform(base_min, base_max)
                
                # Method-specific adjustment
                method_adjustment = method_prefs.get(method, 0.0)
                
                # Dataset-specific variation
                dataset_variation = np.random.normal(0, 0.05)
                
                # Metric-specific adjustment
                metric_adjustments = {
                    "faithfulness": 0.0,  # Base metric
                    "stability": -0.05,   # Generally lower
                    "completeness": -0.03,
                    "sparsity": np.random.uniform(-0.1, 0.1),  # Highly variable
                    "consistency": -0.04,
                    "monotonicity": -0.02,
                    "simplicity": np.random.uniform(-0.05, 0.05)
                }
                metric_adjustment = metric_adjustments.get(metric, 0.0)
                
                # Final score
                final_score = base_performance + method_adjustment + dataset_variation + metric_adjustment
                final_score = max(0.1, min(1.0, final_score))  # Clamp to valid range
                
                dataset_results[method] = final_score
            
            performance_data.append({
                'dataset': dataset_name,
                'results': dataset_results
            })
        
        # Convert to DataFrame for analysis
        data_for_analysis = []
        for dataset_data in performance_data:
            for method, score in dataset_data['results'].items():
                data_for_analysis.append({
                    'dataset': dataset_data['dataset'],
                    'method': method,
                    'score': score
                })
        
        df = pd.DataFrame(data_for_analysis)
        
        # Show sample data
        with st.expander("🔍 Sample Performance Data"):
            sample_df = df.head(20)
            st.dataframe(sample_df)
            
            # Show average performance by method
            avg_performance = df.groupby('method')['score'].agg(['mean', 'std']).round(4)
            avg_performance.columns = ['Average Score', 'Std Dev']
            avg_performance = avg_performance.sort_values('Average Score', ascending=False)
            
            st.markdown("**Average Performance by Method:**")
            st.dataframe(avg_performance)
        
        # Perform statistical analysis based on selected test method
        if test_method == "Wilcoxon Signed-Rank":
            self._perform_wilcoxon_analysis(df, modality, metric, alpha)
        else:  # Friedman + Nemenyi
            self._perform_friedman_nemenyi_analysis(df, modality, metric, alpha)
    
    def _perform_wilcoxon_analysis(self, df: pd.DataFrame, modality: str, metric: str, alpha: float):
        """Perform pairwise Wilcoxon signed-rank tests with CD plot"""
        
        st.markdown("#### 🔬 Wilcoxon Signed-Rank Analysis")
        
        # Calculate rankings for CD plot
        df['ranking'] = df.groupby('dataset')['score'].rank(method='dense', ascending=False)
        
        # Remove any potential duplicates
        df_clean = df.drop_duplicates(subset=['dataset', 'method'])
        
        # Create ranking matrix
        ranking_matrix = df_clean.pivot(index='dataset', columns='method', values='ranking')
        
        methods = list(ranking_matrix.columns)
        n_methods = len(methods)
        
        # Show detailed data for Wilcoxon test explanation
        st.markdown("#### 🔍 Wilcoxon Test Data and Formula")
        
        with st.expander("📊 View Raw Ranking Data"):
            st.markdown("**Ranking Matrix (rows = datasets, columns = methods):**")
            st.dataframe(ranking_matrix.round(3))
            
            st.markdown("**How Wilcoxon Signed-Rank Test Works:**")
            st.markdown("""
            1. **Paired Data**: Compare rankings of two methods across the same datasets
            2. **Calculate Differences**: d_i = rank_method1_i - rank_method2_i for each dataset i
            3. **Rank Differences**: Rank |d_i| from smallest to largest
            4. **Sum Positive/Negative Ranks**: W⁺ = sum of ranks where d_i > 0
            5. **Test Statistic**: W = min(W⁺, W⁻) 
            6. **p-value**: Compare W to critical values or use normal approximation
            """)
        
        # Perform pairwise Wilcoxon tests with detailed calculations
        wilcoxon_results = []
        p_values_matrix = np.ones((n_methods, n_methods))
        detailed_calculations = []
        
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i != j:
                    try:
                        # Get paired data
                        data1 = ranking_matrix[method1].values
                        data2 = ranking_matrix[method2].values
                        
                        # Calculate differences
                        differences = data1 - data2
                        
                        # Check for zero variance (all differences are the same)
                        if np.std(differences) < 1e-10:
                            # Handle case where all differences are identical
                            p_val = 1.0  # No significant difference
                            stat = 0.0
                            st.info(f"Methods {method1} and {method2} have identical rankings across all datasets.")
                        else:
                            # Wilcoxon signed-rank test
                            stat, p_val = stats.wilcoxon(data1, data2, 
                                                        alternative='two-sided',
                                                        zero_method='wilcox')  # Handle zero differences
                        
                        p_values_matrix[i, j] = p_val
                        
                        if i < j:  # Avoid duplicates and store detailed calculations
                            avg_rank1 = data1.mean()
                            avg_rank2 = data2.mean()
                            
                            # Calculate detailed Wilcoxon statistics
                            abs_differences = np.abs(differences[differences != 0])  # Remove zeros
                            if len(abs_differences) > 0:
                                ranks = stats.rankdata(abs_differences)
                                positive_ranks = ranks[differences[differences != 0] > 0]
                                negative_ranks = ranks[differences[differences != 0] < 0]
                                
                                w_plus = np.sum(positive_ranks) if len(positive_ranks) > 0 else 0
                                w_minus = np.sum(negative_ranks) if len(negative_ranks) > 0 else 0
                                w_statistic = min(w_plus, w_minus)
                            else:
                                w_plus = w_minus = w_statistic = 0
                            
                            wilcoxon_results.append({
                                'Method A': method1,
                                'Method B': method2,
                                'Avg Rank A': f"{avg_rank1:.3f}",
                                'Avg Rank B': f"{avg_rank2:.3f}",
                                'Mean Difference': f"{np.mean(differences):.3f}",
                                'W⁺ (Positive Ranks Sum)': f"{w_plus:.1f}",
                                'W⁻ (Negative Ranks Sum)': f"{w_minus:.1f}",
                                'Wilcoxon Statistic (W)': f"{w_statistic:.3f}",
                                'Scipy Statistic': f"{stat:.3f}",
                                'p-value': f"{p_val:.6f}",
                                'Significant': 'Yes' if p_val < alpha else 'No',
                                'Better Method': method1 if avg_rank1 < avg_rank2 else method2
                            })
                            
                            # Store detailed calculation for this pair
                            detailed_calculations.append({
                                'pair': f"{method1} vs {method2}",
                                'method1_ranks': data1,
                                'method2_ranks': data2,
                                'differences': differences,
                                'w_plus': w_plus,
                                'w_minus': w_minus,
                                'statistic': stat,
                                'p_value': p_val
                            })
                            
                    except Exception as e:
                        st.warning(f"Error in Wilcoxon test for {method1} vs {method2}: {str(e)}")
                        p_values_matrix[i, j] = 1.0
        
        # Show detailed calculations for first few pairs
        if detailed_calculations:
            with st.expander("🧮 Detailed Wilcoxon Calculations (First 3 Pairs)"):
                for idx, calc in enumerate(detailed_calculations[:3]):
                    st.markdown(f"**{calc['pair']}:**")
                    
                    # Create a detailed table
                    calc_df = pd.DataFrame({
                        'Dataset': ranking_matrix.index,
                        f"{calc['pair'].split(' vs ')[0]} Rank": calc['method1_ranks'],
                        f"{calc['pair'].split(' vs ')[1]} Rank": calc['method2_ranks'],
                        'Difference (A-B)': calc['differences'],
                        'Abs Difference': np.abs(calc['differences']),
                    })
                    
                    # Add rank of absolute differences
                    non_zero_diffs = calc['differences'][calc['differences'] != 0]
                    if len(non_zero_diffs) > 0:
                        ranks_of_abs = stats.rankdata(np.abs(non_zero_diffs))
                        calc_df['Rank of |Diff|'] = np.where(
                            calc['differences'] != 0,
                            stats.rankdata(np.abs(calc['differences'])),
                            0
                        )
                        calc_df['Sign'] = np.where(calc['differences'] > 0, '+', 
                                                 np.where(calc['differences'] < 0, '-', '0'))
                    else:
                        calc_df['Rank of |Diff|'] = 0
                        calc_df['Sign'] = '0'
                    
                    st.dataframe(calc_df)
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("W⁺ (Sum of + ranks)", f"{calc['w_plus']:.1f}")
                    with col2:
                        st.metric("W⁻ (Sum of - ranks)", f"{calc['w_minus']:.1f}")
                    with col3:
                        st.metric("Test Statistic W", f"{min(calc['w_plus'], calc['w_minus']):.1f}")
                    
                    st.markdown(f"**p-value:** {calc['p_value']:.6f}")
                    st.markdown("---")
        
        # Create p-values DataFrame for CD plot
        p_values_df = pd.DataFrame(p_values_matrix, index=methods, columns=methods)
        
        # Generate Critical Difference Plot
        try:
            # Validate data before creating CD plot
            if ranking_matrix.empty or len(methods) < 2:
                st.warning("Insufficient data for critical difference plot. Need at least 2 methods.")
                return
            
            # Check for data quality issues
            avg_ranks = ranking_matrix.mean().sort_values()
            
            # Check if all rankings are the same (zero variance)
            if avg_ranks.std() < 1e-10:
                st.warning("All methods have identical rankings. Cannot generate meaningful CD plot.")
                # Still show the ranking table
                ranking_table = pd.DataFrame({
                    'Method': avg_ranks.index,
                    'Average Rank': avg_ranks.values,
                    'Rank Position': range(1, len(avg_ranks) + 1)
                })
                st.dataframe(ranking_table)
                return
            
            # Ensure p-values matrix is symmetric and valid
            p_values_df = p_values_df.fillna(1.0)  # Fill NaN with 1.0 (no significance)
            
            # Make matrix symmetric (required by scikit-posthocs)
            for i in range(len(methods)):
                for j in range(len(methods)):
                    if i != j:
                        # Use the maximum p-value for symmetry (more conservative)
                        p_val = max(p_values_df.iloc[i, j], p_values_df.iloc[j, i])
                        p_values_df.iloc[i, j] = p_val
                        p_values_df.iloc[j, i] = p_val
                    else:
                        p_values_df.iloc[i, j] = 1.0  # Diagonal should be 1.0
            
            fig, ax = plt.subplots(figsize=(14, 8))
            
            # Create CD plot using scikit-posthocs
            sp.critical_difference_diagram(
                avg_ranks, 
                p_values_df, 
                ax=ax
            )
            
            ax.set_title(f'Critical Difference Plot - Wilcoxon Signed-Rank Test\n'
                        f'{modality} | {metric.title()} | α = {alpha}', 
                        fontsize=16, pad=20)
            ax.set_xlabel('Average Ranking (1 = Best)', fontsize=14)
            ax.grid(True, alpha=0.3)
            
            st.pyplot(fig)
            plt.close()
            
        except Exception as e:
            st.error(f"Error creating CD plot: {str(e)}")
            
            # Fallback: show ranking table
            avg_ranks = ranking_matrix.mean().sort_values()
            st.markdown("**Method Rankings (Fallback Display):**")
            ranking_table = pd.DataFrame({
                'Method': avg_ranks.index,
                'Average Rank': avg_ranks.values,
                'Rank Position': range(1, len(avg_ranks) + 1)
            })
            st.dataframe(ranking_table)
        
        # Display Wilcoxon test results
        st.markdown("#### 📋 Pairwise Wilcoxon Test Results")
        
        if wilcoxon_results:
            results_df = pd.DataFrame(wilcoxon_results)
            
            # Color code significant results
            def highlight_significant(row):
                if row['Significant'] == 'Yes':
                    return ['background-color: #90EE90'] * len(row)
                else:
                    return ['background-color: #FFE4E1'] * len(row)
            
            styled_df = results_df.style.apply(highlight_significant, axis=1)
            st.dataframe(styled_df, use_container_width=True)
            
            # Summary statistics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                significant_pairs = len([r for r in wilcoxon_results if r['Significant'] == 'Yes'])
                total_pairs = len(wilcoxon_results)
                st.metric("Significant Pairs", f"{significant_pairs}/{total_pairs}")
            
            with col2:
                best_method = avg_ranks.index[0]
                st.metric("Best Performing Method", best_method)
            
            with col3:
                worst_method = avg_ranks.index[-1]
                st.metric("Worst Performing Method", worst_method)
    
    def _perform_friedman_nemenyi_analysis(self, df: pd.DataFrame, modality: str, metric: str, alpha: float):
        """Perform Friedman test followed by Nemenyi post-hoc analysis"""
        
        st.markdown("#### 🔬 Friedman + Nemenyi Analysis")
        
        # Use the existing implementation but with modality-specific title
        df['ranking'] = df.groupby('dataset')['score'].rank(method='dense', ascending=False)
        
        # Remove any potential duplicates
        df_clean = df.drop_duplicates(subset=['dataset', 'method'])
        
        # Create ranking matrix
        ranking_matrix = df_clean.pivot(index='dataset', columns='method', values='ranking')
        
        methods = list(ranking_matrix.columns)
        
        try:
            # Friedman test
            friedman_stat, friedman_p = stats.friedmanchisquare(
                *[ranking_matrix[method].values for method in methods]
            )
            
            st.markdown("##### 📊 Friedman Test Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Friedman Statistic", f"{friedman_stat:.4f}")
                st.metric("p-value", f"{friedman_p:.6f}")
            
            with col2:
                significance = "Significant" if friedman_p < alpha else "Not Significant"
                st.metric("Result", significance)
                st.metric("Alpha Level", f"{alpha}")
            
            if friedman_p < alpha:
                st.success("📈 Significant differences found - proceeding with Nemenyi post-hoc analysis")
                
                # Create CD plot
                fig, ax = plt.subplots(figsize=(14, 8))
                
                avg_ranks = ranking_matrix.mean().sort_values()
                posthoc_results = sp.posthoc_nemenyi_friedman(ranking_matrix)
                
                sp.critical_difference_diagram(
                    avg_ranks, 
                    posthoc_results, 
                    ax=ax
                )
                
                ax.set_title(f'Critical Difference Plot - Friedman + Nemenyi Test\n'
                            f'{modality} | {metric.title()} | α = {alpha}', 
                            fontsize=16, pad=20)
                ax.set_xlabel('Average Ranking (1 = Best)', fontsize=14)
                ax.grid(True, alpha=0.3)
                
                st.pyplot(fig)
                plt.close()
                
                # Show detailed results table (reuse existing code)
                comparison_results = []
                for i, method1 in enumerate(methods):
                    for j, method2 in enumerate(methods):
                        if i < j:
                            p_val = posthoc_results.loc[method1, method2]
                            avg_rank1 = avg_ranks[method1]
                            avg_rank2 = avg_ranks[method2]
                            
                            comparison_results.append({
                                'Method A': method1,
                                'Method B': method2,
                                'Avg Rank A': f"{avg_rank1:.3f}",
                                'Avg Rank B': f"{avg_rank2:.3f}",
                                'p-value': f"{p_val:.6f}",
                                'Significant': 'Yes' if p_val < alpha else 'No',
                                'Better Method': method1 if avg_rank1 < avg_rank2 else method2
                            })
                
                if comparison_results:
                    st.markdown("##### 📋 Nemenyi Post-hoc Results")
                    results_df = pd.DataFrame(comparison_results)
                    
                    def highlight_significant(row):
                        if row['Significant'] == 'Yes':
                            return ['background-color: #90EE90'] * len(row)
                        else:
                            return ['background-color: #FFE4E1'] * len(row)
                    
                    styled_df = results_df.style.apply(highlight_significant, axis=1)
                    st.dataframe(styled_df, use_container_width=True)
            
            else:
                st.info("🔄 No significant differences found between methods.")
                
                # Still show ranking
                avg_ranks = ranking_matrix.mean().sort_values()
                ranking_df = pd.DataFrame({
                    'Method': avg_ranks.index,
                    'Average Rank': avg_ranks.values
                })
                st.dataframe(ranking_df, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error performing Friedman analysis: {str(e)}")


def create_experiment_planner(available_data: Dict[str, Any]) -> ExperimentPlanner:
    """Factory function to create an ExperimentPlanner instance"""
    return ExperimentPlanner(available_data)