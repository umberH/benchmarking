#!/usr/bin/env python3
"""
Streamlit Dashboard for XAI Benchmarking Results
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from pathlib import Path
import numpy as np
from typing import Dict, Any, List
import altair as alt

# Page configuration
st.set_page_config(
    page_title="XAI Benchmarking Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1rem;
        padding-top: 10px;
        padding-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_benchmark_results(file_path: str = "results/benchmark_results.json") -> Dict[str, Any]:
    """Load benchmark results from JSON file"""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        st.warning(f"Results file not found: {file_path}")
        # Try to find a file in results/iterations/ as fallback
        iteration_dir = Path("results/iterations")
        if not iteration_dir.exists():
            st.error("No results/iterations/ directory found.")
            return {}
        iteration_files = sorted(iteration_dir.glob("*.json"))
        if not iteration_files:
            st.error("No iteration files found in results/iterations/.")
            return {}
        # Try to extract dataset name from user selection or just pick the first file
        # (main() will filter by dataset if needed)
        st.info(f"Loading fallback iteration file: {iteration_files[0].name}")
        with open(iteration_files[0], "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        st.error(f"Invalid JSON in results file: {file_path}")
        return {}

def parse_result_key(key: str) -> Dict[str, str]:
    """Parse result key to extract dataset, model, and method"""
    parts = key.split('_')
    if len(parts) >= 3:
        # Handle multi-word dataset names
        if parts[0] == 'adult' and parts[1] == 'income':
            dataset = 'adult_income'
            model = parts[2]
            method = '_'.join(parts[3:]) if len(parts) > 3 else parts[3]
        else:
            dataset = parts[0]
            model = parts[1]
            method = '_'.join(parts[2:]) if len(parts) > 2 else parts[2]
    else:
        dataset, model, method = 'unknown', 'unknown', 'unknown'
    
    return {'dataset': dataset, 'model': model, 'method': method}

def create_metrics_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """Create a DataFrame from evaluation results"""
    metrics_data = []
    
    for key, metrics in results.get('evaluation_results', {}).items():
        parsed = parse_result_key(key)
        row = {
            'Dataset': parsed['dataset'],
            'Model': parsed['model'],
            'Method': parsed['method'],
            **metrics
        }
        metrics_data.append(row)
    
    return pd.DataFrame(metrics_data)

def create_explanation_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """Create a DataFrame from explanation results"""
    explanation_data = []
    
    for key, info in results.get('explanation_results', {}).items():
        parsed = parse_result_key(key)
        row = {
            'Dataset': parsed['dataset'],
            'Model': parsed['model'],
            'Method': parsed['method'],
            'Generation Time (s)': info.get('generation_time', 0),
            'Number of Explanations': info.get('explanation_info', {}).get('n_explanations', 0),
            'Number of Features': len(info.get('explanation_info', {}).get('feature_names', []))
        }
        explanation_data.append(row)
    
    return pd.DataFrame(explanation_data)

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 XAI Benchmarking Dashboard</h1>', unsafe_allow_html=True)
    
    # Load results
    results = load_benchmark_results()
    if not results:
        st.warning("No benchmark results found. Please run the benchmark first.")
        # Try to load from iterations folder for selected dataset
        iteration_dir = Path("results/iterations")
        if iteration_dir.exists():
            iteration_files = sorted(iteration_dir.glob("*.json"))
            if iteration_files:
                st.info(f"Loading fallback iteration file: {iteration_files[0].name}")
                with open(iteration_files[0], "r") as f:
                    results = json.load(f)
    if not results:
        return
    
    # Sidebar for filters
    st.sidebar.header("🔧 Filters & Controls")
    
    # Extract unique values for filters
    metrics_df = create_metrics_dataframe(results)
    explanation_df = create_explanation_dataframe(results)
    
    if not metrics_df.empty:
        datasets = ['All'] + sorted(metrics_df['Dataset'].unique().tolist())
        models = ['All'] + sorted(metrics_df['Model'].unique().tolist())
        methods = ['All'] + sorted(metrics_df['Method'].unique().tolist())
        
        selected_dataset = st.sidebar.selectbox("Dataset", datasets)
        selected_model = st.sidebar.selectbox("Model", models)
        selected_method = st.sidebar.selectbox("Explanation Method", methods)
        
        # Apply filters
        filtered_df = metrics_df.copy()
        if selected_dataset != 'All':
            filtered_df = filtered_df[filtered_df['Dataset'] == selected_dataset]
        if selected_model != 'All':
            filtered_df = filtered_df[filtered_df['Model'] == selected_model]
        if selected_method != 'All':
            filtered_df = filtered_df[filtered_df['Method'] == selected_method]
    else:
        filtered_df = pd.DataFrame()
    
    # Main content with tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Overview", 
        "🎯 Model Performance", 
        "🔍 Explanation Metrics",
        "⏱️ Performance Analysis",
        "📊 Comparative Analysis",
        "📋 Raw Data",
        "🧩 Explanation Visualizations"
    ])
    # --- New Tab: Explanation Visualizations ---
    with tab7:
        st.header("🧩 Explanation Visualizations")
        # Load all iteration files
        iteration_dir = Path("results/iterations")
        iteration_files = sorted(iteration_dir.glob("*.json"))
        if not iteration_files:
            st.warning("No iteration files found in results/iterations/.")
        else:
            file_names = [f.name for f in iteration_files]
            selected_file = st.selectbox("Select Iteration File", file_names)
            st.info(f"Fetching file: {selected_file}")
            with open(iteration_dir / selected_file, "r") as f:
                iteration_data = json.load(f)
            st.subheader("Iteration File Content (truncated)")
            st.json({k: v if k != 'explanation_results' else '...omitted...' for k, v in iteration_data.items()})
            explanation_results = iteration_data.get("explanation_results", {})
            available_methods = list(explanation_results.keys())
            if not available_methods:
                st.warning("No explanation results in this iteration file.")
            else:
                selected_methods = st.multiselect("Select Explanation Methods", available_methods, default=available_methods)
                # For each selected method, show all explanations (or allow multi-instance selection)
                for selected_method in selected_methods:
                    st.markdown(f"---\n### Method: `{selected_method}`")
                    method_data = explanation_results[selected_method]
                    explanations = method_data.get("explanations", [])
                    if not explanations:
                        st.warning(f"No explanations found for method {selected_method}.")
                        continue
                    instance_ids = [e.get("instance_id", i) for i, e in enumerate(explanations)]
                    selected_instances = st.multiselect(f"Select Instances for {selected_method}", instance_ids, default=instance_ids[:5], key=f"{selected_method}_instances")
                    for sid in selected_instances:
                        explanation = explanations[instance_ids.index(sid)]
                        st.markdown(f"#### Instance: {sid}")
                        # Feature importance (SHAP, LIME, IG, Ablation)
                        if "feature_importance" in explanation:
                            st.subheader("Feature Importance (Bar Plot)")
                            feature_names = explanation.get("feature_names")
                            importances = explanation["feature_importance"]
                            if hasattr(importances, 'tolist'):
                                importances = importances.tolist()
                            df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
                            fig = px.bar(df, x="Feature", y="Importance", title=f"Feature Importances for Instance {sid}")
                            st.plotly_chart(fig, use_container_width=True)
                        # Counterfactual
                        if "counterfactual" in explanation:
                            st.subheader("Counterfactual (Table)")
                            orig = explanation.get("original")
                            cf = explanation.get("counterfactual")
                            if orig is not None and cf is not None:
                                diff = [o != c for o, c in zip(orig, cf)]
                                df = pd.DataFrame({"Original": orig, "Counterfactual": cf, "Changed": diff})
                                st.dataframe(df)
                        # Prototype
                        if "prototype" in explanation:
                            st.subheader("Prototype (Table)")
                            orig = explanation.get("original")
                            proto = explanation.get("prototype")
                            if orig is not None and proto is not None:
                                diff = [o != p for o, p in zip(orig, proto)]
                                df = pd.DataFrame({"Original": orig, "Prototype": proto, "Changed": diff})
                                st.dataframe(df)
                        # Importance map (image)
                        if "importance_map" in explanation:
                            st.subheader("Importance Map (Heatmap)")
                            imp_map = np.array(explanation["importance_map"])
                            fig = px.imshow(imp_map, color_continuous_scale='RdBu', title=f"Importance Map for Instance {sid}")
                            st.plotly_chart(fig, use_container_width=True)
                        st.subheader("Raw Explanation Data")
                        st.json(explanation)
    
    with tab1:
        st.header("📈 Experiment Overview")
        
        # Experiment info
        exp_info = results.get('experiment_info', {})
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Timestamp", exp_info.get('timestamp', 'N/A'))
        with col2:
            st.metric("Total Combinations", len(results.get('evaluation_results', {})))
        with col3:
            st.metric("Datasets", len(set(metrics_df['Dataset']) if not metrics_df.empty else []))
        with col4:
            st.metric("Methods", len(set(metrics_df['Method']) if not metrics_df.empty else []))
        
        # Summary statistics
        if not filtered_df.empty:
            st.subheader("📊 Summary Statistics")
            
            # Calculate summary stats
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
            summary_stats = filtered_df[numeric_cols].describe()
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Numeric Metrics Summary:**")
                st.dataframe(summary_stats, use_container_width=True)
            
            with col2:
                # Top performers
                st.write("**Top Performers by Faithfulness:**")
                top_faithful = filtered_df.nlargest(5, 'faithfulness')[['Dataset', 'Model', 'Method', 'faithfulness']]
                st.dataframe(top_faithful, use_container_width=True)
    
    with tab2:
        st.header("🎯 Model Performance Analysis")
        
        if not filtered_df.empty:
            # Model comparison heatmap
            st.subheader("🔥 Model Performance Heatmap")
            
            # Select metric for heatmap
            heatmap_metric = st.selectbox(
                "Select Metric for Heatmap:",
                ['faithfulness', 'monotonicity', 'completeness', 'stability', 'consistency', 'sparsity', 'simplicity'],
                key='heatmap_metric'
            )
            
            # Pivot table for heatmap
            pivot_data = filtered_df.pivot_table(
                values=heatmap_metric,
                index=['Dataset', 'Model'],
                columns='Method',
                aggfunc='mean'
            ).round(3)
            
            # Create heatmap
            fig = px.imshow(
                pivot_data.values,
                x=pivot_data.columns,
                y=[f"{idx[0]}_{idx[1]}" for idx in pivot_data.index],
                color_continuous_scale='RdYlBu',
                aspect='auto'
            )
            fig.update_layout(
                title=f"Model Performance Heatmap ({heatmap_metric.title()})",
                xaxis_title="Explanation Method",
                yaxis_title="Dataset_Model",
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Model comparison bar chart
            st.subheader("📊 Model Comparison")
            
            metric_to_plot = st.selectbox(
                "Select Metric to Compare:",
                ['faithfulness', 'monotonicity', 'completeness', 'stability', 'consistency', 'sparsity', 'simplicity']
            )
            
            if metric_to_plot in filtered_df.columns:
                fig = px.bar(
                    filtered_df.groupby(['Dataset', 'Model'])[metric_to_plot].mean().reset_index(),
                    x='Model',
                    y=metric_to_plot,
                    color='Dataset',
                    title=f"Average {metric_to_plot.title()} by Model and Dataset",
                    barmode='group'
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("🔍 Explanation Method Analysis")
        
        if not filtered_df.empty:
            # Method comparison
            st.subheader("📊 Explanation Method Comparison")
            
            # Radar chart for method comparison
            method_metrics = ['faithfulness', 'monotonicity', 'completeness', 'stability', 'consistency', 'sparsity', 'simplicity']
            
            selected_methods = st.multiselect(
                "Select methods to compare:",
                filtered_df['Method'].unique(),
                default=filtered_df['Method'].unique()[:3]
            )
            
            if selected_methods:
                method_data = filtered_df[filtered_df['Method'].isin(selected_methods)].groupby('Method')[method_metrics].mean()
                
                # Create radar chart
                fig = go.Figure()
                
                for method in selected_methods:
                    values = method_data.loc[method].values.tolist()
                    values += values[:1]  # Close the radar chart
                    
                    fig.add_trace(go.Scatterpolar(
                        r=values,
                        theta=method_metrics + [method_metrics[0]],
                        fill='toself',
                        name=method
                    ))
                
                fig.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )),
                    showlegend=True,
                    title="Explanation Method Performance Radar Chart"
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Method performance table
            st.subheader("📋 Method Performance Summary")
            method_summary = filtered_df.groupby('Method')[method_metrics].agg(['mean', 'std']).round(3)
            st.dataframe(method_summary, use_container_width=True)
    
    with tab4:
        st.header("⏱️ Performance Analysis")
        
        if not filtered_df.empty and not explanation_df.empty:
            # Time complexity analysis
            st.subheader("⏱️ Time Complexity Analysis")
            
            # Merge metrics with explanation data
            merged_df = filtered_df.merge(
                explanation_df[['Dataset', 'Model', 'Method', 'Generation Time (s)', 'Number of Explanations']],
                on=['Dataset', 'Model', 'Method'],
                how='left'
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Time vs Performance scatter plot
                scatter_kwargs = {
                    'x': 'Generation Time (s)',
                    'y': 'faithfulness',
                    'color': 'Method',
                    'hover_data': ['Dataset', 'Model'],
                    'title': "Time Complexity vs Faithfulness"
                }
                
                # Add size parameter only if the column exists
                if 'Number of Explanations' in merged_df.columns:
                    scatter_kwargs['size'] = 'Number of Explanations'
                
                fig = px.scatter(merged_df, **scatter_kwargs)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Time distribution by method
                fig = px.box(
                    merged_df,
                    x='Method',
                    y='Generation Time (s)',
                    title="Generation Time Distribution by Method"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Performance vs Time efficiency
            st.subheader("⚡ Efficiency Analysis")
            
            # Calculate efficiency score (faithfulness / time)
            merged_df['efficiency'] = merged_df['faithfulness'] / (merged_df['Generation Time (s)'] + 1e-6)
            
            fig = px.bar(
                merged_df.groupby('Method')['efficiency'].mean().reset_index(),
                x='Method',
                y='efficiency',
                title="Efficiency Score (Faithfulness / Time) by Method"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        st.header("📊 Comparative Analysis")
        
        if not filtered_df.empty:
            # Correlation analysis
            st.subheader("🔗 Metric Correlations")
            
            numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns
            correlation_matrix = filtered_df[numeric_cols].corr()
            
            fig = px.imshow(
                correlation_matrix,
                color_continuous_scale='RdBu',
                aspect='auto',
                title="Metric Correlation Matrix"
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Multi-dimensional analysis
            st.subheader("🎯 Multi-dimensional Analysis")
            
            # PCA-like visualization using top metrics
            top_metrics = ['faithfulness', 'monotonicity', 'completeness', 'stability']
            
            if len(top_metrics) >= 2:
                fig = px.scatter(
                    filtered_df,
                    x=top_metrics[0],
                    y=top_metrics[1],
                    color='Method',
                    size='consistency',
                    hover_data=['Dataset', 'Model'],
                    title=f"{top_metrics[0].title()} vs {top_metrics[1].title()}"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    with tab6:
        st.header("📋 Raw Data & Export")
        
        # Raw data tables
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Evaluation Metrics")
            if not filtered_df.empty:
                st.dataframe(filtered_df, use_container_width=True)
                
                # Download button
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Metrics CSV",
                    data=csv,
                    file_name="xai_benchmark_metrics.csv",
                    mime="text/csv"
                )
        
        with col2:
            st.subheader("⏱️ Explanation Performance")
            if not explanation_df.empty:
                st.dataframe(explanation_df, use_container_width=True)
                
                # Download button
                csv = explanation_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Performance CSV",
                    data=csv,
                    file_name="xai_benchmark_performance.csv",
                    mime="text/csv"
                )
        
        # Configuration info
        st.subheader("⚙️ Experiment Configuration")
        config = results.get('experiment_info', {}).get('config', {})
        st.json(config)

if __name__ == "__main__":
    main() 