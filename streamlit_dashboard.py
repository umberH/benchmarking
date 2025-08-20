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

# Claude Artifacts-style CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    /* Main theme inspired by Claude Artifacts */
    .main-header {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: #1a1a1a;
        text-align: center;
        margin: 2rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        text-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    
    /* Card components with Claude-style design */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid rgba(226, 232, 240, 0.6);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin: 0.75rem 0;
        transition: all 0.3s ease;
        backdrop-filter: blur(10px);
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        border-color: rgba(102, 126, 234, 0.3);
    }
    
    /* Status indicators with gradients */
    .success-indicator {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 12px;
        font-weight: 600;
        box-shadow: 0 4px 8px rgba(16, 185, 129, 0.3);
    }
    
    .warning-indicator {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 12px;
        font-weight: 600;
        box-shadow: 0 4px 8px rgba(245, 158, 11, 0.3);
    }
    
    .info-indicator {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 12px;
        font-weight: 600;
        box-shadow: 0 4px 8px rgba(59, 130, 246, 0.3);
    }
    
    /* Enhanced tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
        border-radius: 16px;
        padding: 8px;
        box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.06);
    }
    
    .stTabs [data-baseweb="tab"] {
        height: auto;
        padding: 12px 20px;
        border-radius: 12px;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 0.95rem;
        border: none;
        background: transparent;
        color: #64748b;
        transition: all 0.2s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
        transform: translateY(-1px);
    }
    
    /* Sidebar enhancements */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
        border-right: 1px solid rgba(226, 232, 240, 0.8);
    }
    
    /* Button styling with Claude aesthetics */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        transition: all 0.2s ease;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 16px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #5a67d8 0%, #667eea 100%);
    }
    
    /* Input and select styling */
    .stSelectbox > div > div {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 12px;
        transition: all 0.2s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Metric containers with modern design */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid rgba(226, 232, 240, 0.6);
        padding: 1.5rem;
        border-radius: 16px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.02);
        transition: all 0.2s ease;
    }
    
    [data-testid="metric-container"]:hover {
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.1);
        transform: translateY(-1px);
    }
    
    /* Chart containers with glass morphism */
    .js-plotly-plot {
        background: rgba(255, 255, 255, 0.9);
        border-radius: 16px;
        border: 1px solid rgba(226, 232, 240, 0.5);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        overflow: hidden;
        margin: 1rem 0;
    }
    
    /* Custom alert styles */
    .success-alert {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 1px solid #a7f3d0;
        color: #065f46;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .warning-alert {
        background: linear-gradient(135deg, #fffbeb 0%, #fef3c7 100%);
        border: 1px solid #fcd34d;
        color: #92400e;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .error-alert {
        background: linear-gradient(135deg, #fef2f2 0%, #fecaca 100%);
        border: 1px solid #fca5a5;
        color: #991b1b;
        padding: 1rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    /* Hide default Streamlit elements for cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
        border-radius: 8px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 8px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #5a67d8 0%, #667eea 100%);
    }
    
    /* Animation for loading states */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .fade-in {
        animation: fadeIn 0.5s ease-out;
    }
</style>
""", unsafe_allow_html=True)

def show_success(message: str):
    """Display a success message with Claude Artifacts styling"""
    st.markdown(f'<div class="success-alert">{message}</div>', unsafe_allow_html=True)

def show_warning(message: str):
    """Display a warning message with Claude Artifacts styling"""
    st.markdown(f'<div class="warning-alert">{message}</div>', unsafe_allow_html=True)

def show_error(message: str):
    """Display an error message with Claude Artifacts styling"""
    st.markdown(f'<div class="error-alert">{message}</div>', unsafe_allow_html=True)

def show_info(message: str):
    """Display an info message with Claude Artifacts styling"""
    st.markdown(f'<div class="info-indicator">{message}</div>', unsafe_allow_html=True)

def parse_feature_importance(importance_data):
    """Parse feature importance from various formats"""
    if importance_data is None:
        return []
    
    # If it's already a list
    if isinstance(importance_data, list):
        return importance_data
    
    # If it's a numpy array
    if hasattr(importance_data, 'tolist'):
        return importance_data.tolist()
    
    # If it's a string representation
    if isinstance(importance_data, str):
        # Handle string representations like "[0. 1. 0. 0. 0.]"
        try:
            import ast
            import re
            
            # Clean the string
            cleaned = importance_data.strip()
            
            # Remove extra brackets and whitespace
            cleaned = re.sub(r'^\[|\]$', '', cleaned)
            
            # Split by whitespace and convert to floats
            values = []
            for val in cleaned.split():
                try:
                    values.append(float(val))
                except ValueError:
                    continue
            
            if values:
                return values
            
            # Try ast.literal_eval as fallback
            return ast.literal_eval(importance_data)
        
        except Exception:
            try:
                # Last resort: eval (careful with security)
                return eval(importance_data)
            except:
                return []
    
    # Try to convert single values
    try:
        return [float(importance_data)]
    except:
        return []

@st.cache_data
def discover_available_experiments() -> Dict[str, Dict[str, Any]]:
    """Discover all available experiment folders"""
    experiments = {}
    results_dir = Path("results")
    
    # Look for experiment_* folders
    for exp_dir in results_dir.glob("experiment_*"):
        if exp_dir.is_dir():
            # Try to load benchmark results from the experiment folder
            benchmark_file = exp_dir / "benchmark_results.json"
            if benchmark_file.exists():
                try:
                    with open(benchmark_file, "r", encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # Extract timestamp from folder name
                    timestamp = exp_dir.name.replace("experiment_", "")
                    
                    experiments[exp_dir.name] = {
                        "path": str(benchmark_file),
                        "data": data,
                        "timestamp": timestamp,
                        "folder": str(exp_dir),
                        "display_name": f"Experiment {timestamp[:8]} {timestamp[9:15]}",
                        "comprehensive_results_count": len(data.get("comprehensive_results", [])),
                        "description": f"Comprehensive experiment from {timestamp[:8]}"
                    }
                except Exception as e:
                    continue
    
    return experiments

@st.cache_data
def load_experiment_data(experiment_info: Dict[str, Any]) -> Dict[str, Any]:
    """Load data for selected experiment"""
    return experiment_info["data"]

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
    """Create a DataFrame from evaluation results (comprehensive format)"""
    metrics_data = []
    
    # Handle comprehensive_results format (new experiment format)
    if 'comprehensive_results' in results:
        comprehensive_results = results['comprehensive_results']
        
        # Ensure comprehensive_results is a list
        if not isinstance(comprehensive_results, list):
            st.error(f"Expected comprehensive_results to be a list, got {type(comprehensive_results)}")
            return pd.DataFrame()
        
        for i, result in enumerate(comprehensive_results):
            # Ensure each result is a dictionary
            if not isinstance(result, dict):
                st.warning(f"Skipping result {i}: expected dict, got {type(result)}")
                continue
                
            evaluation = result.get('evaluations', {})
            
            # Ensure evaluation is a dictionary
            if not isinstance(evaluation, dict):
                evaluation = {}
            
            # Safely extract metrics with defaults
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
    
    return pd.DataFrame(metrics_data)

def create_explanation_dataframe(results: Dict[str, Any]) -> pd.DataFrame:
    """Create a DataFrame from explanation results (comprehensive format)"""
    explanation_data = []
    
    # Handle comprehensive_results format (new experiment format)
    if 'comprehensive_results' in results:
        comprehensive_results = results['comprehensive_results']
        
        # Ensure comprehensive_results is a list
        if not isinstance(comprehensive_results, list):
            st.error(f"Expected comprehensive_results to be a list, got {type(comprehensive_results)}")
            return pd.DataFrame()
        
        for i, result in enumerate(comprehensive_results):
            # Ensure each result is a dictionary
            if not isinstance(result, dict):
                st.warning(f"Skipping result {i}: expected dict, got {type(result)}")
                continue
                
            explanation_info = result.get('explanation_info', {})
            
            # Ensure explanation_info is a dictionary
            if not isinstance(explanation_info, dict):
                explanation_info = {}
            
            row = {
                'Dataset': result.get('dataset', 'unknown'),
                'Model': result.get('model', 'unknown'),
                'Method': result.get('explanation_method', 'unknown'),
                'Generation Time (s)': explanation_info.get('generation_time', 0),
                'Number of Explanations': explanation_info.get('n_explanations', 0),
                'Number of Features': len(explanation_info.get('feature_names', []))
            }
            explanation_data.append(row)
    
    return pd.DataFrame(explanation_data)

def main():
    # Header with Claude Artifacts styling
    st.markdown('<h1 class="main-header">XAI Benchmarking Dashboard</h1>', unsafe_allow_html=True)
    
    # Welcome section with quick stats
    col1, col2, col3 = st.columns(3)
    with col2:
        st.markdown("""
        <div class="metric-card" style="text-align: center;">
            <h3 style="color: #667eea; margin: 0;">Welcome to XAI Analysis</h3>
            <p style="color: #64748b; margin: 0.5rem 0;">Explore explainable AI methods across datasets and models</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Sidebar for experiment selection
    st.sidebar.header("Experiment Selection")
    
    # Discover available experiments
    available_experiments = discover_available_experiments()
    
    if not available_experiments:
        st.sidebar.error("No experiment data found!")
        show_error("No experiment data available. Please run benchmarking first with: `python main.py --comprehensive`")
        return
    
    # Experiment selection dropdown
    experiment_names = list(available_experiments.keys())
    selected_experiment_name = st.sidebar.selectbox(
        "Select Experiment:",
        experiment_names,
        format_func=lambda x: available_experiments[x]["display_name"],
        help="Choose which experiment to analyze"
    )
    
    selected_experiment_info = available_experiments[selected_experiment_name]
    
    # Display experiment information
    with st.sidebar.expander("Experiment Information"):
        st.write(f"**Timestamp:** {selected_experiment_info['timestamp']}")
        st.write(f"**Total Results:** {selected_experiment_info['comprehensive_results_count']}")
        st.write(f"**Description:** {selected_experiment_info['description']}")
        st.write(f"**Folder:** {selected_experiment_info['folder']}")
    
    # Load selected experiment data
    results = load_experiment_data(selected_experiment_info)
    
    if not results:
        show_error("Failed to load selected experiment data.")
        return
    
    # Sidebar for filters
    st.sidebar.header("Filters & Controls")
    
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
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Overview", 
        "Model Performance", 
        "Explanation Metrics",
        "Performance Analysis",
        "Explanation Visualizations",
        "Detailed Analysis"
    ])
    # --- New Tab: Explanation Visualizations ---
    with tab5:
        st.header("Explanation Visualizations")
        
        # Load detailed explanations from selected experiment
        detailed_explanations_dir = Path(selected_experiment_info['folder']) / "detailed_explanations"
        
        if not detailed_explanations_dir.exists():
            show_warning("No detailed explanations found for this experiment.")
            return
        
        # Get all available combinations
        available_combinations = []
        for dataset_dir in detailed_explanations_dir.iterdir():
            if dataset_dir.is_dir():
                for model_dir in dataset_dir.iterdir():
                    if model_dir.is_dir():
                        for explanation_file in model_dir.glob("*_detailed_explanations.json"):
                            method_name = explanation_file.name.replace("_detailed_explanations.json", "")
                            combination = f"{dataset_dir.name}_{model_dir.name}_{method_name}"
                            available_combinations.append({
                                "display_name": combination,
                                "dataset": dataset_dir.name,
                                "model": model_dir.name,
                                "method": method_name,
                                "file_path": explanation_file
                            })
        
        if not available_combinations:
            show_warning("No detailed explanation files found.")
            return
        
        # Combination selection
        combination_names = [combo["display_name"] for combo in available_combinations]
        selected_combination_name = st.selectbox(
            "Select Dataset-Model-Method Combination:",
            combination_names,
            help="Choose which combination to visualize"
        )
        
        selected_combo = next(c for c in available_combinations if c["display_name"] == selected_combination_name)
        
        # Load the detailed explanation data
        try:
            with open(selected_combo["file_path"], "r", encoding='utf-8') as f:
                explanation_data = json.load(f)
        except Exception as e:
            show_error(f"Failed to load explanation data: {e}")
            return
        
        # Display basic information
        show_info(f"**Dataset:** {selected_combo['dataset']} | **Model:** {selected_combo['model']} | **Method:** {selected_combo['method']}")
        
        # Method-specific visualizations
        method = selected_combo['method']
        
        # Handle different data structures
        if isinstance(explanation_data, dict):
            explanations = explanation_data.get("explanations", [])
        elif isinstance(explanation_data, list):
            # If explanation_data is already a list of explanations
            explanations = explanation_data
        else:
            show_error(f"Unexpected data format: expected dict or list, got {type(explanation_data)}")
            return
        
        if not explanations:
            show_warning("No explanations found in this file.")
            return
        
        st.subheader(f"{method.upper()} Visualizations")
        
        # Create tabs for different types of analysis
        viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
            "Overview", "Individual Predictions", "Feature Analysis", "Method-Specific"
        ])
        
        with viz_tab1:
            st.markdown("### 📋 Explanation Summary")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Explanations", len(explanations))
            with col2:
                if explanations and "feature_names" in explanations[0]:
                    st.metric("Number of Features", len(explanations[0]["feature_names"]))
                else:
                    st.metric("Number of Features", "N/A")
            with col3:
                st.metric("Method Type", method.replace("_", " ").title())
            
            # Show first few explanations as examples
            st.markdown("### 🔍 Sample Explanations")
            sample_size = min(3, len(explanations))
            for i in range(sample_size):
                with st.expander(f"Sample Explanation {i+1}"):
                    st.json(explanations[i])
        
        with viz_tab2:
            st.markdown("### 🎯 Individual Instance Analysis")
            
            # Instance selection
            instance_ids = [str(i) for i in range(len(explanations))]
            selected_instance_idx = st.selectbox(
                "Select Instance to Analyze:",
                range(len(explanations)),
                format_func=lambda x: f"Instance {x}",
                help="Choose an instance to analyze in detail"
            )
            
            explanation = explanations[selected_instance_idx]
            
            # Feature importance visualization
            if "feature_importance" in explanation and "feature_names" in explanation:
                st.markdown("#### 📊 Feature Importance")
                feature_names = explanation["feature_names"]
                importances = explanation["feature_importance"]
                
                if hasattr(importances, 'tolist'):
                    importances = importances.tolist()
                
                # Create DataFrame and sort by absolute importance
                feature_df = pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importances,
                    "Abs_Importance": [abs(x) for x in importances]
                }).sort_values("Abs_Importance", ascending=True)
                
                # Bar plot
                fig = px.bar(
                    feature_df.tail(15),  # Top 15 features
                    x="Importance",
                    y="Feature",
                    orientation='h',
                    title=f"Top 15 Feature Importances (Instance {selected_instance_idx})",
                    color="Importance",
                    color_continuous_scale="RdBu_r"
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Summary statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Most Important Feature", feature_df.iloc[-1]["Feature"])
                    st.metric("Max Importance", f"{max(importances):.4f}")
                with col2:
                    st.metric("Least Important Feature", feature_df.iloc[0]["Feature"])
                    st.metric("Min Importance", f"{min(importances):.4f}")
        
        with viz_tab3:
            st.markdown("### 📈 Feature Analysis Across All Instances")
            
            # Aggregate feature importance across all instances
            if explanations and "feature_importance" in explanations[0] and "feature_names" in explanations[0]:
                feature_names = explanations[0]["feature_names"]
                all_importances = []
                
                for exp in explanations:
                    if "feature_importance" in exp:
                        importances = exp["feature_importance"]
                        if hasattr(importances, 'tolist'):
                            importances = importances.tolist()
                        all_importances.append(importances)
                
                if all_importances:
                    # Calculate statistics
                    importance_array = np.array(all_importances)
                    mean_importance = np.mean(importance_array, axis=0)
                    std_importance = np.std(importance_array, axis=0)
                    
                    # Create summary DataFrame
                    feature_summary = pd.DataFrame({
                        "Feature": feature_names,
                        "Mean_Importance": mean_importance,
                        "Std_Importance": std_importance,
                        "Abs_Mean": np.abs(mean_importance)
                    }).sort_values("Abs_Mean", ascending=False)
                    
                    # Global feature importance plot
                    fig = px.bar(
                        feature_summary.head(20),
                        x="Feature",
                        y="Mean_Importance",
                        error_y="Std_Importance",
                        title="Global Feature Importance (Mean ± Std)",
                        color="Mean_Importance",
                        color_continuous_scale="RdBu_r"
                    )
                    fig.update_xaxes(tickangle=45)
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature importance distribution
                    st.markdown("#### 📊 Feature Importance Distribution")
                    top_features = feature_summary.head(10)["Feature"].tolist()
                    
                    # Create distribution plots for top features
                    selected_features = st.multiselect(
                        "Select features to show distribution:",
                        top_features,
                        default=top_features[:5]
                    )
                    
                    if selected_features:
                        fig = go.Figure()
                        for feature in selected_features:
                            feature_idx = feature_names.index(feature)
                            feature_values = importance_array[:, feature_idx]
                            fig.add_trace(go.Box(
                                y=feature_values,
                                name=feature,
                                boxpoints='outliers'
                            ))
                        
                        fig.update_layout(
                            title="Feature Importance Distributions",
                            yaxis_title="Importance",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        with viz_tab4:
            st.markdown(f"### 🔬 {method.upper()}-Specific Visualizations")
            
            # SHAP-specific visualizations
            if "shap" in method.lower():
                st.markdown("#### 🎯 SHAP Analysis")
                
                # Local vs Global SHAP plots
                if explanations and "feature_importance" in explanations[0]:
                    # Prepare data for SHAP-style plots
                    feature_names = explanations[0].get("feature_names", [])
                    shap_values = []
                    
                    for exp in explanations:
                        if "feature_importance" in exp:
                            importances = exp["feature_importance"]
                            if hasattr(importances, 'tolist'):
                                importances = importances.tolist()
                            shap_values.append(importances)
                    
                    if shap_values and feature_names:
                        shap_array = np.array(shap_values)
                        
                        # Summary plot (global)
                        st.markdown("##### 🌍 Global SHAP Summary")
                        
                        # Calculate feature importance ranking
                        mean_abs_shap = np.mean(np.abs(shap_array), axis=0)
                        feature_ranking = pd.DataFrame({
                            "Feature": feature_names,
                            "Mean_Abs_SHAP": mean_abs_shap
                        }).sort_values("Mean_Abs_SHAP", ascending=True)
                        
                        # SHAP summary plot style
                        fig = px.scatter(
                            x=shap_array[:, :].flatten(),
                            y=np.tile(range(len(feature_names)), len(shap_values)),
                            color=shap_array[:, :].flatten(),
                            color_continuous_scale="RdBu_r",
                            title="SHAP Summary Plot",
                            labels={"x": "SHAP Value", "y": "Feature Index", "color": "SHAP Value"}
                        )
                        fig.update_layout(
                            yaxis=dict(
                                tickmode="array",
                                tickvals=list(range(len(feature_names))),
                                ticktext=feature_names
                            ),
                            height=600
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        
                        # Local SHAP plot
                        st.markdown("##### 🎯 Local SHAP Analysis")
                        instance_for_local = st.selectbox(
                            "Select instance for local SHAP plot:",
                            range(len(explanations)),
                            format_func=lambda x: f"Instance {x}",
                            key="local_shap_instance"
                        )
                        
                        if instance_for_local < len(shap_values):
                            local_shap = shap_values[instance_for_local]
                            local_df = pd.DataFrame({
                                "Feature": feature_names,
                                "SHAP_Value": local_shap
                            }).sort_values("SHAP_Value", key=abs, ascending=True)
                            
                            # Waterfall-style plot
                            fig = px.bar(
                                local_df.tail(15),
                                x="SHAP_Value",
                                y="Feature",
                                orientation='h',
                                title=f"Local SHAP Values (Instance {instance_for_local})",
                                color="SHAP_Value",
                                color_continuous_scale="RdBu_r"
                            )
                            fig.add_vline(x=0, line_dash="dash", line_color="black")
                            fig.update_layout(height=500)
                            st.plotly_chart(fig, use_container_width=True)
            
            # LIME-specific visualizations
            elif "lime" in method.lower():
                st.markdown("#### 🍋 LIME Analysis")
                st.info("LIME explanations show local linear approximations around instances.")
                
                if explanations and "feature_importance" in explanations[0]:
                    # LIME typically shows fewer features
                    instance_idx = st.selectbox(
                        "Select instance for LIME analysis:",
                        range(len(explanations)),
                        format_func=lambda x: f"Instance {x}",
                        key="lime_instance"
                    )
                    
                    explanation = explanations[instance_idx]
                    if "feature_importance" in explanation and "feature_names" in explanation:
                        importances = explanation["feature_importance"]
                        feature_names = explanation["feature_names"]
                        
                        if hasattr(importances, 'tolist'):
                            importances = importances.tolist()
                        
                        # LIME usually shows top contributing features
                        lime_df = pd.DataFrame({
                            "Feature": feature_names,
                            "Contribution": importances
                        }).sort_values("Contribution", key=abs, ascending=False).head(10)
                        
                        fig = px.bar(
                            lime_df,
                            x="Contribution",
                            y="Feature",
                            orientation='h',
                            title=f"LIME Local Explanation (Instance {instance_idx})",
                            color="Contribution",
                            color_continuous_scale="RdBu_r"
                        )
                        fig.add_vline(x=0, line_dash="dash", line_color="black")
                        st.plotly_chart(fig, use_container_width=True)
            
            # Integrated Gradients
            elif "integrated_gradients" in method.lower():
                st.markdown("#### 🎯 Integrated Gradients Analysis")
                st.info("Integrated Gradients shows attribution along the path from baseline to input.")
                
            # Add more method-specific visualizations as needed
            else:
                st.markdown(f"#### 🔧 {method.replace('_', ' ').title()} Analysis")
                st.info(f"Method-specific visualizations for {method} will be displayed here.")
                
                # Generic feature importance for other methods
                if explanations and "feature_importance" in explanations[0]:
                    st.markdown("##### Feature Importance Analysis")
                    instance_idx = st.selectbox(
                        f"Select instance for {method} analysis:",
                        range(len(explanations)),
                        format_func=lambda x: f"Instance {x}",
                        key=f"{method}_instance"
                    )
                    
                    explanation = explanations[instance_idx]
                    if "feature_importance" in explanation and "feature_names" in explanation:
                        importances = explanation["feature_importance"]
                        feature_names = explanation["feature_names"]
                        
                        if hasattr(importances, 'tolist'):
                            importances = importances.tolist()
                        
                        method_df = pd.DataFrame({
                            "Feature": feature_names,
                            "Importance": importances
                        }).sort_values("Importance", key=abs, ascending=False).head(15)
                        
                        fig = px.bar(
                            method_df,
                            x="Importance",
                            y="Feature",
                            orientation='h',
                            title=f"{method.replace('_', ' ').title()} Feature Importance (Instance {instance_idx})",
                            color="Importance",
                            color_continuous_scale="Viridis"
                        )
                        st.plotly_chart(fig, use_container_width=True)
    
    # --- New Tab: Detailed Analysis ---
    with tab6:
        st.header("Detailed Analysis")
        
        # Load detailed explanations directory
        detailed_explanations_dir = Path(selected_experiment_info['folder']) / "detailed_explanations"
        
        if not detailed_explanations_dir.exists():
            show_warning("No detailed explanations found for this experiment. Detailed analysis requires detailed explanation data.")
            return
        
        # Analysis type selection
        analysis_type = st.selectbox(
            "Select Analysis Type:",
            ["Dataset Level Analysis", "Model Level Analysis", "Explanation Method Deep Dive", "Feature Importance Analysis", "Individual Instance Analysis"],
            help="Choose the type of detailed analysis to perform"
        )
        
        if analysis_type == "Dataset Level Analysis":
            st.subheader("Dataset Level Analysis")
            
            # Get available datasets from detailed explanations
            available_datasets = []
            for dataset_dir in detailed_explanations_dir.iterdir():
                if dataset_dir.is_dir():
                    available_datasets.append(dataset_dir.name)
            
            if not available_datasets:
                show_warning("No datasets found in detailed explanations.")
                return
            
            selected_dataset = st.selectbox("Select Dataset:", available_datasets)
            dataset_path = detailed_explanations_dir / selected_dataset
            
            # Get all models and methods for this dataset
            models_methods = {}
            total_explanations = 0
            
            for model_dir in dataset_path.iterdir():
                if model_dir.is_dir():
                    model_name = model_dir.name
                    methods = []
                    for explanation_file in model_dir.glob("*_detailed_explanations.json"):
                        method_name = explanation_file.name.replace("_detailed_explanations.json", "")
                        methods.append(method_name)
                        
                        # Count explanations in this file
                        try:
                            with open(explanation_file, "r", encoding='utf-8') as f:
                                data = json.load(f)
                                if isinstance(data, dict):
                                    explanations = data.get("explanations", [])
                                elif isinstance(data, list):
                                    explanations = data
                                else:
                                    explanations = []
                                total_explanations += len(explanations)
                        except:
                            continue
                    
                    if methods:
                        models_methods[model_name] = methods
            
            # Dataset overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Models Available", len(models_methods))
            with col2:
                all_methods = set()
                for methods in models_methods.values():
                    all_methods.update(methods)
                st.metric("Unique Methods", len(all_methods))
            with col3:
                total_combinations = sum(len(methods) for methods in models_methods.values())
                st.metric("Total Combinations", total_combinations)
            with col4:
                st.metric("Total Explanations", total_explanations)
            
            # Show available combinations
            st.markdown("#### 🔍 Available Model-Method Combinations")
            
            combinations_data = []
            for model, methods in models_methods.items():
                for method in methods:
                    # Try to load explanation file to get more details
                    explanation_file = dataset_path / model / f"{method}_detailed_explanations.json"
                    explanation_count = 0
                    avg_generation_time = 0
                    
                    try:
                        with open(explanation_file, "r", encoding='utf-8') as f:
                            data = json.load(f)
                            if isinstance(data, dict):
                                explanations = data.get("explanations", [])
                                generation_times = [exp.get("generation_time", 0) for exp in explanations if isinstance(exp, dict)]
                                explanation_count = len(explanations)
                                avg_generation_time = np.mean(generation_times) if generation_times else 0
                            elif isinstance(data, list):
                                explanation_count = len(data)
                    except:
                        pass
                    
                    combinations_data.append({
                        "Model": model,
                        "Method": method,
                        "Explanations": explanation_count,
                        "Avg Generation Time": f"{avg_generation_time:.3f}s" if avg_generation_time > 0 else "N/A"
                    })
            
            combinations_df = pd.DataFrame(combinations_data)
            st.dataframe(combinations_df, use_container_width=True)
            
            # Model-Method heatmap
            if not combinations_df.empty:
                st.markdown("#### 🔥 Explanation Count Heatmap")
                
                # Create pivot table for heatmap
                pivot_data = combinations_df.pivot_table(
                    values='Explanations',
                    index='Model',
                    columns='Method',
                    aggfunc='sum',
                    fill_value=0
                )
                
                fig = px.imshow(
                    pivot_data.values,
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    color_continuous_scale='Blues',
                    title=f"Number of Explanations for {selected_dataset}",
                    labels=dict(color="Explanation Count")
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
        elif analysis_type == "🤖 Model Level Analysis":
            st.subheader("🤖 Model Level Analysis")
            
            # Get available models from detailed explanations
            available_models = set()
            for dataset_dir in detailed_explanations_dir.iterdir():
                if dataset_dir.is_dir():
                    for model_dir in dataset_dir.iterdir():
                        if model_dir.is_dir():
                            available_models.add(model_dir.name)
            
            available_models = sorted(list(available_models))
            
            if not available_models:
                show_warning("No models found in detailed explanations.")
                return
            
            selected_model = st.selectbox("Select Model:", available_models)
            
            # Collect model data from detailed explanations
            model_data = {}
            total_explanations = 0
            
            for dataset_dir in detailed_explanations_dir.iterdir():
                if dataset_dir.is_dir():
                    dataset_name = dataset_dir.name
                    model_dir = dataset_dir / selected_model
                    
                    if model_dir.exists():
                        methods = []
                        for explanation_file in model_dir.glob("*_detailed_explanations.json"):
                            method_name = explanation_file.name.replace("_detailed_explanations.json", "")
                            
                            # Load explanation data
                            try:
                                with open(explanation_file, "r", encoding='utf-8') as f:
                                    data = json.load(f)
                                    if isinstance(data, dict):
                                        explanations = data.get("explanations", [])
                                    elif isinstance(data, list):
                                        explanations = data
                                    else:
                                        explanations = []
                                    
                                    methods.append({
                                        "method": method_name,
                                        "explanations": explanations,
                                        "count": len(explanations)
                                    })
                                    total_explanations += len(explanations)
                            except:
                                continue
                        
                        if methods:
                            model_data[dataset_name] = methods
            
            if not model_data:
                show_warning(f"No data found for model {selected_model}")
                return
            
            # Model overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Datasets", len(model_data))
            with col2:
                all_methods = set()
                for dataset_methods in model_data.values():
                    for method_info in dataset_methods:
                        all_methods.add(method_info["method"])
                st.metric("Unique Methods", len(all_methods))
            with col3:
                total_combinations = sum(len(methods) for methods in model_data.values())
                st.metric("Total Combinations", total_combinations)
            with col4:
                st.metric("Total Explanations", total_explanations)
            
            # Dataset-Method analysis
            st.markdown("#### 📊 Dataset-Method Analysis")
            
            analysis_data = []
            for dataset, methods in model_data.items():
                for method_info in methods:
                    method = method_info["method"]
                    explanations = method_info["explanations"]
                    
                    # Calculate statistics from explanations
                    if explanations and isinstance(explanations[0], dict):
                        generation_times = [exp.get("generation_time", 0) for exp in explanations]
                        feature_counts = [len(exp.get("feature_names", [])) for exp in explanations if "feature_names" in exp]
                        
                        analysis_data.append({
                            "Dataset": dataset,
                            "Method": method,
                            "Explanations": len(explanations),
                            "Avg Generation Time": np.mean(generation_times) if generation_times else 0,
                            "Avg Feature Count": np.mean(feature_counts) if feature_counts else 0
                        })
                    else:
                        analysis_data.append({
                            "Dataset": dataset,
                            "Method": method,
                            "Explanations": len(explanations),
                            "Avg Generation Time": 0,
                            "Avg Feature Count": 0
                        })
            
            analysis_df = pd.DataFrame(analysis_data)
            st.dataframe(analysis_df, use_container_width=True)
            
            # Method performance comparison
            if not analysis_df.empty:
                st.markdown("#### 🔥 Method Performance Heatmap")
                
                # Create heatmap based on explanation count
                pivot_data = analysis_df.pivot_table(
                    values='Explanations',
                    index='Dataset',
                    columns='Method',
                    aggfunc='sum',
                    fill_value=0
                )
                
                fig = px.imshow(
                    pivot_data.values,
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    color_continuous_scale='Viridis',
                    title=f"Explanation Coverage for {selected_model}",
                    labels=dict(color="Explanation Count")
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
                
                # Generation time comparison
                st.markdown("#### ⏱️ Generation Time Analysis")
                
                time_data = analysis_df[analysis_df['Avg Generation Time'] > 0]
                if not time_data.empty:
                    fig = px.bar(
                        time_data,
                        x='Method',
                        y='Avg Generation Time',
                        color='Dataset',
                        title=f"Average Generation Time by Method for {selected_model}",
                        barmode='group'
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    show_info("No generation time data available for analysis.")
            
        elif analysis_type == "🔍 Explanation Method Deep Dive":
            st.subheader("🔍 Explanation Method Deep Dive")
            
            # Get all available methods from detailed explanations
            available_methods = set()
            for dataset_dir in detailed_explanations_dir.iterdir():
                if dataset_dir.is_dir():
                    for model_dir in dataset_dir.iterdir():
                        if model_dir.is_dir():
                            for explanation_file in model_dir.glob("*_detailed_explanations.json"):
                                method_name = explanation_file.name.replace("_detailed_explanations.json", "")
                                available_methods.add(method_name)
            
            available_methods = sorted(list(available_methods))
            
            if not available_methods:
                show_warning("No explanation methods found in detailed explanations.")
                return
            
            selected_method = st.selectbox("Select Explanation Method:", available_methods)
            
            # Collect method data across all datasets and models
            method_data = []
            total_explanations = 0
            
            for dataset_dir in detailed_explanations_dir.iterdir():
                if dataset_dir.is_dir():
                    dataset_name = dataset_dir.name
                    for model_dir in dataset_dir.iterdir():
                        if model_dir.is_dir():
                            model_name = model_dir.name
                            explanation_file = model_dir / f"{selected_method}_detailed_explanations.json"
                            
                            if explanation_file.exists():
                                try:
                                    with open(explanation_file, "r", encoding='utf-8') as f:
                                        data = json.load(f)
                                        if isinstance(data, dict):
                                            explanations = data.get("explanations", [])
                                        elif isinstance(data, list):
                                            explanations = data
                                        else:
                                            explanations = []
                                        
                                        if explanations:
                                            # Calculate statistics
                                            generation_times = [exp.get("generation_time", 0) for exp in explanations if isinstance(exp, dict)]
                                            feature_counts = [len(exp.get("feature_names", [])) for exp in explanations if isinstance(exp, dict) and "feature_names" in exp]
                                            
                                            method_data.append({
                                                "Dataset": dataset_name,
                                                "Model": model_name,
                                                "Explanations": len(explanations),
                                                "Avg Generation Time": np.mean(generation_times) if generation_times else 0,
                                                "Avg Feature Count": np.mean(feature_counts) if feature_counts else 0,
                                                "Total Generation Time": np.sum(generation_times) if generation_times else 0
                                            })
                                            total_explanations += len(explanations)
                                except:
                                    continue
            
            if not method_data:
                show_warning(f"No data found for method {selected_method}")
                return
            
            method_df = pd.DataFrame(method_data)
            
            # Method overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Explanations", total_explanations)
            with col2:
                st.metric("Dataset Coverage", len(method_df['Dataset'].unique()))
            with col3:
                st.metric("Model Coverage", len(method_df['Model'].unique()))
            with col4:
                avg_time = method_df['Avg Generation Time'].mean()
                st.metric("Avg Generation Time", f"{avg_time:.3f}s")
            
            # Performance analysis
            st.markdown(f"#### 📊 {selected_method.title()} Performance Analysis")
            st.dataframe(method_df, use_container_width=True)
            
            # Visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                # Generation time by dataset
                fig = px.bar(
                    method_df,
                    x='Dataset',
                    y='Avg Generation Time',
                    color='Model',
                    title=f"Generation Time by Dataset for {selected_method}",
                    barmode='group'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Explanation count by model
                fig = px.bar(
                    method_df,
                    x='Model',
                    y='Explanations',
                    color='Dataset',
                    title=f"Explanation Count by Model for {selected_method}",
                    barmode='group'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "📈 Feature Importance Analysis":
            st.subheader("📈 Feature Importance Analysis")
            
            # Get all combinations with feature importance data
            combinations_with_features = []
            
            for dataset_dir in detailed_explanations_dir.iterdir():
                if dataset_dir.is_dir():
                    dataset_name = dataset_dir.name
                    for model_dir in dataset_dir.iterdir():
                        if model_dir.is_dir():
                            model_name = model_dir.name
                            for explanation_file in model_dir.glob("*_detailed_explanations.json"):
                                method_name = explanation_file.name.replace("_detailed_explanations.json", "")
                                
                                try:
                                    with open(explanation_file, "r", encoding='utf-8') as f:
                                        data = json.load(f)
                                        if isinstance(data, dict):
                                            explanations = data.get("explanations", [])
                                        elif isinstance(data, list):
                                            explanations = data
                                        else:
                                            explanations = []
                                        
                                        # Check if this method has feature importance data
                                        has_feature_importance = False
                                        if explanations and isinstance(explanations[0], dict):
                                            if "feature_importance" in explanations[0] and "feature_names" in explanations[0]:
                                                has_feature_importance = True
                                        
                                        if has_feature_importance:
                                            combinations_with_features.append({
                                                "display_name": f"{dataset_name}_{model_name}_{method_name}",
                                                "dataset": dataset_name,
                                                "model": model_name,
                                                "method": method_name,
                                                "file_path": explanation_file,
                                                "explanations": explanations
                                            })
                                except:
                                    continue
            
            if not combinations_with_features:
                show_warning("No combinations with feature importance data found.")
                return
            
            # Selection
            combination_names = [combo["display_name"] for combo in combinations_with_features]
            selected_combination = st.selectbox(
                "Select Dataset-Model-Method Combination:",
                combination_names,
                help="Choose combination for feature importance analysis"
            )
            
            selected_combo = next(c for c in combinations_with_features if c["display_name"] == selected_combination)
            explanations = selected_combo["explanations"]
            
            show_info(f"**Dataset:** {selected_combo['dataset']} | **Model:** {selected_combo['model']} | **Method:** {selected_combo['method']}")
            
            # Feature importance analysis
            if explanations and "feature_names" in explanations[0]:
                feature_names = explanations[0]["feature_names"]
                
                # Collect all feature importances
                all_importances = []
                for exp in explanations:
                    if "feature_importance" in exp:
                        importance = exp["feature_importance"]
                        if hasattr(importance, 'tolist'):
                            importance = importance.tolist()
                        all_importances.append(importance)
                
                if all_importances:
                    importance_array = np.array(all_importances)
                    
                    # Statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Instances", len(all_importances))
                    with col2:
                        st.metric("Features", len(feature_names))
                    with col3:
                        st.metric("Avg Importance Range", f"{np.mean(np.max(importance_array, axis=1) - np.min(importance_array, axis=1)):.3f}")
                    with col4:
                        st.metric("Most Important Feature", feature_names[np.argmax(np.mean(np.abs(importance_array), axis=0))])
                    
                    # Global feature importance
                    st.markdown("#### 🌍 Global Feature Importance")
                    
                    mean_importance = np.mean(importance_array, axis=0)
                    std_importance = np.std(importance_array, axis=0)
                    
                    feature_df = pd.DataFrame({
                        "Feature": feature_names,
                        "Mean Importance": mean_importance,
                        "Std Importance": std_importance,
                        "Abs Mean Importance": np.abs(mean_importance)
                    }).sort_values("Abs Mean Importance", ascending=False)
                    
                    # Top features bar chart
                    fig = px.bar(
                        feature_df.head(20),
                        x="Mean Importance",
                        y="Feature",
                        error_x="Std Importance",
                        orientation='h',
                        title="Top 20 Most Important Features (Global Average)",
                        color="Mean Importance",
                        color_continuous_scale="RdBu_r"
                    )
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature importance distribution
                    st.markdown("#### 📊 Feature Importance Distribution")
                    
                    top_features = feature_df.head(10)["Feature"].tolist()
                    selected_features = st.multiselect(
                        "Select features to show distribution:",
                        top_features,
                        default=top_features[:5]
                    )
                    
                    if selected_features:
                        fig = go.Figure()
                        for feature in selected_features:
                            feature_idx = feature_names.index(feature)
                            feature_values = importance_array[:, feature_idx]
                            fig.add_trace(go.Box(
                                y=feature_values,
                                name=feature,
                                boxpoints='outliers'
                            ))
                        
                        fig.update_layout(
                            title="Feature Importance Distributions",
                            yaxis_title="Importance Value",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature correlation heatmap
                    st.markdown("#### 🔗 Feature Importance Correlation")
                    
                    if len(feature_names) <= 50:  # Only for manageable number of features
                        corr_matrix = np.corrcoef(importance_array.T)
                        
                        fig = px.imshow(
                            corr_matrix,
                            x=feature_names,
                            y=feature_names,
                            color_continuous_scale="RdBu",
                            title="Feature Importance Correlation Matrix",
                            aspect="auto"
                        )
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        show_info(f"Too many features ({len(feature_names)}) for correlation matrix. Showing top 20 features only.")
                        
                        top_20_indices = np.argsort(np.mean(np.abs(importance_array), axis=0))[-20:]
                        top_20_corr = np.corrcoef(importance_array[:, top_20_indices].T)
                        top_20_names = [feature_names[i] for i in top_20_indices]
                        
                        fig = px.imshow(
                            top_20_corr,
                            x=top_20_names,
                            y=top_20_names,
                            color_continuous_scale="RdBu",
                            title="Top 20 Features Importance Correlation Matrix",
                            aspect="auto"
                        )
                        fig.update_layout(height=600)
                        st.plotly_chart(fig, use_container_width=True)
        
        elif analysis_type == "🔬 Individual Instance Analysis":
            st.subheader("🔬 Individual Instance Analysis")
            
            # Get all combinations with explanations
            combinations_with_explanations = []
            
            for dataset_dir in detailed_explanations_dir.iterdir():
                if dataset_dir.is_dir():
                    dataset_name = dataset_dir.name
                    for model_dir in dataset_dir.iterdir():
                        if model_dir.is_dir():
                            model_name = model_dir.name
                            for explanation_file in model_dir.glob("*_detailed_explanations.json"):
                                method_name = explanation_file.name.replace("_detailed_explanations.json", "")
                                
                                try:
                                    with open(explanation_file, "r", encoding='utf-8') as f:
                                        data = json.load(f)
                                        if isinstance(data, dict):
                                            explanations = data.get("explanations", [])
                                        elif isinstance(data, list):
                                            explanations = data
                                        else:
                                            explanations = []
                                        
                                        if explanations:
                                            combinations_with_explanations.append({
                                                "display_name": f"{dataset_name}_{model_name}_{method_name}",
                                                "dataset": dataset_name,
                                                "model": model_name,
                                                "method": method_name,
                                                "file_path": explanation_file,
                                                "explanations": explanations,
                                                "count": len(explanations)
                                            })
                                except:
                                    continue
            
            if not combinations_with_explanations:
                show_warning("No combinations with individual explanations found.")
                return
            
            # Combination selection
            combination_names = [f"{combo['display_name']} ({combo['count']} instances)" for combo in combinations_with_explanations]
            selected_combination_idx = st.selectbox(
                "Select Dataset-Model-Method Combination:",
                range(len(combination_names)),
                format_func=lambda x: combination_names[x],
                help="Choose combination for individual instance analysis"
            )
            
            selected_combo = combinations_with_explanations[selected_combination_idx]
            explanations = selected_combo["explanations"]
            
            # Get feature names from benchmark results for this combination
            feature_names = None
            for result in results.get('comprehensive_results', []):
                if (result.get('dataset') == selected_combo['dataset'] and 
                    result.get('model') == selected_combo['model'] and 
                    result.get('explanation_method') == selected_combo['method']):
                    feature_names = result.get('explanation_info', {}).get('feature_names', [])
                    break
            
            if not feature_names:
                # Fallback to generic names if not found
                max_features = 0
                for exp in explanations:
                    if isinstance(exp, dict) and 'feature_importance' in exp:
                        importance = exp['feature_importance']
                        if isinstance(importance, str):
                            try:
                                import ast
                                importance = ast.literal_eval(importance.replace('[', '').replace(']', '').split())
                            except:
                                try:
                                    importance = eval(importance)
                                except:
                                    importance = []
                        if hasattr(importance, '__len__'):
                            max_features = max(max_features, len(importance))
                
                feature_names = [f"Feature_{i}" for i in range(max_features)] if max_features > 0 else []
            
            show_info(f"**Dataset:** {selected_combo['dataset']} | **Model:** {selected_combo['model']} | **Method:** {selected_combo['method']} | **Instances:** {len(explanations)} | **Features:** {len(feature_names)}")
            
            # Instance selection
            if explanations:
                # Instance selection with preview
                col1, col2 = st.columns([1, 3])
                
                with col1:
                    st.markdown("#### 📋 Instance Selection")
                    
                    # Show instance list with basic info
                    instance_options = []
                    for i, exp in enumerate(explanations):
                        if isinstance(exp, dict):
                            prediction = exp.get("prediction", "N/A")
                            true_label = exp.get("true_label", "N/A")
                            confidence = exp.get("confidence", exp.get("prediction_confidence", "N/A"))
                            
                            # Format confidence
                            if isinstance(confidence, (int, float)):
                                confidence_str = f"{confidence:.3f}"
                            else:
                                confidence_str = str(confidence)
                            
                            instance_options.append(f"Instance {i}: Pred={prediction}, True={true_label}, Conf={confidence_str}")
                        else:
                            instance_options.append(f"Instance {i}")
                    
                    selected_instance_idx = st.selectbox(
                        "Select Instance:",
                        range(len(instance_options)),
                        format_func=lambda x: instance_options[x],
                        help="Choose an instance to analyze in detail"
                    )
                
                with col2:
                    st.markdown("#### 🔍 Instance Details")
                    
                    selected_explanation = explanations[selected_instance_idx]
                    
                    if isinstance(selected_explanation, dict):
                        # Create tabs for different aspects of the explanation
                        inst_tab1, inst_tab2, inst_tab3, inst_tab4 = st.tabs([
                            "📊 Overview", "🎯 Prediction", "📈 Features", "🧩 Method-Specific"
                        ])
                        
                        with inst_tab1:
                            st.markdown("##### 📋 Instance Overview")
                            
                            # Basic information
                            col_a, col_b, col_c = st.columns(3)
                            
                            with col_a:
                                prediction = selected_explanation.get("prediction", "N/A")
                                st.metric("Prediction", str(prediction))
                            
                            with col_b:
                                true_label = selected_explanation.get("true_label", "N/A")
                                st.metric("True Label", str(true_label))
                            
                            with col_c:
                                confidence = selected_explanation.get("confidence", selected_explanation.get("prediction_confidence", "N/A"))
                                if isinstance(confidence, (int, float)):
                                    st.metric("Confidence", f"{confidence:.3f}")
                                else:
                                    st.metric("Confidence", str(confidence))
                            
                            # Additional metadata
                            if "instance_id" in selected_explanation:
                                st.write(f"**Instance ID:** {selected_explanation['instance_id']}")
                            
                            if "generation_time" in selected_explanation:
                                gen_time = selected_explanation['generation_time']
                                st.write(f"**Generation Time:** {gen_time:.4f}s" if isinstance(gen_time, (int, float)) else f"**Generation Time:** {gen_time}")
                            
                            # Show original instance data if available
                            if "original_instance" in selected_explanation:
                                st.markdown("##### 📄 Original Instance Data")
                                original = selected_explanation["original_instance"]
                                
                                if isinstance(original, dict):
                                    # Show as a nice table
                                    original_df = pd.DataFrame([original]).T
                                    original_df.columns = ["Value"]
                                    st.dataframe(original_df, use_container_width=True)
                                elif isinstance(original, list):
                                    # Show as numbered list
                                    original_df = pd.DataFrame({"Feature Index": range(len(original)), "Value": original})
                                    st.dataframe(original_df, use_container_width=True)
                                else:
                                    st.write(f"**Original Instance:** {original}")
                        
                        with inst_tab2:
                            st.markdown("##### 🎯 Prediction Analysis")
                            
                            # Show prediction breakdown if available
                            if "prediction_probabilities" in selected_explanation:
                                probs = selected_explanation["prediction_probabilities"]
                                if isinstance(probs, dict):
                                    prob_df = pd.DataFrame(list(probs.items()), columns=["Class", "Probability"])
                                    prob_df = prob_df.sort_values("Probability", ascending=False)
                                    
                                    # Bar chart of probabilities
                                    fig = px.bar(
                                        prob_df,
                                        x="Class",
                                        y="Probability",
                                        title="Prediction Probabilities",
                                        color="Probability",
                                        color_continuous_scale="Viridis"
                                    )
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Show as table
                                    st.dataframe(prob_df, use_container_width=True)
                                else:
                                    st.write(f"**Prediction Probabilities:** {probs}")
                            
                            # Prediction correctness
                            prediction = selected_explanation.get("prediction", None)
                            true_label = selected_explanation.get("true_label", None)
                            
                            if prediction is not None and true_label is not None:
                                is_correct = str(prediction) == str(true_label)
                                if is_correct:
                                    st.success("✅ **Prediction is CORRECT**")
                                else:
                                    st.error("❌ **Prediction is INCORRECT**")
                            
                            # Show decision boundary or model confidence if available
                            if "decision_score" in selected_explanation:
                                decision_score = selected_explanation["decision_score"]
                                st.write(f"**Decision Score:** {decision_score}")
                        
                        with inst_tab3:
                            st.markdown("##### 📈 Feature Analysis")
                            
                            # Parse feature importance using our helper function
                            importance_raw = selected_explanation.get("feature_importance", [])
                            importance = parse_feature_importance(importance_raw)
                            
                            if importance and len(importance) > 0 and feature_names and len(feature_names) >= len(importance):
                                # Create feature importance DataFrame
                                feature_df = pd.DataFrame({
                                    "Feature": feature_names[:len(importance)],
                                    "Importance": importance,
                                    "Abs_Importance": [abs(x) for x in importance]
                                }).sort_values("Abs_Importance", ascending=False)
                                
                                # Show non-zero features only
                                non_zero_features = feature_df[feature_df["Abs_Importance"] > 0]
                                
                                if not non_zero_features.empty:
                                    # Top features chart
                                    top_features = non_zero_features.head(15)
                                    fig = px.bar(
                                        top_features,
                                        x="Importance",
                                        y="Feature",
                                        orientation='h',
                                        title=f"Feature Importances (Instance {selected_instance_idx}) - Non-Zero Only",
                                        color="Importance",
                                        color_continuous_scale="RdBu_r"
                                    )
                                    fig.update_layout(height=max(300, len(top_features) * 25))
                                    st.plotly_chart(fig, use_container_width=True)
                                    
                                    # Feature importance table
                                    st.markdown("##### 📊 Feature Importance Details")
                                    st.dataframe(feature_df, use_container_width=True)
                                    
                                    # Summary statistics
                                    col_a, col_b, col_c = st.columns(3)
                                    with col_a:
                                        st.metric("Non-zero Features", len(non_zero_features))
                                    with col_b:
                                        st.metric("Max Importance", f"{feature_df['Abs_Importance'].max():.4f}")
                                    with col_c:
                                        top_feature = feature_df.iloc[0]['Feature']
                                        st.metric("Most Important", top_feature)
                                
                                else:
                                    st.warning("All feature importances are zero for this instance.")
                                    st.dataframe(feature_df, use_container_width=True)
                            
                            # Show top_features if available (alternative format)
                            elif "top_features" in selected_explanation and selected_explanation["top_features"]:
                                top_features_data = selected_explanation["top_features"]
                                if isinstance(top_features_data, list) and len(top_features_data) > 0:
                                    st.markdown("##### 🔝 Top Features (From Explanation)")
                                    
                                    # Parse top features format
                                    top_features_list = []
                                    for feat in top_features_data:
                                        if isinstance(feat, dict):
                                            feature_idx = feat.get("feature_index", 0)
                                            importance_val = feat.get("importance", 0)
                                            feature_name = feature_names[feature_idx] if feature_names and feature_idx < len(feature_names) else f"Feature_{feature_idx}"
                                            
                                            top_features_list.append({
                                                "Feature": feature_name,
                                                "Importance": importance_val,
                                                "Abs_Importance": abs(importance_val)
                                            })
                                    
                                    if top_features_list:
                                        top_features_df = pd.DataFrame(top_features_list)
                                        
                                        # Filter non-zero features
                                        non_zero_top = top_features_df[top_features_df["Abs_Importance"] > 0]
                                        
                                        if not non_zero_top.empty:
                                            fig = px.bar(
                                                non_zero_top,
                                                x="Importance",
                                                y="Feature",
                                                orientation='h',
                                                title=f"Top Features (Instance {selected_instance_idx})",
                                                color="Importance",
                                                color_continuous_scale="RdBu_r"
                                            )
                                            fig.update_layout(height=max(300, len(non_zero_top) * 30))
                                            st.plotly_chart(fig, use_container_width=True)
                                        
                                        st.dataframe(top_features_df, use_container_width=True)
                            
                            # Show raw importance data for debugging
                            elif importance_raw:
                                st.info("Raw feature importance data detected but couldn't parse properly.")
                                st.write(f"**Raw data type:** {type(importance_raw)}")
                                st.write(f"**Raw data:** {importance_raw}")
                                
                                if feature_names:
                                    st.write(f"**Available feature names:** {feature_names}")
                            
                            else:
                                st.info("No feature importance data available for this instance.")
                                if feature_names:
                                    st.write(f"**Available features:** {len(feature_names)} features")
                                    with st.expander("📋 Feature Names"):
                                        for i, name in enumerate(feature_names):
                                            st.write(f"{i}: {name}")
                        
                        with inst_tab4:
                            st.markdown(f"##### 🧩 {selected_combo['method'].title()} Specific Analysis")
                            
                            method = selected_combo['method'].lower()
                            
                            # SHAP-specific analysis
                            if "shap" in method:
                                st.markdown("**SHAP Value Analysis**")
                                
                                shap_values = parse_feature_importance(selected_explanation.get("feature_importance", []))
                                
                                if shap_values and feature_names and len(feature_names) >= len(shap_values):
                                    # Create waterfall effect
                                    baseline = selected_explanation.get("baseline_prediction", 0)
                                    
                                    shap_df = pd.DataFrame({
                                        "Feature": feature_names[:len(shap_values)],
                                        "SHAP_Value": shap_values
                                    }).sort_values("SHAP_Value", key=abs, ascending=False).head(15)
                                        
                                        fig = px.bar(
                                            shap_df,
                                            x="SHAP_Value",
                                            y="Feature",
                                            orientation='h',
                                            title="SHAP Values (Waterfall Style)",
                                            color="SHAP_Value",
                                            color_continuous_scale="RdBu_r"
                                        )
                                        fig.add_vline(x=0, line_dash="dash", line_color="black", line_width=2)
                                        fig.update_layout(height=500)
                                        st.plotly_chart(fig, use_container_width=True)
                                        
                                        # Show cumulative effect
                                        cumulative_effect = baseline + sum(shap_values)
                                        st.write(f"**Baseline Prediction:** {baseline:.4f}")
                                        st.write(f"**Final Prediction:** {cumulative_effect:.4f}")
                                        st.write(f"**Total SHAP Effect:** {sum(shap_values):.4f}")
                            
                            # LIME-specific analysis
                            elif "lime" in method:
                                st.markdown("**LIME Local Explanation**")
                                st.info("LIME provides local linear approximations around this instance.")
                                
                                lime_explanation = parse_feature_importance(selected_explanation.get("feature_importance", []))
                                
                                if lime_explanation and feature_names and len(feature_names) >= len(lime_explanation):
                                    # LIME typically shows fewer features
                                    lime_df = pd.DataFrame({
                                        "Feature": feature_names[:len(lime_explanation)],
                                        "LIME_Weight": lime_explanation
                                    }).sort_values("LIME_Weight", key=abs, ascending=False).head(10)
                                        
                                        fig = px.bar(
                                            lime_df,
                                            x="LIME_Weight",
                                            y="Feature",
                                            orientation='h',
                                            title="LIME Feature Weights",
                                            color="LIME_Weight",
                                            color_continuous_scale="RdBu_r"
                                        )
                                        fig.add_vline(x=0, line_dash="dash", line_color="black")
                                        st.plotly_chart(fig, use_container_width=True)
                            
                            # Counterfactual-specific analysis
                            elif "counterfactual" in method:
                                st.markdown("**Counterfactual Analysis**")
                                
                                if "counterfactual" in selected_explanation:
                                    original = selected_explanation.get("original_instance", [])
                                    counterfactual = selected_explanation["counterfactual"]
                                    
                                    if len(original) == len(counterfactual):
                                        # Show changes
                                        feature_names = selected_explanation.get("feature_names", [f"Feature_{i}" for i in range(len(original))])
                                        
                                        changes_df = pd.DataFrame({
                                            "Feature": feature_names,
                                            "Original": original,
                                            "Counterfactual": counterfactual,
                                            "Changed": [orig != cf for orig, cf in zip(original, counterfactual)]
                                        })
                                        
                                        # Highlight changed features
                                        changed_features = changes_df[changes_df["Changed"]]
                                        
                                        if not changed_features.empty:
                                            st.markdown("**Features that need to change:**")
                                            st.dataframe(changed_features, use_container_width=True)
                                        else:
                                            st.info("No feature changes required for counterfactual.")
                                        
                                        # Show all features
                                        st.markdown("**All Features Comparison:**")
                                        st.dataframe(changes_df, use_container_width=True)
                            
                            # Prototype-specific analysis
                            elif "prototype" in method:
                                st.markdown("**Prototype Analysis**")
                                
                                if "prototype" in selected_explanation:
                                    prototype = selected_explanation["prototype"]
                                    original = selected_explanation.get("original_instance", [])
                                    
                                    if len(original) == len(prototype):
                                        feature_names = selected_explanation.get("feature_names", [f"Feature_{i}" for i in range(len(original))])
                                        
                                        proto_df = pd.DataFrame({
                                            "Feature": feature_names,
                                            "Original": original,
                                            "Prototype": prototype,
                                            "Difference": [abs(orig - proto) if isinstance(orig, (int, float)) and isinstance(proto, (int, float)) else "N/A" for orig, proto in zip(original, prototype)]
                                        })
                                        
                                        st.dataframe(proto_df, use_container_width=True)
                                
                                if "prototype_similarity" in selected_explanation:
                                    similarity = selected_explanation["prototype_similarity"]
                                    st.write(f"**Prototype Similarity:** {similarity:.4f}")
                            
                            # Generic method analysis
                            else:
                                st.markdown(f"**{selected_combo['method'].title()} Analysis**")
                                
                                # Show any method-specific fields
                                method_specific_fields = []
                                for key, value in selected_explanation.items():
                                    if key not in ["prediction", "true_label", "confidence", "feature_names", "feature_importance", "original_instance", "generation_time", "instance_id"]:
                                        method_specific_fields.append((key, value))
                                
                                if method_specific_fields:
                                    for key, value in method_specific_fields:
                                        st.write(f"**{key.replace('_', ' ').title()}:** {value}")
                                else:
                                    st.info(f"No specific analysis available for {selected_combo['method']} method.")
                            
                            # Raw explanation data
                            with st.expander("🔍 Raw Explanation Data"):
                                st.json(selected_explanation)
                    
                    else:
                        st.warning("Selected explanation is not in the expected format.")
                        st.json(selected_explanation)
    
    with tab1:
        st.header("📈 Experiment Overview")
        
        # Experiment info
        exp_info = results.get('experiment_info', {})
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Timestamp", exp_info.get('timestamp', 'N/A'))
        with col2:
            st.metric("Total Combinations", len(results.get('comprehensive_results', [])))
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
    

if __name__ == "__main__":
    main() 