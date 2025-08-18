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

# Custom CSS for Claude Artifacts-style layout
st.markdown("""
<style>
    /* Main container and layout */
    .main-header {
        font-size: 2.8rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    
    /* Card-based layout */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
        margin: 0.5rem 0;
        transition: transform 0.2s;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);
    }
    
    /* Status indicators */
    .success-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(102, 126, 234, 0.25);
    }
    
    .warning-box {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(245, 87, 108, 0.25);
    }
    
    .info-box {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 1.2rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(79, 172, 254, 0.25);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #f8f9fa;
        border-radius: 12px;
        padding: 4px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        transition: all 0.2s;
        box-shadow: 0 2px 4px rgba(102, 126, 234, 0.3);
    }
    
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.4);
    }
    
    /* Selectbox and input styling */
    .stSelectbox > div > div {
        border-radius: 8px;
        border: 2px solid #e9ecef;
        transition: border-color 0.2s;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Metrics styling */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        border: 1px solid #e9ecef;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    
    /* Chart containers */
    .js-plotly-plot {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.07);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Custom gradient backgrounds for different sections */
    .overview-section {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
    }
    
    .analysis-section {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
    }
    
    .performance-section {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    # Header
    st.markdown('<h1 class="main-header">📊 XAI Benchmarking Dashboard</h1>', unsafe_allow_html=True)

@st.cache_data
def load_benchmark_results(file_path: str = "results/benchmark_results.json") -> Dict[str, Any]:
    """Load benchmark results from JSON file"""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {}

@st.cache_data
def discover_available_runs() -> Dict[str, Dict[str, Any]]:
    """Discover all available benchmark runs and data sources"""
    available_runs = {}
    
    # Check new run-based structure first
    runs_dir = Path("results/runs")
    if runs_dir.exists():
        for run_dir in runs_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith('run_'):
                # Try to load run metadata
                metadata_file = run_dir / "run_metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, "r") as f:
                            metadata = json.load(f)
                        
                        # Load benchmark results from run
                        results_file = run_dir / "benchmark_results.json"
                        if results_file.exists():
                            with open(results_file, "r") as f:
                                data = json.load(f)
                            
                            run_display_name = f"{metadata['run_name']} ({metadata['run_type']})"
                            available_runs[run_display_name] = {
                                "path": str(results_file),
                                "type": metadata['run_type'],
                                "data": data,
                                "timestamp": metadata['timestamp'],
                                "combinations": len(data.get("comprehensive_results", [])),
                                "description": f"{metadata['run_type'].title()} run from {metadata['timestamp'][:8]}",
                                "run_metadata": metadata
                            }
                    except Exception as e:
                        continue
                
                # Fallback: try to load benchmark results without metadata
                else:
                    results_file = run_dir / "benchmark_results.json"
                    if results_file.exists():
                        try:
                            with open(results_file, "r") as f:
                                data = json.load(f)
                            
                            run_display_name = f"{run_dir.name} (legacy)"
                            available_runs[run_display_name] = {
                                "path": str(results_file),
                                "type": "unknown",
                                "data": data,
                                "timestamp": data.get("experiment_info", {}).get("timestamp", "Unknown"),
                                "combinations": len(data.get("comprehensive_results", [])),
                                "description": f"Legacy run in {run_dir.name}"
                            }
                        except Exception:
                            continue
    
    # Check main benchmark results (legacy location)
    main_results_path = Path("results/benchmark_results.json")
    if main_results_path.exists():
        try:
            with open(main_results_path, "r") as f:
                data = json.load(f)
                available_runs["Legacy Main Results"] = {
                    "path": str(main_results_path),
                    "type": "legacy",
                    "data": data,
                    "timestamp": data.get("experiment_info", {}).get("timestamp", "Unknown"),
                    "combinations": len(data.get("comprehensive_results", [])),
                    "description": "Legacy main benchmark results (pre-run organization)"
                }
        except Exception as e:
            pass
    
    # Check comprehensive reports (additional sources)
    results_dir = Path("results")
    if results_dir.exists():
        for results_file in results_dir.glob("*_results.json"):
            if results_file.name != "benchmark_results.json":
                try:
                    with open(results_file, "r") as f:
                        data = json.load(f)
                        run_name = results_file.stem.replace("_", " ").title()
                        available_runs[run_name] = {
                            "path": str(results_file),
                            "type": "comprehensive",
                            "data": data,
                            "timestamp": data.get("experiment_info", {}).get("timestamp", "Unknown"),
                            "combinations": len(data.get("evaluation_results", {})),
                            "description": f"Results from {run_name.lower()}"
                        }
                except Exception:
                    continue
    
    # Check iteration files for additional runs
    iteration_dir = Path("results/iterations")
    if iteration_dir.exists():
        iteration_files = sorted(iteration_dir.glob("*.json"))
        
        # Group iterations by experiment/timestamp
        grouped_iterations = {}
        for file in iteration_files:
            try:
                with open(file, "r") as f:
                    data = json.load(f)
                
                # Extract timestamp from filename or data
                filename_parts = file.stem.split("_")
                if len(filename_parts) >= 3:
                    timestamp = f"{filename_parts[1]}_{filename_parts[2]}"
                    dataset = filename_parts[3] if len(filename_parts) > 3 else "unknown"
                    model = filename_parts[4] if len(filename_parts) > 4 else "unknown"
                    method = "_".join(filename_parts[5:]) if len(filename_parts) > 5 else "unknown"
                    
                    run_key = f"Experiment {timestamp}"
                    
                    if run_key not in grouped_iterations:
                        grouped_iterations[run_key] = {
                            "iterations": [],
                            "timestamp": timestamp,
                            "datasets": set(),
                            "models": set(),
                            "methods": set()
                        }
                    
                    grouped_iterations[run_key]["iterations"].append({
                        "file": file,
                        "data": data,
                        "dataset": dataset,
                        "model": model,
                        "method": method
                    })
                    grouped_iterations[run_key]["datasets"].add(dataset)
                    grouped_iterations[run_key]["models"].add(model)
                    grouped_iterations[run_key]["methods"].add(method)
                    
            except Exception:
                continue
        
        # Add grouped iterations as run options
        for run_name, run_info in grouped_iterations.items():
            available_runs[run_name] = {
                "path": "grouped_iterations",
                "type": "iterations",
                "data": run_info,
                "timestamp": run_info["timestamp"],
                "combinations": len(run_info["iterations"]),
                "description": f"{len(run_info['iterations'])} iterations across {len(run_info['datasets'])} datasets"
            }
    
    return available_runs

@st.cache_data
def load_selected_run_data(run_info: Dict[str, Any]) -> Dict[str, Any]:
    """Load and process data for selected run"""
    if run_info["type"] == "comprehensive":
        return run_info["data"]
    elif run_info["type"] == "iterations":
        # Aggregate iteration data into comprehensive format
        combined_data = {
            "evaluation_results": {},
            "experiment_info": {
                "timestamp": run_info["timestamp"],
                "total_iterations": len(run_info["data"]["iterations"])
            }
        }
        
        for iteration in run_info["data"]["iterations"]:
            key = f"{iteration['dataset']}_{iteration['model']}_{iteration['method']}"
            combined_data["evaluation_results"][key] = iteration["data"]
        
        return combined_data
    
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
    
    # Check for comprehensive_results first (new format)
    if 'comprehensive_results' in results:
        comprehensive_results = results['comprehensive_results']
        
        for result in comprehensive_results:
            evaluation = result.get('evaluations', {})
            
            # Safely extract metrics with defaults
            row = {
                'Dataset': result.get('dataset', 'unknown'),
                'Model': result.get('model', 'unknown'), 
                'Method': result.get('explanation_method', 'unknown'),
                'Faithfulness': float(evaluation.get('faithfulness', 0.0)),
                'Stability': float(evaluation.get('stability', 0.0)),
                'Completeness': float(evaluation.get('completeness', 0.0)),
                'Sparsity': float(evaluation.get('sparsity', 0.0)),
                'Monotonicity': float(evaluation.get('monotonicity', 0.0)),
                'Consistency': float(evaluation.get('consistency', 0.0)),
                'Time_Complexity': float(evaluation.get('time_complexity', 0.0)),
                'Simplicity': float(evaluation.get('simplicity', 0.0))
            }
            metrics_data.append(row)
    
    # Fallback to old format (evaluation_results)
    elif 'evaluation_results' in results:
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
    
    # Check for comprehensive_results first (new format)
    if 'comprehensive_results' in results:
        comprehensive_results = results['comprehensive_results']
        
        for result in comprehensive_results:
            explanation_info = result.get('explanation_info', {})
            
            row = {
                'Dataset': result.get('dataset', 'unknown'),
                'Model': result.get('model', 'unknown'),
                'Method': result.get('explanation_method', 'unknown'),
                'Generation Time (s)': explanation_info.get('generation_time', 0),
                'Number of Explanations': explanation_info.get('n_explanations', 0),
                'Number of Features': len(explanation_info.get('feature_names', []))
            }
            explanation_data.append(row)
    
    # Fallback to old format
    elif 'explanation_results' in results:
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
    
    # Dashboard Introduction and Quick Actions
    with st.expander("ℹ️ Dashboard Guide & Quick Actions", expanded=False):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("### 🚀 Quick Start")
            st.markdown("""
            - **📈 Overview**: Experiment summary and key metrics
            - **🎯 Model Performance**: Model accuracy and performance analysis
            - **🔍 Explanation Metrics**: XAI method quality evaluation
            - **📊 Comparative Analysis**: Side-by-side method comparison
            """)
        
        with col2:
            st.markdown("### 🎨 Advanced Features")
            st.markdown("""
            - **🧩 Explanation Visualizations**: Interactive explanation plots
            - **📑 Detailed Reports**: Individual instance explanations
            - **🔍 Individual Explanations**: Search and filter specific instances
            - **📋 Raw Data**: Access to complete datasets
            """)
        
        with col3:
            st.markdown("### 💡 Tips")
            st.markdown("""
            - Use **sidebar filters** to focus on specific combinations
            - **Export** individual explanations for further analysis
            - **Compare methods** using the Comparative Analysis tab
            - **Drill down** to individual instances for detailed insights
            """)
        
        # Quick Status Check
        st.markdown("---")
        st.markdown("### 📊 Data Availability Status")
        
        # Check what data is available
        status_cols = st.columns(4)
        
        # Check benchmark results
        results_exists = Path("results/benchmark_results.json").exists()
        with status_cols[0]:
            status = "✅ Available" if results_exists else "❌ Missing"
            st.metric("Benchmark Results", status)
        
        # Check iterations
        iterations_dir = Path("results/iterations")
        iterations_count = len(list(iterations_dir.glob("*.json"))) if iterations_dir.exists() else 0
        with status_cols[1]:
            st.metric("Iteration Files", f"{iterations_count} files")
        
        # Check detailed explanations
        detailed_dir = Path("results/detailed_explanations")
        detailed_combinations = 0
        if detailed_dir.exists():
            for dataset_dir in detailed_dir.iterdir():
                if dataset_dir.is_dir():
                    for model_dir in dataset_dir.iterdir():
                        if model_dir.is_dir():
                            detailed_combinations += len(list(model_dir.glob("*_detailed_explanations.json")))
        
        with status_cols[2]:
            st.metric("Detailed Explanations", f"{detailed_combinations} combinations")
        
        # Check comprehensive report
        comprehensive_exists = Path("results/comprehensive_report.md").exists()
        with status_cols[3]:
            status = "✅ Available" if comprehensive_exists else "❌ Missing"
            st.metric("Comprehensive Report", status)
        
        if not any([results_exists, iterations_count > 0, detailed_combinations > 0]):
            st.warning("⚠️ No benchmark data found. Please run benchmarking first with: `python main.py --comprehensive`")
    
    # Sidebar for run selection and filters
    st.sidebar.header("🎯 Run Selection")
    
    # Discover available runs
    available_runs = discover_available_runs()
    
    if not available_runs:
        st.sidebar.error("No benchmark data found!")
        st.error("❌ No benchmark data available. Please run benchmarking first with: `python main.py --comprehensive`")
        return
    
    # Run selection dropdown
    run_names = list(available_runs.keys())
    selected_run_name = st.sidebar.selectbox(
        "📂 Select Benchmark Run:",
        run_names,
        help="Choose which benchmark run to analyze"
    )
    
    selected_run_info = available_runs[selected_run_name]
    
    # Display run information
    with st.sidebar.expander("ℹ️ Run Information"):
        st.write(f"**Timestamp:** {selected_run_info['timestamp']}")
        st.write(f"**Combinations:** {selected_run_info['combinations']}")
        st.write(f"**Type:** {selected_run_info['type'].title()}")
        st.write(f"**Description:** {selected_run_info['description']}")
    
    # Load selected run data
    results = load_selected_run_data(selected_run_info)
    
    if not results:
        st.error("Failed to load selected run data.")
        return
    
    # Multi-run comparison option
    if len(available_runs) > 1:
        st.sidebar.header("🔄 Multi-Run Comparison")
        
        enable_comparison = st.sidebar.checkbox("Enable Multi-Run Comparison")
        
        if enable_comparison:
            comparison_runs = st.sidebar.multiselect(
                "Select Additional Runs to Compare:",
                [name for name in run_names if name != selected_run_name],
                help="Select other runs to compare with the main selected run"
            )
            
            if comparison_runs:
                st.sidebar.success(f"Comparing {len(comparison_runs) + 1} runs")
                
                # Load comparison data
                comparison_data = {}
                comparison_data[selected_run_name] = results
                
                for run_name in comparison_runs:
                    comp_run_info = available_runs[run_name]
                    comparison_data[run_name] = load_selected_run_data(comp_run_info)
    
    # Sidebar filters
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
        "🧩 Explanation Visualizations",
        "📑 Detailed Reports"
    ])
    # --- New Tab: Detailed Explanation Reports ---
    with tab7:
        st.header("📑 Detailed Explanation Reports")
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
        
        # High-level experiment summary
        exp_info = results.get('experiment_info', {})
        
        # Key metrics at the top
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📅 Run Date", exp_info.get('timestamp', 'N/A')[:10] if exp_info.get('timestamp') else 'N/A')
        with col2:
            st.metric("🔗 Total Combinations", len(results.get('evaluation_results', {})))
        with col3:
            st.metric("📊 Datasets", len(set(metrics_df['Dataset']) if not metrics_df.empty else []))
        with col4:
            st.metric("🤖 Models", len(set(metrics_df['Model']) if not metrics_df.empty else []))
        with col5:
            st.metric("🔍 XAI Methods", len(set(metrics_df['Method']) if not metrics_df.empty else []))
        
        st.markdown("---")
        
        # Experiment composition breakdown
        if not metrics_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏗️ Experiment Composition")
                
                # Dataset distribution
                dataset_counts = metrics_df['Dataset'].value_counts()
                fig_datasets = px.bar(
                    x=dataset_counts.index, 
                    y=dataset_counts.values,
                    title="Distribution by Dataset",
                    color=dataset_counts.index,
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig_datasets.update_layout(xaxis_title="Dataset", yaxis_title="Count")
                st.plotly_chart(fig_datasets, use_container_width=True)
                
                # Model distribution  
                model_counts = metrics_df['Model'].value_counts()
                fig_models = px.bar(
                    x=model_counts.index, 
                    y=model_counts.values,
                    title="Distribution by Model Type",
                    color=model_counts.index,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                fig_models.update_layout(xaxis_title="Model", yaxis_title="Count")
                st.plotly_chart(fig_models, use_container_width=True)
            
            with col2:
                st.subheader("📊 Performance Overview")
                
                # Success rate analysis
                success_rates = []
                for _, row in metrics_df.iterrows():
                    # Calculate overall success based on multiple metrics
                    faithfulness = row.get('Faithfulness', 0)
                    stability = row.get('Stability', 0)
                    completeness = row.get('Completeness', 0)
                    overall_score = (faithfulness + stability + completeness) / 3
                    success_rates.append({
                        'Combination': f"{row['Dataset']}-{row['Model']}-{row['Method'][:10]}",
                        'Overall Score': overall_score,
                        'Dataset': row['Dataset'],
                        'Model': row['Model'],
                        'Method': row['Method']
                    })
                
                if success_rates:
                    success_df = pd.DataFrame(success_rates)
                    
                    # Top performing combinations
                    top_combinations = success_df.nlargest(10, 'Overall Score')
                    fig_top = px.bar(
                        top_combinations, 
                        x='Overall Score', 
                        y='Combination',
                        color='Dataset',
                        title="Top 10 Performing Combinations",
                        orientation='h'
                    )
                    fig_top.update_layout(height=400)
                    st.plotly_chart(fig_top, use_container_width=True)
                    
                    # Method performance comparison
                    method_performance = success_df.groupby('Method')['Overall Score'].agg(['mean', 'std', 'count']).reset_index()
                    method_performance.columns = ['Method', 'Average Score', 'Std Dev', 'Count']
                    method_performance = method_performance.sort_values('Average Score', ascending=False)
                    
                    fig_methods = px.bar(
                        method_performance.head(10), 
                        x='Method', 
                        y='Average Score',
                        error_y='Std Dev',
                        title="Average Performance by XAI Method",
                        color='Average Score',
                        color_continuous_scale='viridis'
                    )
                    fig_methods.update_xaxes(tickangle=45)
                    st.plotly_chart(fig_methods, use_container_width=True)
            
            st.markdown("---")
            
            # Detailed breakdown tables
            st.subheader("📋 Detailed Breakdown")
            
            tab_datasets, tab_models, tab_methods = st.tabs(["📊 By Dataset", "🤖 By Model", "🔍 By Method"])
            
            with tab_datasets:
                dataset_summary = metrics_df.groupby('Dataset').agg({
                    'Faithfulness': ['mean', 'std', 'count'],
                    'Stability': ['mean', 'std'],
                    'Completeness': ['mean', 'std'],
                    'Sparsity': ['mean', 'std']
                }).round(3)
                dataset_summary.columns = [f'{col[0]} {col[1].title()}' for col in dataset_summary.columns]
                st.dataframe(dataset_summary, use_container_width=True)
                
                # Dataset performance heatmap
                dataset_metrics = metrics_df.groupby('Dataset')[['Faithfulness', 'Stability', 'Completeness', 'Sparsity']].mean()
                fig_heatmap = px.imshow(
                    dataset_metrics.T, 
                    labels=dict(x="Dataset", y="Metric", color="Score"),
                    title="Performance Heatmap by Dataset",
                    color_continuous_scale='RdYlBu_r'
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
            
            with tab_models:
                model_summary = metrics_df.groupby('Model').agg({
                    'Faithfulness': ['mean', 'std', 'count'],
                    'Stability': ['mean', 'std'],
                    'Completeness': ['mean', 'std'],
                    'Sparsity': ['mean', 'std']
                }).round(3)
                model_summary.columns = [f'{col[0]} {col[1].title()}' for col in model_summary.columns]
                st.dataframe(model_summary, use_container_width=True)
                
                # Model comparison radar chart
                model_metrics = metrics_df.groupby('Model')[['Faithfulness', 'Stability', 'Completeness', 'Sparsity']].mean()
                fig_radar = go.Figure()
                
                for model in model_metrics.index:
                    fig_radar.add_trace(go.Scatterpolar(
                        r=model_metrics.loc[model].values.tolist() + [model_metrics.loc[model].values[0]],
                        theta=list(model_metrics.columns) + [model_metrics.columns[0]],
                        fill='toself',
                        name=model
                    ))
                
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="Model Performance Comparison (Radar Chart)",
                    height=500
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            
            with tab_methods:
                method_summary = metrics_df.groupby('Method').agg({
                    'Faithfulness': ['mean', 'std', 'count'],
                    'Stability': ['mean', 'std'],
                    'Completeness': ['mean', 'std'],
                    'Sparsity': ['mean', 'std']
                }).round(3)
                method_summary.columns = [f'{col[0]} {col[1].title()}' for col in method_summary.columns]
                st.dataframe(method_summary, use_container_width=True)
                
                # Method correlation analysis
                method_correlations = metrics_df[['Faithfulness', 'Stability', 'Completeness', 'Sparsity']].corr()
                fig_corr = px.imshow(
                    method_correlations, 
                    labels=dict(color="Correlation"),
                    title="Metric Correlation Analysis",
                    color_continuous_scale='RdBu',
                    zmin=-1, zmax=1
                )
                st.plotly_chart(fig_corr, use_container_width=True)
        
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
                top_faithful = filtered_df.nlargest(5, 'Faithfulness')[['Dataset', 'Model', 'Method', 'Faithfulness']]
                st.dataframe(top_faithful, use_container_width=True)
    
    with tab2:
        st.header("🎯 Model Performance Analysis")
        
        if not filtered_df.empty:
            # Model performance overview
            st.subheader("📊 Model Performance Overview")
            
            # Performance metrics by model
            model_performance = filtered_df.groupby('Model').agg({
                'Faithfulness': ['mean', 'std', 'min', 'max'],
                'Stability': ['mean', 'std', 'min', 'max'],
                'Completeness': ['mean', 'std', 'min', 'max'],
                'Sparsity': ['mean', 'std', 'min', 'max']
            }).round(3)
            
            # Flatten column names
            model_performance.columns = [f'{col[0]} {col[1].title()}' for col in model_performance.columns]
            
            # Display performance table
            st.dataframe(model_performance, use_container_width=True)
            
            # Performance comparison visualizations
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🏆 Model Ranking")
                
                # Calculate composite performance score
                model_scores = filtered_df.groupby('Model')[['Faithfulness', 'Stability', 'Completeness']].mean()
                model_scores['Composite Score'] = model_scores.mean(axis=1)
                model_scores = model_scores.sort_values('Composite Score', ascending=False)
                
                # Ranking chart
                fig_ranking = px.bar(
                    x=model_scores['Composite Score'],
                    y=model_scores.index,
                    orientation='h',
                    title="Model Ranking by Composite Performance",
                    labels={'x': 'Composite Score', 'y': 'Model'},
                    color=model_scores['Composite Score'],
                    color_continuous_scale='viridis'
                )
                fig_ranking.update_layout(height=400)
                st.plotly_chart(fig_ranking, use_container_width=True)
                
                # Performance distribution
                st.subheader("📈 Performance Distribution")
                metric_to_plot = st.selectbox("Select Metric:", ['Faithfulness', 'Stability', 'Completeness', 'Sparsity'])
                
                fig_dist = px.box(
                    filtered_df, 
                    x='Model', 
                    y=metric_to_plot,
                    title=f"{metric_to_plot} Distribution by Model",
                    color='Model'
                )
                fig_dist.update_xaxes(tickangle=45)
                st.plotly_chart(fig_dist, use_container_width=True)
            
            with col2:
                st.subheader("🔥 Performance Heatmap")
                
                # Model performance heatmap
                model_metrics = filtered_df.groupby('Model')[['Faithfulness', 'Stability', 'Completeness', 'Sparsity']].mean()
                
                fig_heatmap = px.imshow(
                    model_metrics.T,
                    labels=dict(x="Model", y="Metric", color="Score"),
                    title="Model Performance Heatmap",
                    color_continuous_scale='RdYlBu_r',
                    aspect='auto'
                )
                st.plotly_chart(fig_heatmap, use_container_width=True)
                
                # Radar chart comparison
                st.subheader("🎯 Multi-Metric Comparison")
                
                fig_radar = go.Figure()
                colors = px.colors.qualitative.Set1
                
                for i, model in enumerate(model_metrics.index):
                    values = model_metrics.loc[model].tolist()
                    values.append(values[0])  # Close the radar chart
                    
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=list(model_metrics.columns) + [model_metrics.columns[0]],
                        fill='toself',
                        name=model,
                        line_color=colors[i % len(colors)]
                    ))
                
                fig_radar.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    title="Model Performance Radar Chart",
                    height=400
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            
            st.markdown("---")
            
            # Detailed model analysis
            st.subheader("🔬 Detailed Model Analysis")
            
            # Model selection for detailed view
            selected_models = st.multiselect(
                "Select Models for Detailed Analysis:", 
                options=filtered_df['Model'].unique(),
                default=filtered_df['Model'].unique()[:3] if len(filtered_df['Model'].unique()) > 3 else filtered_df['Model'].unique()
            )
            
            if selected_models:
                detailed_df = filtered_df[filtered_df['Model'].isin(selected_models)]
                
                # Performance trends and patterns
                tab_trends, tab_datasets, tab_methods = st.tabs(["📈 Performance Trends", "📊 By Dataset", "🔍 By Method"])
                
                with tab_trends:
                    # Performance trend analysis
                    if 'Timestamp' in detailed_df.columns:
                        # Time-based performance trends
                        trend_data = detailed_df.groupby(['Model', 'Timestamp'])[['Faithfulness', 'Stability', 'Completeness']].mean().reset_index()
                        
                        for metric in ['Faithfulness', 'Stability', 'Completeness']:
                            fig_trend = px.line(
                                trend_data, 
                                x='Timestamp', 
                                y=metric, 
                                color='Model',
                                title=f"{metric} Trends Over Time"
                            )
                            st.plotly_chart(fig_trend, use_container_width=True)
                    else:
                        # Performance variability analysis
                        variability_data = []
                        for model in selected_models:
                            model_data = detailed_df[detailed_df['Model'] == model]
                            for metric in ['Faithfulness', 'Stability', 'Completeness', 'Sparsity']:
                                variability_data.append({
                                    'Model': model,
                                    'Metric': metric,
                                    'Mean': model_data[metric].mean(),
                                    'Std': model_data[metric].std(),
                                    'Coefficient of Variation': model_data[metric].std() / model_data[metric].mean() if model_data[metric].mean() > 0 else 0
                                })
                        
                        var_df = pd.DataFrame(variability_data)
                        
                        fig_var = px.bar(
                            var_df, 
                            x='Model', 
                            y='Coefficient of Variation', 
                            color='Metric',
                            title="Performance Variability (Coefficient of Variation)",
                            barmode='group'
                        )
                        st.plotly_chart(fig_var, use_container_width=True)
                        
                        st.dataframe(var_df.pivot(index='Model', columns='Metric', values=['Mean', 'Std', 'Coefficient of Variation']), 
                                   use_container_width=True)
                
                with tab_datasets:
                    # Performance by dataset
                    dataset_performance = detailed_df.groupby(['Model', 'Dataset'])[['Faithfulness', 'Stability', 'Completeness']].mean().reset_index()
                    
                    fig_dataset = px.box(
                        detailed_df, 
                        x='Dataset', 
                        y='Faithfulness', 
                        color='Model',
                        title="Model Performance Across Datasets"
                    )
                    st.plotly_chart(fig_dataset, use_container_width=True)
                    
                    # Dataset-specific model ranking
                    for dataset in detailed_df['Dataset'].unique():
                        st.write(f"**{dataset.title()} Dataset Performance:**")
                        dataset_subset = detailed_df[detailed_df['Dataset'] == dataset]
                        dataset_ranking = dataset_subset.groupby('Model')[['Faithfulness', 'Stability', 'Completeness']].mean()
                        dataset_ranking['Composite'] = dataset_ranking.mean(axis=1)
                        dataset_ranking = dataset_ranking.sort_values('Composite', ascending=False)
                        st.dataframe(dataset_ranking.round(3), use_container_width=True)
                
                with tab_methods:
                    # Performance by XAI method
                    method_performance = detailed_df.groupby(['Model', 'Method'])[['Faithfulness', 'Stability', 'Completeness']].mean().reset_index()
                    
                    fig_method = px.scatter(
                        method_performance, 
                        x='Faithfulness', 
                        y='Stability', 
                        color='Model',
                        size='Completeness',
                        hover_data=['Method'],
                        title="Model Performance by XAI Method (Faithfulness vs Stability)"
                    )
                    st.plotly_chart(fig_method, use_container_width=True)
                    
                    # Method compatibility analysis
                    compatibility_matrix = detailed_df.pivot_table(
                        index='Model', 
                        columns='Method', 
                        values='Faithfulness', 
                        aggfunc='mean'
                    ).fillna(0)
                    
                    fig_compat = px.imshow(
                        compatibility_matrix,
                        labels=dict(x="XAI Method", y="Model", color="Faithfulness"),
                        title="",
                        color_continuous_scale='viridis'
                    )
                    st.plotly_chart(fig_compat, use_container_width=True)
            
            # Select metric for heatmap
            heatmap_metric = st.selectbox(
                "Select Metric for Heatmap:",
                ['Faithfulness', 'Monotonicity', 'Completeness', 'Stability', 'Consistency', 'Sparsity', 'Simplicity'],
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
            
    
    with tab3:
        st.header("🔍 Explanation Method Analysis")
        
        if not filtered_df.empty:
            # Method comparison
            st.subheader("📊 Explanation Method Comparison")
            
            # Radar chart for method comparison
            method_metrics = ['Faithfulness', 'Monotonicity', 'Completeness', 'Stability', 'Consistency', 'Sparsity', 'Simplicity']
            
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
            
            # Check if required columns exist before merging
            explanation_cols = ['Dataset', 'Model', 'Method']
            if 'Generation Time (s)' in explanation_df.columns:
                explanation_cols.append('Generation Time (s)')
            if 'Number of Explanations' in explanation_df.columns:
                explanation_cols.append('Number of Explanations')
            
            # Merge metrics with explanation data
            merged_df = filtered_df.merge(
                explanation_df[explanation_cols],
                on=['Dataset', 'Model', 'Method'],
                how='left'
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Time vs Performance scatter plot
                if 'Generation Time (s)' in merged_df.columns:
                    scatter_kwargs = {
                        'x': 'Generation Time (s)',
                        'y': 'Faithfulness',
                        'color': 'Method',
                        'hover_data': ['Dataset', 'Model'],
                        'title': "Time Complexity vs Faithfulness"
                    }
                    
                    # Add size parameter only if the column exists
                    if 'Number of Explanations' in merged_df.columns:
                        scatter_kwargs['size'] = 'Number of Explanations'
                    
                    fig = px.scatter(merged_df, **scatter_kwargs)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("⚠️ Generation time data not available for current selection")
            
            with col2:
                # Time distribution by method
                if 'Generation Time (s)' in merged_df.columns:
                    # Convert to milliseconds for better readability when values are very small
                    max_time = merged_df['Generation Time (s)'].max()
                    
                    if max_time < 0.1:  # If max time is less than 100ms
                        # Convert to milliseconds
                        merged_df['Generation Time (ms)'] = merged_df['Generation Time (s)'] * 1000
                        
                        fig = px.box(
                            merged_df,
                            x='Method',
                            y='Generation Time (ms)',
                            title="Generation Time Distribution by Method (milliseconds)"
                        )
                        
                        # Set appropriate y-axis range for millisecond values
                        max_time_ms = merged_df['Generation Time (ms)'].max()
                        if max_time_ms > 0:
                            fig.update_yaxes(range=[0, max_time_ms * 1.2])
                        
                        # Show mean values as annotations
                        method_means = merged_df.groupby('Method')['Generation Time (ms)'].mean()
                        for i, (method, mean_val) in enumerate(method_means.items()):
                            fig.add_annotation(
                                x=i,
                                y=mean_val,
                                text=f"μ={mean_val:.2f}ms",
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1,
                                arrowwidth=2,
                                arrowcolor="red",
                                font=dict(size=10, color="red")
                            )
                    
                    else:
                        # Use original seconds for larger values
                        fig = px.box(
                            merged_df,
                            x='Method',
                            y='Generation Time (s)',
                            title="Generation Time Distribution by Method (seconds)"
                        )
                        
                        # Show mean values as annotations
                        method_means = merged_df.groupby('Method')['Generation Time (s)'].mean()
                        for i, (method, mean_val) in enumerate(method_means.items()):
                            fig.add_annotation(
                                x=i,
                                y=mean_val,
                                text=f"μ={mean_val:.3f}s",
                                showarrow=True,
                                arrowhead=2,
                                arrowsize=1,
                                arrowwidth=2,
                                arrowcolor="red",
                                font=dict(size=10, color="red")
                            )
                    
                    # Rotate x-axis labels for better readability
                    fig.update_xaxes(tickangle=45)
                    
                    # Add grid for better readability
                    fig.update_layout(
                        showlegend=False,
                        yaxis=dict(gridcolor='lightgray', gridwidth=1),
                        xaxis=dict(gridcolor='lightgray', gridwidth=1)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show summary statistics
                    time_stats = merged_df.groupby('Method')['Generation Time (s)'].agg(['mean', 'std', 'min', 'max']).round(4)
                    st.subheader("📊 Time Statistics by Method")
                    st.dataframe(time_stats, use_container_width=True)
                    
                else:
                    st.info("⚠️ Generation time data not available for current selection")
            
            # Performance vs Time efficiency
            st.subheader("⚡ Efficiency Analysis")
            
            if 'Generation Time (s)' in merged_df.columns:
                # Calculate efficiency score (faithfulness / time)
                merged_df['efficiency'] = merged_df['Faithfulness'] / (merged_df['Generation Time (s)'] + 1e-6)
                
                fig = px.bar(
                    merged_df.groupby('Method')['efficiency'].mean().reset_index(),
                    x='Method',
                    y='efficiency',
                    title="Efficiency Score (Faithfulness / Time) by Method"
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("⚠️ Efficiency analysis requires generation time data")
    
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
            
            # Create multiple visualizations in columns
            col1, col2 = st.columns(2)
            
            # PCA-like visualization using top metrics
            top_metrics = ['Faithfulness', 'Monotonicity', 'Completeness', 'Stability']
            
            with col1:
                if len(top_metrics) >= 2:
                    fig = px.scatter(
                        filtered_df,
                        x=top_metrics[0],
                        y=top_metrics[1],
                        color='Method',
                        size='Consistency',
                        hover_data=['Dataset', 'Model'],
                        title=f"{top_metrics[0].title()} vs {top_metrics[1].title()}"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Performance vs Robustness scatter
                st.subheader("🔄 Performance vs Robustness")
                fig_robust = px.scatter(
                    filtered_df,
                    x='Faithfulness',
                    y='Stability',
                    color='Dataset',
                    size='Completeness',
                    hover_data=['Method', 'Model'],
                    title="Performance vs Robustness Analysis"
                )
                st.plotly_chart(fig_robust, use_container_width=True)
            
            with col2:
                # 3D visualization
                st.subheader("🎯 3D Performance Space")
                fig_3d = px.scatter_3d(
                    filtered_df,
                    x='Faithfulness',
                    y='Stability', 
                    z='Completeness',
                    color='Method',
                    size='Consistency',
                    hover_data=['Dataset', 'Model'],
                    title="3D Performance Analysis"
                )
                st.plotly_chart(fig_3d, use_container_width=True)
                
                # Parallel coordinates plot
                st.subheader("🌐 Parallel Coordinates")
                parallel_metrics = ['Faithfulness', 'Stability', 'Completeness', 'Sparsity', 'Consistency']
                fig_parallel = px.parallel_coordinates(
                    filtered_df,
                    dimensions=parallel_metrics,
                    color='Faithfulness',
                    title="Multi-metric Parallel Coordinates"
                )
                st.plotly_chart(fig_parallel, use_container_width=True)
    
    # --- Tab 6: Explanation Visualizations ---
    with tab6:
        st.header("🧩 Explanation Visualizations")
        
        # Check if detailed explanations exist
        detailed_dir = Path("results/detailed_explanations")
        
        if not detailed_dir.exists():
            st.warning("No detailed explanation reports found. Run comprehensive benchmarking with `--comprehensive` flag to generate detailed reports.")
            return
        
        # Add view mode selector at the top
        view_mode = st.radio("Select View Mode:", 
                           ["📄 Summary Reports", "🔍 Individual Explanations"],
                           horizontal=True)
        
        # Get available combinations - filter based on view mode
        combinations = []
        for dataset_dir in detailed_dir.iterdir():
            if dataset_dir.is_dir():
                dataset_name = dataset_dir.name
                for model_dir in dataset_dir.iterdir():
                    if model_dir.is_dir():
                        model_name = model_dir.name
                        # For visualization, only include combinations with individual explanation files
                        if view_mode == "🔍 Individual Explanations":
                            for method_file in model_dir.glob("*_detailed_explanations.json"):
                                method_name = method_file.stem.replace("_detailed_explanations", "")
                                combinations.append((dataset_name, model_name, method_name))
                        else:
                            # For reports, use the report files
                            for method_file in model_dir.glob("*_detailed_report.md"):
                                method_name = method_file.stem.replace("_detailed_report", "")
                                combinations.append((dataset_name, model_name, method_name))
        
        if not combinations:
            st.warning("No detailed explanation reports found in the results directory.")
            return
        
        # Selection interface
        st.subheader("🔍 Select Report to View")
        
        col1, col2, col3 = st.columns(3)
        
        datasets = sorted(set(combo[0] for combo in combinations))
        with col1:
            selected_dataset = st.selectbox("Select Dataset", ["All"] + datasets, key="detailed_dataset")
        
        # Filter models based on dataset
        if selected_dataset == "All":
            filtered_combos = combinations
        else:
            filtered_combos = [combo for combo in combinations if combo[0] == selected_dataset]
        
        models = sorted(set(combo[1] for combo in filtered_combos))
        with col2:
            selected_model = st.selectbox("Select Model", ["All"] + models, key="detailed_model")
        
        # Filter methods based on dataset and model
        if selected_model == "All":
            final_combos = filtered_combos
        else:
            final_combos = [combo for combo in filtered_combos if combo[1] == selected_model]
        
        methods = sorted(set(combo[2] for combo in final_combos))
        with col3:
            selected_method = st.selectbox("Select Method", ["All"] + methods, key="detailed_method")
        
        # Apply final filter
        if selected_method != "All":
            final_combos = [combo for combo in final_combos if combo[2] == selected_method]
        
        # Display reports
        if final_combos:
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
    
    # --- New Tab: Detailed Explanation Reports ---
    with tab7:
        st.header("📑 Detailed Explanation Reports")
        
        # Check if detailed explanations exist
        detailed_dir = Path("results/detailed_explanations")
        
        if not detailed_dir.exists():
            st.warning("No detailed explanation reports found. Run comprehensive benchmarking with `--comprehensive` flag to generate detailed reports.")
            return
        
        # Add view mode selector at the top
        view_mode = st.radio("Select View Mode:", 
                           ["📄 Summary Reports", "🔍 Individual Explanations"],
                           horizontal=True)
        
        # Get available combinations - filter based on view mode
        combinations = []
        for dataset_dir in detailed_dir.iterdir():
            if dataset_dir.is_dir():
                dataset_name = dataset_dir.name
                for model_dir in dataset_dir.iterdir():
                    if model_dir.is_dir():
                        model_name = model_dir.name
                        # For visualization, only include combinations with individual explanation files
                        if view_mode == "🔍 Individual Explanations":
                            for method_file in model_dir.glob("*_detailed_explanations.json"):
                                method_name = method_file.stem.replace("_detailed_explanations", "")
                                combinations.append((dataset_name, model_name, method_name))
                        else:
                            # For reports, use the report files
                            for method_file in model_dir.glob("*_detailed_report.md"):
                                method_name = method_file.stem.replace("_detailed_report", "")
                                combinations.append((dataset_name, model_name, method_name))
        
        if not combinations:
            st.warning("No detailed explanation reports found in the results directory.")
            return
        
        # Selection interface
        st.subheader("🔍 Select Report to View")
        
        col1, col2, col3 = st.columns(3)
        
        datasets = sorted(set(combo[0] for combo in combinations))
        with col1:
            selected_dataset = st.selectbox("Select Dataset", ["All"] + datasets, key="detailed_dataset")
        
        # Filter models based on dataset
        if selected_dataset == "All":
            filtered_combos = combinations
        else:
            filtered_combos = [combo for combo in combinations if combo[0] == selected_dataset]
        
        models = sorted(set(combo[1] for combo in filtered_combos))
        with col2:
            selected_model = st.selectbox("Select Model", ["All"] + models, key="detailed_model")
        
        # Filter methods based on dataset and model
        if selected_model == "All":
            final_combos = filtered_combos
        else:
            final_combos = [combo for combo in filtered_combos if combo[1] == selected_model]
        
        methods = sorted(set(combo[2] for combo in final_combos))
        with col3:
            selected_method = st.selectbox("Select Method", ["All"] + methods, key="detailed_method")
        
        # Apply final filter
        if selected_method != "All":
            final_combos = [combo for combo in final_combos if combo[2] == selected_method]
        
        # Display reports
        if final_combos:
            # Overview table
            st.subheader("📊 Reports Overview")
            
            overview_data = []
            for dataset, model, method in final_combos:
                report_path = detailed_dir / dataset / model / f"{method}_detailed_report.md"
                if report_path.exists():
                    try:
                        with open(report_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Extract summary stats
                        total_instances = "N/A"
                        valid_explanations = "N/A"
                        errors = "N/A"
                        accuracy = "N/A"
                        
                        # Parse summary statistics
                        import re
                        total_match = re.search(r'\*\*Total Instances:\*\*\s*(\d+)', content)
                        if total_match:
                            total_instances = int(total_match.group(1))
                        
                        valid_match = re.search(r'\*\*Valid Explanations:\*\*\s*(\d+)', content)
                        if valid_match:
                            valid_explanations = int(valid_match.group(1))
                        
                        error_match = re.search(r'\*\*Errors:\*\*\s*(\d+)', content)
                        if error_match:
                            errors = int(error_match.group(1))
                        
                        acc_match = re.search(r'\*\*Model Accuracy:\*\*\s*([\d.]+)', content)
                        if acc_match:
                            accuracy = float(acc_match.group(1))
                        
                        overview_data.append({
                            'Dataset': dataset,
                            'Model': model,
                            'Method': method,
                            'Total Instances': total_instances,
                            'Valid Explanations': valid_explanations,
                            'Errors': errors,
                            'Model Accuracy': accuracy
                        })
                    except Exception as e:
                        st.error(f"Error reading {report_path}: {e}")
            
            if overview_data:
                overview_df = pd.DataFrame(overview_data)
                st.dataframe(overview_df, use_container_width=True)
                
                # Visualizations
                col1, col2 = st.columns(2)
                
                with col1:
                    # Success rate chart
                    overview_df['Success Rate'] = (overview_df['Valid Explanations'].astype(float) / 
                                                 overview_df['Total Instances'].astype(float) * 100).round(2)
                    fig = px.bar(overview_df, x='Method', y='Success Rate', color='Dataset',
                               title="Explanation Success Rate by Method")
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Model accuracy vs explanation success
                    fig = px.scatter(overview_df, x='Model Accuracy', y='Success Rate', 
                                   color='Method', size='Valid Explanations',
                                   hover_data=['Dataset', 'Model'],
                                   title="Model Accuracy vs Explanation Success")
                    st.plotly_chart(fig, use_container_width=True)
            
            # Individual explanation viewer - Fix 3
            st.subheader("🔍 Individual Explanation Viewer")
            
            if view_mode == "🔍 Individual Explanations":
                # Individual explanation browser with enhanced visualization
                if len(final_combos) > 0:
                    # Create formatted labels in the style you requested
                    combo_labels = []
                    combo_keys = []
                    for combo in final_combos:
                        dataset, model, method = combo
                        # Format: adult_income_decision_tree_bayesian_rule_list
                        formatted_key = f"{dataset}_{model}_{method}"
                        combo_keys.append(formatted_key)
                        # Display label: Adult Income → Decision Tree → Bayesian Rule List
                        display_label = f"{dataset.replace('_', ' ').title()} → {model.replace('_', ' ').title()} → {method.replace('_', ' ').title()}"
                        combo_labels.append(display_label)
                    
                    selected_idx = st.selectbox("Select Explanation Method Combination:", 
                                              range(len(combo_labels)), 
                                              format_func=lambda x: f"📊 {combo_labels[x]}",
                                              key="individual_combo")
                    
                    if selected_idx is not None:
                        selected_combo_key = combo_keys[selected_idx]
                        st.info(f"🔍 **Selected:** `{selected_combo_key}`")
                    
                    if selected_idx is not None:
                        dataset, model, method = final_combos[selected_idx]
                        json_path = detailed_dir / dataset / model / f"{method}_detailed_explanations.json"
                        
                        if json_path.exists():
                            try:
                                with open(json_path, 'r', encoding='utf-8') as f:
                                    explanations_data = json.load(f)
                                
                                # Enhanced analysis and visualization
                                st.success(f"📊 **Loaded {len(explanations_data)} individual explanations** for `{selected_combo_key}`")
                                
                                # Create overview statistics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                # Calculate statistics
                                total_instances = len(explanations_data)
                                correct_predictions = sum(1 for exp in explanations_data if exp.get('correct_prediction', False))
                                has_importance = sum(1 for exp in explanations_data if exp.get('feature_importance') and len(exp['feature_importance']) > 0)
                                avg_confidence = np.mean([exp.get('explanation_confidence', 0) for exp in explanations_data])
                                
                                with col1:
                                    st.metric("📋 Total Instances", total_instances)
                                with col2:
                                    accuracy = (correct_predictions / total_instances) * 100 if total_instances > 0 else 0
                                    st.metric("🎯 Model Accuracy", f"{accuracy:.1f}%")
                                with col3:
                                    feature_coverage = (has_importance / total_instances) * 100 if total_instances > 0 else 0
                                    st.metric("📊 Feature Coverage", f"{feature_coverage:.1f}%")
                                with col4:
                                    st.metric("⭐ Avg Confidence", f"{avg_confidence:.3f}")
                                
                                # Visualization tabs for different analysis
                                viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs(["📈 Overview", "🎯 Predictions", "📊 Features", "🔍 Individual"])
                                
                                with viz_tab1:
                                    # Overview Analysis
                                    st.subheader("📈 Explanation Method Overview")
                                    
                                    # Create overview charts
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        # Prediction accuracy distribution
                                        accuracy_data = []
                                        for exp in explanations_data:
                                            accuracy_data.append({
                                                'Instance': exp.get('instance_id', 0),
                                                'Correct': 'Correct' if exp.get('correct_prediction', False) else 'Incorrect',
                                                'Confidence': exp.get('explanation_confidence', 0)
                                            })
                                        
                                        if accuracy_data:
                                            acc_df = pd.DataFrame(accuracy_data)
                                            fig_acc = px.histogram(acc_df, x='Correct', color='Correct',
                                                                 title=f"Prediction Accuracy Distribution ({selected_combo_key})")
                                            st.plotly_chart(fig_acc, use_container_width=True)
                                    
                                    with col2:
                                        # Confidence distribution
                                        if accuracy_data:
                                            fig_conf = px.histogram(acc_df, x='Confidence', color='Correct',
                                                                  title="Explanation Confidence Distribution",
                                                                  nbins=20)
                                            st.plotly_chart(fig_conf, use_container_width=True)
                                    
                                    # Method-specific insights
                                    method_lower = method.lower()
                                    if any(rule_method in method_lower for rule_method in ['bayesian_rule_list', 'corels']):
                                        st.info("📋 **Rule-based Method**: This method generates logical rules for decision making.")
                                    elif 'shap' in method_lower:
                                        st.info("🎯 **SHAP Method**: Shows feature contributions to individual predictions.")
                                    elif 'lime' in method_lower:
                                        st.info("🔍 **LIME Method**: Local interpretable model-agnostic explanations.")
                                
                                with viz_tab2:
                                    # Prediction Analysis
                                    st.subheader("🎯 Prediction Analysis")
                                    
                                    # Prediction vs True Label Analysis
                                    pred_data = []
                                    for exp in explanations_data:
                                        pred_data.append({
                                            'Instance_ID': exp.get('instance_id', 0),
                                            'True_Label': exp.get('true_label', 0),
                                            'Prediction': exp.get('prediction', 0),
                                            'Correct': exp.get('correct_prediction', False),
                                            'Prob_Class_0': exp.get('prediction_proba', [0, 0])[0] if len(exp.get('prediction_proba', [])) > 0 else 0,
                                            'Prob_Class_1': exp.get('prediction_proba', [0, 0])[1] if len(exp.get('prediction_proba', [])) > 1 else 0
                                        })
                                    
                                    if pred_data:
                                        pred_df = pd.DataFrame(pred_data)
                                        
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            # Confusion matrix style
                                            fig_conf_matrix = px.scatter(pred_df, x='True_Label', y='Prediction', 
                                                                        color='Correct', size='Prob_Class_1',
                                                                        title="Predictions vs True Labels")
                                            st.plotly_chart(fig_conf_matrix, use_container_width=True)
                                        
                                        with col2:
                                            # Probability distribution
                                            fig_prob = px.scatter(pred_df, x='Prob_Class_0', y='Prob_Class_1',
                                                                color='Correct', hover_data=['Instance_ID'],
                                                                title="Prediction Probability Space")
                                            st.plotly_chart(fig_prob, use_container_width=True)
                                        
                                        # Summary statistics
                                        st.subheader("📊 Prediction Statistics")
                                        summary_col1, summary_col2, summary_col3 = st.columns(3)
                                        
                                        with summary_col1:
                                            st.metric("True Positives", len(pred_df[(pred_df['True_Label'] == 1) & (pred_df['Prediction'] == 1)]))
                                        with summary_col2:
                                            st.metric("False Positives", len(pred_df[(pred_df['True_Label'] == 0) & (pred_df['Prediction'] == 1)]))
                                        with summary_col3:
                                            st.metric("False Negatives", len(pred_df[(pred_df['True_Label'] == 1) & (pred_df['Prediction'] == 0)]))
                                
                                with viz_tab3:
                                    # Feature Analysis
                                    st.subheader("📊 Feature Importance Analysis")
                                    
                                    # Aggregate feature importance across all instances
                                    all_importance = []
                                    feature_stats = {}
                                    
                                    for exp in explanations_data:
                                        importance = exp.get('feature_importance', [])
                                        if importance:
                                            for i, imp_val in enumerate(importance):
                                                feature_name = f'Feature_{i}'
                                                if feature_name not in feature_stats:
                                                    feature_stats[feature_name] = []
                                                feature_stats[feature_name].append(imp_val)
                                                all_importance.append({
                                                    'Instance': exp.get('instance_id', 0),
                                                    'Feature': feature_name,
                                                    'Importance': imp_val,
                                                    'Correct': exp.get('correct_prediction', False)
                                                })
                                    
                                    if all_importance:
                                        imp_df = pd.DataFrame(all_importance)
                                        
                                        col1, col2 = st.columns(2)
                                        
                                        with col1:
                                            # Feature importance distribution
                                            avg_importance = imp_df.groupby('Feature')['Importance'].mean().sort_values(ascending=False)
                                            fig_feat = px.bar(x=avg_importance.index[:10], y=avg_importance.values[:10],
                                                            title="Top 10 Features by Average Importance")
                                            st.plotly_chart(fig_feat, use_container_width=True)
                                        
                                        with col2:
                                            # Feature importance by prediction correctness
                                            fig_feat_correct = px.box(imp_df, x='Feature', y='Importance', color='Correct',
                                                                    title="Feature Importance by Prediction Accuracy")
                                            fig_feat_correct.update_xaxes(tickangle=45)
                                            st.plotly_chart(fig_feat_correct, use_container_width=True)
                                        
                                        # Feature statistics table
                                        st.subheader("📋 Feature Statistics")
                                        if feature_stats:
                                            stats_data = []
                                            for feature, values in feature_stats.items():
                                                stats_data.append({
                                                    'Feature': feature,
                                                    'Mean': np.mean(values),
                                                    'Std': np.std(values),
                                                    'Min': np.min(values),
                                                    'Max': np.max(values),
                                                    'Count': len(values)
                                                })
                                            stats_df = pd.DataFrame(stats_data)
                                            st.dataframe(stats_df.round(4), use_container_width=True)
                                    else:
                                        st.warning("⚠️ No feature importance data available for this explanation method.")
                                        
                                        # Show method-specific information
                                        method_lower = method.lower()
                                        if any(rule_method in method_lower for rule_method in ['bayesian_rule_list', 'corels']):
                                            st.info("📋 This is a rule-based method. It generates decision rules rather than feature importance scores.")
                                        elif any(concept_method in method_lower for concept_method in ['tcav', 'concept']):
                                            st.info("🔗 This is a concept-based method. It works with high-level concepts rather than individual features.")
                                
                                with viz_tab4:
                                    # Individual Explanation Viewer
                                    st.subheader("🔍 Individual Explanation Details")
                                    
                                    # Simple instance selector
                                    instance_ids = [exp.get('instance_id', i) for i, exp in enumerate(explanations_data)]
                                    selected_instance_id = st.selectbox("Select Instance:", instance_ids, 
                                                                      format_func=lambda x: f"Instance {x}")
                                    
                                    # Find selected explanation
                                    selected_explanation = None
                                    for exp in explanations_data:
                                        if exp.get('instance_id') == selected_instance_id:
                                            selected_explanation = exp
                                            break
                                    
                                    if selected_explanation:
                                        # Display instance details
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Instance ID", selected_explanation.get('instance_id', 'N/A'))
                                        with col2:
                                            st.metric("True Label", selected_explanation.get('true_label', 'N/A'))
                                        with col3:
                                            pred_val = selected_explanation.get('prediction', 'N/A')
                                            is_correct = selected_explanation.get('correct_prediction', False)
                                            status = "✅" if is_correct else "❌"
                                            st.metric("Prediction", f"{pred_val} {status}")
                                        with col4:
                                            conf_val = selected_explanation.get('explanation_confidence', 0)
                                            st.metric("Confidence", f"{conf_val:.3f}")
                                        
                                        # Prediction probabilities
                                        proba = selected_explanation.get('prediction_proba', [])
                                        if proba and len(proba) >= 2:
                                            st.subheader("📊 Prediction Probabilities")
                                            prob_col1, prob_col2 = st.columns(2)
                                            with prob_col1:
                                                st.metric("Class 0 Probability", f"{proba[0]:.3f}")
                                            with prob_col2:
                                                st.metric("Class 1 Probability", f"{proba[1]:.3f}")
                                            
                                            # Probability bar chart
                                            prob_df = pd.DataFrame({
                                                'Class': ['Class 0', 'Class 1'],
                                                'Probability': proba[:2]
                                            })
                                            fig_prob_bar = px.bar(prob_df, x='Class', y='Probability', 
                                                                title=f"Prediction Probabilities - Instance {selected_instance_id}")
                                            st.plotly_chart(fig_prob_bar, use_container_width=True)
                                        
                                        # Feature importance for this instance
                                        importance = selected_explanation.get('feature_importance', [])
                                        if importance and len(importance) > 0:
                                            st.subheader("🎯 Feature Importance")
                                            
                                            # Create feature importance chart
                                            feat_df = pd.DataFrame({
                                                'Feature': [f'Feature_{i}' for i in range(len(importance))],
                                                'Importance': importance
                                            })
                                            feat_df = feat_df.sort_values('Importance', key=abs, ascending=False)
                                            
                                            fig_feat_instance = px.bar(feat_df.head(10), x='Feature', y='Importance',
                                                                     title=f"Top 10 Feature Importance - Instance {selected_instance_id}",
                                                                     color='Importance',
                                                                     color_continuous_scale='RdBu')
                                            fig_feat_instance.update_xaxes(tickangle=45)
                                            st.plotly_chart(fig_feat_instance, use_container_width=True)
                                        
                                        # Show explanation metadata if available
                                        metadata = selected_explanation.get('explanation_metadata', {})
                                        if metadata:
                                            st.subheader("📖 Explanation Metadata")
                                            st.json(metadata)
                                        
                                        # Download individual explanation
                                        explanation_json = json.dumps(selected_explanation, indent=2)
                                        st.download_button(
                                            label=f"📥 Download Instance {selected_instance_id} Explanation",
                                            data=explanation_json,
                                            file_name=f"{selected_combo_key}_instance_{selected_instance_id}.json",
                                            mime="application/json"
                                        )
                                            
                            except Exception as e:
                                st.error(f"Error loading explanations: {e}")
                        else:
                            st.warning(f"Individual explanations file not found: {json_path}")
                else:
                    st.info("Please select a dataset, model, and method combination to view individual explanations.")
            
            else:  # Summary Reports mode
                # Detailed report viewer
                st.subheader("📄 Detailed Report Viewer")
            
            if len(final_combos) == 1:
                # Single report - show directly
                dataset, model, method = final_combos[0]
                report_path = detailed_dir / dataset / model / f"{method}_detailed_report.md"
                
                if report_path.exists():
                    with open(report_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    st.markdown("---")
                    st.markdown(content)
                    
                    # Download button
                    st.download_button(
                        label=f"📥 Download {method} Report",
                        data=content,
                        file_name=f"{dataset}_{model}_{method}_report.md",
                        mime="text/markdown"
                    )
            
            elif len(final_combos) > 1:
                # Multiple reports - show selector
                combo_labels = [f"{combo[0]} → {combo[1]} → {combo[2]}" for combo in final_combos]
                selected_combo_idx = st.selectbox("Select specific report to view:", 
                                                range(len(combo_labels)), 
                                                format_func=lambda x: combo_labels[x])
                
                if selected_combo_idx is not None:
                    dataset, model, method = final_combos[selected_combo_idx]
                    report_path = detailed_dir / dataset / model / f"{method}_detailed_report.md"
                    
                    if report_path.exists():
                        with open(report_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        st.markdown("---")
                        st.markdown(content)
                        
                        # Download button
                        st.download_button(
                            label=f"📥 Download {method} Report",
                            data=content,
                            file_name=f"{dataset}_{model}_{method}_report.md",
                            mime="text/markdown"
                        )
        else:
            st.info("No reports match the selected criteria.")

if __name__ == "__main__":
    main()

