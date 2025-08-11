# XAI Benchmarking Dashboard

A comprehensive Streamlit dashboard for visualizing XAI benchmarking results with interactive charts and analysis tools.

## Features

### 📈 Overview Tab
- Experiment summary with key metrics
- Summary statistics for all numeric metrics
- Top performers by faithfulness

### 🎯 Model Performance Tab
- Interactive heatmap showing model performance across datasets and methods
- Bar charts comparing models by different metrics
- Filterable by dataset, model, and explanation method

### 🔍 Explanation Metrics Tab
- Radar charts comparing explanation methods across multiple metrics
- Method performance summary tables
- Multi-method comparison visualizations

### ⏱️ Performance Analysis Tab
- Time complexity vs faithfulness scatter plots
- Generation time distribution by method
- Efficiency analysis (faithfulness/time ratio)

### 📊 Comparative Analysis Tab
- Metric correlation matrix heatmap
- Multi-dimensional scatter plots
- Statistical relationships between metrics

### 📋 Raw Data Tab
- Raw data tables for metrics and performance
- CSV download functionality
- Experiment configuration display

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements_dashboard.txt
```

2. Make sure you have benchmark results in `results/benchmark_results.json`

## Usage

Run the dashboard:
```bash
streamlit run streamlit_dashboard.py
```

The dashboard will automatically load your benchmark results and display them in an interactive web interface.

## Navigation

- **Sidebar Filters**: Use the sidebar to filter results by dataset, model, or explanation method
- **Tabs**: Navigate between different analysis views using the tabs
- **Interactive Charts**: Hover over charts for detailed information, zoom, and pan
- **Download**: Export filtered data as CSV files from the Raw Data tab

## Customization

You can modify the dashboard by:
- Adding new visualization types in the respective tab sections
- Changing the color schemes in the plotly charts
- Adding new metrics or analysis methods
- Customizing the CSS styling in the main function

## Data Structure

The dashboard expects your `benchmark_results.json` to contain:
- `evaluation_results`: Dictionary with keys like "dataset_model_method" and metric values
- `explanation_results`: Dictionary with generation times and explanation info
- `experiment_info`: Metadata about the experiment

## Troubleshooting

- If no data appears, check that your `results/benchmark_results.json` file exists and is valid JSON
- If charts don't render, ensure all required dependencies are installed
- For performance issues with large datasets, consider filtering the data in the sidebar 