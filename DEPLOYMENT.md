# XAI Benchmarking Dashboard - Cloud Deployment Guide

## Quick Deploy to Streamlit Cloud

1. **Fork/Clone this repository** to your GitHub account

2. **Go to [Streamlit Cloud](https://share.streamlit.io/)**

3. **Deploy the app**:
   - Click "New app"
   - Select your GitHub repository
   - Set main file path: `streamlit_dashboard.py`
   - Deploy!

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run streamlit_dashboard.py
```

## Dashboard Features

- 📊 **Overview**: Summary metrics and top performers
- 🎯 **Model Performance**: Detailed model analysis and rankings
- 🔍 **Explanation Methods**: Method comparison and radar charts  
- ⏱️ **Performance Analysis**: Time complexity and efficiency
- 📊 **Comparative Analysis**: Multi-dimensional analysis with 3D plots
- 📋 **Raw Data**: Data export and individual explanation viewer

## Key Visualizations

- **SHAP Force Plots**: Interactive force plot simulations
- **3D Performance Space**: Multi-metric 3D visualization
- **Parallel Coordinates**: Multi-dimensional analysis
- **Method-specific Analysis**: SHAP, LIME, Integrated Gradients
- **Time vs Performance**: Efficiency analysis

## Data Structure

The dashboard expects result files in this structure:
```
results/
├── benchmark_results.json     # Main results file
├── runs/                     # Run-based organization
│   └── run_TIMESTAMP_TYPE/   # Individual run folders
└── detailed_explanations/    # Individual explanation data
```

## Cloud Deployment Notes

- Result files are excluded from git (see `.gitignore`)
- Upload your own result files or use the sample data generation
- The dashboard gracefully handles missing data files
- All required dependencies are in `requirements.txt`

## Environment Variables (Optional)

For production deployment, you can set:
- `STREAMLIT_SERVER_PORT` - Custom port (default: 8501)
- `STREAMLIT_SERVER_ADDRESS` - Custom address (default: 0.0.0.0)