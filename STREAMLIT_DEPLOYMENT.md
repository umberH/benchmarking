# 🚀 Streamlit Dashboard Deployment Guide

This guide explains how to deploy the XAI Benchmarking Dashboard on Streamlit Community Cloud directly from this repository.

## 📁 Repository Structure for Deployment

```
benchmarking/
├── streamlit_dashboard.py              # Main dashboard application
├── requirements_streamlit_deploy.txt   # Streamlit-specific dependencies  
├── .streamlit/config.toml              # Streamlit configuration
├── results/                            # Experiment data
│   ├── experiment_*/
│   │   ├── benchmark_results.json     # ✅ Essential for dashboard
│   │   ├── comprehensive_report.md    # ✅ Essential for dashboard
│   │   └── detailed_explanations/     # ✅ Used by visualizations
└── src/                               # Source code (not needed for dashboard)
```

## 🌐 Streamlit Cloud Deployment Steps

### 1. Push Repository to GitHub
```bash
git add .
git commit -m "Add Streamlit deployment configuration"
git push origin main
```

### 2. Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select your repository: `benchmarking`
5. Set configuration:
   - **Main file path**: `streamlit_dashboard.py`
   - **Requirements file**: `requirements_streamlit_deploy.txt`
   - **Python version**: 3.9+
6. Click "Deploy!"

### 3. Dashboard Features
The deployed dashboard includes:
- 📊 **Experiment Selection**: Choose from available experiment folders
- 🎯 **Model Performance**: Compare models across datasets
- 🔍 **Explanation Metrics**: Analyze XAI method effectiveness  
- 📈 **Interactive Visualizations**: Plotly charts and analysis
- 🧩 **Method-Specific Visualizations**: SHAP, LIME, and other method plots
- 📑 **Detailed Analysis**: Deep dive into datasets, models, and metrics

## 🔧 Local Testing

Test the dashboard locally before deployment:
```bash
# Install Streamlit dependencies
pip install -r requirements_streamlit_deploy.txt

# Run the dashboard locally  
streamlit run streamlit_dashboard.py

# Access at http://localhost:8501
```

## 📊 Data Management

### Essential Files for Dashboard
- `results/experiment_*/benchmark_results.json` - Main experimental results
- `results/experiment_*/comprehensive_report.md` - Summary reports  
- `results/experiment_*/detailed_explanations/` - Individual explanations

### Optimized for Cloud Deployment
The `.gitignore` is configured to:
- ✅ Include essential experiment results
- ❌ Exclude large iteration files  
- ✅ Keep detailed explanations for visualizations
- ❌ Exclude Python cache and temp files

## 🚀 Updating Results

When you run new experiments:
```bash
# Run comprehensive benchmarking
python main.py --comprehensive

# New experiment folder will be created automatically
# Dashboard will detect it on next deployment

# Commit and push updates
git add results/experiment_*
git commit -m "Update experiment results"
git push origin main

# Streamlit Cloud will auto-redeploy
```

## ⚙️ Configuration

### Streamlit Configuration (.streamlit/config.toml)
- Custom theme colors matching dashboard design
- Optimized server settings for cloud deployment
- Disabled usage statistics for privacy

### Requirements (requirements_streamlit_deploy.txt)
- Minimal dependencies for faster deployment
- Specific versions for stability
- Includes all visualization libraries

## 🔍 Troubleshooting

### Common Issues:
1. **Import Errors**: Check `requirements_streamlit_deploy.txt` includes all dependencies
2. **File Not Found**: Ensure experiment folders exist in `results/`
3. **Memory Issues**: Large datasets may need optimization
4. **Slow Loading**: Consider reducing detailed explanation file sizes

### Performance Tips:
- Use `@st.cache_data` decorators (already implemented)
- Limit detailed explanations to essential samples
- Enable file size limits in config.toml

## 📈 Usage Analytics

After deployment, you can:
- Monitor app usage in Streamlit Cloud dashboard
- View logs for debugging
- Update app settings and configurations
- Manage deployments and scaling

## 🎯 Next Steps

1. **Test locally**: `streamlit run streamlit_dashboard.py`
2. **Deploy to cloud**: Follow steps above
3. **Share dashboard**: Get public URL from Streamlit Cloud
4. **Update regularly**: Push new experiment results as needed