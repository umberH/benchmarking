# 🚀 Streamlit Cloud Deployment Guide

## Essential Files for GitHub (Commit these files)

### 📂 Core Application Files
```
streamlit_dashboard.py              # Main dashboard application
requirements_streamlit_cloud.txt    # Streamlit Cloud dependencies
.streamlit/config.toml              # Streamlit configuration
```

### 📂 Source Code (Required for dashboard to work)
```
src/
├── __init__.py
├── benchmark.py
├── evaluation/
├── explanations/
├── models/
├── utils/
└── datasets/
```

### 📂 Configuration Files
```
configs/
└── default_config.yaml
```

### 📂 Documentation
```
README.md
DASHBOARD_README.md
STREAMLIT_DEPLOYMENT_GUIDE.md
```

### 📂 Sample Data Structure (Create empty folders)
```
results/
├── benchmark_results.json         # Empty sample file
├── iterations/                     # Empty folder
└── detailed_explanations/          # Empty folder
```

## ❌ Files to EXCLUDE from GitHub (.gitignore)

Add these to your `.gitignore` file:

```gitignore
# Results and data (too large for GitHub)
results/*.json
results/iterations/*.json
results/detailed_explanations/*/*
results/*.md
data/
models/
checkpoints/

# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
.DS_Store?
._*
.Spotlight-V100
.Trashes
ehthumbs.db
Thumbs.db
```

## 🌐 Streamlit Cloud Deployment Steps

### 1. Prepare Repository
```bash
# Create .gitignore file
echo "results/*.json" >> .gitignore
echo "results/iterations/*.json" >> .gitignore
echo "results/detailed_explanations/*/*" >> .gitignore
echo "data/" >> .gitignore

# Commit essential files only
git add streamlit_dashboard.py
git add requirements_streamlit_cloud.txt
git add .streamlit/config.toml
git add src/
git add configs/
git add README.md
git add DASHBOARD_README.md
git add .gitignore

git commit -m "Add Streamlit dashboard for deployment"
git push origin main
```

### 2. Deploy on Streamlit Cloud

1. **Go to [share.streamlit.io](https://share.streamlit.io)**

2. **Connect GitHub Repository**
   - Link your GitHub account
   - Select your repository: `yourusername/benchmarking`

3. **Configure App Settings**
   - **Branch**: `main`
   - **Main file path**: `streamlit_dashboard.py`
   - **Python version**: `3.9` or `3.10`

4. **Set Requirements File**
   - The deployment will automatically use `requirements_streamlit_cloud.txt`

5. **Deploy**
   - Click "Deploy!" button
   - Wait for deployment (usually 2-5 minutes)

### 3. Advanced Configuration (Optional)

Create `streamlit_app.py` as an alias (some platforms prefer this name):
```python
# streamlit_app.py
from streamlit_dashboard import main

if __name__ == "__main__":
    main()
```

### 4. Post-Deployment

1. **Test the deployed app**
   - Verify all tabs load correctly
   - Test with sample data

2. **Share the URL**
   - Your app will be available at: `https://share.streamlit.io/yourusername/benchmarking/main/streamlit_dashboard.py`

## 📝 Sample Data for Testing

To test the dashboard without running full benchmarks, create minimal sample files:

```bash
# Create sample structure
mkdir -p results/iterations
mkdir -p results/detailed_explanations/sample_dataset/sample_model

# Create minimal sample results file
echo '{"evaluation_results": {}, "experiment_info": {"timestamp": "2024-01-01"}}' > results/benchmark_results.json
```

## 🔧 Troubleshooting

### Common Issues:

1. **ImportError**: Make sure all source files are committed
2. **FileNotFoundError**: Dashboard handles missing files gracefully
3. **Memory Issues**: Streamlit Cloud has limited resources - dashboard optimized for efficiency

### Dashboard Features Available Without Data:
- ✅ Interface loads correctly
- ✅ All tabs are accessible  
- ✅ Shows "No data available" messages
- ✅ Displays data status in guide section

### Dashboard Features Available With Data:
- ✅ All visualizations and analysis
- ✅ Individual explanation viewer
- ✅ Method-specific visualizations
- ✅ Export capabilities

## 🎯 Recommended Workflow

1. **Development**: Run benchmarks locally, use full dashboard
2. **Sharing**: Deploy dashboard to Streamlit Cloud for demos
3. **Data Loading**: Users can upload their own results files if needed

## 🔐 Security Note

Never commit sensitive data or API keys. The dashboard only uses publicly shareable visualization code.