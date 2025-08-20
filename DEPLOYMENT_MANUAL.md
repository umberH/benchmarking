# 🚀 Manual Setup Guide: Streamlit Dashboard Deployment

This guide will help you create a separate Git repository for your Streamlit dashboard and deploy it to Streamlit Community Cloud.

## 📋 Prerequisites
- Git installed and configured
- GitHub account
- Access to your experiment results

## 🎯 Step-by-Step Process

### Step 1: Create New Directory Structure

```bash
# Create new directory (outside your current project)
mkdir ../xai-dashboard-deploy
cd ../xai-dashboard-deploy

# Initialize new git repository
git init
```

### Step 2: Copy Essential Files

Copy these files from your main project:

```bash
# Main dashboard file
cp ../benchmarking/streamlit_dashboard.py .

# Dependencies  
cp ../benchmarking/requirements_streamlit_deploy.txt requirements.txt

# Streamlit configuration
mkdir .streamlit
cp ../benchmarking/.streamlit/config.toml .streamlit/
```

### Step 3: Create Results Structure

```bash
# Create results directory
mkdir results

# Copy each experiment folder
cp -r ../benchmarking/results/experiment_20250819_175132 results/
# Repeat for other experiment folders you want to include
```

**Important**: Only copy essential files to keep repository size manageable:
- `benchmark_results.json` (essential)
- `comprehensive_report.md` (optional)
- `detailed_explanations/` (for individual analysis - may be large)

### Step 4: Create Deployment Files

#### Create README.md:
```markdown
# XAI Benchmarking Dashboard

Interactive Streamlit dashboard for XAI benchmarking analysis.

## Features
- Experiment overview and comparison
- Model performance analysis
- Individual instance explanations
- Feature importance analysis
- Method-specific visualizations

## Usage
Select an experiment from the sidebar and explore different analysis tabs.

## Data
Analyzes results from comprehensive XAI benchmarking experiments.
```

#### Create .gitignore:
```gitignore
# Python
__pycache__/
*.py[cod]
.Python
env/
venv/

# Streamlit
.streamlit/secrets.toml

# OS files
.DS_Store
Thumbs.db

# Large files (optional - if you want to exclude large explanation files)
results/*/detailed_explanations/*/*.json
```

### Step 5: Initial Commit

```bash
# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: XAI Dashboard for Streamlit deployment

- Interactive dashboard with comprehensive XAI analysis
- Experiment selection and filtering capabilities  
- Individual instance explanation viewer
- Method-specific visualizations (SHAP, LIME, etc.)
- Feature importance analysis
- Ready for Streamlit Community Cloud deployment"
```

### Step 6: Create GitHub Repository

1. **Go to GitHub**: https://github.com/new
2. **Repository name**: `xai-dashboard-deploy` (or your preferred name)
3. **Visibility**: Public (required for free Streamlit Cloud)
4. **Don't initialize** with README (you already have one)
5. **Click "Create repository"**

### Step 7: Connect and Push to GitHub

```bash
# Add GitHub remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/xai-dashboard-deploy.git

# Set main branch
git branch -M main

# Push to GitHub
git push -u origin main
```

### Step 8: Deploy on Streamlit Community Cloud

1. **Go to Streamlit Cloud**: https://share.streamlit.io
2. **Sign in** with your GitHub account
3. **Click "New app"**
4. **Select your repository**: `xai-dashboard-deploy`
5. **Set configuration**:
   - **Main file path**: `streamlit_dashboard.py`
   - **Requirements file**: `requirements.txt` (automatically detected)
   - **Python version**: 3.9+ (default)
6. **Click "Deploy!"**

## 🔄 Updating Your Dashboard

When you make changes or have new experiment results:

```bash
# In your dashboard repository
cd ../xai-dashboard-deploy

# Copy updated files
cp ../benchmarking/streamlit_dashboard.py .

# Copy new experiment results
cp -r ../benchmarking/results/experiment_NEW_TIMESTAMP results/

# Commit and push
git add .
git commit -m "Update dashboard with new experiment results"
git push origin main
```

Streamlit Cloud will automatically redeploy when you push changes.

## 📊 Directory Structure

Your final dashboard repository should look like:

```
xai-dashboard-deploy/
├── streamlit_dashboard.py          # Main dashboard app
├── requirements.txt               # Dependencies
├── README.md                     # Documentation
├── .gitignore                    # Git ignore rules
├── .streamlit/
│   └── config.toml              # Streamlit config
└── results/
    ├── experiment_20250819_175132/
    │   ├── benchmark_results.json
    │   ├── comprehensive_report.md
    │   └── detailed_explanations/
    └── experiment_20250819_083558/
        ├── benchmark_results.json
        └── detailed_explanations/
```

## 🎯 Tips for Success

### Keep Repository Size Manageable:
- Include only essential experiment results
- Consider excluding very large detailed explanation files if needed
- Use `.gitignore` to control what gets uploaded

### Optimize for Streamlit Cloud:
- Test locally first: `streamlit run streamlit_dashboard.py`
- Monitor memory usage (Streamlit Cloud has limits)
- Use caching in your dashboard code (already implemented)

### Deployment Best Practices:
- Use meaningful commit messages
- Tag important versions: `git tag v1.0.0`
- Keep dashboard code and experiment code separate
- Update dashboard when you have significant new results

## 🆘 Troubleshooting

### Common Issues:

1. **Repository too large**: Remove unnecessary files, use `.gitignore`
2. **Import errors**: Check `requirements.txt` includes all dependencies
3. **File not found**: Ensure experiment files are properly copied
4. **Memory issues**: Reduce dataset size or optimize caching

### Getting Help:
- Streamlit Community: https://discuss.streamlit.io
- Streamlit Docs: https://docs.streamlit.io
- GitHub Issues: Create issues in your repository

## 🚀 You're Ready!

Once deployed, you'll have a public URL where anyone can access your XAI benchmarking dashboard. Share it with colleagues, include it in papers, or use it for presentations!

Example URL: `https://your-username-xai-dashboard-deploy-streamlit-app-hash.streamlit.app`