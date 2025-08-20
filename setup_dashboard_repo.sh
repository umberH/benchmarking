#!/bin/bash

# Script to set up separate git repository for Streamlit dashboard deployment

echo "🚀 Setting up separate Streamlit dashboard repository..."

# Step 1: Create new directory for dashboard repo
DASHBOARD_DIR="../xai-dashboard-deploy"
mkdir -p "$DASHBOARD_DIR"
cd "$DASHBOARD_DIR"

echo "📁 Created dashboard directory: $DASHBOARD_DIR"

# Step 2: Initialize new git repository
git init
echo "✅ Initialized new git repository"

# Step 3: Copy essential files for dashboard
echo "📋 Copying essential files..."

# Copy main dashboard file
cp ../benchmarking/streamlit_dashboard.py .

# Copy streamlit-specific requirements
cp ../benchmarking/requirements_streamlit_deploy.txt requirements.txt

# Copy streamlit configuration
mkdir -p .streamlit
cp ../benchmarking/.streamlit/config.toml .streamlit/

echo "✅ Copied dashboard files"

# Step 4: Copy experiment results (structure only)
echo "📊 Setting up results structure..."

mkdir -p results

# Copy experiment folders with essential files only
for exp_dir in ../benchmarking/results/experiment_*/; do
    if [ -d "$exp_dir" ]; then
        exp_name=$(basename "$exp_dir")
        echo "Copying experiment: $exp_name"
        
        mkdir -p "results/$exp_name"
        
        # Copy essential JSON file
        if [ -f "$exp_dir/benchmark_results.json" ]; then
            cp "$exp_dir/benchmark_results.json" "results/$exp_name/"
        fi
        
        # Copy markdown report
        if [ -f "$exp_dir/comprehensive_report.md" ]; then
            cp "$exp_dir/comprehensive_report.md" "results/$exp_name/"
        fi
        
        # Copy detailed explanations structure (with size limit)
        if [ -d "$exp_dir/detailed_explanations" ]; then
            echo "Copying detailed explanations for $exp_name..."
            mkdir -p "results/$exp_name/detailed_explanations"
            
            # Copy detailed explanations with size filtering
            find "$exp_dir/detailed_explanations" -name "*_detailed_explanations.json" -size -1M -exec cp --parents {} "results/$exp_name/" \;
        fi
    fi
done

echo "✅ Results structure created"

# Step 5: Create deployment-specific files
echo "📝 Creating deployment files..."

# Create README for deployment repo
cat > README.md << 'EOF'
# XAI Benchmarking Dashboard

Interactive Streamlit dashboard for analyzing XAI (Explainable AI) benchmarking results.

## 🚀 Live Dashboard
This dashboard is deployed on Streamlit Community Cloud.

## ✨ Features
- **Experiment Overview**: Summary statistics and metrics
- **Model Performance**: Compare models across datasets  
- **Explanation Methods**: Analyze different XAI techniques
- **Performance Analysis**: Time complexity and efficiency metrics
- **Individual Instance Analysis**: Detailed explanation exploration
- **Feature Importance**: Deep dive into feature contributions

## 📊 Data Structure
The dashboard analyzes experiment data from:
- `results/experiment_*/benchmark_results.json` - Main experimental results
- `results/experiment_*/detailed_explanations/` - Individual explanations
- `results/experiment_*/comprehensive_report.md` - Summary reports

## 🎯 Usage
1. Select an experiment from the sidebar
2. Use filters to focus on specific datasets/models/methods
3. Explore different analysis tabs
4. Drill down into individual instances for detailed explanations

## 🔧 Local Development
```bash
pip install -r requirements.txt
streamlit run streamlit_dashboard.py
```

## 📈 Supported Analysis
- SHAP visualizations (waterfall plots, summary plots)
- LIME local explanations  
- Feature attribution analysis
- Counterfactual explanations
- Prototype analysis
- Performance benchmarking
EOF

# Create .gitignore for deployment
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/

# Streamlit
.streamlit/secrets.toml

# Large files - keep structure but exclude very large explanation files
results/*/detailed_explanations/*/*/*.json
!results/*/detailed_explanations/*/*_detailed_explanations.json

# OS files
.DS_Store
Thumbs.db
*.tmp
*.log

# IDE
.vscode/
.idea/
EOF

echo "✅ Created deployment files"

# Step 6: Create initial commit
echo "📦 Creating initial commit..."

git add .
git commit -m "Initial commit: XAI Dashboard for Streamlit deployment

- Streamlit dashboard with comprehensive XAI analysis
- Experiment selection and filtering
- Individual instance explanation analysis  
- Method-specific visualizations (SHAP, LIME, etc.)
- Feature importance analysis
- Performance benchmarking

Ready for Streamlit Community Cloud deployment"

echo "✅ Initial commit created"

# Step 7: Display next steps
echo ""
echo "🎉 Dashboard repository setup complete!"
echo ""
echo "📍 Current location: $(pwd)"
echo ""
echo "🔗 Next steps to deploy:"
echo "1. Create a new repository on GitHub:"
echo "   - Go to https://github.com/new"
echo "   - Name: 'xai-dashboard-deploy' (or your preferred name)"
echo "   - Make it public for Streamlit Cloud"
echo ""
echo "2. Connect and push to GitHub:"
echo "   git remote add origin https://github.com/YOURUSERNAME/xai-dashboard-deploy.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. Deploy on Streamlit Cloud:"
echo "   - Go to https://share.streamlit.io"
echo "   - Connect your GitHub account"
echo "   - Select your 'xai-dashboard-deploy' repository"
echo "   - Set main file: streamlit_dashboard.py"
echo "   - Deploy!"
echo ""
echo "📊 Repository contents:"
ls -la

echo ""
echo "📈 Results included:"
find results -name "*.json" | wc -l | xargs echo "JSON files:"
find results -name "*.md" | wc -l | xargs echo "Markdown files:"

echo ""
echo "🚀 Ready for deployment!"
EOF