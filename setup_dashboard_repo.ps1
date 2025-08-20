# PowerShell script to set up Streamlit dashboard repository

Write-Host "🚀 Setting up separate Streamlit dashboard repository..." -ForegroundColor Green

# Step 1: Create new directory for dashboard repo
$DashboardDir = "..\xai-dashboard-deploy"
New-Item -ItemType Directory -Path $DashboardDir -Force | Out-Null
Set-Location $DashboardDir

Write-Host "📁 Created dashboard directory: $DashboardDir" -ForegroundColor Yellow

# Step 2: Initialize new git repository  
git init
Write-Host "✅ Initialized new git repository" -ForegroundColor Green

# Step 3: Copy essential files
Write-Host "📋 Copying essential files..." -ForegroundColor Yellow

Copy-Item "..\benchmarking\streamlit_dashboard.py" .
Copy-Item "..\benchmarking\requirements_streamlit_deploy.txt" "requirements.txt"

New-Item -ItemType Directory -Path ".streamlit" -Force | Out-Null
Copy-Item "..\benchmarking\.streamlit\config.toml" ".streamlit\"

Write-Host "✅ Copied dashboard files" -ForegroundColor Green

# Step 4: Set up results structure
Write-Host "📊 Setting up results structure..." -ForegroundColor Yellow
New-Item -ItemType Directory -Path "results" -Force | Out-Null

# Copy experiment results
$ExperimentDirs = Get-ChildItem "..\benchmarking\results\experiment_*" -Directory
foreach ($ExpDir in $ExperimentDirs) {
    $ExpName = $ExpDir.Name
    Write-Host "Copying experiment: $ExpName" -ForegroundColor Cyan
    
    $TargetDir = "results\$ExpName"
    New-Item -ItemType Directory -Path $TargetDir -Force | Out-Null
    
    # Copy essential files
    $BenchmarkFile = "$($ExpDir.FullName)\benchmark_results.json"
    if (Test-Path $BenchmarkFile) {
        Copy-Item $BenchmarkFile "$TargetDir\"
    }
    
    $ReportFile = "$($ExpDir.FullName)\comprehensive_report.md"  
    if (Test-Path $ReportFile) {
        Copy-Item $ReportFile "$TargetDir\"
    }
    
    # Copy detailed explanations (if exists and not too large)
    $DetailedDir = "$($ExpDir.FullName)\detailed_explanations"
    if (Test-Path $DetailedDir) {
        Write-Host "Copying detailed explanations for $ExpName..." -ForegroundColor Gray
        Copy-Item $DetailedDir "$TargetDir\" -Recurse -Force
    }
}

Write-Host "✅ Results structure created" -ForegroundColor Green

# Step 5: Create deployment files
Write-Host "📝 Creating deployment files..." -ForegroundColor Yellow

# Create README.md
$ReadmeContent = @"
# XAI Benchmarking Dashboard

Interactive Streamlit dashboard for analyzing XAI (Explainable AI) benchmarking results.

## 🚀 Features
- **Experiment Overview**: Summary statistics and metrics
- **Model Performance**: Compare models across datasets  
- **Explanation Methods**: Analyze different XAI techniques
- **Performance Analysis**: Time complexity and efficiency metrics
- **Individual Instance Analysis**: Detailed explanation exploration
- **Feature Importance**: Deep dive into feature contributions

## 📊 Data Structure
The dashboard analyzes experiment data from:
- ``results/experiment_*/benchmark_results.json`` - Main experimental results
- ``results/experiment_*/detailed_explanations/`` - Individual explanations
- ``results/experiment_*/comprehensive_report.md`` - Summary reports

## 🎯 Usage
1. Select an experiment from the sidebar
2. Use filters to focus on specific datasets/models/methods
3. Explore different analysis tabs
4. Drill down into individual instances for detailed explanations

## 🔧 Local Development
``````bash
pip install -r requirements.txt
streamlit run streamlit_dashboard.py
``````

## 📈 Supported Analysis
- SHAP visualizations (waterfall plots, summary plots)
- LIME local explanations  
- Feature attribution analysis
- Counterfactual explanations
- Prototype analysis
- Performance benchmarking
"@

Set-Content -Path "README.md" -Value $ReadmeContent

# Create .gitignore
$GitignoreContent = @"
# Python
__pycache__/
*.py[cod]
*`$py.class
*.so
.Python
env/
venv/
ENV/

# Streamlit
.streamlit/secrets.toml

# OS files
.DS_Store
Thumbs.db
*.tmp
*.log

# IDE
.vscode/
.idea/
"@

Set-Content -Path ".gitignore" -Value $GitignoreContent

Write-Host "✅ Created deployment files" -ForegroundColor Green

# Step 6: Create initial commit
Write-Host "📦 Creating initial commit..." -ForegroundColor Yellow

git add .
git commit -m "Initial commit: XAI Dashboard for Streamlit deployment - Ready for cloud deployment"

Write-Host "✅ Initial commit created" -ForegroundColor Green

# Step 7: Display next steps
Write-Host ""
Write-Host "🎉 Dashboard repository setup complete!" -ForegroundColor Green
Write-Host ""
Write-Host "📍 Current location: $(Get-Location)" -ForegroundColor Yellow
Write-Host ""
Write-Host "🔗 Next steps to deploy:" -ForegroundColor Cyan
Write-Host "1. Create a new repository on GitHub:" -ForegroundColor White
Write-Host "   - Go to https://github.com/new" -ForegroundColor Gray
Write-Host "   - Name: 'xai-dashboard-deploy' (or your preferred name)" -ForegroundColor Gray
Write-Host "   - Make it public for Streamlit Cloud" -ForegroundColor Gray
Write-Host ""
Write-Host "2. Connect and push to GitHub:" -ForegroundColor White
Write-Host "   git remote add origin https://github.com/YOURUSERNAME/xai-dashboard-deploy.git" -ForegroundColor Gray
Write-Host "   git branch -M main" -ForegroundColor Gray
Write-Host "   git push -u origin main" -ForegroundColor Gray
Write-Host ""
Write-Host "3. Deploy on Streamlit Cloud:" -ForegroundColor White
Write-Host "   - Go to https://share.streamlit.io" -ForegroundColor Gray
Write-Host "   - Connect your GitHub account" -ForegroundColor Gray
Write-Host "   - Select your 'xai-dashboard-deploy' repository" -ForegroundColor Gray
Write-Host "   - Set main file: streamlit_dashboard.py" -ForegroundColor Gray
Write-Host "   - Deploy!" -ForegroundColor Gray
Write-Host ""
Write-Host "📊 Repository contents:" -ForegroundColor Yellow
Get-ChildItem | Format-Table Name, Length, LastWriteTime

$JsonCount = (Get-ChildItem -Recurse -Filter "*.json" | Measure-Object).Count
$MdCount = (Get-ChildItem -Recurse -Filter "*.md" | Measure-Object).Count

Write-Host ""
Write-Host "📈 Results included:" -ForegroundColor Yellow
Write-Host "JSON files: $JsonCount" -ForegroundColor Gray
Write-Host "Markdown files: $MdCount" -ForegroundColor Gray

Write-Host ""
Write-Host "🚀 Ready for deployment!" -ForegroundColor Green
Write-Host ""
Write-Host "📖 See DEPLOYMENT_MANUAL.md for detailed instructions" -ForegroundColor Cyan

# Keep window open
Read-Host "Press Enter to continue..."