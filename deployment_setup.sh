#!/bin/bash

# Script to set up dashboard deployment within the same repository

echo "Setting up Streamlit deployment in current repository..."

# Create deployment-specific requirements
cat > requirements_streamlit_deploy.txt << 'EOF'
streamlit>=1.28.0
pandas>=1.5.0
plotly>=5.15.0
numpy>=1.24.0
altair>=5.0.0
EOF

# Create Streamlit config if it doesn't exist
mkdir -p .streamlit
cat > .streamlit/config.toml << 'EOF'
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[server]
headless = true
port = 8501
enableCORS = false
enableXsrfProtection = false
maxUploadSize = 200

[browser]
gatherUsageStats = false
EOF

# Update .gitignore for deployment (keep essential files only)
if [ -f .gitignore ]; then
    # Add deployment-specific ignores while keeping essential files
    cat >> .gitignore << 'EOF'

# Streamlit deployment - keep essential files only
results/detailed_explanations/*/*/*.json
results/iterations/*.json
!results/experiment_*/benchmark_results.json
!results/experiment_*/comprehensive_report.md

# Streamlit cache
.streamlit/secrets.toml
EOF
else
    # Create new .gitignore
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

# Keep experiment structure but limit large files for deployment
results/detailed_explanations/*/*/*.json
results/iterations/*.json

# Keep essential files for dashboard
!results/experiment_*/benchmark_results.json
!results/experiment_*/comprehensive_report.md

# OS files
.DS_Store
Thumbs.db
*.tmp
*.log
EOF
fi

echo "Deployment setup complete!"
echo ""
echo "Files created:"
echo "- requirements_streamlit_deploy.txt"
echo "- .streamlit/config.toml"
echo "- Updated .gitignore"
echo ""
echo "To deploy on Streamlit Cloud:"
echo "1. Push your repository to GitHub"
echo "2. Go to share.streamlit.io"
echo "3. Connect your GitHub repository"
echo "4. Set main file: streamlit_dashboard.py"
echo "5. Set requirements file: requirements_streamlit_deploy.txt"
echo "6. Deploy!"