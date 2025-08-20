@echo off
echo 🚀 Setting up separate Streamlit dashboard repository...

REM Step 1: Create new directory for dashboard repo
set DASHBOARD_DIR=..\xai-dashboard-deploy
mkdir "%DASHBOARD_DIR%" 2>nul
cd "%DASHBOARD_DIR%"

echo 📁 Created dashboard directory: %DASHBOARD_DIR%

REM Step 2: Initialize new git repository
git init
echo ✅ Initialized new git repository

REM Step 3: Copy essential files
echo 📋 Copying essential files...

copy ..\benchmarking\streamlit_dashboard.py .
copy ..\benchmarking\requirements_streamlit_deploy.txt requirements.txt

mkdir .streamlit 2>nul
copy ..\benchmarking\.streamlit\config.toml .streamlit\

echo ✅ Copied dashboard files

REM Step 4: Copy experiment results structure
echo 📊 Setting up results structure...
mkdir results 2>nul

echo Copying experiment results...
REM Note: You'll need to manually copy the results folders or use the manual steps below

echo ✅ Basic structure created

echo.
echo 🎉 Basic dashboard repository setup complete!
echo.
echo 📍 Current location: %cd%
echo.
echo 🔗 Next steps:
echo 1. Manually copy experiment results (see manual instructions)
echo 2. Create GitHub repository
echo 3. Push to GitHub  
echo 4. Deploy on Streamlit Cloud
echo.
echo 📖 See DEPLOYMENT_MANUAL.md for detailed instructions

pause