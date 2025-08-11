# XAI Benchmarking Framework - Setup and Execution Guide

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Running Tests](#running-tests)
5. [Using the Framework](#using-the-framework)
6. [Configuration](#configuration)
7. [Troubleshooting](#troubleshooting)

## Prerequisites

Before setting up the XAI Benchmarking Framework, ensure you have:

- **Python 3.8 or higher**
- **pip** (Python package installer)
- **Git** (for cloning the repository)

## Installation

### 1. Clone or Download the Project

If you have the project files locally, navigate to the project directory:
```bash
cd path/to/benchmarking
```

### 2. Create a Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv xai_benchmark_env

# Activate virtual environment
# On Windows:
xai_benchmark_env\Scripts\activate
# On macOS/Linux:
source xai_benchmark_env/bin/activate
```

### 3. Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**Note**: Some packages might take time to install, especially PyTorch and TensorFlow. If you encounter issues with specific packages, you can install them individually or use conda for better dependency management.

### 4. Verify Installation

```bash
# Check if Python can import the main modules
python -c "import numpy, pandas, sklearn, torch, shap, lime; print('All packages imported successfully!')"
```

## Quick Start

### 1. Basic Test Run

Run the test script to verify everything is working:

```bash
python test_framework.py
```

This will:
- Test data loading functionality
- Test model creation
- Test explanation method creation
- Run a simple end-to-end pipeline

### 2. Run the Main Framework

```bash
# Run with default configuration
python main.py --config configs/default_config.yaml

# Run in interactive mode
python main.py --config configs/default_config.yaml --interactive

# Run with custom output directory
python main.py --config configs/default_config.yaml --output-dir results/my_experiment
```

### 3. Run Examples

```bash
# Run example usage scenarios
python example_usage.py
```

## Running Tests

### Automated Test Suite

```bash
# Run comprehensive tests
python test_framework.py
```

The test suite will check:
- ✅ Data loading and preprocessing
- ✅ Model creation and training
- ✅ Explanation method creation
- ✅ End-to-end pipeline execution

### Individual Component Testing

You can also test individual components:

```python
# Test data loading
from src.data.data_manager import DataManager
from src.utils.config import load_config

config = load_config("configs/default_config.yaml")
data_manager = DataManager(config)
datasets = data_manager.get_available_datasets()
print(f"Available datasets: {datasets}")
```

## Using the Framework

### 1. Command Line Interface

The framework provides a command-line interface with the following options:

```bash
python main.py [OPTIONS]

Options:
  --config TEXT          Configuration file path (default: configs/default_config.yaml)
  --output-dir TEXT      Output directory for results (default: results)
  --interactive          Run in interactive mode
  --verbose              Enable verbose logging
  --help                 Show help message
```

### 2. Interactive Mode

Interactive mode allows you to:
- Select datasets interactively
- Choose models for training
- Select explanation methods
- Configure evaluation metrics

```bash
python main.py --interactive
```

### 3. Programmatic Usage

You can also use the framework programmatically:

```python
from src.benchmark import XAIBenchmark
from src.utils.config import load_config
from pathlib import Path

# Load configuration
config = load_config("configs/default_config.yaml")

# Create output directory
output_dir = Path("results/my_experiment")
output_dir.mkdir(parents=True, exist_ok=True)

# Initialize and run benchmark
benchmark = XAIBenchmark(config, output_dir)
benchmark.run_full_pipeline()
```

## Configuration

### Configuration File Structure

The framework uses YAML configuration files. The main configuration sections are:

```yaml
# Data configuration
data:
  tabular_datasets: [...]
  image_datasets: [...]
  text_datasets: [...]

# Model configuration
models:
  tabular: [...]
  image: [...]
  text: [...]

# Explanation methods
explanations:
  feature_attribution: [...]
  example_based: [...]
  concept_based: [...]
  perturbation: [...]

# Evaluation metrics
evaluation:
  fidelity: [...]
  time_complexity: [...]
  stability: [...]
  comprehensibility: [...]

# Experiment settings
experiment:
  preprocessing: {...}
  training: {...}
  explanation: {...}
  evaluation: {...}

# Output settings
output:
  results_dir: "results"
  save_models: true
  save_explanations: true
  generate_plots: true
```

### Creating Custom Configurations

1. Copy the default configuration:
```bash
cp configs/default_config.yaml configs/my_config.yaml
```

2. Modify the configuration file according to your needs

3. Run with your custom configuration:
```bash
python main.py --config configs/my_config.yaml
```

## Supported Features

### Data Types
- **Tabular Data**: CSV files, sklearn datasets
- **Image Data**: PNG, JPG files, torchvision datasets
- **Text Data**: Text files, sklearn text datasets

### Machine Learning Models
- **Simple ML**: Decision Trees, Random Forest, Gradient Boosting, MLP
- **Neural Networks**: CNN, Vision Transformer
- **Text Models**: BERT (simplified), LSTM (simplified)

### Explanation Methods
- **Feature Attribution**: SHAP, LIME, Integrated Gradients
- **Example-based**: Prototypes, Counterfactuals
- **Concept-based**: TCAV, Concept Bottleneck Models
- **Perturbation**: Occlusion, Feature Ablation

### Evaluation Metrics
- **Fidelity**: Faithfulness, Monotonicity, Completeness
- **Time Complexity**: Explanation time, Training time
- **Stability**: Consistency, Robustness
- **Comprehensibility**: Sparsity, Simplicity

## Output and Results

### Results Directory Structure

```
results/
├── models/              # Trained models
├── explanations/        # Generated explanations
├── evaluation/          # Evaluation metrics
├── plots/              # Generated visualizations
├── reports/            # Detailed reports
└── summary.json        # Summary of results
```

### Generated Reports

The framework generates several types of reports:

1. **Detailed Results**: JSON files with comprehensive metrics
2. **Summary Report**: High-level overview of performance
3. **Comparison Report**: Comparison between different methods
4. **Visualizations**: Plots and charts for analysis

## Troubleshooting

### Common Issues

#### 1. Import Errors
**Problem**: `ModuleNotFoundError` when importing packages
**Solution**: Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

#### 2. CUDA/GPU Issues
**Problem**: PyTorch CUDA errors
**Solution**: Install CPU-only version or check CUDA compatibility:
```bash
# CPU-only PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### 3. Memory Issues
**Problem**: Out of memory errors
**Solution**: Reduce batch sizes or use smaller datasets in configuration

#### 4. Configuration Errors
**Problem**: YAML parsing errors
**Solution**: Check YAML syntax and ensure all required sections are present

### Getting Help

1. **Check the logs**: Look at the generated log files for detailed error messages
2. **Run tests**: Use `python test_framework.py` to identify specific issues
3. **Check configuration**: Verify your configuration file syntax
4. **Reduce complexity**: Start with simple configurations and gradually add complexity

### Performance Tips

1. **Use appropriate data sizes**: Start with small datasets for testing
2. **Enable GPU acceleration**: If available, use GPU for faster training
3. **Optimize batch sizes**: Adjust batch sizes based on your hardware
4. **Use parallel processing**: Enable multiprocessing where supported

## Next Steps

After successful setup and testing:

1. **Explore the codebase**: Understand the modular structure
2. **Customize configurations**: Adapt to your specific use cases
3. **Add new models**: Implement additional ML models
4. **Add new explanation methods**: Extend with new XAI techniques
5. **Add new evaluation metrics**: Implement custom evaluation criteria
6. **Scale experiments**: Run larger-scale benchmarking studies

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the code documentation
3. Run the test suite to identify problems
4. Check the generated log files for detailed error information 