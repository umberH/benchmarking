# XAI Benchmarking Framework

A comprehensive benchmarking framework for Explainable AI (XAI) that supports multiple data types, machine learning models, and explanation methods.

## Features

- **Multi-modal Data Support**: Tabular, Image, and Text data
- **Classification Tasks**: Binary and Multi-class classification
- **ML Models**: From simple models (Decision Trees, Gradient Boosting) to complex transformers
- **Explanation Methods**: Feature attribution, example-based, counterfactual, concept-based, and perturbation methods
- **Evaluation Metrics**: Fidelity, time complexity, and other XAI-specific metrics

## Project Structure

```
benchmarking/
├── data/                   # Data storage and preprocessing
├── models/                 # ML model implementations
├── explanations/           # XAI explanation methods
├── evaluation/            # Evaluation metrics and benchmarks
├── configs/               # Configuration files
├── utils/                 # Utility functions
├── notebooks/             # Jupyter notebooks for analysis
└── results/               # Benchmark results and visualizations
```

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the main benchmarking pipeline:
```bash
python main.py --config configs/default_config.yaml
```

3. Or use the interactive interface:
```bash
python interactive_benchmark.py
```

## Configuration

Edit `configs/default_config.yaml` to customize:
- Dataset selection
- Model parameters
- Explanation methods
- Evaluation metrics

## Supported Components

### Data Types
- **Tabular**: CSV, Parquet files
- **Image**: PNG, JPG, with preprocessing
- **Text**: CSV with text columns, JSON

### ML Models
- **Simple Models**: Decision Trees, Random Forest, Gradient Boosting
- **Neural Networks**: MLP, CNN, RNN
- **Transformers**: BERT, ViT, Tabular Transformers

### Explanation Methods
- **Feature Attribution**: SHAP, LIME, Integrated Gradients
- **Example-based**: Prototypes, Counterfactuals
- **Concept-based**: TCAV, Concept Bottleneck Models
- **Perturbation**: Occlusion, Feature Ablation

### Evaluation Metrics
- **Fidelity**: Faithfulness, Monotonicity
- **Time Complexity**: Explanation generation time
- **Stability**: Consistency across runs
- **Comprehensibility**: Human interpretability scores

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 