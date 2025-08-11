"""
Configuration utilities for XAI benchmarking
"""

import yaml
import logging
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration structure
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True if valid, raises ValueError if invalid
    """
    required_sections = ['data', 'models', 'explanations', 'evaluation']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate data section
    required_data_keys = ['tabular_datasets', 'image_datasets', 'text_datasets']
    for key in required_data_keys:
        if key not in config['data']:
            raise ValueError(f"Missing '{key}' in data configuration")
    
    # Validate models section
    required_model_keys = ['tabular', 'image', 'text']
    for key in required_model_keys:
        if key not in config['models']:
            raise ValueError(f"Missing '{key}' in models configuration")
    
    # Validate explanations section
    required_explanation_keys = ['feature_attribution', 'example_based', 'concept_based', 'perturbation']
    for key in required_explanation_keys:
        if key not in config['explanations']:
            raise ValueError(f"Missing '{key}' in explanations configuration")
    
    return True


def save_config(config: Dict[str, Any], output_path: str):
    """
    Save configuration to YAML file
    
    Args:
        config: Configuration dictionary
        output_path: Output file path
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2) 