"""
Model Persistence Utility
Handles saving and loading of trained models to avoid retraining
"""

import os
import pickle
import json
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import torch
import joblib
from datetime import datetime


class ModelPersistence:
    """
    Handles model saving and loading for the benchmarking framework
    """

    def __init__(self, storage_dir: Path = None):
        """
        Initialize ModelPersistence

        Args:
            storage_dir: Directory to store models (default: ./saved_models)
        """
        self.logger = logging.getLogger(__name__)
        self.storage_dir = Path(storage_dir) if storage_dir else Path("saved_models")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # Create subdirectories for organization
        self.models_dir = self.storage_dir / "models"
        self.metadata_dir = self.storage_dir / "metadata"
        self.models_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)

        self.logger.info(f"Model persistence initialized with storage at: {self.storage_dir}")

    def _generate_model_id(self, dataset_name: str, model_type: str,
                          model_config: Dict[str, Any]) -> str:
        """
        Generate unique ID for model based on dataset, type, and config

        Args:
            dataset_name: Name of the dataset
            model_type: Type of model (e.g., 'random_forest', 'neural_network')
            model_config: Model configuration dictionary

        Returns:
            Unique model ID string
        """
        # Create a stable hash from config
        config_str = json.dumps(model_config, sort_keys=True)
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]

        # Create model ID
        model_id = f"{dataset_name}_{model_type}_{config_hash}"
        return model_id

    def save_model(self, model: Any, dataset_name: str, model_type: str,
                  model_config: Dict[str, Any],
                  performance_metrics: Dict[str, Any] = None) -> str:
        """
        Save a trained model with metadata

        Args:
            model: Trained model object
            dataset_name: Name of the dataset used for training
            model_type: Type of model
            model_config: Model configuration
            performance_metrics: Model performance metrics

        Returns:
            Model ID for future loading
        """
        try:
            model_id = self._generate_model_id(dataset_name, model_type, model_config)
            model_path = self.models_dir / f"{model_id}.pkl"
            metadata_path = self.metadata_dir / f"{model_id}_metadata.json"

            # Prepare metadata
            metadata = {
                "model_id": model_id,
                "dataset_name": dataset_name,
                "model_type": model_type,
                "model_config": model_config,
                "performance_metrics": performance_metrics or {},
                "saved_timestamp": datetime.now().isoformat(),
                "framework": self._detect_framework(model)
            }

            # Save model based on framework
            if metadata["framework"] == "pytorch":
                # For PyTorch models
                torch_path = self.models_dir / f"{model_id}.pt"
                if hasattr(model, 'state_dict'):
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'model_config': model_config
                    }, torch_path)
                    metadata["model_file"] = str(torch_path.name)
                else:
                    # Fallback to pickle for wrapped PyTorch models
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    metadata["model_file"] = str(model_path.name)

            elif metadata["framework"] == "sklearn":
                # For scikit-learn models
                joblib.dump(model, model_path)
                metadata["model_file"] = str(model_path.name)

            else:
                # Generic pickle fallback
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                metadata["model_file"] = str(model_path.name)

            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Model saved successfully: {model_id}")
            self.logger.info(f"Performance metrics: Test Acc={metadata['performance_metrics'].get('test_accuracy', 'N/A')}")

            return model_id

        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            raise

    def load_model(self, dataset_name: str, model_type: str,
                  model_config: Dict[str, Any]) -> Tuple[Optional[Any], Optional[Dict[str, Any]]]:
        """
        Load a previously saved model

        Args:
            dataset_name: Name of the dataset
            model_type: Type of model
            model_config: Model configuration

        Returns:
            Tuple of (model, metadata) or (None, None) if not found
        """
        try:
            model_id = self._generate_model_id(dataset_name, model_type, model_config)
            metadata_path = self.metadata_dir / f"{model_id}_metadata.json"

            # Check if model exists
            if not metadata_path.exists():
                self.logger.info(f"No saved model found for: {model_id}")
                return None, None

            # Load metadata
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)

            # Load model based on framework
            model_file = self.models_dir / metadata["model_file"]

            if not model_file.exists():
                self.logger.warning(f"Model file not found: {model_file}")
                return None, None

            if metadata["framework"] == "pytorch" and model_file.suffix == ".pt":
                # Load PyTorch model
                checkpoint = torch.load(model_file, map_location='cpu')
                # Note: The actual model recreation would need to be handled by the model factory
                # This returns the state dict for the model factory to use
                model = checkpoint

            elif metadata["framework"] == "sklearn":
                # Load scikit-learn model
                model = joblib.load(model_file)

            else:
                # Load generic pickle
                with open(model_file, 'rb') as f:
                    model = pickle.load(f)

            self.logger.info(f"Model loaded successfully: {model_id}")
            self.logger.info(f"Model was saved on: {metadata['saved_timestamp']}")
            self.logger.info(f"Previous performance: Test Acc={metadata['performance_metrics'].get('test_accuracy', 'N/A')}")

            return model, metadata

        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return None, None

    def model_exists(self, dataset_name: str, model_type: str,
                    model_config: Dict[str, Any]) -> bool:
        """
        Check if a model exists for given configuration

        Args:
            dataset_name: Name of the dataset
            model_type: Type of model
            model_config: Model configuration

        Returns:
            True if model exists, False otherwise
        """
        model_id = self._generate_model_id(dataset_name, model_type, model_config)
        metadata_path = self.metadata_dir / f"{model_id}_metadata.json"
        return metadata_path.exists()

    def get_model_info(self, dataset_name: str, model_type: str,
                      model_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Get information about a saved model without loading it

        Args:
            dataset_name: Name of the dataset
            model_type: Type of model
            model_config: Model configuration

        Returns:
            Model metadata or None if not found
        """
        model_id = self._generate_model_id(dataset_name, model_type, model_config)
        metadata_path = self.metadata_dir / f"{model_id}_metadata.json"

        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                return json.load(f)

        return None

    def list_saved_models(self) -> List[Dict[str, Any]]:
        """
        List all saved models with their metadata

        Returns:
            List of model metadata dictionaries
        """
        models = []

        for metadata_file in self.metadata_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                    models.append(metadata)
            except Exception as e:
                self.logger.warning(f"Error reading metadata file {metadata_file}: {e}")

        # Sort by saved timestamp
        models.sort(key=lambda x: x.get('saved_timestamp', ''), reverse=True)

        return models

    def cleanup_old_models(self, days: int = 30):
        """
        Remove models older than specified days

        Args:
            days: Number of days to keep models
        """
        from datetime import datetime, timedelta

        cutoff_date = datetime.now() - timedelta(days=days)
        removed_count = 0

        for metadata_file in self.metadata_dir.glob("*_metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)

                saved_date = datetime.fromisoformat(metadata['saved_timestamp'])

                if saved_date < cutoff_date:
                    # Remove model and metadata
                    model_file = self.models_dir / metadata['model_file']
                    if model_file.exists():
                        model_file.unlink()
                    metadata_file.unlink()
                    removed_count += 1

            except Exception as e:
                self.logger.warning(f"Error processing file {metadata_file}: {e}")

        self.logger.info(f"Cleaned up {removed_count} old models")

    def _detect_framework(self, model: Any) -> str:
        """
        Detect the ML framework of the model

        Args:
            model: Model object

        Returns:
            Framework name ('pytorch', 'sklearn', 'tensorflow', 'unknown')
        """
        model_class = str(type(model))

        if 'torch' in model_class or hasattr(model, 'parameters'):
            return 'pytorch'
        elif 'sklearn' in model_class or hasattr(model, 'fit') and hasattr(model, 'predict'):
            return 'sklearn'
        elif 'tensorflow' in model_class or 'keras' in model_class:
            return 'tensorflow'
        else:
            return 'unknown'

    def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get statistics about model storage

        Returns:
            Dictionary with storage statistics
        """
        total_size = 0
        model_count = 0

        for model_file in self.models_dir.glob("*"):
            if model_file.is_file():
                total_size += model_file.stat().st_size
                model_count += 1

        return {
            "total_models": model_count,
            "total_size_mb": total_size / (1024 * 1024),
            "storage_directory": str(self.storage_dir)
        }


