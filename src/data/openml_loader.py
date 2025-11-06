"""
OpenML Dataset Loader
Integrates OpenML datasets into the XAI benchmarking framework
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import warnings

try:
    from sklearn.datasets import fetch_openml as sklearn_fetch_openml
    OPENML_AVAILABLE = True
except ImportError:
    OPENML_AVAILABLE = False


class OpenMLLoader:
    """
    Loads datasets from OpenML for the benchmarking framework
    """

    def __init__(self, cache_dir: str = "data/openml_cache"):
        """
        Initialize OpenML loader

        Args:
            cache_dir: Directory to cache downloaded datasets
        """
        self.logger = logging.getLogger(__name__)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        if not OPENML_AVAILABLE:
            self.logger.warning("OpenML not available. Install with: pip install openml")

        # OpenML dataset mappings for confirmed available datasets
        # Format: dataset_name -> (openml_id, openml_name, type)
        self.dataset_mappings = {
            # Binary tabular datasets (5)
            "adult_income": (1590, "adult", "tabular"),
            "breast_cancer": (15, "breast-w", "tabular"),
            "heart_disease": (43672, "Heart-Disease-Dataset-(Comprehensive)", "tabular"),
            "german_credit": (31, "credit-g", "tabular"),
            # Note: compas not available on OpenML, will use fallback

            # Multi-class tabular datasets (5)
            "iris": (61, "iris", "tabular"),
            "wine_quality": (287, "wine_quality", "tabular"),
            "diabetes": (37, "diabetes", "tabular"),
            "wine_classification": (187, "wine", "tabular"),
            "digits": (28, "optdigits", "tabular"),

            # Image datasets (3)
            "mnist": (554, "mnist_784", "image"),
            "cifar10": (40927, "cifar_10", "image"),
            "fashion_mnist": (40996, "Fashion-MNIST", "image")
        }

    def is_available(self) -> bool:
        """Check if OpenML is available"""
        return OPENML_AVAILABLE

    def load_dataset(self, dataset_id: int,
                    dataset_name: str = None,
                    dataset_type: str = "tabular") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load dataset from OpenML using sklearn's fetch_openml

        Args:
            dataset_id: OpenML dataset ID
            dataset_name: Optional name for logging
            dataset_type: Type of dataset ('tabular' or 'image')

        Returns:
            Tuple of (X, y, metadata)
        """
        if not OPENML_AVAILABLE:
            raise ImportError("sklearn not available. Install with: pip install scikit-learn")

        try:
            self.logger.info(f"Loading OpenML dataset {dataset_id} ({dataset_name or 'unnamed'}) via sklearn")

            # Download from OpenML using sklearn with memory-efficient settings
            # Use parser='liac-arff' for large datasets to avoid pandas memory issues
            try:
                data = sklearn_fetch_openml(
                    data_id=dataset_id,
                    as_frame=True,
                    parser='liac-arff',  # More memory efficient than 'auto' for large datasets
                    cache=True,
                    data_home=str(self.cache_dir)
                )
            except (MemoryError, Exception) as e:
                # If liac-arff fails, raise the error for fallback handling
                self.logger.error(f"Memory error or parser error loading dataset {dataset_id}: {e}")
                raise MemoryError(f"Dataset {dataset_id} too large to load in memory") from e

            X = data.data
            y = data.target

            # Handle missing values
            if X.isnull().any().any():
                self.logger.warning(f"Dataset contains missing values. Filling with median/mode...")
                X = self._handle_missing_values(X)

            # Convert target to numeric if needed
            y_array = self._encode_target(y)

            # Convert features to numeric
            X_numeric = self._convert_to_numeric(X)

            # Get feature names
            feature_names = list(X.columns) if hasattr(X, 'columns') else None

            # Create metadata
            metadata = {
                'name': dataset_name or f'openml_{dataset_id}',
                'description': f'OpenML dataset ID {dataset_id}',
                'openml_id': dataset_id,
                'feature_names': feature_names,
                'target_name': 'target',
                'n_samples': X_numeric.shape[0],
                'n_features': X_numeric.shape[1],
                'n_classes': len(np.unique(y_array)),
                'task_type': self._infer_task_type(y_array),
                'source': 'openml',
                'dataset_type': dataset_type
            }

            self.logger.info(f"Successfully loaded {metadata['name']}: "
                           f"{metadata['n_samples']} samples, {metadata['n_features']} features, "
                           f"{metadata['n_classes']} classes")

            return X_numeric.values, y_array, metadata

        except Exception as e:
            self.logger.error(f"Failed to load OpenML dataset {dataset_id}: {e}")
            raise

    def load_by_name(self, dataset_name: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load dataset by name using predefined mappings

        Args:
            dataset_name: Name of the dataset

        Returns:
            Tuple of (X, y, metadata)
        """
        if dataset_name not in self.dataset_mappings:
            raise ValueError(f"Dataset '{dataset_name}' not found in mappings. "
                           f"Available: {list(self.dataset_mappings.keys())}")

        dataset_id, openml_name, dataset_type = self.dataset_mappings[dataset_name]
        return self.load_dataset(dataset_id, dataset_name, dataset_type)

    def _infer_task_type(self, y: np.ndarray) -> str:
        """Infer task type from target values"""
        if y.dtype.kind in ['f']:  # float
            return 'regression'
        elif y.dtype.kind in ['i', 'U', 'S']:  # int, unicode, string
            n_unique = len(np.unique(y))
            if n_unique == 2:
                return 'binary_classification'
            elif n_unique > 2:
                return 'multiclass_classification'

        return 'unknown'

    def preprocess_for_classification(self, y: np.ndarray,
                                    n_classes: Optional[int] = None) -> np.ndarray:
        """
        Preprocess target for classification tasks

        Args:
            y: Target values
            n_classes: Number of classes to bin into (for regression->classification)

        Returns:
            Processed target values
        """
        if y.dtype.kind == 'f' and n_classes is not None:
            # Convert regression to classification by binning
            bins = np.quantile(y, np.linspace(0, 1, n_classes + 1))
            bins = np.unique(bins)  # Remove duplicates
            y_binned = np.digitize(y, bins[1:-1])
            self.logger.info(f"Converted regression to {len(np.unique(y_binned))}-class classification")
            return y_binned

        # Ensure labels start from 0
        unique_labels = np.unique(y)
        if not np.array_equal(unique_labels, np.arange(len(unique_labels))):
            label_mapping = {label: i for i, label in enumerate(unique_labels)}
            y_mapped = np.array([label_mapping[label] for label in y])
            self.logger.info(f"Remapped labels: {dict(zip(unique_labels, range(len(unique_labels))))}")
            return y_mapped

        return y

    def _handle_missing_values(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        X_filled = X.copy()

        for col in X_filled.columns:
            if X_filled[col].isnull().any():
                if X_filled[col].dtype in ['int64', 'float64']:
                    # Numeric: fill with median
                    X_filled[col].fillna(X_filled[col].median(), inplace=True)
                else:
                    # Categorical: fill with mode
                    mode_val = X_filled[col].mode()
                    if len(mode_val) > 0:
                        X_filled[col].fillna(mode_val[0], inplace=True)
                    else:
                        X_filled[col].fillna('unknown', inplace=True)

        return X_filled

    def _encode_target(self, y: pd.Series) -> np.ndarray:
        """Encode target variable to numeric"""
        if y.dtype == 'object' or y.dtype.name == 'category':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            self.logger.info(f"Encoded target labels: {dict(zip(le.classes_, le.transform(le.classes_)))}")
            return y_encoded
        else:
            return y.values

    def _convert_to_numeric(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert DataFrame columns to numeric types"""
        df_numeric = df.copy()

        for col in df_numeric.columns:
            if df_numeric[col].dtype == 'object' or df_numeric[col].dtype.name == 'category':
                # Categorical column - encode
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                df_numeric[col] = le.fit_transform(df_numeric[col].astype(str))

        return df_numeric

    def get_available_datasets(self) -> Dict[str, Tuple[int, str, str]]:
        """Get available dataset mappings"""
        return self.dataset_mappings.copy()

    def add_dataset_mapping(self, name: str, openml_id: int):
        """Add a new dataset mapping"""
        self.dataset_mappings[name] = openml_id
        self.logger.info(f"Added dataset mapping: {name} -> {openml_id}")

    def search_datasets(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Search for datasets on OpenML

        Args:
            query: Search query
            limit: Maximum number of results

        Returns:
            List of dataset information
        """
        if not OPENML_AVAILABLE:
            raise ImportError("OpenML not available")

        try:
            datasets = openml.datasets.list_datasets(
                output_format='dataframe',
                size=limit
            )

            # Filter by query
            if query:
                mask = datasets['name'].str.contains(query, case=False, na=False)
                datasets = datasets[mask]

            results = []
            for _, row in datasets.head(limit).iterrows():
                results.append({
                    'id': row['did'],
                    'name': row['name'],
                    'description': row.get('description', ''),
                    'instances': row.get('NumberOfInstances', 0),
                    'features': row.get('NumberOfFeatures', 0),
                    'classes': row.get('NumberOfClasses', 0)
                })

            return results

        except Exception as e:
            self.logger.error(f"Failed to search datasets: {e}")
            return []

    def validate_dataset(self, X: np.ndarray, y: np.ndarray,
                        metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate downloaded dataset

        Args:
            X: Feature matrix
            y: Target vector
            metadata: Dataset metadata

        Returns:
            Validation results
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'errors': []
        }

        # Basic shape validation
        if X.shape[0] != len(y):
            validation['errors'].append(f"Shape mismatch: X has {X.shape[0]} samples, y has {len(y)}")
            validation['is_valid'] = False

        # Check for missing values
        if np.isnan(X).any():
            missing_ratio = np.isnan(X).sum() / X.size
            validation['warnings'].append(f"Missing values detected: {missing_ratio:.2%}")

        # Check dataset size
        if X.shape[0] < 100:
            validation['warnings'].append(f"Small dataset: only {X.shape[0]} samples")

        # Check class balance for classification
        if metadata.get('task_type') in ['binary_classification', 'multiclass_classification']:
            unique, counts = np.unique(y, return_counts=True)
            min_ratio = min(counts) / sum(counts)
            if min_ratio < 0.1:
                validation['warnings'].append(f"Class imbalance detected: minimum class ratio {min_ratio:.2%}")

        return validation


# Convenience functions for integration
def load_openml_dataset(dataset_id: int, cache_dir: str = "data/openml_cache") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Convenience function to load OpenML dataset"""
    loader = OpenMLLoader(cache_dir)
    return loader.load_dataset(dataset_id)


def load_openml_by_name(dataset_name: str, cache_dir: str = "data/openml_cache") -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Convenience function to load OpenML dataset by name"""
    loader = OpenMLLoader(cache_dir)
    return loader.load_by_name(dataset_name)