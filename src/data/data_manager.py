"""
Data manager for handling different data types in XAI benchmarking
"""

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import time

from .datasets import TabularDataset, ImageDataset, TextDataset
from ..utils.data_splitting import DataSplitter


class DataManager:
    """
    Manages data loading and preprocessing for different data types
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize data manager
        
        Args:
            config: Configuration dictionary for data management
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.datasets = {}
        
        # Initialize data splitter
        self.data_splitter = DataSplitter(config)
        
        # Default data paths
        self.data_paths = {
            'tabular': Path('data/tabular'),
            'image': Path('data/image'),
            'text': Path('data/text')
        }
        
        # Update with config paths if provided
        if 'data_paths' in config:
            self.data_paths.update(config['data_paths'])
    
    def get_available_datasets(self) -> List[str]:
        """Get list of available datasets"""
        available = []
        
        # Check tabular datasets
        if self.data_paths['tabular'].exists():
            for file_path in self.data_paths['tabular'].glob('*.csv'):
                available.append(f"tabular_{file_path.stem}")
        
        # Check image datasets
        if self.data_paths['image'].exists():
            for folder in self.data_paths['image'].iterdir():
                if folder.is_dir():
                    available.append(f"image_{folder.name}")
        
        # Check text datasets
        if self.data_paths['text'].exists():
            for file_path in self.data_paths['text'].glob('*.csv'):
                available.append(f"text_{file_path.stem}")
        
        return available
    
    def split_dataset(self, dataset, split_strategy: str = None, **kwargs) -> Dict[str, Any]:
        """
        Split dataset using enhanced splitting strategies
        
        Args:
            dataset: Dataset object with get_data() method
            split_strategy: Splitting strategy to use
            **kwargs: Additional arguments for specific strategies
            
        Returns:
            Dictionary containing split indices and metadata
        """
        # Get default strategy from config
        if split_strategy is None:
            split_strategy = self.config.get('experiment', {}).get('data_splitting', {}).get('default_strategy', 'stratified')
        
        # Get data from dataset
        X_train, X_test, y_train, y_test = dataset.get_data()
        
        # Combine all data for splitting
        X_combined = np.vstack([X_train, X_test]) if isinstance(X_train, np.ndarray) else pd.concat([X_train, X_test])
        y_combined = np.concatenate([y_train, y_test]) if isinstance(y_train, np.ndarray) else pd.concat([y_train, y_test])
        
        # Apply splitting strategy
        split_result = self.data_splitter.split_data(X_combined, y_combined, split_strategy, **kwargs)
        
        self.logger.info(f"Applied {split_strategy} splitting strategy to dataset {dataset.name}")
        self.logger.info(self.data_splitter.get_split_summary(split_result))
        
        return split_result
    
    def load_datasets(self) -> Dict[str, Any]:
        """Load all configured datasets"""
        datasets = {}
        
        # Load mandatory datasets first
        mandatory_datasets = self._get_mandatory_datasets()
        for dataset_name, dataset_config in mandatory_datasets.items():
            self.logger.info(f"Loading mandatory dataset: {dataset_name}")
            dataset = self._load_mandatory_dataset(dataset_name, dataset_config)
            datasets[dataset_name] = dataset
        
        # Load other configured datasets if they exist
        # Note: For now, we focus on mandatory datasets
        # Additional dataset loading logic can be added here later
        
        return datasets
    
    def _get_mandatory_datasets(self) -> Dict[str, Dict[str, Any]]:
        """Get configuration for mandatory datasets"""
        mandatory_datasets = {}
        
        # Check tabular datasets
        for dataset_config in self.config.get('data', {}).get('tabular_datasets', []):
            if dataset_config.get('mandatory', False):
                mandatory_datasets[dataset_config['name']] = {
                    'type': 'tabular',
                    'source': dataset_config['source'],
                    'description': dataset_config['description']
                }
        
        # Check image datasets
        for dataset_config in self.config.get('data', {}).get('image_datasets', []):
            if dataset_config.get('mandatory', False):
                mandatory_datasets[dataset_config['name']] = {
                    'type': 'image',
                    'source': dataset_config['source'],
                    'description': dataset_config['description']
                }
        
        # Check text datasets
        for dataset_config in self.config.get('data', {}).get('text_datasets', []):
            if dataset_config.get('mandatory', False):
                mandatory_datasets[dataset_config['name']] = {
                    'type': 'text',
                    'source': dataset_config['source'],
                    'description': dataset_config['description']
                }
        
        return mandatory_datasets
    
    def _load_mandatory_dataset(self, dataset_name: str, dataset_config: Dict[str, Any]) -> Any:
        """Load a mandatory dataset by name"""
        dataset_type = dataset_config['type']
        source = dataset_config['source']
        
        if dataset_name == 'adult_income':
            return self._load_adult_income_dataset()
        elif dataset_name == 'compas':
            return self._load_compas_dataset()
        elif dataset_name == 'mnist':
            return self._load_mnist_dataset()
        elif dataset_name == 'imdb':
            return self._load_imdb_dataset()
        else:
            raise ValueError(f"Unknown mandatory dataset: {dataset_name}")
    
    def load_dataset(self, dataset_name: str) -> Any:
        """Load a specific dataset by name"""
        # Check if it's a mandatory dataset first
        mandatory_datasets = self._get_mandatory_datasets()
        if dataset_name in mandatory_datasets:
            return self._load_mandatory_dataset(dataset_name, mandatory_datasets[dataset_name])
        
        # Try to parse as type_id format
        if '_' in dataset_name:
            dataset_type, dataset_id = dataset_name.split('_', 1)
            
            if dataset_type == 'tabular':
                config = {
                    'name': dataset_name,
                    'type': 'tabular',
                    'file_path': str(self.data_paths['tabular'] / f"{dataset_id}.csv"),
                    'target_column': 'target',  # Default, should be configurable
                    'test_size': 0.2
                }
                return self._load_tabular_dataset(config)
            
            elif dataset_type == 'image':
                config = {
                    'name': dataset_name,
                    'type': 'image',
                    'data_path': str(self.data_paths['image'] / dataset_id),
                    'test_size': 0.2
                }
                return self._load_image_dataset(config)
            
            elif dataset_type == 'text':
                config = {
                    'name': dataset_name,
                    'type': 'text',
                    'file_path': str(self.data_paths['text'] / f"{dataset_id}.csv"),
                    'text_column': 'text',  # Default, should be configurable
                    'target_column': 'target',
                    'test_size': 0.2
                }
                return self._load_text_dataset(config)
        
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _load_tabular_dataset(self, config: Dict[str, Any]) -> TabularDataset:
        """Load tabular dataset"""
        file_path = Path(config['file_path'])
        if not file_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {file_path}")
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Extract features and target
        target_column = config.get('target_column', 'target')
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Split data
        test_size = config.get('test_size', 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        return TabularDataset(
            name=config['name'],
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            feature_names=list(X.columns),
            config=config
        )
    
    def _load_image_dataset(self, config: Dict[str, Any]) -> ImageDataset:
        """Load image dataset"""
        data_path = Path(config['data_path'])
        if not data_path.exists():
            raise FileNotFoundError(f"Image dataset path not found: {data_path}")
        
        return ImageDataset(
            name=config['name'],
            data_path=data_path,
            config=config
        )
    
    def _load_text_dataset(self, config: Dict[str, Any]) -> TextDataset:
        """Load text dataset"""
        file_path = Path(config['file_path'])
        if not file_path.exists():
            raise FileNotFoundError(f"Text dataset file not found: {file_path}")
        
        # Load data
        df = pd.read_csv(file_path)
        
        # Extract text and target
        text_column = config.get('text_column', 'text')
        target_column = config.get('target_column', 'target')
        
        if text_column not in df.columns:
            raise ValueError(f"Text column '{text_column}' not found in dataset")
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        texts = df[text_column].tolist()
        targets = df[target_column].tolist()
        
        # Split data
        test_size = config.get('test_size', 0.2)
        train_texts, test_texts, train_targets, test_targets = train_test_split(
            texts, targets, test_size=test_size, random_state=42, stratify=targets
        )
        
        return TextDataset(
            name=config['name'],
            train_texts=train_texts,
            test_texts=test_texts,
            train_targets=train_targets,
            test_targets=test_targets,
            config=config
        )
    
    def _load_adult_income_dataset(self) -> TabularDataset:
        """Load Adult Income dataset from UCI"""
        try:
            # Load from UCI repository
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            column_names = [
                'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
                'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
                'hours-per-week', 'native-country', 'income'
            ]
            
            self.logger.info("Loading Adult Income dataset from UCI...")
            df = pd.read_csv(url, names=column_names, skipinitialspace=True)
            
            # Clean the data
            df = df.replace('?', np.nan)
            df = df.dropna()
            
            # Convert target to binary
            df['income'] = (df['income'] == '>50K').astype(int)
            
            # Select features for simplicity (numerical + categorical)
            feature_columns = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
            X = df[feature_columns]
            y = df['income']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            self.logger.info(f"Adult Income dataset loaded: {len(X_train)} train, {len(X_test)} test samples")
            
            return TabularDataset(
                name='adult_income',
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                feature_names=feature_columns,
                config={'source': 'uci', 'description': 'Adult Income dataset'}
            )
            
        except Exception as e:
            self.logger.error(f"Could not load Adult Income from UCI: {e}")
            raise
    
    def _load_compas_dataset(self) -> TabularDataset:
        """Load COMPAS dataset from online source"""
        try:
            # Load COMPAS dataset from GitHub repository
            url = "https://raw.githubusercontent.com/propublica/compas-analysis/master/compas-scores-two-years.csv"
            
            self.logger.info("Loading COMPAS dataset from ProPublica...")
            df = pd.read_csv(url)
            
            # Select relevant features for recidivism prediction
            feature_columns = ['age', 'priors_count', 'days_b_screening_arrest']
            target_column = 'two_year_recid'
            
            # Check if required columns exist
            if not all(col in df.columns for col in feature_columns + [target_column]):
                # Use available numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                if target_column in df.columns:
                    available_features = [col for col in numeric_cols if col != target_column]
                    feature_columns = available_features[:5]  # Use first 5 numeric features
                else:
                    raise ValueError(f"Target column '{target_column}' not found in COMPAS dataset")
            
            X = df[feature_columns]
            y = df[target_column]
            
            # Handle missing values
            X = X.fillna(X.mean())
            y = y.fillna(0)  # Assume no recidivism for missing values
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            self.logger.info(f"COMPAS dataset loaded: {len(X_train)} train, {len(X_test)} test samples")
            
            return TabularDataset(
                name='compas',
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                feature_names=list(X.columns),
                config={'source': 'propublica', 'description': 'COMPAS dataset'}
            )
            
        except Exception as e:
            self.logger.error(f"Could not load COMPAS dataset: {e}")
            raise
    
    def _load_mnist_dataset(self) -> ImageDataset:
        """Load MNIST dataset from torchvision"""
        try:
            import torchvision
            from torchvision import transforms
            
            self.logger.info("Loading MNIST dataset from torchvision...")
            
            # Define transforms
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization
            ])
            
            # Load MNIST
            train_dataset = torchvision.datasets.MNIST(
                root='./data', train=True, download=True, transform=transform
            )
            test_dataset = torchvision.datasets.MNIST(
                root='./data', train=False, download=True, transform=transform
            )
            
            # Convert to numpy arrays for compatibility
            X_train = []
            y_train = []
            for i in range(min(1000, len(train_dataset))):  # Limit for memory
                img, label = train_dataset[i]
                X_train.append(img.numpy())
                y_train.append(label)
            
            X_test = []
            y_test = []
            for i in range(min(200, len(test_dataset))):  # Limit for memory
                img, label = test_dataset[i]
                X_test.append(img.numpy())
                y_test.append(label)
            
            X_train = np.array(X_train)
            X_test = np.array(X_test)
            y_train = np.array(y_train)
            y_test = np.array(y_test)
            
            self.logger.info(f"MNIST dataset loaded: {len(X_train)} train, {len(X_test)} test samples")
            
            return ImageDataset(
                name='mnist',
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                config={'source': 'torchvision', 'description': 'MNIST dataset'}
            )
            
        except Exception as e:
            self.logger.error(f"Could not load MNIST dataset: {e}")
            raise
    
    def _load_imdb_dataset(self) -> TextDataset:
        """Load IMDB dataset from Hugging Face"""
        try:
            from datasets import load_dataset
            
            self.logger.info("Loading IMDB dataset from Hugging Face...")
            
            # Load IMDB dataset from Hugging Face
            dataset = load_dataset('imdb')
            
            # Extract train and test data
            train_data = dataset['train']
            test_data = dataset['test']
            
            # Build balanced subsets to ensure at least two classes
            def build_balanced_subset(split, total, seed=42):
                import random
                rng = random.Random(seed)
                texts = split['text']
                labels = split['label']
                idx0 = [i for i, y in enumerate(labels) if y == 0]
                idx1 = [i for i, y in enumerate(labels) if y == 1]
                n_per = max(1, total // 2)
                rng.shuffle(idx0)
                rng.shuffle(idx1)
                sel0 = idx0[:n_per]
                sel1 = idx1[:n_per]
                sel = sel0 + sel1
                rng.shuffle(sel)
                sel = sel[: 2 * n_per]
                return [texts[i] for i in sel], [labels[i] for i in sel]
            
            train_texts, train_targets = build_balanced_subset(train_data, total=1000)
            test_texts, test_targets = build_balanced_subset(test_data, total=200)
            
            self.logger.info(f"IMDB dataset loaded: {len(train_texts)} train, {len(test_texts)} test samples")
            
            return TextDataset(
                name='imdb',
                train_texts=train_texts,
                test_texts=test_texts,
                train_targets=train_targets,
                test_targets=test_targets,
                config={'source': 'huggingface', 'description': 'IMDB Movie Reviews dataset'}
            )
            
        except Exception as e:
            self.logger.error(f"Could not load IMDB dataset: {e}")
            raise
    
 