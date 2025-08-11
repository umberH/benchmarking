#!/usr/bin/env python3
"""
Enhanced Data Splitting Utilities for XAI Benchmarking Framework

This module provides comprehensive data splitting strategies including:
- Stratified Splits (maintains class distribution)
- Time-based Splits (for temporal data)
- Cross-validation (K-fold, stratified K-fold, Leave-one-out)
- Custom Split Strategies (domain-specific splits)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import warnings
from datetime import datetime, timedelta

try:
    from sklearn.model_selection import (
        train_test_split, StratifiedShuffleSplit, TimeSeriesSplit,
        KFold, StratifiedKFold, LeaveOneOut, GroupShuffleSplit,
        GroupKFold, StratifiedGroupKFold, ShuffleSplit
    )
    from sklearn.preprocessing import LabelEncoder
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available. Some splitting methods may not work.")


class DataSplitter:
    """
    Comprehensive data splitting utility for XAI benchmarking
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize DataSplitter with configuration
        
        Args:
            config: Configuration dictionary for data splitting
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default splitting configuration
        self.default_config = {
            'test_size': 0.2,
            'validation_size': 0.1,
            'random_state': 42,
            'stratify': True,
            'shuffle': True,
            'n_splits': 5,
            'n_repeats': 3,
            'time_column': None,
            'group_column': None,
            'min_samples_per_class': 2,
            'max_imbalance_ratio': 0.3
        }
        
        # Update with provided config
        if 'data_splitting' in self.config:
            self.default_config.update(self.config['data_splitting'])
    
    def split_data(self, X: Union[np.ndarray, pd.DataFrame], 
                   y: Union[np.ndarray, pd.Series],
                   split_strategy: str = "stratified",
                   **kwargs) -> Dict[str, Any]:
        """
        Split data according to specified strategy
        
        Args:
            X: Feature data
            y: Target data
            split_strategy: Splitting strategy to use
            **kwargs: Additional arguments for specific strategies
            
        Returns:
            Dictionary containing split indices and metadata
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn is required for data splitting")
        
        # Convert to numpy arrays if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
        
        # Validate inputs
        self._validate_inputs(X, y)
        
        # Apply splitting strategy
        if split_strategy == "stratified":
            return self._stratified_split(X, y, **kwargs)
        elif split_strategy == "time_based":
            return self._time_based_split(X, y, **kwargs)
        elif split_strategy == "cross_validation":
            return self._cross_validation_split(X, y, **kwargs)
        elif split_strategy == "custom":
            return self._custom_split(X, y, **kwargs)
        elif split_strategy == "group_based":
            return self._group_based_split(X, y, **kwargs)
        elif split_strategy == "holdout":
            return self._holdout_split(X, y, **kwargs)
        else:
            raise ValueError(f"Unknown split strategy: {split_strategy}")
    
    def _validate_inputs(self, X: np.ndarray, y: np.ndarray):
        """Validate input data"""
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        
        if len(X) < 10:
            self.logger.warning("Very small dataset detected. Some splitting strategies may not work well.")
        
        # Check for class balance
        unique_classes, counts = np.unique(y, return_counts=True)
        if len(unique_classes) < 2:
            raise ValueError("At least 2 classes required for splitting")
        
        # Check for severe class imbalance
        min_count = np.min(counts)
        max_count = np.max(counts)
        imbalance_ratio = min_count / max_count
        
        if imbalance_ratio < self.default_config['max_imbalance_ratio']:
            self.logger.warning(f"Severe class imbalance detected (ratio: {imbalance_ratio:.3f})")
        
        # Check minimum samples per class
        if min_count < self.default_config['min_samples_per_class']:
            self.logger.warning(f"Some classes have fewer than {self.default_config['min_samples_per_class']} samples")
    
    def _stratified_split(self, X: np.ndarray, y: np.ndarray, 
                         test_size: float = None, validation_size: float = None,
                         n_splits: int = None, **kwargs) -> Dict[str, Any]:
        """
        Stratified split maintaining class distribution
        
        Args:
            X: Feature data
            y: Target data
            test_size: Proportion of test set
            validation_size: Proportion of validation set
            n_splits: Number of splits for cross-validation
            
        Returns:
            Dictionary with split indices and metadata
        """
        test_size = test_size or self.default_config['test_size']
        validation_size = validation_size or self.default_config['validation_size']
        n_splits = n_splits or self.default_config['n_splits']
        
        # Single stratified split
        if validation_size == 0:
            train_idx, test_idx = train_test_split(
                np.arange(len(X)), 
                test_size=test_size,
                random_state=self.default_config['random_state'],
                stratify=y,
                shuffle=self.default_config['shuffle']
            )
            
            return {
                'split_type': 'stratified_single',
                'train_indices': train_idx,
                'test_indices': test_idx,
                'validation_indices': None,
                'metadata': {
                    'test_size': test_size,
                    'train_samples': len(train_idx),
                    'test_samples': len(test_idx),
                    'class_distribution_train': self._get_class_distribution(y[train_idx]),
                    'class_distribution_test': self._get_class_distribution(y[test_idx])
                }
            }
        
        # Train/validation/test split
        else:
            # First split: train+val vs test
            train_val_idx, test_idx = train_test_split(
                np.arange(len(X)),
                test_size=test_size,
                random_state=self.default_config['random_state'],
                stratify=y,
                shuffle=self.default_config['shuffle']
            )
            
            # Second split: train vs validation
            val_size_adjusted = validation_size / (1 - test_size)
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_size_adjusted,
                random_state=self.default_config['random_state'],
                stratify=y[train_val_idx],
                shuffle=self.default_config['shuffle']
            )
            
            return {
                'split_type': 'stratified_tvt',
                'train_indices': train_idx,
                'validation_indices': val_idx,
                'test_indices': test_idx,
                'metadata': {
                    'test_size': test_size,
                    'validation_size': validation_size,
                    'train_samples': len(train_idx),
                    'validation_samples': len(val_idx),
                    'test_samples': len(test_idx),
                    'class_distribution_train': self._get_class_distribution(y[train_idx]),
                    'class_distribution_validation': self._get_class_distribution(y[val_idx]),
                    'class_distribution_test': self._get_class_distribution(y[test_idx])
                }
            }
    
    def _time_based_split(self, X: np.ndarray, y: np.ndarray,
                         time_column: str = None, test_size: float = None,
                         gap: int = 0, **kwargs) -> Dict[str, Any]:
        """
        Time-based split for temporal data
        
        Args:
            X: Feature data (must be DataFrame with time column)
            y: Target data
            time_column: Column name containing timestamps
            test_size: Proportion of test set
            gap: Gap between train and test sets (in time units)
            
        Returns:
            Dictionary with split indices and metadata
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a DataFrame for time-based splitting")
        
        time_column = time_column or self.default_config['time_column']
        if time_column is None or time_column not in X.columns:
            raise ValueError(f"Time column '{time_column}' not found in data")
        
        test_size = test_size or self.default_config['test_size']
        
        # Sort by time
        X_sorted = X.sort_values(time_column).reset_index(drop=True)
        y_sorted = y[X_sorted.index]
        
        # Calculate split point
        split_idx = int(len(X_sorted) * (1 - test_size))
        
        # Apply gap if specified
        if gap > 0:
            split_idx = max(0, split_idx - gap)
        
        train_idx = np.arange(split_idx)
        test_idx = np.arange(split_idx, len(X_sorted))
        
        # Map back to original indices
        train_idx_original = X_sorted.iloc[train_idx].index.values
        test_idx_original = X_sorted.iloc[test_idx].index.values
        
        return {
            'split_type': 'time_based',
            'train_indices': train_idx_original,
            'test_indices': test_idx_original,
            'validation_indices': None,
            'metadata': {
                'test_size': test_size,
                'gap': gap,
                'time_column': time_column,
                'train_samples': len(train_idx_original),
                'test_samples': len(test_idx_original),
                'train_time_range': (X_sorted.iloc[0][time_column], X_sorted.iloc[split_idx-1][time_column]),
                'test_time_range': (X_sorted.iloc[split_idx][time_column], X_sorted.iloc[-1][time_column])
            }
        }
    
    def _cross_validation_split(self, X: np.ndarray, y: np.ndarray,
                               cv_type: str = "stratified_kfold", n_splits: int = None,
                               n_repeats: int = None, **kwargs) -> Dict[str, Any]:
        """
        Cross-validation splits
        
        Args:
            X: Feature data
            y: Target data
            cv_type: Type of cross-validation
            n_splits: Number of folds
            n_repeats: Number of repeats (for repeated CV)
            
        Returns:
            Dictionary with CV fold indices and metadata
        """
        n_splits = n_splits or self.default_config['n_splits']
        n_repeats = n_repeats or self.default_config['n_repeats']
        
        if cv_type == "stratified_kfold":
            cv = StratifiedKFold(n_splits=n_splits, shuffle=self.default_config['shuffle'],
                               random_state=self.default_config['random_state'])
        elif cv_type == "kfold":
            cv = KFold(n_splits=n_splits, shuffle=self.default_config['shuffle'],
                      random_state=self.default_config['random_state'])
        elif cv_type == "leave_one_out":
            cv = LeaveOneOut()
        elif cv_type == "stratified_shuffle_split":
            cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=self.default_config['test_size'],
                                      random_state=self.default_config['random_state'])
        else:
            raise ValueError(f"Unknown CV type: {cv_type}")
        
        # Generate CV splits
        cv_splits = []
        for train_idx, test_idx in cv.split(X, y):
            cv_splits.append({
                'train_indices': train_idx,
                'test_indices': test_idx,
                'train_samples': len(train_idx),
                'test_samples': len(test_idx),
                'class_distribution_train': self._get_class_distribution(y[train_idx]),
                'class_distribution_test': self._get_class_distribution(y[test_idx])
            })
        
        return {
            'split_type': f'cross_validation_{cv_type}',
            'cv_splits': cv_splits,
            'n_splits': len(cv_splits),
            'metadata': {
                'cv_type': cv_type,
                'n_splits': n_splits,
                'n_repeats': n_repeats,
                'shuffle': self.default_config['shuffle'],
                'random_state': self.default_config['random_state']
            }
        }
    
    def _group_based_split(self, X: np.ndarray, y: np.ndarray,
                          groups: np.ndarray = None, group_column: str = None,
                          test_size: float = None, **kwargs) -> Dict[str, Any]:
        """
        Group-based split (e.g., by patient, subject, etc.)
        
        Args:
            X: Feature data
            y: Target data
            groups: Group labels
            group_column: Column name containing group labels
            test_size: Proportion of test set
            
        Returns:
            Dictionary with split indices and metadata
        """
        if groups is None and group_column is None:
            group_column = self.default_config['group_column']
        
        if groups is None and group_column is not None:
            if not isinstance(X, pd.DataFrame):
                raise ValueError("X must be a DataFrame for group-based splitting")
            if group_column not in X.columns:
                raise ValueError(f"Group column '{group_column}' not found in data")
            groups = X[group_column].values
        
        if groups is None:
            raise ValueError("Groups must be provided for group-based splitting")
        
        test_size = test_size or self.default_config['test_size']
        
        # Use GroupShuffleSplit
        gss = GroupShuffleSplit(n_splits=1, test_size=test_size,
                               random_state=self.default_config['random_state'])
        
        train_idx, test_idx = next(gss.split(X, y, groups))
        
        return {
            'split_type': 'group_based',
            'train_indices': train_idx,
            'test_indices': test_idx,
            'validation_indices': None,
            'metadata': {
                'test_size': test_size,
                'group_column': group_column,
                'train_samples': len(train_idx),
                'test_samples': len(test_idx),
                'unique_groups_train': len(np.unique(groups[train_idx])),
                'unique_groups_test': len(np.unique(groups[test_idx])),
                'class_distribution_train': self._get_class_distribution(y[train_idx]),
                'class_distribution_test': self._get_class_distribution(y[test_idx])
            }
        }
    
    def _custom_split(self, X: np.ndarray, y: np.ndarray,
                     custom_indices: Dict[str, np.ndarray] = None,
                     split_function: callable = None, **kwargs) -> Dict[str, Any]:
        """
        Custom split using provided indices or function
        
        Args:
            X: Feature data
            y: Target data
            custom_indices: Dictionary with 'train', 'test', 'validation' indices
            split_function: Custom function to generate splits
            
        Returns:
            Dictionary with split indices and metadata
        """
        if custom_indices is not None:
            # Use provided indices
            train_idx = custom_indices.get('train', np.array([]))
            test_idx = custom_indices.get('test', np.array([]))
            val_idx = custom_indices.get('validation', None)
            
        elif split_function is not None:
            # Use custom function
            result = split_function(X, y, **kwargs)
            if isinstance(result, dict):
                train_idx = result.get('train_indices', np.array([]))
                test_idx = result.get('test_indices', np.array([]))
                val_idx = result.get('validation_indices', None)
            else:
                raise ValueError("Custom split function must return a dictionary")
        else:
            raise ValueError("Either custom_indices or split_function must be provided")
        
        return {
            'split_type': 'custom',
            'train_indices': train_idx,
            'test_indices': test_idx,
            'validation_indices': val_idx,
            'metadata': {
                'train_samples': len(train_idx),
                'test_samples': len(test_idx),
                'validation_samples': len(val_idx) if val_idx is not None else 0,
                'class_distribution_train': self._get_class_distribution(y[train_idx]),
                'class_distribution_test': self._get_class_distribution(y[test_idx]),
                'class_distribution_validation': self._get_class_distribution(y[val_idx]) if val_idx is not None else None
            }
        }
    
    def _holdout_split(self, X: np.ndarray, y: np.ndarray,
                      test_size: float = None, validation_size: float = None,
                      **kwargs) -> Dict[str, Any]:
        """
        Simple holdout split (no stratification)
        
        Args:
            X: Feature data
            y: Target data
            test_size: Proportion of test set
            validation_size: Proportion of validation set
            
        Returns:
            Dictionary with split indices and metadata
        """
        test_size = test_size or self.default_config['test_size']
        validation_size = validation_size or self.default_config['validation_size']
        
        if validation_size == 0:
            train_idx, test_idx = train_test_split(
                np.arange(len(X)),
                test_size=test_size,
                random_state=self.default_config['random_state'],
                shuffle=self.default_config['shuffle']
            )
            
            return {
                'split_type': 'holdout_single',
                'train_indices': train_idx,
                'test_indices': test_idx,
                'validation_indices': None,
                'metadata': {
                    'test_size': test_size,
                    'train_samples': len(train_idx),
                    'test_samples': len(test_idx),
                    'class_distribution_train': self._get_class_distribution(y[train_idx]),
                    'class_distribution_test': self._get_class_distribution(y[test_idx])
                }
            }
        else:
            # Train/validation/test split
            train_val_idx, test_idx = train_test_split(
                np.arange(len(X)),
                test_size=test_size,
                random_state=self.default_config['random_state'],
                shuffle=self.default_config['shuffle']
            )
            
            val_size_adjusted = validation_size / (1 - test_size)
            train_idx, val_idx = train_test_split(
                train_val_idx,
                test_size=val_size_adjusted,
                random_state=self.default_config['random_state'],
                shuffle=self.default_config['shuffle']
            )
            
            return {
                'split_type': 'holdout_tvt',
                'train_indices': train_idx,
                'validation_indices': val_idx,
                'test_indices': test_idx,
                'metadata': {
                    'test_size': test_size,
                    'validation_size': validation_size,
                    'train_samples': len(train_idx),
                    'validation_samples': len(val_idx),
                    'test_samples': len(test_idx),
                    'class_distribution_train': self._get_class_distribution(y[train_idx]),
                    'class_distribution_validation': self._get_class_distribution(y[val_idx]),
                    'class_distribution_test': self._get_class_distribution(y[test_idx])
                }
            }
    
    def _get_class_distribution(self, y: np.ndarray) -> Dict[str, int]:
        """Get class distribution as dictionary"""
        unique_classes, counts = np.unique(y, return_counts=True)
        return {f"class_{cls}": count for cls, count in zip(unique_classes, counts)}
    
    def get_split_summary(self, split_result: Dict[str, Any]) -> str:
        """Generate a human-readable summary of the split"""
        split_type = split_result['split_type']
        metadata = split_result['metadata']
        
        summary = f"Data Split Summary:\n"
        summary += f"Type: {split_type}\n"
        
        if 'cv_splits' in split_result:
            # Cross-validation
            summary += f"Number of folds: {len(split_result['cv_splits'])}\n"
            avg_train = np.mean([split['train_samples'] for split in split_result['cv_splits']])
            avg_test = np.mean([split['test_samples'] for split in split_result['cv_splits']])
            summary += f"Average train samples per fold: {avg_train:.0f}\n"
            summary += f"Average test samples per fold: {avg_test:.0f}\n"
        else:
            # Single split
            summary += f"Train samples: {metadata.get('train_samples', 'N/A')}\n"
            summary += f"Test samples: {metadata.get('test_samples', 'N/A')}\n"
            if metadata.get('validation_samples'):
                summary += f"Validation samples: {metadata['validation_samples']}\n"
        
        return summary


def create_split_config(split_strategy: str, **kwargs) -> Dict[str, Any]:
    """
    Create a configuration dictionary for data splitting
    
    Args:
        split_strategy: The splitting strategy to use
        **kwargs: Strategy-specific parameters
        
    Returns:
        Configuration dictionary
    """
    config = {
        'split_strategy': split_strategy,
        'parameters': kwargs
    }
    
    return config


def validate_split_result(split_result: Dict[str, Any]) -> bool:
    """
    Validate a split result
    
    Args:
        split_result: Result from DataSplitter.split_data()
        
    Returns:
        True if valid, False otherwise
    """
    required_keys = ['split_type', 'metadata']
    
    # Check required keys
    for key in required_keys:
        if key not in split_result:
            return False
    
    # Check for CV splits or single split
    if 'cv_splits' in split_result:
        if not isinstance(split_result['cv_splits'], list):
            return False
        for split in split_result['cv_splits']:
            if not all(key in split for key in ['train_indices', 'test_indices']):
                return False
    else:
        if not all(key in split_result for key in ['train_indices', 'test_indices']):
            return False
    
    return True 