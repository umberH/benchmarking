"""
Data validation utilities for XAI benchmarking
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path
import warnings
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


class DataValidator:
    """
    Comprehensive data validation for XAI benchmarking datasets
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize data validator
        
        Args:
            config: Configuration dictionary with validation thresholds
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default validation thresholds
        self.thresholds = {
            'class_imbalance_ratio': 0.2,  # Minimum ratio of minority to majority class
            'min_dataset_size': 100,  # Minimum samples for reliable evaluation
            'max_missing_ratio': 0.3,  # Maximum allowed missing values ratio
            'outlier_threshold': 3.0,  # Z-score threshold for outlier detection
            'feature_correlation_threshold': 0.95,  # Maximum correlation between features
            'min_features': 2,  # Minimum number of features
            'max_features': 1000,  # Maximum number of features (for performance)
        }
        
        # Update with config if provided
        if 'validation' in self.config:
            self.thresholds.update(self.config['validation'])
    
    def validate_dataset(self, dataset, dataset_name: str) -> Dict[str, Any]:
        """
        Comprehensive dataset validation with robust type conversion
        """
        self.logger.info(f"Validating dataset: {dataset_name}")
        validation_results = {
            'dataset_name': dataset_name,
            'is_valid': True,
            'warnings': [],
            'errors': [],
            'recommendations': [],
            'validation_details': {}
        }
        try:
            # Get dataset info
            dataset_info = dataset.get_info()
            validation_results['dataset_info'] = dataset_info
            # Get data
            X_train, X_test, y_train, y_test = dataset.get_data()
            # Defensive: ensure correct types for validation
            dtype = dataset_info.get('type', '').lower()
            import numpy as np
            import pandas as pd
            if dtype == 'image':
                # Ensure np.ndarray, never convert to DataFrame
                if not isinstance(X_train, np.ndarray):
                    X_train = np.array(X_train)
                if not isinstance(X_test, np.ndarray):
                    X_test = np.array(X_test)
                # Defensive: if 4D, keep as is; if 3D, add channel dim
                if X_train.ndim == 3:
                    X_train = X_train[:, None, :, :]
                if X_test.ndim == 3:
                    X_test = X_test[:, None, :, :]
                # Robust check: image data should be 4D (N, C, H, W)
                if X_train.ndim != 4:
                    validation_results['errors'].append(f"Image data must be 4D (N, C, H, W), got shape {X_train.shape}")
                if X_test.ndim != 4:
                    validation_results['errors'].append(f"Image test data must be 4D (N, C, H, W), got shape {X_test.shape}")
                # Do NOT convert image data to DataFrame or enforce 2D
            elif dtype == 'text':
                # Ensure list of str
                if not isinstance(X_train, list):
                    X_train = list(X_train)
                if not isinstance(X_test, list):
                    X_test = list(X_test)
            elif dtype == 'tabular':
                # Ensure DataFrame or np.ndarray, but only if 2D
                if not (isinstance(X_train, pd.DataFrame) or (isinstance(X_train, np.ndarray) and X_train.ndim == 2)):
                    X_train = pd.DataFrame(X_train)
                if not (isinstance(X_test, pd.DataFrame) or (isinstance(X_test, np.ndarray) and X_test.ndim == 2)):
                    X_test = pd.DataFrame(X_test)
            # Run general validation checks
            self._check_dataset_size(X_train, X_test, y_train, y_test, validation_results)
            self._check_class_balance(y_train, y_test, validation_results)
            self._check_data_quality(X_train, X_test, validation_results)
            # Only run tabular-specific checks for tabular data
            if dtype == 'tabular':
                self._check_feature_distributions(X_train, validation_results)
                self._check_feature_correlations(X_train, validation_results)
                self._check_outliers(X_train, validation_results)
            # Overall validation status
            validation_results['is_valid'] = len(validation_results['errors']) == 0
            # Generate recommendations
            self._generate_recommendations(validation_results)
        except Exception as e:
            self.logger.error(f"Error validating dataset {dataset_name}: {e}")
            validation_results['errors'].append(f"Validation failed: {str(e)}")
            validation_results['is_valid'] = False
        return validation_results
    
    def _check_dataset_size(self, X_train, X_test, y_train, y_test, results: Dict[str, Any]):
        """Check if dataset size meets minimum requirements"""
        n_train = len(X_train)
        n_test = len(X_test)
        n_total = n_train + n_test
        
        results['validation_details']['dataset_size'] = {
            'n_train': n_train,
            'n_test': n_test,
            'n_total': n_total,
            'train_test_ratio': n_train / n_total if n_total > 0 else 0
        }
        
        if n_total < self.thresholds['min_dataset_size']:
            results['errors'].append(
                f"Dataset too small: {n_total} samples < {self.thresholds['min_dataset_size']} minimum"
            )
        elif n_total < self.thresholds['min_dataset_size'] * 2:
            results['warnings'].append(
                f"Dataset size ({n_total}) is close to minimum threshold ({self.thresholds['min_dataset_size']})"
            )
        
        if n_test < 10:
            results['warnings'].append(f"Test set very small: {n_test} samples")
    
    def _check_class_balance(self, y_train, y_test, results: Dict[str, Any]):
        """Check for class imbalance in target variables"""
        # Handle different data types
        if hasattr(y_train, 'values'):
            y_train = y_train.values
        if hasattr(y_test, 'values'):
            y_test = y_test.values
        
        # Get unique classes and counts
        train_classes, train_counts = np.unique(y_train, return_counts=True)
        test_classes, test_counts = np.unique(y_test, return_counts=True)
        
        # Calculate imbalance ratios
        train_imbalance_ratio = np.min(train_counts) / np.max(train_counts)
        test_imbalance_ratio = np.min(test_counts) / np.max(test_counts)
        
        results['validation_details']['class_balance'] = {
            'train_classes': train_classes.tolist(),
            'train_counts': train_counts.tolist(),
            'test_classes': test_classes.tolist(),
            'test_counts': test_counts.tolist(),
            'train_imbalance_ratio': float(train_imbalance_ratio),
            'test_imbalance_ratio': float(test_imbalance_ratio)
        }
        
        # Check for severe imbalance
        if train_imbalance_ratio < self.thresholds['class_imbalance_ratio']:
            results['warnings'].append(
                f"Severe class imbalance in training set: ratio {train_imbalance_ratio:.3f} < {self.thresholds['class_imbalance_ratio']}"
            )
        
        if test_imbalance_ratio < self.thresholds['class_imbalance_ratio']:
            results['warnings'].append(
                f"Severe class imbalance in test set: ratio {test_imbalance_ratio:.3f} < {self.thresholds['class_imbalance_ratio']}"
            )
        
        # Check for missing classes in test set
        missing_classes = set(train_classes) - set(test_classes)
        if missing_classes:
            results['errors'].append(f"Missing classes in test set: {missing_classes}")
    
    def _check_data_quality(self, X_train, X_test, results: Dict[str, Any]):
        """Check for data quality issues like missing values, with robust error handling."""
        try:
            # Handle different data types
            if isinstance(X_train, np.ndarray):
                # For image/text data, check for NaN/Inf values, handle None safely
                train_has_nan = False
                train_has_inf = False
                test_has_nan = False
                test_has_inf = False
                if X_train is not None:
                    try:
                        train_has_nan = np.isnan(X_train).any()
                        train_has_inf = np.isinf(X_train).any()
                    except Exception:
                        train_has_nan = False
                        train_has_inf = False
                if X_test is not None:
                    try:
                        test_has_nan = np.isnan(X_test).any()
                        test_has_inf = np.isinf(X_test).any()
                    except Exception:
                        test_has_nan = False
                        test_has_inf = False
                results['validation_details']['data_quality'] = {
                    'train_has_nan': bool(train_has_nan),
                    'train_has_inf': bool(train_has_inf),
                    'test_has_nan': bool(test_has_nan),
                    'test_has_inf': bool(test_has_inf),
                    'data_type': 'array',
                    'train_missing_ratio': None,
                    'test_missing_ratio': None
                }
                if train_has_nan or test_has_nan:
                    results['warnings'].append("NaN values detected in data")
                if train_has_inf or test_has_inf:
                    results['warnings'].append("Infinite values detected in data")
            else:
                # For tabular data, convert to DataFrame and check missing values
                if not isinstance(X_train, pd.DataFrame):
                    X_train = pd.DataFrame(X_train)
                if not isinstance(X_test, pd.DataFrame):
                    X_test = pd.DataFrame(X_test)
                # Check missing values
                train_missing = X_train.isnull().sum()
                test_missing = X_test.isnull().sum()
                # Defensive: avoid division by zero
                if X_train.shape[0] == 0 or X_train.shape[1] == 0:
                    raise ValueError("Training data is empty or malformed (zero rows or columns).")
                if X_test.shape[0] == 0 or X_test.shape[1] == 0:
                    raise ValueError("Test data is empty or malformed (zero rows or columns).")
                train_missing_ratio = train_missing.sum() / (X_train.shape[0] * X_train.shape[1])
                test_missing_ratio = test_missing.sum() / (X_test.shape[0] * X_test.shape[1])
                results['validation_details']['data_quality'] = {
                    'train_missing_ratio': float(train_missing_ratio) if train_missing_ratio is not None else 0.0,
                    'test_missing_ratio': float(test_missing_ratio) if test_missing_ratio is not None else 0.0,
                    'train_missing_by_feature': train_missing.to_dict(),
                    'test_missing_by_feature': test_missing.to_dict(),
                    'data_type': 'tabular',
                    'train_has_nan': None,
                    'train_has_inf': None,
                    'test_has_nan': None,
                    'test_has_inf': None
                }
                if train_missing_ratio is not None and train_missing_ratio > self.thresholds['max_missing_ratio']:
                    results['errors'].append(
                        f"Too many missing values in training set: {train_missing_ratio:.3f} > {self.thresholds['max_missing_ratio']}"
                    )
                if test_missing_ratio is not None and test_missing_ratio > self.thresholds['max_missing_ratio']:
                    results['errors'].append(
                        f"Too many missing values in test set: {test_missing_ratio:.3f} > {self.thresholds['max_missing_ratio']}"
                    )
                # Check for infinite values
                train_infinite = np.isinf(X_train.select_dtypes(include=[np.number])).sum().sum()
                test_infinite = np.isinf(X_test.select_dtypes(include=[np.number])).sum().sum()
                if train_infinite > 0:
                    results['warnings'].append(f"Found {train_infinite} infinite values in training set")
                if test_infinite > 0:
                    results['warnings'].append(f"Found {test_infinite} infinite values in test set")
        except Exception as e:
            # Always set a data_quality entry, even on error
            results['validation_details']['data_quality'] = {
                'train_missing_ratio': None,
                'test_missing_ratio': None,
                'train_missing_by_feature': None,
                'test_missing_by_feature': None,
                'data_type': 'error',
                'train_has_nan': None,
                'train_has_inf': None,
                'test_has_nan': None,
                'test_has_inf': None
            }
            results['errors'].append(f"Validation failed: {str(e)}")
            self.logger.error(f"Error during data quality check: {e}")
    
    def _check_feature_distributions(self, X_train, results: Dict[str, Any]):
        """Check feature distributions for potential issues"""
        if isinstance(X_train, np.ndarray):
            # For image/text data, check basic statistics
            if X_train.ndim > 2:
                # Flatten for analysis
                X_flat = X_train.reshape(X_train.shape[0], -1)
            else:
                X_flat = X_train
            
            # Calculate basic statistics
            mean_val = np.mean(X_flat)
            std_val = np.std(X_flat)
            min_val = np.min(X_flat)
            max_val = np.max(X_flat)
            
            # Check for constant features
            feature_stds = np.std(X_flat, axis=0)
            constant_features = np.sum(feature_stds == 0)
            
            distribution_info = {
                'n_features': X_flat.shape[1],
                'mean_value': float(mean_val),
                'std_value': float(std_val),
                'min_value': float(min_val),
                'max_value': float(max_val),
                'constant_features': int(constant_features),
                'data_type': 'array'
            }
            
            if constant_features > 0:
                results['warnings'].append(f"Constant features detected: {constant_features} features")
        else:
            # For tabular data, convert to DataFrame and check distributions
            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train)
            
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns
            
            distribution_info = {}
            for col in numeric_cols:
                values = X_train[col].dropna()
                if len(values) > 0:
                    # Basic statistics
                    mean_val = values.mean()
                    std_val = values.std()
                    skew_val = stats.skew(values)
                    kurt_val = stats.kurtosis(values)
                    
                    distribution_info[col] = {
                        'mean': float(mean_val),
                        'std': float(std_val),
                        'skewness': float(skew_val),
                        'kurtosis': float(kurt_val),
                        'n_unique': int(values.nunique()),
                        'is_constant': bool(values.nunique() <= 1)
                    }
                    
                    # Check for constant features
                    if values.nunique() <= 1:
                        results['warnings'].append(f"Feature '{col}' is constant (no variance)")
                    
                    # Check for highly skewed distributions
                    if abs(skew_val) > 3:
                        results['warnings'].append(f"Feature '{col}' is highly skewed (skewness: {skew_val:.3f})")
        
        results['validation_details']['feature_distributions'] = distribution_info
    
    def _check_feature_correlations(self, X_train, results: Dict[str, Any]):
        """Check for highly correlated features"""
        if isinstance(X_train, np.ndarray):
            # For image/text data, skip correlation analysis
            results['validation_details']['feature_correlations'] = {
                'data_type': 'array',
                'correlation_analysis': 'skipped_for_array_data'
            }
        else:
            # For tabular data, check correlations
            if not isinstance(X_train, pd.DataFrame):
                X_train = pd.DataFrame(X_train)
            
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 1:
                # Calculate correlation matrix
                corr_matrix = X_train[numeric_cols].corr()
                
                # Find highly correlated feature pairs
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_val = abs(corr_matrix.iloc[i, j])
                        if corr_val > self.thresholds['feature_correlation_threshold']:
                            high_corr_pairs.append({
                                'feature1': corr_matrix.columns[i],
                                'feature2': corr_matrix.columns[j],
                                'correlation': float(corr_val)
                            })
                
                results['validation_details']['feature_correlations'] = {
                    'high_corr_pairs': high_corr_pairs,
                    'max_correlation': float(corr_matrix.abs().max().max()),
                    'data_type': 'tabular'
                }
                
                if high_corr_pairs:
                    results['warnings'].append(
                        f"Found {len(high_corr_pairs)} highly correlated feature pairs (|r| > {self.thresholds['feature_correlation_threshold']})"
                    )
    
    def _check_outliers(self, X_train, results: Dict[str, Any]):
        """Detect outliers using Z-score method"""
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        
        numeric_cols = X_train.select_dtypes(include=[np.number]).columns
        
        outlier_info = {}
        total_outliers = 0
        
        for col in numeric_cols:
            values = X_train[col].dropna()
            if len(values) > 0 and values.std() > 0:
                # Calculate Z-scores
                z_scores = np.abs((values - values.mean()) / values.std())
                outliers = z_scores > self.thresholds['outlier_threshold']
                
                n_outliers = outliers.sum()
                outlier_ratio = n_outliers / len(values)
                
                outlier_info[col] = {
                    'n_outliers': int(n_outliers),
                    'outlier_ratio': float(outlier_ratio),
                    'max_z_score': float(z_scores.max())
                }
                
                total_outliers += n_outliers
                
                if outlier_ratio > 0.1:  # More than 10% outliers
                    results['warnings'].append(
                        f"High outlier ratio in feature '{col}': {outlier_ratio:.3f}"
                    )
        
        results['validation_details']['outliers'] = {
            'by_feature': outlier_info,
            'total_outliers': total_outliers
        }
    
    def _generate_recommendations(self, results: Dict[str, Any]):
        """Generate recommendations based on validation results"""
        recommendations = []
        
        try:
            # Dataset size recommendations
            if 'dataset_size' in results['validation_details']:
                size_info = results['validation_details']['dataset_size']
                if size_info.get('n_total', 0) < self.thresholds['min_dataset_size'] * 2:
                    recommendations.append("Consider collecting more data or using data augmentation techniques")
            
            # Class balance recommendations
            if 'class_balance' in results['validation_details']:
                balance_info = results['validation_details']['class_balance']
                train_imbalance = balance_info.get('train_imbalance_ratio')
                if train_imbalance is not None and train_imbalance < 0.3:
                    recommendations.append("Consider using class balancing techniques (SMOTE, undersampling, etc.)")
                    recommendations.append("Use metrics like F1-score instead of accuracy for imbalanced datasets")
            
            # Data quality recommendations (only for tabular data)
            if 'data_quality' in results['validation_details']:
                quality_info = results['validation_details']['data_quality']
                if quality_info.get('data_type') == 'tabular':
                    train_missing_ratio = quality_info.get('train_missing_ratio')
                    if train_missing_ratio is not None and train_missing_ratio > 0.1:
                        recommendations.append("Implement proper missing value imputation strategies")
                    if train_missing_ratio is not None and train_missing_ratio > 0.3:
                        recommendations.append("Consider removing features with too many missing values")
            
            # Feature recommendations (only for tabular data)
            if 'feature_distributions' in results['validation_details']:
                dist_info = results['validation_details']['feature_distributions']
                if isinstance(dist_info, dict) and dist_info.get('data_type') != 'array':
                    constant_features = [col for col, info in dist_info.items() 
                                       if isinstance(info, dict) and info.get('is_constant', False)]
                    if constant_features:
                        recommendations.append(f"Remove constant features: {constant_features}")
            
            if 'feature_correlations' in results['validation_details']:
                corr_info = results['validation_details']['feature_correlations']
                if isinstance(corr_info, dict) and corr_info.get('high_corr_pairs'):
                    recommendations.append("Consider feature selection to remove highly correlated features")
            
            # Outlier recommendations (only for tabular data)
            if 'outliers' in results['validation_details']:
                outlier_info = results['validation_details']['outliers']
                if isinstance(outlier_info, dict) and outlier_info.get('total_outliers', 0) > 0:
                    recommendations.append("Consider outlier detection and treatment strategies")
        
        except Exception as e:
            self.logger.warning(f"Error generating recommendations: {e}")
            recommendations.append("Unable to generate specific recommendations due to validation errors")
        
        results['recommendations'] = recommendations
    
    def generate_validation_report(self, validation_results: Dict[str, Any], output_dir: Path = None) -> str:
        """
        Generate a comprehensive validation report
        
        Args:
            validation_results: Results from validate_dataset()
            output_dir: Directory to save report files
            
        Returns:
            Formatted report string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"DATA VALIDATION REPORT: {validation_results['dataset_name']}")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Overall status
        status = "✅ VALID" if validation_results['is_valid'] else "❌ INVALID"
        report_lines.append(f"Overall Status: {status}")
        report_lines.append("")
        
        # Dataset info
        if 'dataset_info' in validation_results:
            info = validation_results['dataset_info']
            report_lines.append("Dataset Information:")
            report_lines.append(f"  - Type: {info.get('type', 'Unknown')}")
            report_lines.append(f"  - Features: {info.get('n_features', 'Unknown')}")
            report_lines.append(f"  - Classes: {info.get('n_classes', 'Unknown')}")
            report_lines.append("")
        
        # Validation details
        details = validation_results['validation_details']
        
        # Dataset size
        if 'dataset_size' in details:
            size_info = details['dataset_size']
            report_lines.append("Dataset Size:")
            report_lines.append(f"  - Training samples: {size_info['n_train']}")
            report_lines.append(f"  - Test samples: {size_info['n_test']}")
            report_lines.append(f"  - Total samples: {size_info['n_total']}")
            report_lines.append(f"  - Train/Test ratio: {size_info['train_test_ratio']:.3f}")
            report_lines.append("")
        
        # Class balance
        if 'class_balance' in details:
            balance_info = details['class_balance']
            report_lines.append("Class Balance:")
            report_lines.append(f"  - Training imbalance ratio: {balance_info['train_imbalance_ratio']:.3f}")
            report_lines.append(f"  - Test imbalance ratio: {balance_info['test_imbalance_ratio']:.3f}")
            report_lines.append("")
        
        # Data quality
        if 'data_quality' in details:
            quality_info = details['data_quality']
            report_lines.append("Data Quality:")
            train_ratio = quality_info.get('train_missing_ratio', 0.0)
            test_ratio = quality_info.get('test_missing_ratio', 0.0)
            train_ratio = train_ratio if train_ratio is not None else 0.0
            test_ratio = test_ratio if test_ratio is not None else 0.0
            report_lines.append(f"  - Training missing ratio: {train_ratio:.3f}")
            report_lines.append(f"  - Test missing ratio: {test_ratio:.3f}")
            report_lines.append("")
        
        # Warnings and errors
        if validation_results['warnings']:
            report_lines.append("⚠️  WARNINGS:")
            for warning in validation_results['warnings']:
                report_lines.append(f"  - {warning}")
            report_lines.append("")
        
        if validation_results['errors']:
            report_lines.append("❌ ERRORS:")
            for error in validation_results['errors']:
                report_lines.append(f"  - {error}")
            report_lines.append("")
        
        # Recommendations
        if validation_results['recommendations']:
            report_lines.append("💡 RECOMMENDATIONS:")
            for rec in validation_results['recommendations']:
                report_lines.append(f"  - {rec}")
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        # Save report if output directory provided
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            report_file = output_dir / f"validation_report_{validation_results['dataset_name']}.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            
            # Save detailed results as JSON
            json_file = output_dir / f"validation_details_{validation_results['dataset_name']}.json"
            import json
            with open(json_file, 'w') as f:
                json.dump(validation_results, f, indent=2, default=str)
        
        return report


def validate_all_datasets(datasets: Dict[str, Any], config: Dict[str, Any] = None, output_dir: Path = None) -> Dict[str, Any]:
    """
    Validate all datasets in a dataset dictionary
    
    Args:
        datasets: Dictionary of dataset objects
        config: Configuration dictionary
        output_dir: Directory to save validation reports
        
    Returns:
        Dictionary containing validation results for all datasets
    """
    validator = DataValidator(config)
    all_results = {}
    
    for dataset_name, dataset in datasets.items():
        results = validator.validate_dataset(dataset, dataset_name)
        all_results[dataset_name] = results
        
        # Print summary
        status = "✅" if results['is_valid'] else "❌"
        print(f"{status} {dataset_name}: {len(results['warnings'])} warnings, {len(results['errors'])} errors")
        
        # Generate report
        if output_dir:
            validator.generate_validation_report(results, output_dir)
    
    return all_results 