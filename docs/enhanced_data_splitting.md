# Enhanced Data Splitting for XAI Benchmarking Framework

## Overview

The Enhanced Data Splitting module provides comprehensive data splitting strategies for the XAI benchmarking framework. It supports various splitting approaches to handle different data characteristics and requirements.

## Features

### 1. Stratified Splits
- **Purpose**: Maintains class distribution across train/test splits
- **Use Cases**: Multi-class datasets, imbalanced datasets
- **Benefits**: Prevents bias in evaluation due to class imbalance

### 2. Time-based Splits
- **Purpose**: Handles temporal data and prevents data leakage
- **Use Cases**: Time series data, sequential data, financial data
- **Features**: Configurable gaps between train and test sets

### 3. Cross-validation
- **Types**: K-fold, Stratified K-fold, Leave-one-out, Stratified Shuffle Split
- **Use Cases**: Small datasets, model selection, hyperparameter tuning
- **Benefits**: More robust evaluation with limited data

### 4. Group-based Splits
- **Purpose**: Splits by groups (e.g., patients, subjects, users)
- **Use Cases**: Medical data, user behavior data, multi-subject studies
- **Benefits**: Prevents data leakage between related samples

### 5. Custom Split Strategies
- **Purpose**: Domain-specific splitting requirements
- **Use Cases**: Specialized evaluation protocols, external validation sets
- **Features**: Flexible configuration and custom functions

### 6. Holdout Splits
- **Purpose**: Simple train/test splits without stratification
- **Use Cases**: Large datasets, quick prototyping
- **Features**: Optional validation set support

## Configuration

### Basic Configuration

```yaml
experiment:
  data_splitting:
    # Default splitting strategy
    default_strategy: "stratified"
    
    # Split sizes
    test_size: 0.2
    validation_size: 0.1
    
    # General settings
    random_state: 42
    stratify: true
    shuffle: true
    
    # Cross-validation settings
    n_splits: 5
    n_repeats: 3
    
    # Time-based splitting
    time_column: null
    gap: 0
    
    # Group-based splitting
    group_column: null
    
    # Validation thresholds
    min_samples_per_class: 2
    max_imbalance_ratio: 0.3
```

### Strategy-Specific Configuration

```yaml
strategies:
  stratified:
    description: "Maintains class distribution across splits"
    supports_validation: true
    supports_cv: true
  time_based:
    description: "Time-based splits for temporal data"
    supports_validation: false
    supports_cv: false
    requires_time_column: true
  cross_validation:
    description: "K-fold cross-validation"
    supports_validation: false
    supports_cv: true
    cv_types:
      - "stratified_kfold"
      - "kfold"
      - "leave_one_out"
      - "stratified_shuffle_split"
  group_based:
    description: "Group-based splits (e.g., by patient, subject)"
    supports_validation: false
    supports_cv: false
    requires_group_column: true
  custom:
    description: "Custom split strategies"
    supports_validation: true
    supports_cv: true
  holdout:
    description: "Simple holdout split"
    supports_validation: true
    supports_cv: false
```

## Usage Examples

### Basic Usage

```python
from src.utils.data_splitting import DataSplitter
from src.utils.config import load_config

# Load configuration
config = load_config("configs/default_config.yaml")

# Initialize splitter
splitter = DataSplitter(config)

# Split data
result = splitter.split_data(X, y, "stratified", test_size=0.2)
```

### Stratified Splitting

```python
# Basic stratified split
result = splitter.split_data(X, y, "stratified", test_size=0.2)

# With validation set
result = splitter.split_data(X, y, "stratified", 
                           test_size=0.2, validation_size=0.1)
```

### Time-based Splitting

```python
# For temporal data
result = splitter.split_data(X_df, y, "time_based", 
                           time_column="timestamp", test_size=0.2)

# With gap between train and test
result = splitter.split_data(X_df, y, "time_based", 
                           time_column="timestamp", test_size=0.2, gap=30)
```

### Cross-validation

```python
# Stratified K-fold
result = splitter.split_data(X, y, "cross_validation", 
                           cv_type="stratified_kfold", n_splits=5)

# Leave-one-out
result = splitter.split_data(X, y, "cross_validation", 
                           cv_type="leave_one_out")
```

### Group-based Splitting

```python
# Split by groups
result = splitter.split_data(X_df, y, "group_based", 
                           group_column="patient_id", test_size=0.2)
```

### Custom Splitting

```python
# Using custom indices
custom_indices = {
    'train': np.arange(0, 700),
    'test': np.arange(700, 850),
    'validation': np.arange(850, 1000)
}
result = splitter.split_data(X, y, "custom", custom_indices=custom_indices)

# Using custom function
def my_split_function(X, y, **kwargs):
    # Custom splitting logic
    return {
        'train_indices': train_idx,
        'test_indices': test_idx,
        'validation_indices': val_idx
    }

result = splitter.split_data(X, y, "custom", split_function=my_split_function)
```

### Integration with DataManager

```python
from src.data.data_manager import DataManager

# Initialize data manager
data_manager = DataManager(config)

# Load dataset
datasets = data_manager.load_datasets()
dataset = datasets['adult_income']

# Apply splitting strategy
result = data_manager.split_dataset(dataset, "stratified", test_size=0.2)
```

## Output Format

All splitting methods return a standardized dictionary format:

```python
{
    'split_type': 'stratified_single',
    'train_indices': np.array([...]),
    'test_indices': np.array([...]),
    'validation_indices': None,  # or np.array([...])
    'metadata': {
        'test_size': 0.2,
        'train_samples': 800,
        'test_samples': 200,
        'class_distribution_train': {'class_0': 400, 'class_1': 400},
        'class_distribution_test': {'class_0': 100, 'class_1': 100}
    }
}
```

For cross-validation:

```python
{
    'split_type': 'cross_validation_stratified_kfold',
    'cv_splits': [
        {
            'train_indices': np.array([...]),
            'test_indices': np.array([...]),
            'train_samples': 800,
            'test_samples': 200,
            'class_distribution_train': {...},
            'class_distribution_test': {...}
        },
        # ... more folds
    ],
    'n_splits': 5,
    'metadata': {
        'cv_type': 'stratified_kfold',
        'n_splits': 5,
        'shuffle': True,
        'random_state': 42
    }
}
```

## Validation and Quality Checks

The module includes several validation features:

### Input Validation
- Checks for minimum dataset size
- Validates class balance
- Ensures sufficient samples per class
- Warns about severe class imbalance

### Result Validation
```python
from src.utils.data_splitting import validate_split_result

# Validate split result
is_valid = validate_split_result(result)
```

### Quality Metrics
- Class distribution across splits
- Sample counts
- Time ranges (for temporal data)
- Group distribution (for group-based splits)

## Best Practices

### 1. Choose Appropriate Strategy

- **Stratified**: Use for classification tasks with class imbalance
- **Time-based**: Use for temporal data to prevent leakage
- **Group-based**: Use when samples are related (patients, users)
- **Cross-validation**: Use for small datasets or model selection
- **Holdout**: Use for large datasets or quick prototyping

### 2. Configuration Guidelines

```yaml
# For balanced datasets
data_splitting:
  default_strategy: "stratified"
  test_size: 0.2
  validation_size: 0.1

# For temporal data
data_splitting:
  default_strategy: "time_based"
  time_column: "timestamp"
  gap: 30  # days/units between train and test

# For small datasets
data_splitting:
  default_strategy: "cross_validation"
  cv_type: "stratified_kfold"
  n_splits: 5

# For group-based data
data_splitting:
  default_strategy: "group_based"
  group_column: "patient_id"
  test_size: 0.2
```

### 3. Error Handling

```python
try:
    result = splitter.split_data(X, y, "stratified")
except ValueError as e:
    print(f"Validation error: {e}")
except ImportError as e:
    print(f"Missing dependency: {e}")
```

## Testing

Run the test suite to verify functionality:

```bash
python test_data_splitting.py
```

The test suite covers:
- Basic splitting strategies
- Time-based splitting
- Group-based splitting
- Custom splitting
- DataManager integration
- Configuration creation

## Troubleshooting

### Common Issues

1. **"At least 2 classes required"**
   - Ensure your target variable has multiple classes
   - Check for data preprocessing issues

2. **"Time column not found"**
   - Verify the time column exists in your DataFrame
   - Check column name spelling

3. **"Groups must be provided"**
   - Ensure group column is specified for group-based splitting
   - Verify group column exists in data

4. **"Very small dataset detected"**
   - Consider using cross-validation for small datasets
   - Adjust minimum sample thresholds if appropriate

### Performance Considerations

- For large datasets, use holdout splits
- For small datasets, use cross-validation
- Time-based splitting requires sorting (O(n log n))
- Group-based splitting may be slower for many groups

## Future Enhancements

Planned features:
- Nested cross-validation for hyperparameter tuning
- Stratified group-based splitting
- Multi-label stratification
- Automated strategy selection based on data characteristics
- Integration with external validation sets
- Support for regression tasks 