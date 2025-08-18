# Detailed Explanation Report

**Dataset:** imdb  
**Model:** lstm  
**Explanation Method:** attention_visualization  
**Generated:** 2025-08-16 07:02:04  

## Summary Statistics

- **Total Instances:** 200
- **Valid Explanations:** 200
- **Errors:** 0
- **Model Accuracy:** 0.8150
- **Average Feature Importance:** 0.0104
- **Feature Importance Std:** 0.0033
- **Max Feature Importance:** 0.2195

## Prediction Analysis

- **Correct Predictions:** 163 (81.5%)
- **Incorrect Predictions:** 37 (18.5%)

## Feature Importance Analysis

### Most Frequently Important Features

| Feature Index | Frequency | Avg Importance | Percentage |
|---------------|-----------|----------------|------------|
| 1 | 68 | 0.0150 | 34.0% |
| 2 | 43 | 0.0163 | 21.5% |
| 0 | 33 | 0.0139 | 16.5% |
| 98 | 22 | 0.0126 | 11.0% |
| 50 | 21 | 0.0136 | 10.5% |
| 31 | 17 | 0.0137 | 8.5% |
| 97 | 17 | 0.0134 | 8.5% |
| 4 | 17 | 0.0148 | 8.5% |
| 99 | 16 | 0.0128 | 8.0% |
| 43 | 14 | 0.0154 | 7.0% |

## Sample Explanations

### Correct Predictions (Sample)

#### Instance 0

- **True Label:** 0.0
- **Prediction:** 0.0
- **Prediction Probabilities:** ['0.781', '0.219']
- **Top Features:**
  - Feature 50: 0.0142
  - Feature 39: 0.0130
  - Feature 48: 0.0130

#### Instance 1

- **True Label:** 1.0
- **Prediction:** 1.0
- **Prediction Probabilities:** ['0.356', '0.644']
- **Top Features:**
  - Feature 80: 0.0130
  - Feature 32: 0.0125
  - Feature 42: 0.0125

#### Instance 3

- **True Label:** 1.0
- **Prediction:** 1.0
- **Prediction Probabilities:** ['0.397', '0.603']
- **Top Features:**
  - Feature 32: 0.0126
  - Feature 77: 0.0126
  - Feature 98: 0.0123

#### Instance 4

- **True Label:** 1.0
- **Prediction:** 1.0
- **Prediction Probabilities:** ['0.323', '0.677']
- **Top Features:**
  - Feature 7: 0.0303
  - Feature 0: 0.0292
  - Feature 43: 0.0285

#### Instance 5

- **True Label:** 1.0
- **Prediction:** 1.0
- **Prediction Probabilities:** ['0.341', '0.659']
- **Top Features:**
  - Feature 41: 0.0156
  - Feature 73: 0.0136
  - Feature 46: 0.0126

### Incorrect Predictions (Sample)

#### Instance 2

- **True Label:** 1.0
- **Prediction:** 0.0
- **Prediction Probabilities:** ['0.503', '0.497']
- **Top Features:**
  - Feature 1: 0.0128
  - Feature 12: 0.0117
  - Feature 69: 0.0117

#### Instance 7

- **True Label:** 0.0
- **Prediction:** 1.0
- **Prediction Probabilities:** ['0.413', '0.587']
- **Top Features:**
  - Feature 92: 0.0151
  - Feature 30: 0.0147
  - Feature 81: 0.0137

#### Instance 8

- **True Label:** 1.0
- **Prediction:** 0.0
- **Prediction Probabilities:** ['0.656', '0.344']
- **Top Features:**
  - Feature 21: 0.0156
  - Feature 71: 0.0132
  - Feature 16: 0.0127

## Detailed Results Table

| Instance ID | True Label | Prediction | Correct | Top Feature | Top Importance |
|-------------|------------|------------|---------|-------------|----------------|
