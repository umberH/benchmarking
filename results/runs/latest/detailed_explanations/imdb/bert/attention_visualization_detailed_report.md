# Detailed Explanation Report

**Dataset:** imdb  
**Model:** bert  
**Explanation Method:** attention_visualization  
**Generated:** 2025-08-16 07:01:26  

## Summary Statistics

- **Total Instances:** 200
- **Valid Explanations:** 200
- **Errors:** 0
- **Model Accuracy:** 0.8100
- **Average Feature Importance:** 0.0104
- **Feature Importance Std:** 0.0032
- **Max Feature Importance:** 0.1541

## Prediction Analysis

- **Correct Predictions:** 162 (81.0%)
- **Incorrect Predictions:** 38 (19.0%)

## Feature Importance Analysis

### Most Frequently Important Features

| Feature Index | Frequency | Avg Importance | Percentage |
|---------------|-----------|----------------|------------|
| 1 | 69 | 0.0149 | 34.5% |
| 2 | 48 | 0.0160 | 24.0% |
| 0 | 36 | 0.0141 | 18.0% |
| 50 | 25 | 0.0135 | 12.5% |
| 98 | 25 | 0.0129 | 12.5% |
| 97 | 22 | 0.0134 | 11.0% |
| 99 | 20 | 0.0133 | 10.0% |
| 4 | 17 | 0.0160 | 8.5% |
| 31 | 16 | 0.0134 | 8.0% |
| 44 | 15 | 0.0159 | 7.5% |

## Sample Explanations

### Correct Predictions (Sample)

#### Instance 0

- **True Label:** 0.0
- **Prediction:** 0.0
- **Prediction Probabilities:** ['0.753', '0.247']
- **Top Features:**
  - Feature 50: 0.0142
  - Feature 39: 0.0130
  - Feature 48: 0.0130

#### Instance 1

- **True Label:** 1.0
- **Prediction:** 1.0
- **Prediction Probabilities:** ['0.320', '0.680']
- **Top Features:**
  - Feature 80: 0.0130
  - Feature 32: 0.0125
  - Feature 42: 0.0125

#### Instance 3

- **True Label:** 1.0
- **Prediction:** 1.0
- **Prediction Probabilities:** ['0.350', '0.650']
- **Top Features:**
  - Feature 32: 0.0126
  - Feature 77: 0.0126
  - Feature 98: 0.0123

#### Instance 4

- **True Label:** 1.0
- **Prediction:** 1.0
- **Prediction Probabilities:** ['0.259', '0.741']
- **Top Features:**
  - Feature 7: 0.0303
  - Feature 0: 0.0292
  - Feature 43: 0.0285

#### Instance 5

- **True Label:** 1.0
- **Prediction:** 1.0
- **Prediction Probabilities:** ['0.371', '0.629']
- **Top Features:**
  - Feature 41: 0.0156
  - Feature 73: 0.0136
  - Feature 46: 0.0126

### Incorrect Predictions (Sample)

#### Instance 2

- **True Label:** 1.0
- **Prediction:** 0.0
- **Prediction Probabilities:** ['0.510', '0.490']
- **Top Features:**
  - Feature 1: 0.0127
  - Feature 97: 0.0120
  - Feature 12: 0.0116

#### Instance 7

- **True Label:** 0.0
- **Prediction:** 1.0
- **Prediction Probabilities:** ['0.405', '0.595']
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
