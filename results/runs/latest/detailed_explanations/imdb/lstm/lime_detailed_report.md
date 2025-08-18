# Detailed Explanation Report

**Dataset:** imdb  
**Model:** lstm  
**Explanation Method:** lime  
**Generated:** 2025-08-16 07:01:42  

## Summary Statistics

- **Total Instances:** 200
- **Valid Explanations:** 200
- **Errors:** 0
- **Model Accuracy:** 0.8150
- **Average Feature Importance:** 0.0200
- **Feature Importance Std:** 0.0236
- **Max Feature Importance:** 0.5366

## Prediction Analysis

- **Correct Predictions:** 163 (81.5%)
- **Incorrect Predictions:** 37 (18.5%)

## Feature Importance Analysis

### Most Frequently Important Features

| Feature Index | Frequency | Avg Importance | Percentage |
|---------------|-----------|----------------|------------|
| 4 | 38 | 0.0661 | 19.0% |
| 2 | 34 | 0.0553 | 17.0% |
| 7 | 31 | 0.0708 | 15.5% |
| 1 | 29 | 0.0685 | 14.5% |
| 5 | 29 | 0.0877 | 14.5% |
| 8 | 28 | 0.0681 | 14.0% |
| 15 | 28 | 0.0696 | 14.0% |
| 3 | 26 | 0.0745 | 13.0% |
| 13 | 26 | 0.0630 | 13.0% |
| 22 | 26 | 0.0585 | 13.0% |

## Sample Explanations

### Correct Predictions (Sample)

#### Instance 0

- **True Label:** 0.0
- **Prediction:** 0.0
- **Prediction Probabilities:** ['0.781', '0.219']
- **Top Features:**
  - Feature 0: 0.0392
  - Feature 1: 0.0384
  - Feature 2: 0.0376

#### Instance 1

- **True Label:** 1.0
- **Prediction:** 1.0
- **Prediction Probabilities:** ['0.356', '0.644']
- **Top Features:**
  - Feature 14: 0.0471
  - Feature 31: 0.0464
  - Feature 30: 0.0437

#### Instance 3

- **True Label:** 1.0
- **Prediction:** 1.0
- **Prediction Probabilities:** ['0.397', '0.603']
- **Top Features:**
  - Feature 43: 0.0973
  - Feature 35: 0.0569
  - Feature 6: 0.0483

#### Instance 4

- **True Label:** 1.0
- **Prediction:** 1.0
- **Prediction Probabilities:** ['0.323', '0.677']
- **Top Features:**
  - Feature 0: 0.0444
  - Feature 1: 0.0434
  - Feature 2: 0.0424

#### Instance 5

- **True Label:** 1.0
- **Prediction:** 1.0
- **Prediction Probabilities:** ['0.341', '0.659']
- **Top Features:**
  - Feature 44: 0.0798
  - Feature 3: 0.0770
  - Feature 48: 0.0673

### Incorrect Predictions (Sample)

#### Instance 2

- **True Label:** 1.0
- **Prediction:** 0.0
- **Prediction Probabilities:** ['0.503', '0.497']
- **Top Features:**
  - Feature 23: 0.1286
  - Feature 35: 0.0790
  - Feature 1: 0.0776

#### Instance 7

- **True Label:** 0.0
- **Prediction:** 1.0
- **Prediction Probabilities:** ['0.413', '0.587']
- **Top Features:**
  - Feature 2: 0.0947
  - Feature 46: 0.0934
  - Feature 5: 0.0797

#### Instance 8

- **True Label:** 1.0
- **Prediction:** 0.0
- **Prediction Probabilities:** ['0.656', '0.344']
- **Top Features:**
  - Feature 11: 0.0519
  - Feature 41: 0.0482
  - Feature 37: 0.0460

## Detailed Results Table

| Instance ID | True Label | Prediction | Correct | Top Feature | Top Importance |
|-------------|------------|------------|---------|-------------|----------------|
