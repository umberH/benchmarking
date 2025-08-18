# Detailed Explanation Report

**Dataset:** imdb  
**Model:** bert  
**Explanation Method:** lime  
**Generated:** 2025-08-16 07:01:06  

## Summary Statistics

- **Total Instances:** 200
- **Valid Explanations:** 200
- **Errors:** 0
- **Model Accuracy:** 0.8100
- **Average Feature Importance:** 0.0200
- **Feature Importance Std:** 0.0214
- **Max Feature Importance:** 0.3324

## Prediction Analysis

- **Correct Predictions:** 162 (81.0%)
- **Incorrect Predictions:** 38 (19.0%)

## Feature Importance Analysis

### Most Frequently Important Features

| Feature Index | Frequency | Avg Importance | Percentage |
|---------------|-----------|----------------|------------|
| 4 | 45 | 0.0616 | 22.5% |
| 1 | 45 | 0.0550 | 22.5% |
| 2 | 42 | 0.0544 | 21.0% |
| 3 | 35 | 0.0616 | 17.5% |
| 0 | 30 | 0.0533 | 15.0% |
| 13 | 27 | 0.0795 | 13.5% |
| 17 | 27 | 0.0675 | 13.5% |
| 38 | 26 | 0.0586 | 13.0% |
| 12 | 25 | 0.0747 | 12.5% |
| 7 | 24 | 0.0633 | 12.0% |

## Sample Explanations

### Correct Predictions (Sample)

#### Instance 0

- **True Label:** 0.0
- **Prediction:** 0.0
- **Prediction Probabilities:** ['0.753', '0.247']
- **Top Features:**
  - Feature 39: 0.0923
  - Feature 4: 0.0870
  - Feature 13: 0.0718

#### Instance 1

- **True Label:** 1.0
- **Prediction:** 1.0
- **Prediction Probabilities:** ['0.320', '0.680']
- **Top Features:**
  - Feature 0: 0.0392
  - Feature 1: 0.0384
  - Feature 2: 0.0376

#### Instance 3

- **True Label:** 1.0
- **Prediction:** 1.0
- **Prediction Probabilities:** ['0.350', '0.650']
- **Top Features:**
  - Feature 30: 0.0916
  - Feature 43: 0.0785
  - Feature 17: 0.0669

#### Instance 4

- **True Label:** 1.0
- **Prediction:** 1.0
- **Prediction Probabilities:** ['0.259', '0.741']
- **Top Features:**
  - Feature 26: 0.0512
  - Feature 13: 0.0506
  - Feature 42: 0.0503

#### Instance 5

- **True Label:** 1.0
- **Prediction:** 1.0
- **Prediction Probabilities:** ['0.371', '0.629']
- **Top Features:**
  - Feature 48: 0.0783
  - Feature 22: 0.0734
  - Feature 20: 0.0689

### Incorrect Predictions (Sample)

#### Instance 2

- **True Label:** 1.0
- **Prediction:** 0.0
- **Prediction Probabilities:** ['0.510', '0.490']
- **Top Features:**
  - Feature 4: 0.1292
  - Feature 23: 0.0988
  - Feature 35: 0.0498

#### Instance 7

- **True Label:** 0.0
- **Prediction:** 1.0
- **Prediction Probabilities:** ['0.405', '0.595']
- **Top Features:**
  - Feature 0: 0.0392
  - Feature 1: 0.0384
  - Feature 2: 0.0376

#### Instance 8

- **True Label:** 1.0
- **Prediction:** 0.0
- **Prediction Probabilities:** ['0.656', '0.344']
- **Top Features:**
  - Feature 7: 0.1287
  - Feature 49: 0.1230
  - Feature 20: 0.0451

## Detailed Results Table

| Instance ID | True Label | Prediction | Correct | Top Feature | Top Importance |
|-------------|------------|------------|---------|-------------|----------------|
