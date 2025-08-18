# Comprehensive XAI Benchmarking Report

Generated on: 2025-08-16 07:02:04

## Summary

- **Datasets**: 4
- **Models**: 8
- **Explanation Methods**: 17
- **Evaluation Metrics**: 16
- **Total Combinations**: 96

### Datasets
- **adult_income** (tabular)
- **compas** (tabular)
- **imdb** (text)
- **mnist** (image)

### Models
- **bert** (bert)
- **cnn** (cnn)
- **decision_tree** (decision_tree)
- **gradient_boosting** (gradient_boosting)
- **lstm** (lstm)
- **mlp** (mlp)
- **random_forest** (random_forest)
- **vit** (vit)

### Explanation Methods
- **attention_visualization**
- **bayesian_rule_list**
- **causal_shap**
- **concept_bottleneck**
- **corels**
- **counterfactual**
- **feature_ablation**
- **influence_functions**
- **integrated_gradients**
- **lime**
- **occlusion**
- **prototype**
- **shap**
- **shap_interactive**
- **shapley_flow**
- **tcav**
- **text_occlusion**

## Model Performance Summary

Training and test set performance for each model on each dataset.

| Dataset | Model | Train Accuracy | Test Accuracy | Train Loss | Test Loss | Other Metrics |
|---------|-------|----------------|---------------|------------|-----------|---------------|
| adult_income | decision_tree | 0.8405 | 0.8326 | N/A | N/A | train_f1: 0.8247; test_f1: 0.8159; train_precision: 0.8365; test_precision: 0.8264; train_recall: 0.8405; test_recall: 0.8326; overfitting_gap: 0.0079; overfitting_severity: low; class_accuracies: [0.957845950121386, 0.45472703062583225]; n_classes: 2.0000; n_train_samples: 24129.0000; n_test_samples: 6033.0000; training_time: 0.0357; model_complexity: {'n_parameters': 13, 'model_size_bytes': 48, 'model_size_mb': 4.57763671875e-05, 'complexity_level': 'simple'} |
| adult_income | random_forest | 0.8425 | 0.8333 | N/A | N/A | train_f1: 0.8267; test_f1: 0.8160; train_precision: 0.8392; test_precision: 0.8278; train_recall: 0.8425; test_recall: 0.8333; overfitting_gap: 0.0092; overfitting_severity: low; class_accuracies: [0.9602736702714633, 0.45006657789613846]; n_classes: 2.0000; n_train_samples: 24129.0000; n_test_samples: 6033.0000; training_time: 1.0598; model_complexity: {'n_parameters': 19, 'model_size_bytes': 48, 'model_size_mb': 4.57763671875e-05, 'complexity_level': 'simple'} |
| adult_income | gradient_boosting | 0.8387 | 0.8356 | N/A | N/A | train_f1: 0.8229; test_f1: 0.8189; train_precision: 0.8340; test_precision: 0.8305; train_recall: 0.8387; test_recall: 0.8356; overfitting_gap: 0.0031; overfitting_severity: low; class_accuracies: [0.9607150739351137, 0.4580559254327563]; n_classes: 2.0000; n_train_samples: 24129.0000; n_test_samples: 6033.0000; training_time: 1.0073; model_complexity: {'n_parameters': 20, 'model_size_bytes': 48, 'model_size_mb': 4.57763671875e-05, 'complexity_level': 'simple'} |
| adult_income | mlp | 0.8257 | 0.8236 | N/A | N/A | train_f1: 0.8112; test_f1: 0.8083; train_precision: 0.8161; test_precision: 0.8137; train_recall: 0.8257; test_recall: 0.8236; overfitting_gap: 0.0021; overfitting_severity: low; class_accuracies: [0.9452659457073493, 0.4567243675099867]; n_classes: 2.0000; n_train_samples: 24129.0000; n_test_samples: 6033.0000; training_time: 7.4767; model_complexity: {'n_parameters': 23, 'model_size_bytes': 48, 'model_size_mb': 4.57763671875e-05, 'complexity_level': 'simple'} |
| compas | decision_tree | 0.7375 | 0.6736 | N/A | N/A | train_f1: 0.7347; test_f1: 0.6706; train_precision: 0.7381; test_precision: 0.6721; train_recall: 0.7375; test_recall: 0.6736; overfitting_gap: 0.0639; overfitting_severity: low; class_accuracies: [0.755359394703657, 0.5738461538461539]; n_classes: 2.0000; n_train_samples: 5771.0000; n_test_samples: 1443.0000; training_time: 0.0080; model_complexity: {'n_parameters': 13, 'model_size_bytes': 48, 'model_size_mb': 4.57763671875e-05, 'complexity_level': 'simple'} |
| compas | random_forest | 0.7538 | 0.6826 | N/A | N/A | train_f1: 0.7516; test_f1: 0.6797; train_precision: 0.7543; test_precision: 0.6813; train_recall: 0.7538; test_recall: 0.6826; overfitting_gap: 0.0712; overfitting_severity: low; class_accuracies: [0.7629255989911727, 0.5846153846153846]; n_classes: 2.0000; n_train_samples: 5771.0000; n_test_samples: 1443.0000; training_time: 0.2688; model_complexity: {'n_parameters': 19, 'model_size_bytes': 48, 'model_size_mb': 4.57763671875e-05, 'complexity_level': 'simple'} |
| compas | gradient_boosting | 0.7054 | 0.6951 | N/A | N/A | train_f1: 0.7025; test_f1: 0.6924; train_precision: 0.7049; test_precision: 0.6941; train_recall: 0.7054; test_recall: 0.6951; overfitting_gap: 0.0103; overfitting_severity: low; class_accuracies: [0.7730138713745272, 0.6]; n_classes: 2.0000; n_train_samples: 5771.0000; n_test_samples: 1443.0000; training_time: 0.2391; model_complexity: {'n_parameters': 20, 'model_size_bytes': 48, 'model_size_mb': 4.57763671875e-05, 'complexity_level': 'simple'} |
| compas | mlp | 0.6881 | 0.6854 | N/A | N/A | train_f1: 0.6862; test_f1: 0.6837; train_precision: 0.6868; test_precision: 0.6840; train_recall: 0.6881; test_recall: 0.6854; overfitting_gap: 0.0027; overfitting_severity: low; class_accuracies: [0.7490542244640606, 0.6076923076923076]; n_classes: 2.0000; n_train_samples: 5771.0000; n_test_samples: 1443.0000; training_time: 1.4459; model_complexity: {'n_parameters': 23, 'model_size_bytes': 48, 'model_size_mb': 4.57763671875e-05, 'complexity_level': 'simple'} |
| mnist | cnn | 1.0000 | 0.9900 | N/A | N/A | train_f1: 1.0000; test_f1: 0.9900; train_precision: 1.0000; test_precision: 0.9904; train_recall: 1.0000; test_recall: 0.9900; overfitting_gap: 0.0100; overfitting_severity: low; class_accuracies: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.95, 0.9583333333333334, 1.0, 1.0]; n_classes: 10.0000; n_train_samples: 1000.0000; n_test_samples: 200.0000; training_time: 5.4178; model_complexity: {'n_parameters': 688138, 'model_size_bytes': 2752552, 'model_size_mb': 2.6250381469726562, 'complexity_level': 'complex'} |
| mnist | vit | 0.8090 | 0.7350 | N/A | N/A | train_f1: 0.8058; test_f1: 0.7252; train_precision: 0.8150; test_precision: 0.7439; train_recall: 0.8090; test_recall: 0.7350; overfitting_gap: 0.0740; overfitting_severity: low; class_accuracies: [0.9411764705882353, 0.9285714285714286, 0.5, 0.6875, 0.8571428571428571, 0.35, 0.5, 0.875, 0.8, 0.7619047619047619]; n_classes: 10.0000; n_train_samples: 1000.0000; n_test_samples: 200.0000; training_time: 7.7056; model_complexity: {'n_parameters': 3231242, 'model_size_bytes': 12924968, 'model_size_mb': 12.326210021972656, 'complexity_level': 'complex'} |
| imdb | bert | 0.9180 | 0.8100 | N/A | N/A | train_f1: 0.9180; test_f1: 0.8099; train_precision: 0.9180; test_precision: 0.8105; train_recall: 0.9180; test_recall: 0.8100; overfitting_gap: 0.1080; overfitting_severity: moderate; class_accuracies: [0.79, 0.83]; n_classes: 2.0000; n_train_samples: 1000.0000; n_test_samples: 200.0000; training_time: 0.3603; model_complexity: {'n_parameters': 15, 'model_size_bytes': 48, 'model_size_mb': 4.57763671875e-05, 'complexity_level': 'simple'} |
| imdb | lstm | 0.8870 | 0.8150 | N/A | N/A | train_f1: 0.8870; test_f1: 0.8149; train_precision: 0.8870; test_precision: 0.8158; train_recall: 0.8870; test_recall: 0.8150; overfitting_gap: 0.0720; overfitting_severity: low; class_accuracies: [0.84, 0.79]; n_classes: 2.0000; n_train_samples: 1000.0000; n_test_samples: 200.0000; training_time: 0.5865; model_complexity: {'n_parameters': 4, 'model_size_bytes': 48, 'model_size_mb': 4.57763671875e-05, 'complexity_level': 'simple'} |

## XAI Evaluation Results Table

Each row represents a unique combination of Dataset, Model, and Explanation Method with their evaluation metrics.

| Dataset | Model | Explanation Method | Detailed Report | Time Complexity | Faithfulness | Monotonicity | Completeness | Stability | Consistency | Sparsity | Simplicity | Advanced Identity | Advanced Separability | Advanced Non Sensitivity | Advanced Compactness | Advanced Correctness | Advanced Entropy | Advanced Gini Coefficient | Advanced Kl Divergence |
|---------|-------|-------------------|-----------------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|--------|
| adult_income | decision_tree | shap | N/A | 0.0011 | 0.1900 | 0.0200 | 0.0100 | 0.0000 | 0.4651 | 0.0300 | 0.9520 | 0.9432 | 0.1706 | 1.0000 | 0.1775 | 0.6240 | 0.0215 | 0.1420 | 0.1685 |
| adult_income | decision_tree | lime | N/A | 0.0128 | 0.1000 | 0.0267 | 0.0000 | 0.0000 | 0.5440 | 0.0840 | 0.9400 | 0.8511 | 0.2117 | 1.0000 | 0.2230 | 0.5898 | 0.0402 | 0.1800 | 0.1998 |
| adult_income | decision_tree | causal_shap | N/A | 0.0196 | 0.2400 | 0.0200 | 0.0200 | 0.0000 | 0.5571 | 0.0360 | 0.9508 | 1.0000 | 0.1790 | 1.0000 | 0.1871 | 0.5816 | 0.0243 | 0.1508 | 0.1757 |
| adult_income | decision_tree | shapley_flow | N/A | 0.0108 | 0.1000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5600 | 0.0000 | 0.0000 | 0.0000 |
| adult_income | decision_tree | shap_interactive | N/A | 0.0330 | 0.3000 | 0.0333 | 0.0033 | 0.0000 | 0.3873 | 0.1000 | 0.9407 | 0.9163 | 0.2813 | 1.0000 | 0.2840 | 0.6260 | 0.0839 | 0.2307 | 0.2494 |
| adult_income | decision_tree | prototype | N/A | 0.0010 | 0.7100 | 0.8436 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| adult_income | decision_tree | counterfactual | N/A | 0.0009 | 0.7200 | 0.1716 | 1.0000 | 0.5949 | 0.0657 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| adult_income | decision_tree | bayesian_rule_list | N/A | 0.0006 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| adult_income | decision_tree | corels | N/A | 0.0003 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| adult_income | decision_tree | feature_ablation | N/A | 0.0006 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4632 | 0.0240 | 0.9440 | 1.0000 | 0.2245 | 1.0000 | 0.2300 | 0.6040 | 0.0172 | 0.1840 | 0.2228 |
| adult_income | random_forest | shap | N/A | 0.0361 | 0.1900 | 0.0200 | 0.0000 | 0.0000 | 0.4696 | 0.0140 | 0.9440 | 0.9438 | 0.1647 | 1.0000 | 0.1675 | 0.6170 | 0.0291 | 0.1340 | 0.1609 |
| adult_income | random_forest | lime | N/A | 0.0211 | 0.1200 | 0.0400 | 0.0000 | 0.0000 | 0.4209 | 0.0120 | 0.9607 | 0.9091 | 0.1478 | 1.0000 | 0.1495 | 0.5749 | 0.0155 | 0.1207 | 0.1445 |
| adult_income | random_forest | causal_shap | N/A | 0.7504 | 0.2400 | 0.0200 | 0.0000 | 0.0000 | 0.5377 | 0.0440 | 0.9491 | 0.9950 | 0.1900 | 1.0000 | 0.1961 | 0.5808 | 0.0413 | 0.1591 | 0.1787 |
| adult_income | random_forest | shapley_flow | N/A | 0.3477 | 0.1333 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5600 | 0.0000 | 0.0000 | 0.0000 |
| adult_income | random_forest | shap_interactive | N/A | 0.8816 | 0.2333 | 0.0333 | 0.0923 | 0.0000 | 0.4137 | 0.0933 | 0.9290 | 0.9020 | 0.2582 | 1.0000 | 0.2464 | 0.6256 | 0.1132 | 0.2033 | 0.2202 |
| adult_income | random_forest | prototype | N/A | 0.0043 | 0.7100 | 0.8102 | 1.0000 | 0.9644 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| adult_income | random_forest | counterfactual | N/A | 0.0051 | 0.7250 | 0.1994 | 1.0000 | 0.6720 | 0.2167 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| adult_income | random_forest | bayesian_rule_list | N/A | 0.0045 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| adult_income | random_forest | corels | N/A | 0.0041 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| adult_income | random_forest | feature_ablation | N/A | 0.0301 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4509 | 0.0120 | 0.9320 | 1.0000 | 0.2132 | 1.0000 | 0.2150 | 0.6040 | 0.0309 | 0.1720 | 0.2091 |
| adult_income | gradient_boosting | shap | N/A | 0.0026 | 0.2100 | 0.0300 | 0.0000 | 0.0000 | 0.4896 | 0.0220 | 0.9400 | 0.9432 | 0.1825 | 1.0000 | 0.1875 | 0.6440 | 0.0316 | 0.1500 | 0.1784 |
| adult_income | gradient_boosting | lime | N/A | 0.0158 | 0.1400 | 0.0400 | 0.0000 | 0.0000 | 0.4489 | 0.0240 | 0.9551 | 0.9091 | 0.1903 | 1.0000 | 0.1943 | 0.6151 | 0.0155 | 0.1551 | 0.1845 |
| adult_income | gradient_boosting | causal_shap | N/A | 0.0534 | 0.2600 | 0.0200 | 0.0000 | 0.0000 | 0.6324 | 0.0360 | 0.9554 | 1.0000 | 0.1797 | 1.0000 | 0.1848 | 0.6125 | 0.0331 | 0.1479 | 0.1669 |
| adult_income | gradient_boosting | shap_interactive | N/A | 0.0530 | 0.3333 | 0.0333 | 0.0000 | 0.0000 | 0.5154 | 0.0867 | 0.9360 | 0.9503 | 0.2770 | 1.0000 | 0.2781 | 0.6562 | 0.1022 | 0.2222 | 0.2312 |
| adult_income | gradient_boosting | prototype | N/A | 0.0009 | 0.7100 | 0.8099 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| adult_income | gradient_boosting | counterfactual | N/A | 0.0008 | 0.6950 | 0.1864 | 1.0000 | 0.8546 | 0.5849 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| adult_income | gradient_boosting | bayesian_rule_list | N/A | 0.0010 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| adult_income | gradient_boosting | corels | N/A | 0.0006 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| adult_income | gradient_boosting | feature_ablation | N/A | 0.0011 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4954 | 0.0200 | 0.9360 | 1.0000 | 0.2404 | 1.0000 | 0.2450 | 0.6380 | 0.0223 | 0.1960 | 0.2377 |
| adult_income | mlp | shap | N/A | 0.0009 | 0.2200 | 0.0250 | 0.0000 | 0.0000 | 0.4396 | 0.0260 | 0.9420 | 0.9032 | 0.1970 | 1.0000 | 0.2025 | 0.6400 | 0.0258 | 0.1620 | 0.1942 |
| adult_income | mlp | lime | N/A | 0.0114 | 0.0600 | 0.0333 | 0.0000 | 0.0000 | 0.6304 | 0.0720 | 0.9685 | 0.8913 | 0.1010 | 1.0000 | 0.1101 | 0.5718 | 0.0302 | 0.0885 | 0.0898 |
| adult_income | mlp | integrated_gradients | N/A | 0.0459 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5833 | 0.0000 | 0.0000 | 0.0000 |
| adult_income | mlp | causal_shap | N/A | 0.0212 | 0.2400 | 0.0100 | 0.0000 | 0.0000 | 0.6613 | 0.0360 | 0.9680 | 1.0000 | 0.1607 | 1.0000 | 0.1646 | 0.5962 | 0.0325 | 0.1310 | 0.1475 |
| adult_income | mlp | shapley_flow | N/A | 0.0087 | 0.1000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.5833 | 0.0000 | 0.0000 | 0.0000 |
| adult_income | mlp | shap_interactive | N/A | 0.0255 | 0.3000 | 0.0150 | 0.0361 | 0.0000 | 0.3809 | 0.1333 | 0.9571 | 0.8680 | 0.2911 | 1.0000 | 0.2805 | 0.6466 | 0.1438 | 0.2297 | 0.2229 |
| adult_income | mlp | prototype | N/A | 0.0009 | 0.7150 | 0.8053 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| adult_income | mlp | counterfactual | N/A | 0.0006 | 0.7050 | 0.2038 | 1.0000 | 0.9448 | 0.5673 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| adult_income | mlp | influence_functions | N/A | 0.0199 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| adult_income | mlp | bayesian_rule_list | N/A | 0.0007 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| adult_income | mlp | corels | N/A | 0.0004 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| adult_income | mlp | feature_ablation | N/A | 0.0006 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.4377 | 0.0240 | 0.9320 | 1.0000 | 0.2845 | 1.0000 | 0.2900 | 0.6360 | 0.0172 | 0.2320 | 0.2828 |
| compas | decision_tree | shap | N/A | 0.0005 | 0.6500 | 0.0300 | 0.0000 | 0.0000 | 0.6861 | 0.3500 | 0.7000 | 0.6292 | 0.3712 | 1.0000 | 0.5250 | 0.6090 | 0.1525 | 0.3500 | 0.4975 |
| compas | decision_tree | lime | N/A | 0.0144 | 0.3200 | 0.0800 | 0.0000 | 0.0000 | 0.7452 | 0.6133 | 0.6873 | 0.5208 | 0.6413 | 1.0000 | 0.9101 | 0.5907 | 0.0269 | 0.6073 | 0.8931 |
| compas | decision_tree | causal_shap | N/A | 0.0097 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.5955 | 0.3067 | 0.7535 | 1.0000 | 0.2218 | 1.0000 | 0.3031 | 0.3975 | 0.2334 | 0.2133 | 0.2466 |
| compas | decision_tree | shapley_flow | N/A | 0.0044 | 0.3333 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.3733 | 0.0000 | 0.0000 | 0.0000 |
| compas | decision_tree | shap_interactive | N/A | 0.0103 | 0.6667 | 0.0222 | 0.0717 | 0.0000 | 0.6939 | 0.5444 | 0.7005 | 0.6612 | 0.4414 | 1.0000 | 0.4878 | 0.5084 | 0.5052 | 0.3449 | 0.3615 |
| compas | decision_tree | prototype | N/A | 0.0003 | 0.6500 | 0.7255 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| compas | decision_tree | counterfactual | N/A | 0.0003 | 0.6100 | 0.2632 | 1.0000 | 0.9795 | 0.9764 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| compas | decision_tree | bayesian_rule_list | N/A | 0.0006 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| compas | decision_tree | corels | N/A | 0.0003 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| compas | decision_tree | feature_ablation | N/A | 0.0003 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6747 | 0.3667 | 0.6667 | 1.0000 | 0.3889 | 0.9999 | 0.5500 | 0.5260 | 0.1840 | 0.3667 | 0.5160 |
| compas | random_forest | shap | N/A | 0.0212 | 0.6300 | 0.0383 | 0.0000 | 0.0000 | 0.7164 | 0.3233 | 0.6933 | 0.6548 | 0.3429 | 1.0000 | 0.4850 | 0.6240 | 0.1777 | 0.3233 | 0.4523 |
| compas | random_forest | lime | N/A | 0.0201 | 0.3400 | 0.0900 | 0.0000 | 0.0000 | 0.8332 | 0.5200 | 0.6888 | 0.5750 | 0.4879 | 1.0000 | 0.6935 | 0.5713 | 0.1296 | 0.4688 | 0.6504 |
| compas | random_forest | causal_shap | N/A | 0.4260 | 0.4800 | 0.0300 | 0.0000 | 0.0000 | 0.6272 | 0.3267 | 0.7456 | 1.0000 | 0.2151 | 0.9943 | 0.2852 | 0.4497 | 0.2803 | 0.2061 | 0.2197 |
| compas | random_forest | shapley_flow | N/A | 0.1864 | 0.4333 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.3733 | 0.0000 | 0.0000 | 0.0000 |
| compas | random_forest | shap_interactive | N/A | 0.4056 | 0.6000 | 0.0611 | 0.0843 | 0.0000 | 0.6631 | 0.4444 | 0.6272 | 0.7818 | 0.3159 | 0.9989 | 0.3517 | 0.4556 | 0.5052 | 0.2548 | 0.2615 |
| compas | random_forest | prototype | N/A | 0.0032 | 0.7100 | 0.6912 | 1.0000 | 0.9954 | 0.6216 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| compas | random_forest | counterfactual | N/A | 0.0032 | 0.6150 | 0.3043 | 1.0000 | 0.9513 | 0.4769 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| compas | random_forest | bayesian_rule_list | N/A | 0.0034 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| compas | random_forest | corels | N/A | 0.0032 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| compas | random_forest | feature_ablation | N/A | 0.0115 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7118 | 0.2933 | 0.6333 | 1.0000 | 0.3111 | 1.0000 | 0.4400 | 0.5640 | 0.2671 | 0.2933 | 0.3929 |
| compas | gradient_boosting | shap | N/A | 0.0010 | 0.6100 | 0.0250 | 0.0000 | 0.0000 | 0.7096 | 0.3233 | 0.7133 | 0.6705 | 0.3429 | 1.0000 | 0.4850 | 0.6490 | 0.1551 | 0.3233 | 0.4549 |
| compas | gradient_boosting | lime | N/A | 0.0100 | 0.4000 | 0.1400 | 0.0000 | 0.3151 | 0.9147 | 0.5333 | 0.7228 | 0.5814 | 0.5510 | 1.0000 | 0.7841 | 0.6346 | 0.0433 | 0.5228 | 0.7567 |
| compas | gradient_boosting | causal_shap | N/A | 0.0213 | 0.4600 | 0.0300 | 0.0000 | 0.0000 | 0.6126 | 0.3200 | 0.6787 | 1.0000 | 0.1766 | 1.0000 | 0.2399 | 0.4526 | 0.3214 | 0.1787 | 0.1786 |
| compas | gradient_boosting | shap_interactive | N/A | 0.0221 | 0.6333 | 0.0611 | 0.0292 | 0.0000 | 0.6215 | 0.4889 | 0.6794 | 0.7303 | 0.3533 | 0.9998 | 0.3962 | 0.4872 | 0.4803 | 0.2910 | 0.2864 |
| compas | gradient_boosting | prototype | N/A | 0.0003 | 0.6550 | 0.6883 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| compas | gradient_boosting | counterfactual | N/A | 0.0004 | 0.6300 | 0.3109 | 1.0000 | 0.7165 | 0.0151 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| compas | gradient_boosting | bayesian_rule_list | N/A | 0.0005 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| compas | gradient_boosting | corels | N/A | 0.0003 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| compas | gradient_boosting | feature_ablation | N/A | 0.0007 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.6815 | 0.3333 | 0.6933 | 1.0000 | 0.3536 | 1.0000 | 0.5000 | 0.5980 | 0.1767 | 0.3333 | 0.4633 |
| compas | mlp | shap | N/A | 0.0005 | 0.5800 | 0.0100 | 0.0000 | 0.0000 | 0.7432 | 0.3800 | 0.8000 | 0.6556 | 0.4031 | 1.0000 | 0.5700 | 0.6430 | 0.0126 | 0.3800 | 0.5674 |
| compas | mlp | lime | N/A | 0.0084 | 0.0400 | 0.0000 | 0.0000 | 0.0000 | 0.7918 | 0.3733 | 0.7848 | 0.6000 | 0.3567 | 1.0000 | 0.5103 | 0.5465 | 0.0876 | 0.3448 | 0.4724 |
| compas | mlp | integrated_gradients | N/A | 0.0233 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.4200 | 0.0000 | 0.0000 | 0.0000 |
| compas | mlp | causal_shap | N/A | 0.0104 | 0.3400 | 0.0000 | 0.0000 | 0.0000 | 0.5699 | 0.3000 | 0.8075 | 1.0000 | 0.2579 | 1.0000 | 0.3468 | 0.5068 | 0.1461 | 0.2375 | 0.3139 |
| compas | mlp | shapley_flow | N/A | 0.0052 | 0.2333 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 1.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.4200 | 0.0000 | 0.0000 | 0.0000 |
| compas | mlp | shap_interactive | N/A | 0.0114 | 0.4667 | 0.0111 | 0.0384 | 0.0000 | 0.6465 | 0.5111 | 0.7949 | 0.6119 | 0.4629 | 1.0000 | 0.5295 | 0.5891 | 0.3644 | 0.3702 | 0.4356 |
| compas | mlp | prototype | N/A | 0.0002 | 0.6250 | 0.6891 | 1.0000 | 0.9330 | 0.9960 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| compas | mlp | counterfactual | N/A | 0.0002 | 0.6950 | 0.3141 | 1.0000 | 0.9896 | 0.3550 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| compas | mlp | influence_functions | N/A | 0.0172 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| compas | mlp | bayesian_rule_list | N/A | 0.0004 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| compas | mlp | corels | N/A | 0.0002 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| compas | mlp | feature_ablation | N/A | 0.0003 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.7167 | 0.3800 | 0.8000 | 1.0000 | 0.4031 | 1.0000 | 0.5700 | 0.5800 | 0.0126 | 0.3800 | 0.5674 |
| mnist | cnn | prototype | N/A | 0.0012 | 0.9900 | 0.65905195 | 1.0000 | 0.4759 | 0.7362 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mnist | cnn | counterfactual | N/A | 0.0026 | 0.9900 | 0.46621537 | 1.0000 | 0.7973 | 0.8631 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mnist | cnn | tcav | N/A | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| mnist | cnn | concept_bottleneck | N/A | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| mnist | cnn | occlusion | N/A | 0.0227 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| mnist | vit | tcav | N/A | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| mnist | vit | concept_bottleneck | N/A | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| mnist | vit | occlusion | N/A | 0.0721 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| imdb | bert | lime | N/A | 0.0552 | 0.8400 | 0.0000 | 0.0000 | 0.0806 | 0.1373 | 0.7400 | 0.4467 | 1.0000 | 0.4675 | 0.0000 | 0.4545 | 0.5942 | 0.9050 | 0.4467 | 0.0950 |
| imdb | bert | text_occlusion | N/A | 0.0316 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0552 | 0.0046 | 0.8016 | 1.0000 | 0.0890 | 0.0000 | 0.0824 | 0.6360 | 0.2137 | 0.0816 | 0.0663 |
| imdb | bert | attention_visualization | N/A | 0.0811 | 0.0000 | 0.0000 | 0.0000 | 0.7141 | 0.0947 | 0.7005 | 0.0381 | 1.0000 | 0.0360 | 0.0000 | 0.0055 | 0.5880 | 0.9994 | 0.0381 | 0.0006 |
| imdb | lstm | lime | N/A | 0.0583 | 0.8000 | 0.0000 | 0.0000 | 0.0000 | 0.1244 | 0.7400 | 0.4779 | 1.0000 | 0.5296 | 0.0000 | 0.5099 | 0.5680 | 0.8884 | 0.4779 | 0.1116 |
| imdb | lstm | text_occlusion | N/A | 0.0345 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0627 | 0.0062 | 0.7822 | 1.0000 | 0.1734 | 0.0000 | 0.1638 | 0.6500 | 0.2449 | 0.1622 | 0.1351 |
| imdb | lstm | attention_visualization | N/A | 0.0861 | 0.0000 | 0.0000 | 0.0000 | 0.7135 | 0.0924 | 0.6971 | 0.0385 | 1.0000 | 0.0371 | 0.0000 | 0.0059 | 0.5600 | 0.9994 | 0.0385 | 0.0006 |

## Detailed Explanation Analysis

Summary of detailed explanations generated for the entire test set.

| Dataset | Model | Method | Test Instances | Valid Explanations | Accuracy | Avg Feature Importance | Detailed Files |
|---------|-------|--------|----------------|-------------------|----------|----------------------|----------------|
| adult_income | decision_tree | shap | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | decision_tree | lime | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | decision_tree | causal_shap | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | decision_tree | shapley_flow | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | decision_tree | shap_interactive | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | decision_tree | prototype | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | decision_tree | counterfactual | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | decision_tree | bayesian_rule_list | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | decision_tree | corels | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | decision_tree | feature_ablation | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | random_forest | shap | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | random_forest | lime | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | random_forest | causal_shap | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | random_forest | shapley_flow | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | random_forest | shap_interactive | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | random_forest | prototype | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | random_forest | counterfactual | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | random_forest | bayesian_rule_list | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | random_forest | corels | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | random_forest | feature_ablation | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | gradient_boosting | shap | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | gradient_boosting | lime | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | gradient_boosting | causal_shap | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | gradient_boosting | shap_interactive | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | gradient_boosting | prototype | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | gradient_boosting | counterfactual | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | gradient_boosting | bayesian_rule_list | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | gradient_boosting | corels | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | gradient_boosting | feature_ablation | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | mlp | shap | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | mlp | lime | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | mlp | integrated_gradients | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | mlp | causal_shap | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | mlp | shapley_flow | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | mlp | shap_interactive | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | mlp | prototype | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | mlp | counterfactual | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | mlp | influence_functions | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | mlp | bayesian_rule_list | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | mlp | corels | 0 | 0 | 0.000 | 0.0000 | N/A |
| adult_income | mlp | feature_ablation | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | decision_tree | shap | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | decision_tree | lime | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | decision_tree | causal_shap | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | decision_tree | shapley_flow | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | decision_tree | shap_interactive | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | decision_tree | prototype | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | decision_tree | counterfactual | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | decision_tree | bayesian_rule_list | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | decision_tree | corels | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | decision_tree | feature_ablation | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | random_forest | shap | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | random_forest | lime | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | random_forest | causal_shap | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | random_forest | shapley_flow | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | random_forest | shap_interactive | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | random_forest | prototype | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | random_forest | counterfactual | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | random_forest | bayesian_rule_list | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | random_forest | corels | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | random_forest | feature_ablation | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | gradient_boosting | shap | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | gradient_boosting | lime | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | gradient_boosting | causal_shap | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | gradient_boosting | shap_interactive | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | gradient_boosting | prototype | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | gradient_boosting | counterfactual | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | gradient_boosting | bayesian_rule_list | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | gradient_boosting | corels | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | gradient_boosting | feature_ablation | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | mlp | shap | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | mlp | lime | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | mlp | integrated_gradients | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | mlp | causal_shap | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | mlp | shapley_flow | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | mlp | shap_interactive | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | mlp | prototype | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | mlp | counterfactual | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | mlp | influence_functions | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | mlp | bayesian_rule_list | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | mlp | corels | 0 | 0 | 0.000 | 0.0000 | N/A |
| compas | mlp | feature_ablation | 0 | 0 | 0.000 | 0.0000 | N/A |
| mnist | cnn | prototype | 0 | 0 | 0.000 | 0.0000 | N/A |
| mnist | cnn | counterfactual | 0 | 0 | 0.000 | 0.0000 | N/A |
| mnist | cnn | tcav | 0 | 0 | 0.000 | 0.0000 | N/A |
| mnist | cnn | concept_bottleneck | 0 | 0 | 0.000 | 0.0000 | N/A |
| mnist | cnn | occlusion | 0 | 0 | 0.000 | 0.0000 | N/A |
| mnist | vit | tcav | 0 | 0 | 0.000 | 0.0000 | N/A |
| mnist | vit | concept_bottleneck | 0 | 0 | 0.000 | 0.0000 | N/A |
| mnist | vit | occlusion | 0 | 0 | 0.000 | 0.0000 | N/A |
| imdb | bert | lime | 0 | 0 | 0.000 | 0.0000 | N/A |
| imdb | bert | text_occlusion | 0 | 0 | 0.000 | 0.0000 | N/A |
| imdb | bert | attention_visualization | 0 | 0 | 0.000 | 0.0000 | N/A |
| imdb | lstm | lime | 0 | 0 | 0.000 | 0.0000 | N/A |
| imdb | lstm | text_occlusion | 0 | 0 | 0.000 | 0.0000 | N/A |
| imdb | lstm | attention_visualization | 0 | 0 | 0.000 | 0.0000 | N/A |

## Model Performance Analysis by Dataset

### adult_income

#### Model Performance Summary

| Model | Train Accuracy | Test Accuracy | Train Loss | Test Loss |
|-------|----------------|---------------|------------|----------|
| decision_tree | 0.8405 | 0.8326 | N/A | N/A |
| random_forest | 0.8425 | 0.8333 | N/A | N/A |
| gradient_boosting | 0.8387 | 0.8356 | N/A | N/A |
| mlp | 0.8257 | 0.8236 | N/A | N/A |

#### XAI Evaluation Results

| Model | Explanation Method | Time Complexity | Faithfulness | Monotonicity |
|-------|-------------------|--------|--------|--------|
| decision_tree | shap | 0.0011 | 0.1900 | 0.0200 |
| decision_tree | lime | 0.0128 | 0.1000 | 0.0267 |
| decision_tree | causal_shap | 0.0196 | 0.2400 | 0.0200 |
| decision_tree | shapley_flow | 0.0108 | 0.1000 | 0.0000 |
| decision_tree | shap_interactive | 0.0330 | 0.3000 | 0.0333 |
| decision_tree | prototype | 0.0010 | 0.7100 | 0.8436 |
| decision_tree | counterfactual | 0.0009 | 0.7200 | 0.1716 |
| decision_tree | bayesian_rule_list | 0.0006 | 0.0000 | 0.0000 |
| decision_tree | corels | 0.0003 | 0.0000 | 0.0000 |
| decision_tree | feature_ablation | 0.0006 | 0.0000 | 0.0000 |
| random_forest | shap | 0.0361 | 0.1900 | 0.0200 |
| random_forest | lime | 0.0211 | 0.1200 | 0.0400 |
| random_forest | causal_shap | 0.7504 | 0.2400 | 0.0200 |
| random_forest | shapley_flow | 0.3477 | 0.1333 | 0.0000 |
| random_forest | shap_interactive | 0.8816 | 0.2333 | 0.0333 |
| random_forest | prototype | 0.0043 | 0.7100 | 0.8102 |
| random_forest | counterfactual | 0.0051 | 0.7250 | 0.1994 |
| random_forest | bayesian_rule_list | 0.0045 | 0.0000 | 0.0000 |
| random_forest | corels | 0.0041 | 0.0000 | 0.0000 |
| random_forest | feature_ablation | 0.0301 | 0.0000 | 0.0000 |
| gradient_boosting | shap | 0.0026 | 0.2100 | 0.0300 |
| gradient_boosting | lime | 0.0158 | 0.1400 | 0.0400 |
| gradient_boosting | causal_shap | 0.0534 | 0.2600 | 0.0200 |
| gradient_boosting | shap_interactive | 0.0530 | 0.3333 | 0.0333 |
| gradient_boosting | prototype | 0.0009 | 0.7100 | 0.8099 |
| gradient_boosting | counterfactual | 0.0008 | 0.6950 | 0.1864 |
| gradient_boosting | bayesian_rule_list | 0.0010 | 0.0000 | 0.0000 |
| gradient_boosting | corels | 0.0006 | 0.0000 | 0.0000 |
| gradient_boosting | feature_ablation | 0.0011 | 0.0000 | 0.0000 |
| mlp | shap | 0.0009 | 0.2200 | 0.0250 |
| mlp | lime | 0.0114 | 0.0600 | 0.0333 |
| mlp | integrated_gradients | 0.0459 | 0.0000 | 0.0000 |
| mlp | causal_shap | 0.0212 | 0.2400 | 0.0100 |
| mlp | shapley_flow | 0.0087 | 0.1000 | 0.0000 |
| mlp | shap_interactive | 0.0255 | 0.3000 | 0.0150 |
| mlp | prototype | 0.0009 | 0.7150 | 0.8053 |
| mlp | counterfactual | 0.0006 | 0.7050 | 0.2038 |
| mlp | influence_functions | 0.0199 | 0.0000 | 0.0000 |
| mlp | bayesian_rule_list | 0.0007 | 0.0000 | 0.0000 |
| mlp | corels | 0.0004 | 0.0000 | 0.0000 |
| mlp | feature_ablation | 0.0006 | 0.0000 | 0.0000 |

### compas

#### Model Performance Summary

| Model | Train Accuracy | Test Accuracy | Train Loss | Test Loss |
|-------|----------------|---------------|------------|----------|
| decision_tree | 0.7375 | 0.6736 | N/A | N/A |
| random_forest | 0.7538 | 0.6826 | N/A | N/A |
| gradient_boosting | 0.7054 | 0.6951 | N/A | N/A |
| mlp | 0.6881 | 0.6854 | N/A | N/A |

#### XAI Evaluation Results

| Model | Explanation Method | Time Complexity | Faithfulness | Monotonicity |
|-------|-------------------|--------|--------|--------|
| decision_tree | shap | 0.0005 | 0.6500 | 0.0300 |
| decision_tree | lime | 0.0144 | 0.3200 | 0.0800 |
| decision_tree | causal_shap | 0.0097 | 0.5000 | 0.0000 |
| decision_tree | shapley_flow | 0.0044 | 0.3333 | 0.0000 |
| decision_tree | shap_interactive | 0.0103 | 0.6667 | 0.0222 |
| decision_tree | prototype | 0.0003 | 0.6500 | 0.7255 |
| decision_tree | counterfactual | 0.0003 | 0.6100 | 0.2632 |
| decision_tree | bayesian_rule_list | 0.0006 | 0.0000 | 0.0000 |
| decision_tree | corels | 0.0003 | 0.0000 | 0.0000 |
| decision_tree | feature_ablation | 0.0003 | 0.0000 | 0.0000 |
| random_forest | shap | 0.0212 | 0.6300 | 0.0383 |
| random_forest | lime | 0.0201 | 0.3400 | 0.0900 |
| random_forest | causal_shap | 0.4260 | 0.4800 | 0.0300 |
| random_forest | shapley_flow | 0.1864 | 0.4333 | 0.0000 |
| random_forest | shap_interactive | 0.4056 | 0.6000 | 0.0611 |
| random_forest | prototype | 0.0032 | 0.7100 | 0.6912 |
| random_forest | counterfactual | 0.0032 | 0.6150 | 0.3043 |
| random_forest | bayesian_rule_list | 0.0034 | 0.0000 | 0.0000 |
| random_forest | corels | 0.0032 | 0.0000 | 0.0000 |
| random_forest | feature_ablation | 0.0115 | 0.0000 | 0.0000 |
| gradient_boosting | shap | 0.0010 | 0.6100 | 0.0250 |
| gradient_boosting | lime | 0.0100 | 0.4000 | 0.1400 |
| gradient_boosting | causal_shap | 0.0213 | 0.4600 | 0.0300 |
| gradient_boosting | shap_interactive | 0.0221 | 0.6333 | 0.0611 |
| gradient_boosting | prototype | 0.0003 | 0.6550 | 0.6883 |
| gradient_boosting | counterfactual | 0.0004 | 0.6300 | 0.3109 |
| gradient_boosting | bayesian_rule_list | 0.0005 | 0.0000 | 0.0000 |
| gradient_boosting | corels | 0.0003 | 0.0000 | 0.0000 |
| gradient_boosting | feature_ablation | 0.0007 | 0.0000 | 0.0000 |
| mlp | shap | 0.0005 | 0.5800 | 0.0100 |
| mlp | lime | 0.0084 | 0.0400 | 0.0000 |
| mlp | integrated_gradients | 0.0233 | 0.0000 | 0.0000 |
| mlp | causal_shap | 0.0104 | 0.3400 | 0.0000 |
| mlp | shapley_flow | 0.0052 | 0.2333 | 0.0000 |
| mlp | shap_interactive | 0.0114 | 0.4667 | 0.0111 |
| mlp | prototype | 0.0002 | 0.6250 | 0.6891 |
| mlp | counterfactual | 0.0002 | 0.6950 | 0.3141 |
| mlp | influence_functions | 0.0172 | 0.0000 | 0.0000 |
| mlp | bayesian_rule_list | 0.0004 | 0.0000 | 0.0000 |
| mlp | corels | 0.0002 | 0.0000 | 0.0000 |
| mlp | feature_ablation | 0.0003 | 0.0000 | 0.0000 |

### imdb

#### Model Performance Summary

| Model | Train Accuracy | Test Accuracy | Train Loss | Test Loss |
|-------|----------------|---------------|------------|----------|
| bert | 0.9180 | 0.8100 | N/A | N/A |
| lstm | 0.8870 | 0.8150 | N/A | N/A |

#### XAI Evaluation Results

| Model | Explanation Method | Time Complexity | Faithfulness | Monotonicity |
|-------|-------------------|--------|--------|--------|
| bert | lime | 0.0552 | 0.8400 | 0.0000 |
| bert | text_occlusion | 0.0316 | 0.0000 | 0.0000 |
| bert | attention_visualization | 0.0811 | 0.0000 | 0.0000 |
| lstm | lime | 0.0583 | 0.8000 | 0.0000 |
| lstm | text_occlusion | 0.0345 | 0.0000 | 0.0000 |
| lstm | attention_visualization | 0.0861 | 0.0000 | 0.0000 |

### mnist

#### Model Performance Summary

| Model | Train Accuracy | Test Accuracy | Train Loss | Test Loss |
|-------|----------------|---------------|------------|----------|
| cnn | 1.0000 | 0.9900 | N/A | N/A |
| vit | 0.8090 | 0.7350 | N/A | N/A |

#### XAI Evaluation Results

| Model | Explanation Method | Time Complexity | Faithfulness | Monotonicity |
|-------|-------------------|--------|--------|--------|
| cnn | prototype | 0.0012 | 0.9900 | 0.65905195 |
| cnn | counterfactual | 0.0026 | 0.9900 | 0.46621537 |
| cnn | tcav | 0.0000 | 0.0000 | 0.0000 |
| cnn | concept_bottleneck | 0.0000 | 0.0000 | 0.0000 |
| cnn | occlusion | 0.0227 | 0.0000 | 0.0000 |
| vit | tcav | 0.0000 | 0.0000 | 0.0000 |
| vit | concept_bottleneck | 0.0000 | 0.0000 | 0.0000 |
| vit | occlusion | 0.0721 | 0.0000 | 0.0000 |

## Best Performing Models by Dataset

Ranking models by test accuracy on each dataset.

### adult_income - Model Rankings

| Rank | Model | Test Accuracy |
|------|-------|---------------|
| 1 | gradient_boosting | 0.8356 |
| 2 | random_forest | 0.8333 |
| 3 | decision_tree | 0.8326 |
| 4 | mlp | 0.8236 |

### compas - Model Rankings

| Rank | Model | Test Accuracy |
|------|-------|---------------|
| 1 | gradient_boosting | 0.6951 |
| 2 | mlp | 0.6854 |
| 3 | random_forest | 0.6826 |
| 4 | decision_tree | 0.6736 |

### imdb - Model Rankings

| Rank | Model | Test Accuracy |
|------|-------|---------------|
| 1 | lstm | 0.8150 |
| 2 | bert | 0.8100 |

### mnist - Model Rankings

| Rank | Model | Test Accuracy |
|------|-------|---------------|
| 1 | cnn | 0.9900 |
| 2 | vit | 0.7350 |

## Top Performing XAI Combinations

### Best Time Complexity

| Rank | Dataset | Model | Explanation | Score |
|------|---------|-------|-------------|-------|
| 1 | adult_income | random_forest | shap_interactive | 0.8816 |
| 2 | adult_income | random_forest | causal_shap | 0.7504 |
| 3 | compas | random_forest | causal_shap | 0.4260 |
| 4 | compas | random_forest | shap_interactive | 0.4056 |
| 5 | adult_income | random_forest | shapley_flow | 0.3477 |
| 6 | compas | random_forest | shapley_flow | 0.1864 |
| 7 | imdb | lstm | attention_visualization | 0.0861 |
| 8 | imdb | bert | attention_visualization | 0.0811 |
| 9 | mnist | vit | occlusion | 0.0721 |
| 10 | imdb | lstm | lime | 0.0583 |

### Best Faithfulness

| Rank | Dataset | Model | Explanation | Score |
|------|---------|-------|-------------|-------|
| 1 | mnist | cnn | prototype | 0.9900 |
| 2 | mnist | cnn | counterfactual | 0.9900 |
| 3 | imdb | bert | lime | 0.8400 |
| 4 | imdb | lstm | lime | 0.8000 |
| 5 | adult_income | random_forest | counterfactual | 0.7250 |
| 6 | adult_income | decision_tree | counterfactual | 0.7200 |
| 7 | adult_income | mlp | prototype | 0.7150 |
| 8 | adult_income | decision_tree | prototype | 0.7100 |
| 9 | adult_income | random_forest | prototype | 0.7100 |
| 10 | adult_income | gradient_boosting | prototype | 0.7100 |

### Best Monotonicity

| Rank | Dataset | Model | Explanation | Score |
|------|---------|-------|-------------|-------|
| 1 | adult_income | decision_tree | prototype | 0.8436 |
| 2 | adult_income | random_forest | prototype | 0.8102 |
| 3 | adult_income | gradient_boosting | prototype | 0.8099 |
| 4 | adult_income | mlp | prototype | 0.8053 |
| 5 | compas | decision_tree | prototype | 0.7255 |
| 6 | compas | random_forest | prototype | 0.6912 |
| 7 | compas | mlp | prototype | 0.6891 |
| 8 | compas | gradient_boosting | prototype | 0.6883 |
| 9 | compas | mlp | counterfactual | 0.3141 |
| 10 | compas | gradient_boosting | counterfactual | 0.3109 |

### Best Completeness

| Rank | Dataset | Model | Explanation | Score |
|------|---------|-------|-------------|-------|
| 1 | adult_income | decision_tree | prototype | 1.0000 |
| 2 | adult_income | decision_tree | counterfactual | 1.0000 |
| 3 | adult_income | random_forest | prototype | 1.0000 |
| 4 | adult_income | random_forest | counterfactual | 1.0000 |
| 5 | adult_income | gradient_boosting | prototype | 1.0000 |
| 6 | adult_income | gradient_boosting | counterfactual | 1.0000 |
| 7 | adult_income | mlp | prototype | 1.0000 |
| 8 | adult_income | mlp | counterfactual | 1.0000 |
| 9 | compas | decision_tree | prototype | 1.0000 |
| 10 | compas | decision_tree | counterfactual | 1.0000 |

### Best Stability

| Rank | Dataset | Model | Explanation | Score |
|------|---------|-------|-------------|-------|
| 1 | adult_income | decision_tree | shapley_flow | 1.0000 |
| 2 | adult_income | random_forest | shapley_flow | 1.0000 |
| 3 | adult_income | mlp | integrated_gradients | 1.0000 |
| 4 | adult_income | mlp | shapley_flow | 1.0000 |
| 5 | compas | decision_tree | shapley_flow | 1.0000 |
| 6 | compas | random_forest | shapley_flow | 1.0000 |
| 7 | compas | mlp | integrated_gradients | 1.0000 |
| 8 | compas | mlp | shapley_flow | 1.0000 |
| 9 | compas | random_forest | prototype | 0.9954 |
| 10 | compas | mlp | counterfactual | 0.9896 |

