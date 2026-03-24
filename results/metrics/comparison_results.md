# Model Comparison Results

**Dataset:** Food-101 (20 classes, 12750 train, 5000 test)

**Cross-validation:** Stratified 5-Fold


| Model             | Features   | Reducer   |   CV Acc |   CV F1 |   Test Acc |   Test F1 |   Time (s) |
|:------------------|:-----------|:----------|---------:|--------:|-----------:|----------:|-----------:|
| logistic          | cnn        | none      |   0.8075 |  0.8074 |     0.8478 |    0.8479 |        8.9 |
| mlp_sklearn       | cnn        | none      |   0.8039 |  0.8036 |     0.8468 |    0.8463 |       17.4 |
| svm_linear        | cnn        | none      |   0.8023 |  0.8025 |     0.845  |    0.8451 |      267.8 |
| svm_rbf           | cnn        | none      |   0.7725 |  0.7762 |     0.8312 |    0.8329 |      547.7 |
| perceptron        | cnn        | none      |   0.7302 |  0.7295 |     0.7858 |    0.7857 |       27.2 |
| random_forest     | cnn        | none      |   0.7189 |  0.7155 |     0.7752 |    0.7728 |       88.1 |
| gradient_boosting | cnn        | pca       |   0.6707 |  0.6707 |     0.7314 |    0.7314 |      900.4 |
| knn               | cnn        | none      |   0.6151 |  0.6146 |     0.667  |    0.6667 |       30.1 |
| naive_bayes       | cnn        | none      |   0.5649 |  0.5603 |     0.617  |    0.6137 |       12.7 |
| decision_tree     | cnn        | none      |   0.4749 |  0.4756 |     0.517  |    0.5165 |       69.6 |
| svm_rbf           | fused      | pca       |   0.2835 |  0.281  |     0.305  |    0.3038 |      323.6 |
| svm_rbf           | hog        | pca       |   0.262  |  0.2599 |     0.285  |    0.2838 |      246.7 |
| svm_rbf           | histogram  | none      |   0.2657 |  0.2598 |     0.283  |    0.2766 |      183.6 |


## Key Findings

- **Best model:** logistic on cnn features — 84.78% accuracy

- **CNN features dominate:** Best CNN-based (84.78%) vs best handcrafted (30.50%)

- **Fastest model:** logistic (8.9s)
