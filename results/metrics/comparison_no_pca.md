# Table 1: Model Comparison — No PCA (Raw Dimensions)

**Dataset:** Food-101 (20 classes)
**Cross-validation:** Stratified 5-Fold


| Model         | Features   |   CV Acc |   CV F1 |   Test Acc |   Test F1 |   Time (s) |
|:--------------|:-----------|---------:|--------:|-----------:|----------:|-----------:|
| mlp_sklearn   | histogram  |   0.2466 |  0.2362 |     0.2666 |    0.2643 |      142.1 |
| mlp_sklearn   | fused      |   0.2472 |  0.2423 |     0.2636 |    0.261  |     3199.1 |
| logistic      | histogram  |   0.2274 |  0.2181 |     0.2386 |    0.2299 |        9   |
| mlp_sklearn   | hog        |   0.2097 |  0.211  |     0.2304 |    0.2276 |     3356.2 |
| logistic      | lbp        |   0.2211 |  0.2021 |     0.2252 |    0.2063 |        3.6 |
| naive_bayes   | fused      |   0.2094 |  0.1935 |     0.212  |    0.1967 |       82.3 |
| knn           | histogram  |   0.1865 |  0.1697 |     0.2052 |    0.1866 |       39.2 |
| mlp_sklearn   | lbp        |   0.1901 |  0.1857 |     0.2024 |    0.2001 |      144.9 |
| logistic      | glcm       |   0.1925 |  0.1816 |     0.2018 |    0.1919 |       20.2 |
| naive_bayes   | hog        |   0.1911 |  0.1744 |     0.1924 |    0.176  |       99.2 |
| kde           | histogram  |   0.1638 |  0.1481 |     0.1864 |    0.1665 |       92.7 |
| mlp_sklearn   | glcm       |   0.1713 |  0.1707 |     0.1862 |    0.1858 |      160   |
| kde           | lbp        |   0.1685 |  0.1422 |     0.1804 |    0.1515 |       22.5 |
| knn           | lbp        |   0.157  |  0.1465 |     0.1706 |    0.1589 |       10   |
| naive_bayes   | histogram  |   0.1671 |  0.14   |     0.1702 |    0.1429 |        0.6 |
| decision_tree | histogram  |   0.1531 |  0.15   |     0.156  |    0.1546 |       27.1 |
| decision_tree | fused      |   0.1446 |  0.1428 |     0.155  |    0.1455 |     4667.8 |
| perceptron    | fused      |   0.1491 |  0.1478 |     0.155  |    0.1551 |      838.2 |
| naive_bayes   | lbp        |   0.1367 |  0.1028 |     0.1414 |    0.1063 |        0.6 |
| kde           | glcm       |   0.133  |  0.1123 |     0.137  |    0.1136 |       26.6 |
| decision_tree | lbp        |   0.1231 |  0.1152 |     0.1312 |    0.1258 |        4.3 |
| perceptron    | histogram  |   0.1173 |  0.1125 |     0.1312 |    0.1198 |        3.4 |
| knn           | glcm       |   0.1235 |  0.1186 |     0.1308 |    0.1258 |        7.6 |
| perceptron    | hog        |   0.1224 |  0.1211 |     0.1246 |    0.1239 |      816.6 |
| decision_tree | glcm       |   0.1084 |  0.1012 |     0.1196 |    0.113  |        7.3 |
| perceptron    | lbp        |   0.11   |  0.099  |     0.1194 |    0.1068 |        1.1 |
| perceptron    | glcm       |   0.0866 |  0.0689 |     0.1064 |    0.0865 |        1.3 |
| knn           | fused      |   0.0987 |  0.0777 |     0.1036 |    0.0834 |     4638.9 |
| naive_bayes   | glcm       |   0.1011 |  0.054  |     0.1016 |    0.0503 |        0.5 |
| knn           | hog        |   0.0949 |  0.0737 |     0.0958 |    0.0746 |     4662.9 |
| decision_tree | hog        |   0.0767 |  0.0751 |     0.074  |    0.0744 |     6538.3 |


**Best:** mlp_sklearn on histogram — 26.66% accuracy
