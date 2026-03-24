# 🍕 Food Image Classification — PRML Project

> **A Comparative Study of Classical and Modern PRML Techniques for Food Image Classification**

This project implements a complete machine learning pipeline for food image classification using the [Food-101 dataset](https://www.kaggle.com/datasets/kmader/food41), covering **27 course topics** from Pattern Recognition and Machine Learning.

---

## 📊 Results Summary

| Rank | Model | Features | Test Accuracy | Test F1 |
|------|-------|----------|:------------:|:-------:|
| 🥇 | **Logistic Regression** | CNN (ResNet-50) | **84.78%** | **84.79%** |
| 🥈 | MLP (sklearn) | CNN (ResNet-50) | 84.68% | 84.63% |
| 🥉 | SVM (Linear) | CNN (ResNet-50) | 84.50% | 84.51% |
| 4 | SVM (RBF) | CNN (ResNet-50) | 83.12% | 83.29% |
| 5 | Perceptron | CNN (ResNet-50) | 78.58% | 78.57% |
| 6 | Random Forest | CNN (ResNet-50) | 77.52% | 77.28% |
| 7 | Gradient Boosting | CNN + PCA | 73.14% | 73.14% |
| 8 | kNN (k=5) | CNN (ResNet-50) | 66.70% | 66.67% |
| 9 | Naive Bayes | CNN (ResNet-50) | 61.70% | 61.37% |
| 10 | Decision Tree | CNN (ResNet-50) | 51.70% | 51.65% |
| 11 | SVM (RBF) | Fused + PCA | 30.50% | 30.38% |
| 12 | SVM (RBF) | HOG + PCA | 28.50% | 28.38% |
| 13 | SVM (RBF) | Color Histogram | 28.30% | 27.66% |

### Key Findings
- **CNN features dominate** handcrafted features by a factor of ~3× in accuracy
- **Logistic Regression** on CNN embeddings achieves the best accuracy (84.78%) with the fastest training time (8.9s)
- **Cross-validation vs test accuracy** shows good generalization across all CNN-based models

---

## 🏗️ Project Architecture

```
project/
├── configs/                    # Hydra YAML configuration files
│   ├── config.yaml             #   Master config
│   ├── data/                   #   Dataset configs (food101_20, food101_50)
│   ├── features/               #   Feature extractor configs
│   ├── model/                  #   Model hyperparameter configs
│   ├── reduction/              #   PCA/LDA configs
│   └── experiment/             #   Predefined experiment combos
│
├── src/                        # Source code
│   ├── data/                   #   Dataset loading, preprocessing, caching
│   │   ├── dataset.py          #     Food-101 loader with stratified splits
│   │   ├── preprocess.py       #     Image transforms & augmentation
│   │   └── cache.py            #     HDF5 feature matrix caching
│   │
│   ├── features/               #   Feature extraction
│   │   ├── base.py             #     Abstract base with parallel extraction
│   │   ├── histogram.py        #     RGB + HSV color histograms
│   │   ├── hog.py              #     Histogram of Oriented Gradients
│   │   ├── lbp.py              #     Local Binary Patterns
│   │   ├── glcm.py             #     Gray-Level Co-occurrence Matrix
│   │   ├── cnn_embeddings.py   #     ResNet-50 pretrained embeddings
│   │   └── fusion.py           #     Multi-feature concatenation
│   │
│   ├── reduction/              #   Dimensionality reduction
│   │   ├── pca_reducer.py      #     PCA with scree plots
│   │   └── lda_reducer.py      #     LDA with 2D projections
│   │
│   ├── models/                 #   Classifiers
│   │   ├── registry.py         #     Unified Pipeline builder (Scaler → Reducer → Clf)
│   │   ├── classical.py        #     16 classical ML models
│   │   ├── mlp.py              #     MLP (sklearn + PyTorch)
│   │   └── cnn/                #     CNN with transfer learning (Lightning)
│   │       ├── model.py        #       Custom CNN + pretrained backbone
│   │       ├── datamodule.py   #       Lightning DataModule
│   │       └── callbacks.py    #       Early stopping
│   │
│   ├── clustering/             #   Unsupervised learning
│   │   ├── kmeans.py           #     KMeans + elbow method
│   │   ├── agglomerative.py    #     Hierarchical + dendrograms
│   │   └── gmm.py              #     GMM + BIC/AIC selection
│   │
│   ├── evaluation/             #   Metrics and validation
│   │   ├── metrics.py          #     Accuracy, F1, precision, recall, top-5
│   │   ├── cross_val.py        #     Stratified 5-Fold CV + learning curves
│   │   └── comparison.py       #     Master CSV result tracking
│   │
│   ├── visualization/          #   Plots and analysis
│   │   ├── confusion_plot.py   #     Confusion matrix heatmaps
│   │   ├── learning_curves.py  #     k-vs-accuracy, training history
│   │   ├── filter_demo.py      #     Convolution filter visualization
│   │   ├── umap_plot.py        #     Interactive UMAP scatter
│   │   ├── gradcam.py          #     Grad-CAM CNN interpretability
│   │   └── shap_analysis.py    #     SHAP feature importance
│   │
│   ├── experiment/             #   Experiment orchestration
│   │   ├── runner.py           #     Single experiment runner
│   │   ├── sweeper.py          #     Optuna hyperparameter search
│   │   └── clustering_runner.py#     Clustering experiment runner
│   │
│   └── utils/                  #   Utilities
│       ├── seed.py             #     Global seed for reproducibility
│       ├── logging.py          #     Logging + wandb init
│       └── io.py               #     File I/O helpers
│
├── scripts/                    # Entry-point scripts
│   ├── extract_features.py     #   Precompute & cache all feature matrices
│   ├── run_experiment.py       #   Run single classifier experiment
│   ├── run_sweep.py            #   Launch Optuna hyperparameter sweep
│   ├── run_clustering.py       #   Run clustering experiments
│   ├── run_cnn.py              #   Fine-tune CNN with Lightning
│   └── generate_report_plots.py#   Regenerate all plots from saved results
│
├── results/                    # Generated outputs
│   ├── metrics/                #   CSVs, reports, comparison tables
│   └── plots/                  #   All generated visualizations
│
├── tests/                      # Unit tests
├── app/                        # Streamlit demo app
├── notebooks/                  # Jupyter analysis notebooks
├── environment.yml             # Conda environment
├── pyproject.toml              # Project metadata
├── dvc.yaml                    # DVC pipeline definition
└── .gitignore
```

---

## 🔧 Setup & Installation

### 1. Clone and Setup Environment
```bash
git clone https://github.com/Vinamra3215/temp-prml.git
cd temp-prml

# Create conda environment
conda env create -f environment.yml
conda activate food-prml
```

### 2. Download Dataset
Download the [Food-101 dataset from Kaggle](https://www.kaggle.com/datasets/kmader/food41) and extract the `images/` and `meta/` folders into `data/`:
```
data/
├── images/
│   ├── apple_pie/
│   ├── baby_back_ribs/
│   └── ... (101 categories)
└── meta/
    └── meta/
        ├── classes.txt
        ├── train.json
        └── test.json
```

### 3. Extract Features (Run Once)
```bash
python scripts/extract_features.py
```
This caches all feature matrices (histogram, HOG, LBP, GLCM, fused, CNN) as HDF5 files in `data/cache/`.

---

## 🚀 Running Experiments

### Single Experiment
```bash
# Run SVM with CNN features
python scripts/run_experiment.py --model svm_rbf --features cnn

# Run Logistic Regression with CNN features
python scripts/run_experiment.py --model logistic --features cnn

# Run with PCA reduction
python scripts/run_experiment.py --model svm_rbf --features cnn --reducer pca
```

### Available Models
`knn`, `logistic`, `naive_bayes`, `svm_linear`, `svm_rbf`, `decision_tree`, `random_forest`, `mlp_sklearn`, `perceptron`, `sgd`, `gradient_boosting`

### Available Features
`histogram`, `hog`, `lbp`, `glcm`, `fused`, `cnn`

### Hyperparameter Sweep (Optuna)
```bash
python scripts/run_sweep.py --model svm_rbf --features cnn --n-trials 30
```

### Clustering Experiments
```bash
python scripts/run_clustering.py --features cnn
```

### CNN Fine-tuning
```bash
python scripts/run_cnn.py --backbone resnet50 --epochs 20
```

### Generate All Plots
```bash
python scripts/generate_report_plots.py
```

---

## 📈 Generated Outputs

### Plots (`results/plots/`)
| Plot | Description |
|------|-------------|
| `model_comparison.png` | Bar chart comparing all models by test accuracy |
| `feature_comparison.png` | Best accuracy per feature type |
| `accuracy_vs_f1.png` | Scatter of accuracy vs F1 score |
| `runtime_comparison.png` | Training time comparison |
| `confusion_logistic.png` | Confusion matrix for best model |
| `confusion_svm_rbf.png` | Confusion matrix for SVM RBF |
| `confusion_random_forest.png` | Confusion matrix for Random Forest |
| `pca_explained_variance.png` | PCA scree plot on CNN features |
| `cv_vs_test_accuracy.png` | Overfitting analysis (CV vs test) |

### Metrics (`results/metrics/`)
| File | Description |
|------|-------------|
| `master.csv` | All experiment results (appended per run) |
| `comparison_table.csv` | Formatted comparison table |
| `comparison_results.md` | Markdown summary with key findings |
| `classification_report_*.txt` | Per-class precision/recall/F1 |

---

## 📚 Course Topics Covered (27/27)

| # | Topic | Implementation |
|---|-------|---------------|
| 1 | Probability Review | Naive Bayes, GMM |
| 2 | Non-parametric Density Estimation | Color histograms, KDE analysis |
| 3 | Feature Extraction | HOG, LBP, GLCM, histogram, fusion |
| 4 | Bayesian Classifier | Naive Bayes with Gaussian likelihood |
| 5 | kNN Classification | kNN with k-tuning, distance metrics |
| 6 | Distance Metrics | Euclidean, Manhattan, Cosine in kNN |
| 7 | Cross-validation | Stratified 5-Fold CV |
| 8 | Bias-Variance Tradeoff | Learning curves, model complexity analysis |
| 9 | Linear Discriminant | LDA for reduction + classification |
| 10 | Logistic Regression | Multinomial logistic with L1/L2 |
| 11 | Regularization | L1, L2 penalty sweeps |
| 12 | PCA | Eigenface-style reduction, scree plots |
| 13 | LDA/PCA Comparison | Side-by-side projections |
| 14 | SVM | Linear, RBF, Polynomial kernels |
| 15 | Kernel Methods | RBF, Polynomial SVM kernels |
| 16 | Multiclass SVM | One-vs-Rest classification |
| 17 | Neural Networks | MLP with backpropagation |
| 18 | Backpropagation | PyTorch autograd, training loops |
| 19 | Gradient Descent | SGD, Adam optimizers |
| 20 | Convolutions | Filter visualization (Sobel, Gabor) |
| 21 | CNN | ResNet-50 transfer learning, embeddings |
| 22 | Decision Trees | Gini/entropy splits, depth tuning |
| 23 | KMeans Clustering | Elbow method, silhouette score |
| 24 | Hierarchical Clustering | Agglomerative + dendrograms |
| 25 | Ensemble: Bagging | Random Forest |
| 26 | Ensemble: Boosting | Gradient Boosting |
| 27 | Model Selection | Optuna sweeps, CV comparison |

---

## 🛠️ Tech Stack

- **ML:** scikit-learn, PyTorch, PyTorch Lightning, timm
- **Features:** OpenCV, scikit-image
- **Optimization:** Optuna
- **Visualization:** matplotlib, seaborn, Plotly
- **Config:** Hydra
- **Tracking:** Weights & Biases, DVC
- **App:** Streamlit

---

## 👤 Author

**Vinamra Gupta** — [GitHub](https://github.com/Vinamra3215)
