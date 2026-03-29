# 🍕 Food Image Classification — PRML Project

End-to-end machine learning pipeline for classifying food images from 20 categories of the [Food-101 dataset](https://www.kaggle.com/datasets/dansbecker/food-101).

**No pretrained models** — uses only classical ML algorithms and handcrafted feature extraction, covering core PRML course topics including dimensionality reduction, kernel methods, density estimation, and neural networks.

---

## 📊 Project Highlights

| Aspect | Detail |
|--------|--------|
| **Dataset** | Food-101 (20 classes, ~15K images) |
| **Features** | Color Histogram, HOG, LBP, GLCM, Fused |
| **Models** | KNN, Logistic Regression, Naive Bayes, Decision Tree, Gradient Boosting, Perceptron, MLP, KDE (Parzen Window) |
| **Hyperparameters** | Selected via **GridSearchCV** (not hardcoded) |
| **Dimensionality Reduction** | PCA (200 components) |
| **Experiment Tracking** | Local CSV + optional Weights & Biases |
| **Optimization** | Optuna (Bayesian / TPE sampler) |

---

## 🏗 Architecture

```
project/
├── configs/                    # Hydra YAML configurations
│   ├── model/                  # Per-model configs (knn, logistic, kde, etc.)
│   └── experiment/             # Experiment sweep configs
│
├── scripts/                    # Entry points
│   ├── extract_features.py     # Cache all handcrafted features (HDF5)
│   ├── run_all_experiments.py  # ★ Main pipeline: 4 phases, GridSearchCV
│   ├── run_experiment.py       # Single model experiment
│   ├── run_sweep.py            # Optuna hyperparameter sweep
│   └── generate_report_plots.py # Generate 8 presentation plots
│
├── src/
│   ├── data/
│   │   ├── dataset.py          # Food101Dataset (Kaggle folder layout)
│   │   └── cache.py            # HDF5 feature caching
│   │
│   ├── features/               # Feature extractors
│   │   ├── histogram.py        # Color histogram (HSV, 32 bins)
│   │   ├── hog.py              # Histogram of Oriented Gradients
│   │   ├── lbp.py              # Local Binary Patterns
│   │   ├── glcm.py             # Gray-Level Co-occurrence Matrix
│   │   └── fusion.py           # Fused multi-feature vector
│   │
│   ├── models/
│   │   ├── registry.py         # Model + reducer registry, PARAM_GRIDS
│   │   └── classical.py        # Classical models + KDEClassifier
│   │
│   ├── evaluation/
│   │   ├── metrics.py          # Accuracy, F1, precision, recall
│   │   ├── cross_val.py        # Stratified K-Fold CV
│   │   └── comparison.py       # Master CSV result logger
│   │
│   ├── experiment/
│   │   ├── runner.py           # Core runner with GridSearchCV + W&B
│   │   └── sweeper.py          # Optuna sweep with per-model search spaces
│   │
│   ├── analysis/
│   │   └── model_insights.py   # Confusion analysis, auto summary, ranking
│   │
│   ├── reduction/
│   │   └── pca_analysis.py     # PCA variance analysis
│   │
│   └── utils/
│       ├── seed.py             # Reproducibility (numpy, sklearn)
│       └── logging.py          # W&B integration + CSV experiment log
│
├── results/
│   ├── plots/                  # 8 curated analysis plots
│   └── metrics/                # CSV results, sweep JSONs, summaries
│
├── tests/                      # Unit tests
├── notebooks/                  # Jupyter exploration notebooks
└── app/                        # Streamlit demo app
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
conda env create -f environment.yml
conda activate food-prml
```

### 2. Prepare Data

Download [Food-101 from Kaggle](https://www.kaggle.com/datasets/dansbecker/food-101) and extract to:
```
data/
├── images/
│   ├── apple_pie/
│   ├── baby_back_ribs/
│   └── ...
└── meta/
    └── meta/
        ├── train.json
        └── test.json
```

### 3. Extract Features

```bash
python scripts/extract_features.py
```

Extracts 5 feature types (Histogram, HOG, LBP, GLCM, Fused) and caches as HDF5 files in `data/cache/`.

### 4. Run All Experiments

```bash
# Full pipeline — all 4 phases
python scripts/run_all_experiments.py

# With Weights & Biases tracking
python scripts/run_all_experiments.py --wandb

# Run specific phase only
python scripts/run_all_experiments.py --phase 1   # Table 1: No PCA
python scripts/run_all_experiments.py --phase 2   # Table 2: PCA-200
python scripts/run_all_experiments.py --phase 3   # MLP loss curve
python scripts/run_all_experiments.py --phase 4   # Auto summary
```

### 5. Generate Report Plots

```bash
python scripts/generate_report_plots.py
```

---

## 🧪 Experiment Pipeline

The pipeline runs in **4 phases**:

### Phase 1 — No PCA (Raw Dimensions)
All 8 models × 5 feature types = **40 experiments**
- Hyperparameters selected via **GridSearchCV** (5-fold CV, F1 macro scoring)
- Results saved to `master_no_pca.csv`

### Phase 2 — With PCA (200 Components)
Same 40 experiments with PCA dimensionality reduction applied uniformly
- Fair comparison: all models see the same 200-dimensional input
- Results saved to `master_with_pca.csv`

### Phase 3 — MLP Training Loss Curve
- Trains MLP (512→256 hidden units) on fused features
- Records per-epoch training and validation loss
- Saves to `cnn_history.json`

### Phase 4 — Auto Summary
- Generates text summary with best model, feature analysis, efficiency metrics
- Saves to `summary_*.txt`

---

## 🔧 Hyperparameter Tuning

**All hyperparameters are found via GridSearchCV** — nothing is hardcoded.

| Model | Search Space |
|-------|-------------|
| **KNN** | k ∈ {1,3,5,7,9,11,15,21}, weights ∈ {uniform, distance}, metric ∈ {euclidean, manhattan} |
| **Logistic Regression** | C ∈ {0.01, 0.1, 1.0, 10.0} |
| **Naive Bayes** | var_smoothing ∈ {1e-9, 1e-8, 1e-7, 1e-6} |
| **Decision Tree** | max_depth ∈ {5,10,20,30,None}, min_samples_split ∈ {2,5,10} |
| **Gradient Boosting** | n_estimators ∈ {50,100,200}, max_depth ∈ {3,5,7}, lr ∈ {0.01,0.1,0.2} |
| **MLP** | layers ∈ {(128,),(256,128),(512,256,128)}, activation ∈ {relu, tanh} |
| **Perceptron** | penalty ∈ {None, l1, l2}, alpha ∈ {1e-4, 1e-3, 1e-2} |
| **KDE (Parzen)** | bandwidth ∈ {0.1, 0.5, 1.0, 2.0, 5.0}, kernel ∈ {gaussian, tophat, epanechnikov} |

Additionally, **Optuna** (Bayesian optimization with TPE sampler) can be used for deeper sweeps:

```bash
python scripts/run_sweep.py --model knn --features fused --n-trials 30
python scripts/run_sweep.py --model logistic --features hog --wandb
```

---

## 📈 Analysis Plots (8 Total)

| # | Plot | Purpose | Viva Topic |
|---|------|---------|------------|
| 1 | `model_comparison.png` | All models ranked by accuracy | Model selection |
| 2 | `feature_comparison.png` | Best accuracy per feature type | Feature ablation |
| 3 | `pca_explained_variance.png` | Variance vs PCA components | Dimensionality reduction |
| 4 | `accuracy_vs_f1.png` | Accuracy-F1 correlation | Metric analysis |
| 5 | `cv_vs_test_accuracy.png` | Generalization gap | Overfitting / Bias-Variance |
| 6 | `confusion_best_model.png` | Error patterns for best model | Error analysis |
| 7 | `runtime_comparison.png` | Computational cost | Efficiency tradeoffs |
| 8 | `cnn_loss_curve.png` | Training vs validation loss | Neural network convergence |

---

## 🎓 PRML Course Topics Covered

| # | Topic | Implementation |
|---|-------|---------------|
| 1 | Bayesian Decision Theory | KDE classifier (generative Bayesian model) |
| 2 | Density Estimation & Parzen Window | `KDEClassifier` with multiple kernels |
| 3 | Dimensionality Reduction (PCA) | PCA-200 applied uniformly; variance analysis |
| 4 | Linear Discriminant Analysis | LDA available as reducer in registry |
| 5 | K-Nearest Neighbors | KNN with GridSearchCV over k, weights, metric |
| 6 | Naive Bayes Classifier | GaussianNB with variance smoothing tuning |
| 7 | Logistic Regression | Multinomial logistic with L2 regularization |
| 8 | Perceptron & Linear Models | Perceptron, SGDClassifier |
| 9 | Neural Networks (MLP) | MLPClassifier with loss curve tracking |
| 10 | Decision Trees | Pruned via max_depth, min_samples tuning |
| 11 | Ensemble Methods | Gradient Boosting (boosting analysis) |
| 12 | Model Evaluation | Stratified K-Fold CV, confusion matrix, F1 |
| 13 | Hyperparameter Optimization | GridSearchCV + Optuna (Bayesian) |
| 14 | Feature Engineering | HOG, LBP, GLCM, Color Histogram, Fusion |
| 15 | Bias-Variance Tradeoff | CV vs Test accuracy analysis |
| 16 | Experiment Tracking | W&B integration, CSV logging |

---

## 🛠 Feature Extractors

| Feature | Description | Dimensions |
|---------|-------------|:----------:|
| **Color Histogram** | HSV color distribution (32 bins/channel) | 96 |
| **HOG** | Edge orientation histograms (8×8 cells) | ~8,100 |
| **LBP** | Local texture patterns | ~52 |
| **GLCM** | Texture statistics (contrast, correlation, etc.) | ~24 |
| **Fused** | Concatenation of all above | ~8,272 |

---

## 📦 Dependencies

```
scikit-learn >= 1.3
numpy
pandas
matplotlib
seaborn
h5py
tqdm
optuna          # Bayesian hyperparameter optimization
wandb           # Experiment tracking (optional)
hydra-core      # Configuration management
```

---

## 🧑‍💻 Single Experiment

```bash
# Run one model on one feature type
python scripts/run_experiment.py --model knn --features hog

# With PCA reduction and W&B
python scripts/run_experiment.py --model logistic --features fused --reducer pca --pca-components 200 --wandb
```

---

## 📁 Output Structure

After running the full pipeline:

```
results/
├── metrics/
│   ├── master_no_pca.csv           # Table 1: all models, raw features
│   ├── master_with_pca.csv         # Table 2: all models, PCA-200
│   ├── experiment_log.csv          # Detailed log with timestamps + params
│   ├── comparison_no_pca.md        # Formatted markdown comparison
│   ├── comparison_with_pca.md
│   ├── cnn_history.json            # MLP training loss curve data
│   ├── sweep_*.json                # Optuna sweep results
│   ├── summary_*.txt               # Auto-generated experiment summaries
│   └── classification_report_*.txt # Per-model classification reports
│
└── plots/
    ├── model_comparison.png
    ├── feature_comparison.png
    ├── pca_explained_variance.png
    ├── accuracy_vs_f1.png
    ├── cv_vs_test_accuracy.png
    ├── confusion_best_model.png
    ├── runtime_comparison.png
    └── cnn_loss_curve.png
```

---

## 🔬 Key Design Decisions

1. **No pretrained models**: All features are handcrafted (HOG, LBP, etc.) — no ResNet, ViT, or transfer learning. This keeps the project within classical ML / PRML scope.

2. **GridSearchCV for all models**: No hyperparameter is hardcoded. Every model's parameters are discovered through cross-validated grid search, ensuring reproducible and justifiable results.

3. **Two comparison tables**: Models are evaluated both with and without PCA to ensure fair comparison — you can't compare a model on 8,000-d features vs one on 200-d features.

4. **KDE (Parzen Window) Classifier**: Custom implementation of density-estimation-based classification, directly implementing Bayesian decision theory from PRML course.

5. **Reproducibility**: Fixed seed (42), stratified splits, cached features, detailed experiment logging.

---

## 📜 License

MIT License — Educational Project
