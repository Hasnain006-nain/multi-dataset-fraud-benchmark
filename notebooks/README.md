# 📓 Notebooks

## Complete Experimental Workflow for Multi-Dataset Fraud Detection Benchmark

This directory contains the Jupyter notebook used to conduct all experiments, evaluations, and visualizations for the multi-dataset fraud detection benchmark study.

---

## 📁 Directory Structure

```
Notebooks/
│
└── Code.ipynb              # Main experimental notebook
```

---

## 📓 Main Notebook: `Code.ipynb`

The notebook provides a complete, end-to-end experimental workflow for evaluating machine learning and deep learning approaches for financial fraud detection across three diverse datasets.

### Notebook Overview

| Section | Description |
|---------|-------------|
| **1. Installation** | Install required dependencies |
| **2. Mount Drive** | Configure Google Drive and output folders |
| **3. Imports & Settings** | Import libraries and set figure styles |
| **4. Data Preparation** | Leakage-proof train/validation/test split |
| **5. Workflow & Figures** | Visualize class distributions |
| **6. Resampling** | Apply SMOTE/ADASYN to training only |
| **7. Evaluation Helpers** | Metrics, threshold selection, latency, bootstrap |
| **8. Model Training** | Train 12 model configurations |
| **9. MLP with Focal Loss** | Optional GPU-based deep learning |
| **10. Validation & Testing** | Threshold tuning and test evaluation |
| **11. Cost Sensitivity** | Analysis at different FN:FP ratios |
| **12. Temporal Drift** | Chronological validation |
| **13. Publication Figures** | Generate all manuscript figures |
| **14. SHAP Explainability** | Global and local explanations |
| **15. Save Models** | Export models and create manifest |

---

## 🔬 Included Experiments

### Data Processing
- Duplicate removal and label validation
- Leakage-column identification and removal
- Missing/infinite value handling
- Train/validation/test split (stratified)
- Preprocessing fitted on training only

### Resampling Strategies
- **SMOTE** — Synthetic Minority Over-sampling Technique
- **ADASYN** — Adaptive Synthetic Sampling
- Training-only application (no validation/test leakage)

### Models Evaluated

| Category | Models |
|----------|--------|
| **Linear** | Logistic Regression (balanced) |
| **Tree Ensembles** | Random Forest, Balanced RF, EasyEnsemble |
| **Gradient Boosting** | XGBoost, XGBoost + SPW, LightGBM, CatBoost |
| **Resampled Boosting** | XGBoost + SMOTE, XGBoost + ADASYN |
| **Deep Learning** | MLP + Focal Loss, MLP + Focal + SMOTE |

### Evaluation Framework
- Cost-sensitive threshold selection on validation
- MCC, Recall, Precision, F1, ROC-AUC, PR-AUC
- Expected cost per 10,000 transactions
- False alerts and missed frauds per 10,000
- Top-k alert budget analysis
- Bootstrap 95% confidence intervals
- McNemar-Holm statistical testing

### Deployment Analysis
- Repeated inference latency profiling
- Training time benchmarking
- Model size and efficiency comparison

### Explainability
- **SHAP Global** — Feature importance rankings
- **SHAP Local** — Individual case explanations
- Calibration analysis (Brier score, ECE)

### Temporal Validation
- Chronological split validation
- Feature drift analysis (PSI, KS statistics)

---

## 📊 Generated Outputs

The notebook automatically generates and saves:

### Figures (`../Figures/`)
| Output | Description |
|--------|-------------|
| `Fig1.png` | Workflow diagram |
| `02_class_distribution_before_resampling.png` | Original class distribution |
| `03_class_distribution_after_resampling.png` | After SMOTE/ADASYN |
| `04_model_performance_recall_mcc_prauc.png` | Model comparison |
| `05_precision_recall_curves.png` | PR curves per dataset |
| `06_cost_sensitive_performance.png` | Cost sensitivity |
| `07_topk_alert_budget_recall.png` | Review budget analysis |
| `08_latency_p95_microseconds.png` | Inference latency |
| `09_temporal_drift_performance.png` | Chronological validation |
| `10_calibration_curves.png` | Calibration plots |
| `11_mcnemar_holm_heatmap.png` | Statistical significance |
| `12_shap_global_*.png` | SHAP global importance |
| `13_shap_local_*.png` | SHAP local explanations |

### Results (`../Results/`)
| Output | Description |
|--------|-------------|
| `00_environment.csv` | Experiment configuration |
| `01_dataset_audit.csv` | Dataset characteristics |
| `02_class_distribution_before_resampling.csv` | Class distribution |
| `03_training_only_resampling_audit.csv` | Resampling details |
| `04_class_distribution_after_resampling.csv` | After resampling |
| `05_training_times.csv` | Training duration |
| `06_mlp_training_history.csv` | MLP loss curves |
| `07_validation_threshold_cost_sweep.csv` | Threshold sweep |
| `08_selected_thresholds.csv` | Optimal thresholds |
| `09_test_results_default_and_cost_tuned.csv` | Test results |
| `10_topk_alert_budget_metrics.csv` | Review budget |
| `11_repeated_latency_protocol.csv` | Latency measurements |
| `12_cost_sensitivity_results.csv` | Cost sensitivity |
| `13_bootstrap_95ci.csv` | Confidence intervals |
| `14_calibration_metrics.csv` | Brier score, ECE |
| `15_mcnemar_holm_results.csv` | Statistical tests |
| `16_temporal_drift_results.csv` | Chronological results |
| `17_feature_drift_psi_ks.csv` | Feature drift |
| `18_shap_global_feature_importance.csv` | SHAP global |
| `19_shap_local_explanations.csv` | SHAP local |
| `20_manuscript_model_summary.csv` | Summary table |

### Models (`../Models/`)
- Preprocessor per dataset
- Trained models (joblib format)
- Output manifest

---

## 🧠 Frameworks and Libraries

### Core Libraries
```python
pandas, numpy, scipy, matplotlib, seaborn
```

### Machine Learning
```python
scikit-learn, xgboost, lightgbm, catboost
```

### Deep Learning
```python
torch (PyTorch)
```

### Imbalance Handling
```python
imbalanced-learn (SMOTE, ADASYN)
```

### Explainability
```python
shap
```

### Statistics
```python
statsmodels (McNemar test)
```

### Utilities
```python
joblib, psutil, warnings
```

---

## ▶️ Running the Notebook

### 1. Clone the Repository

```bash
git clone https://github.com/hasnain006/Multi-Dataset-Fraud-Detection-Benchmark.git
cd Multi-Dataset-Fraud-Detection-Benchmark
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch Jupyter

```bash
jupyter notebook
```

### 4. Open Notebook

Navigate to:
```
Notebooks/Code.ipynb
```

### 5. Run Cells Sequentially

Execute cells in order from top to bottom.

---

## ⚙️ Hardware Requirements

| Component | Recommendation |
|-----------|----------------|
| **CPU** | 4+ cores |
| **RAM** | 16+ GB (32+ GB recommended) |
| **GPU** | Optional (CUDA compatible for MLP training) |
| **Storage** | 5+ GB free space |

---

## 📊 Notebook Parameters

Key configurable parameters at the top of the notebook:

```python
RANDOM_STATE = 42
RUN_DEEP_MLP = True
BOOTSTRAP_ROUNDS = 300
SHAP_SAMPLE_SIZE = 500
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
MAX_ROWS = {"D1_Kaggle_CC": None, "D2_Online_Fraud": 300_000, "D3_PaySim": 200_000}
FN_COSTS = [10, 50, 100, 500]
PRIMARY_FN_COST = 100
```

---

## 📝 Output Locations

All outputs are saved to organized directories:

| Output Type | Location |
|-------------|----------|
| Figures | `../Figures/` |
| Results | `../Results/` |
| Models | `../Models/` |
| Logs | `../Results/04_logs/` |


---

## 📄 License

This notebook is shared under the MIT License. Please cite the repository if you use this code in your research.

---

<div align="center">

## ⭐ If you found this notebook useful, please consider starring the repository.

</div>
