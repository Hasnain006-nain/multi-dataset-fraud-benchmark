# 📈 Figures

## Visualization Gallery for Multi-Dataset Fraud Detection Benchmark

This folder contains all publication-ready figures generated during the multi-dataset fraud detection benchmark experiments. Visualizations cover data analysis, model performance, deployment metrics, explainability, and statistical validation.

---

## 📊 Figure Index

| Figure | Name | Description |
|--------|------|-------------|
| **Fig 1** | `Fig1.png` | Workflow diagram of the leakage-proof experimental pipeline |
| **Fig 2** | `02_class_distribution_before_resampling.png` | Training class distribution before resampling (log scale) |
| **Fig 3** | `03_class_distribution_after_resampling.png` | Training class distribution after SMOTE/ADASYN resampling |
| **Fig 4** | `04_model_performance_recall_mcc_prauc.png` | Model performance comparison: Recall, MCC, PR-AUC |
| **Fig 5** | `05_precision_recall_curves.png` | Precision-Recall curves across all models and datasets |
| **Fig 6** | `06_cost_sensitive_performance.png` | Cost-sensitive performance at different FN/FP ratios |
| **Fig 7** | `07_topk_alert_budget_recall.png` | Top-k alert budget analysis for investigator workload |
| **Fig 8** | `08_latency_p95_microseconds.png` | P95 inference latency per transaction |
| **Fig 9** | `09_temporal_drift_performance.png` | Chronological validation performance metrics |
| **Fig 10** | `10_calibration_curves.png` | Calibration curves for principal models |
| **Fig 11** | `11_mcnemar_holm_heatmap.png` | McNemar-Holm statistical significance heatmap |
| **Fig 12a** | `12_shap_global_D1_Kaggle_CC.png` | SHAP global feature importance — D1 (Kaggle CC) |
| **Fig 12b** | `12_shap_global_D2_Online_Fraud.png` | SHAP global feature importance — D2 (Online Fraud) |
| **Fig 12c** | `12_shap_global_D3_PaySim.png` | SHAP global feature importance — D3 (PaySim) |
| **Fig 13a** | `13_shap_local_true_positive_D1_Kaggle_CC.png` | Local SHAP explanation — True Positive (D1) |
| **Fig 13b** | `13_shap_local_true_positive_D2_Online_Fraud.png` | Local SHAP explanation — True Positive (D2) |
| **Fig 13c** | `13_shap_local_true_positive_D3_PaySim.png` | Local SHAP explanation — True Positive (D3) |
| **Fig 13d** | `13_shap_local_false_positive_D1_Kaggle_CC.png` | Local SHAP explanation — False Positive (D1) |
| **Fig 13e** | `13_shap_local_false_positive_D2_Online_Fraud.png` | Local SHAP explanation — False Positive (D2) |
| **Fig 13f** | `13_shap_local_false_positive_D3_PaySim.png` | Local SHAP explanation — False Positive (D3) |

---

## 🗂️ Figure Categories

### 1️⃣ Workflow & Data Analysis

| Figure | Description |
|--------|-------------|
| `Fig1.png` | Complete experimental pipeline showing leakage-proof workflow |
| `02_class_distribution_before_resampling.png` | Shows severe class imbalance across all three datasets |
| `03_class_distribution_after_resampling.png` | Demonstrates SMOTE/ADASYN balancing on training data only |

---

### 2️⃣ Model Performance

| Figure | Description |
|--------|-------------|
| `04_model_performance_recall_mcc_prauc.png` | Side-by-side comparison of key metrics |
| `05_precision_recall_curves.png` | PR curves showing discrimination quality |
| `06_cost_sensitive_performance.png` | Expected cost at different FN cost ratios |
| `07_topk_alert_budget_recall.png` | Fraud captured under limited investigator review budgets |
| `09_temporal_drift_performance.png` | Performance on chronological future test splits |

---

### 3️⃣ Deployment & Latency

| Figure | Description |
|--------|-------------|
| `08_latency_p95_microseconds.png` | P95 inference latency across models and datasets |

---

### 4️⃣ Calibration & Statistical Validation

| Figure | Description |
|--------|-------------|
| `10_calibration_curves.png` | Predicted probability vs observed fraud rate |
| `11_mcnemar_holm_heatmap.png` | Holm-adjusted p-values for pairwise comparisons |

---

### 5️⃣ Explainability (SHAP)

#### Global Feature Importance

| Figure | Description |
|--------|-------------|
| `12_shap_global_D1_Kaggle_CC.png` | Top features: V14, V4, V11, V8, V3 |
| `12_shap_global_D2_Online_Fraud.png` | Top features: amt, transaction hour, dayofweek, category |
| `12_shap_global_D3_PaySim.png` | Top features: newbalanceOrig, oldbalanceOrg, amount, type |

#### Local Explanations

| Figure | Description |
|--------|-------------|
| `13_shap_local_true_positive_D1_Kaggle_CC.png` | Feature contributions for a correctly detected fraud |
| `13_shap_local_true_positive_D2_Online_Fraud.png` | Feature contributions for a correctly detected fraud |
| `13_shap_local_true_positive_D3_PaySim.png` | Feature contributions for a correctly detected fraud |
| `13_shap_local_false_positive_D1_Kaggle_CC.png` | Feature contributions for a false alert |
| `13_shap_local_false_positive_D2_Online_Fraud.png` | Feature contributions for a false alert |
| `13_shap_local_false_positive_D3_PaySim.png` | Feature contributions for a false alert |

---

## 🛠️ Figure Generation

All figures were generated using the following Python libraries:

```python
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
```

### Key Visualization Settings

- **Figure DPI**: 150 (screen) / 300 (publication)
- **Color Palette**: Colorblind-friendly palettes
- **Font Size**: Minimum 8pt for publication readiness
- **Format**: PNG and PDF (where applicable)
- **Style**: Clean white background with subtle gridlines

---

## 📥 Usage

### Viewing Figures

```bash
# Open any figure directly
open Figures/04_model_performance_recall_mcc_prauc.png

# Or use a file browser to navigate to the Figures folder
```

### Embedding in README

```markdown
![Model Performance](Figures/04_model_performance_recall_mcc_prauc.png)
```

### Regenerating Figures

To regenerate all figures, run the notebook:

```bash
jupyter notebook Notebooks/Code.ipynb
```

All figures are generated in the **Figure Generation** and **Publication-Ready Figures** sections.

---

## 📝 Figure Descriptions

### Fig 1: Workflow Diagram
The complete leakage-proof experimental pipeline showing data flow from raw datasets to validated benchmark results.

### Fig 2-3: Class Distribution
Demonstrates the extreme class imbalance in fraud detection and the effect of training-only resampling.

### Fig 4-5: Performance Metrics
Comprehensive model comparison using recall, MCC, PR-AUC, and PR curves.

### Fig 6-7: Operational Metrics
Cost-sensitive performance and investigator workload analysis for real-world deployment.

### Fig 8: Latency Analysis
P95 inference latency measurements for production readiness assessment.

### Fig 9: Temporal Drift
Performance on chronological test splits to evaluate model stability over time.

### Fig 10: Calibration
Assessment of predicted probability quality and calibration errors.

### Fig 11: Statistical Significance
McNemar-Holm adjusted p-values for pairwise model comparisons.

### Fig 12-13: SHAP Explainability
Global feature importance and local case-level explanations for model interpretability.



---

## 📄 License

These figures are shared under the MIT License. Please cite the repository if you use these visualizations in your research.

---

<div align="center">

## ⭐ If you found these figures useful, please consider starring the repository.

</div>
