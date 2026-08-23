# 📊 Results

## Experimental Outputs and Benchmark Metrics

This directory contains all experimental outputs, benchmark metrics, statistical analyses, and evaluation summaries generated during the multi-dataset fraud detection benchmark study.

---

## 📁 Directory Structure

```
Results/
│
├── 00_environment.csv              # Experiment environment configuration
├── 01_dataset_audit.csv            # Dataset characteristics and splits
├── 02_class_distribution_before_resampling.csv  # Original class distribution
├── 03_training_only_resampling_audit.csv  # SMOTE/ADASYN resampling details
├── 04_class_distribution_after_resampling.csv  # Balanced training distribution
├── 05_training_times.csv           # Model training duration
├── 06_mlp_training_history.csv     # MLP training loss curves
├── 07_validation_threshold_cost_sweep.csv  # Threshold sweep results
├── 08_selected_thresholds.csv      # Cost-optimal thresholds per model
├── 09_test_results_default_and_cost_tuned.csv  # Complete test results
├── 10_topk_alert_budget_metrics.csv  # Review budget analysis
├── 11_repeated_latency_protocol.csv  # Inference latency measurements
├── 12_cost_sensitivity_results.csv  # Cost ratio sensitivity analysis
├── 13_bootstrap_95ci.csv           # Bootstrap confidence intervals
├── 14_calibration_metrics.csv      # Brier score and ECE
├── 15_mcnemar_holm_results.csv     # Statistical significance testing
├── 16_temporal_drift_results.csv   # Chronological validation
├── 17_feature_drift_psi_ks.csv     # Feature distribution drift
├── 18_shap_global_feature_importance.csv  # Global SHAP importance
├── 19_shap_local_explanations.csv  # Local SHAP case studies
└── 20_manuscript_model_summary.csv # Summary table for manuscript
```

---

## 📋 Result File Descriptions

### 🔧 Environment & Setup

| File | Description |
|------|-------------|
| `00_environment.csv` | Hardware, software, and experiment configuration parameters |

### 📊 Data & Preprocessing

| File | Description |
|------|-------------|
| `01_dataset_audit.csv` | Dataset sizes, fraud rates, features, and split information |
| `02_class_distribution_before_resampling.csv` | Original class distribution per dataset and split |
| `03_training_only_resampling_audit.csv` | SMOTE/ADASYN execution time and sample counts |
| `04_class_distribution_after_resampling.csv` | Balanced class distribution after training-only resampling |

### 🤖 Training

| File | Description |
|------|-------------|
| `05_training_times.csv` | Training time per model per dataset |
| `06_mlp_training_history.csv` | MLP training and validation loss per epoch |

### 🎯 Threshold Selection

| File | Description |
|------|-------------|
| `07_validation_threshold_cost_sweep.csv` | Full threshold sweep results on validation data |
| `08_selected_thresholds.csv` | Cost-optimal thresholds selected per model |

### 🏆 Test Performance

| File | Description |
|------|-------------|
| `09_test_results_default_and_cost_tuned.csv` | Complete test metrics for all models |
| `10_topk_alert_budget_metrics.csv` | Fraud recall under limited investigator review budgets |

### ⚡ Deployment

| File | Description |
|------|-------------|
| `11_repeated_latency_protocol.csv` | P95 inference latency per transaction |

### 📈 Sensitivity & Uncertainty

| File | Description |
|------|-------------|
| `12_cost_sensitivity_results.csv` | Expected cost at different FN:FP cost ratios |
| `13_bootstrap_95ci.csv` | Bootstrap confidence intervals for key metrics |

### 🎯 Calibration & Statistics

| File | Description |
|------|-------------|
| `14_calibration_metrics.csv` | Brier score and Expected Calibration Error (ECE) |
| `15_mcnemar_holm_results.csv` | Holm-adjusted McNemar p-values for pairwise comparisons |

### ⏰ Temporal Analysis

| File | Description |
|------|-------------|
| `16_temporal_drift_results.csv` | Performance on chronological future test splits |
| `17_feature_drift_psi_ks.csv` | PSI and KS statistics per feature |

### 🔍 Explainability

| File | Description |
|------|-------------|
| `18_shap_global_feature_importance.csv` | Mean absolute SHAP values per feature |
| `19_shap_local_explanations.csv` | Local SHAP breakdowns for selected cases |

### 📄 Summary

| File | Description |
|------|-------------|
| `20_manuscript_model_summary.csv` | Consolidated best results for publication |

---

## 📊 Evaluation Metrics

### Primary Metrics

| Metric | Description |
|--------|-------------|
| **Recall** | Fraud detection rate: `TP / (TP + FN)` |
| **Precision** | Alert accuracy: `TP / (TP + FP)` |
| **MCC** | Matthews Correlation Coefficient (balanced measure) |
| **PR-AUC** | Area under Precision-Recall curve |
| **ROC-AUC** | Area under ROC curve |

### Operational Metrics

| Metric | Description |
|--------|-------------|
| **Expected Cost /10k** | Normalized cost: `(c_FP × FP + c_FN × FN) / N × 10,000` |
| **False Alerts /10k** | False positives per 10,000 transactions |
| **Missed Frauds /10k** | False negatives per 10,000 transactions |
| **Top-k Recall** | Fraud captured in top k% of scored transactions |

### Deployment Metrics

| Metric | Description |
|--------|-------------|
| **Training Time** | Model fitting duration in seconds |
| **Latency** | P95 inference time per transaction (microseconds) |

### Calibration & Statistical Metrics

| Metric | Description |
|--------|-------------|
| **Brier Score** | Mean squared error of predicted probabilities |
| **ECE** | Expected Calibration Error (10 bins) |
| **McNemar p-value** | Paired statistical significance test |

---

## 📈 Key Results Summary

### Best Model Performance

| Dataset | Best Model | Threshold | Recall | MCC | PR-AUC | Cost/10k |
|---------|------------|-----------|--------|-----|--------|----------|
| **D1 (Kaggle CC)** | LightGBM | 0.02 | 0.7895 | **0.8295** | 0.8197 | 354.39 |
| **D2 (Online Fraud)** | XGBoost | 0.03 | 0.9417 | **0.6474** | 0.9111 | 499.17 |
| **D3 (PaySim)** | LightGBM | 0.10 | 0.9867 | **0.9364** | 0.9904 | 327.00 |

### Chronological Performance (XGBoost + SMOTE)

| Dataset | Recall | MCC | PR-AUC | Cost/10k |
|---------|--------|-----|--------|----------|
| D1 | 0.7568 | 0.8200 | 0.7847 | 318.44 |
| D2 | 0.9239 | 0.4190 | 0.6721 | 352.00 |
| D3 | 0.9987 | 0.8531 | 0.9861 | 95.50 |

### Inference Latency (P95 microseconds)

| Model | D1 | D2 | D3 |
|-------|-----|-----|-----|
| **Logistic Regression** | 0.12 | 0.09 | 0.07 |
| **CatBoost** | 0.59 | 1.04 | 0.46 |
| **XGBoost** | 1.18 | 1.72 | 1.36 |
| **XGBoost + SMOTE** | 1.19 | 3.63 | 1.06 |
| **LightGBM** | 4.40 | 8.24 | 5.13 |

---

## 📥 Usage

### Loading Results in Python

```python
import pandas as pd

# Load test results
results = pd.read_csv("09_test_results_default_and_cost_tuned.csv")

# Filter cost-tuned results
cost_tuned = results[results.threshold_policy == "validation_cost_tuned"]

# View top models by MCC
top_models = cost_tuned.sort_values(["dataset", "mcc"], ascending=[True, False])
print(top_models[["dataset", "model_display", "mcc", "recall", "expected_cost_per_10k"]])
```

### Loading Performance Summary

```python
summary = pd.read_csv("20_manuscript_model_summary.csv")
print(summary[["dataset", "model_display", "threshold", "recall", "mcc", "pr_auc"]])
```

### Loading SHAP Results

```python
shap_global = pd.read_csv("18_shap_global_feature_importance.csv")
shap_local = pd.read_csv("19_shap_local_explanations.csv")
```

---

## 📊 Statistical Validation

### McNemar-Holm Results (vs XGBoost + SMOTE)

| Dataset | Comparison Model | p-value (Holm) | Significant |
|---------|------------------|----------------|-------------|
| **D1** | Logistic Regression | 7.36e-55 | ✅ Yes |
| **D1** | EasyEnsemble | 2.70e-11 | ✅ Yes |
| **D1** | XGBoost | 0.547 | ❌ No |
| **D1** | LightGBM | 0.177 | ❌ No |
| **D2** | XGBoost | 2.74e-169 | ✅ Yes |
| **D2** | LightGBM | 8.02e-130 | ✅ Yes |
| **D3** | XGBoost | 0.519 | ❌ No |
| **D3** | CatBoost | 1.0 | ❌ No |


---

## 📄 License

These results are shared under the MIT License. Please cite the repository if you use these results in your research.

---

<div align="center">

## ⭐ If you found these results useful, please consider starring the repository.

</div>
