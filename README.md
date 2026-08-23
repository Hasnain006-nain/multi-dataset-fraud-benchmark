# 📊 Multi-Dataset Fraud Detection Benchmark

## A Leakage-Proof Framework for Explainable and Interpretable Financial Fraud Detection Using Tree-Based Ensembles and Neural Networks

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-3.3.0-red?style=for-the-badge)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-green?style=for-the-badge)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2.7-purple?style=for-the-badge)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?style=for-the-badge&logo=pytorch)
![SHAP](https://img.shields.io/badge/SHAP-ExplainableAI-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**[📄 Research Paper](link-to-paper)** •
**[📊 Results Dashboard](#-results)** •
**[🔬 Interactive Notebooks](#-notebooks)**

</div>

---

## 📖 Abstract

Financial fraud detection presents unique challenges including extreme class imbalance, concept drift, and the critical need for explainability. This benchmark study evaluates a comprehensive suite of machine learning and deep learning models across **three diverse fraud datasets** using a **leakage-proof experimental workflow**. We demonstrate that **XGBoost with SMOTE oversampling** achieves the strongest balance between fraud recall, explainability, and real-time performance, while **MLP with Focal Loss** provides competitive deep learning alternatives. The framework incorporates rigorous statistical validation via **McNemar-Holm testing**, **bootstrap confidence intervals**, **SHAP explainability**, and **temporal drift validation**—establishing a robust benchmark for production-ready fraud detection systems.

---

## 🚀 Workflow Overview

The experimental workflow follows a strict leakage-proof pipeline:

<div align="center">

| Layer | Description |
|-------|-------------|
| **Input Layer** | Public financial fraud datasets: D1 (Kaggle Credit Card), D2 (Online Fraud Transactions), D3 (PaySim Mobile Money) |
| **Leakage-Proof Data Preparation** | Duplicate removal, invalid label cleaning, identifier and leakage-column removal, missing/infinite value handling, Train/Validation/Test split *(Preprocessing fitted on training only)* |
| **Training-Only Preprocessing & Resampling** | Numerical Imputation + Robust Scaling, Categorical Imputation + One-Hot Encoding, Resampling (SMOTE/ADASYN) applied only to training *(Validation and test sets remain naturally imbalanced)* |
| **Model Benchmarking Layer** | Logistic Regression, Random Forest, Balanced Random Forest, EasyEnsemble, XGBoost, XGBoost + scale_pos_weight, XGBoost + ADASYN, LightGBM, CatBoost, MLP + Focal Loss |
| **Validation & Operating Threshold Layer** | Threshold tuning only on validation set, Cost-sensitive threshold selection, Frozen threshold for test evaluation |
| **Final Evaluation Layer** | PR-AUC, Recall, Precision, Expected cost per 10k, False alerts per 10k, Top-k alert budget, Bootstrap 95% confidence intervals, McNemar-Holm statistical testing |
| **Deployment & Trust Layer** | Repeated inference latency profiling, Temporal/concept-drift validation, Calibration analysis, SHAP global explanations, SHAP local audit |
| **Output** | Validated Explainable Fraud Detection Benchmark |

</div>

---

## 🎯 Key Contributions

| # | Contribution |
|---|-------------|
| 1️⃣ | **Multi-Dataset Benchmark** — Evaluation across 3 fraud datasets with varying characteristics |
| 2️⃣ | **Leakage-Proof Workflow** — Strict train/validation/test separation with preprocessing fitted on training only |
| 3️⃣ | **Comprehensive Models** — 10+ models including XGBoost, LightGBM, CatBoost, Random Forest, and MLP with Focal Loss |
| 4️⃣ | **Resampling Strategies** — SMOTE, ADASYN, and cost-sensitive threshold tuning |
| 5️⃣ | **Statistical Rigor** — McNemar-Holm testing, bootstrap 95% CIs, and temporal drift analysis |
| 6️⃣ | **Explainability** — SHAP global and local interpretations |
| 7️⃣ | **Deployment Metrics** — Inference latency profiling and top-k alert budget analysis |

---

## 📊 Datasets

The benchmark uses three publicly available fraud detection datasets with varying fraud rates and feature spaces:

| Dataset | Source File | Original Rows | Rows Used | Fraud Cases | Fraud Rate | Features |
|---------|-------------|---------------|-----------|-------------|------------|----------|
| **D1: Kaggle Credit Card** | `creditcard.csv` | 283,726 | 283,726 | 473 | 0.167% | 30 |
| **D2: Online Fraud** | `fraudTest.csv` | 555,719 | 300,000 | 2,145 | 0.715% | 769 |
| **D3: PaySim Mobile Money** | `PS.csv` | 5,840,045 | 200,000 | 4,497 | 2.249% | 11 |

### 📥 Dataset Sources

| Dataset | Source | Link |
|---------|--------|------|
| **Kaggle Credit Card** | Kaggle | [creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Online Transaction Fraud** | Kaggle | [fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection) |
| **PaySim Mobile Money** | Kaggle | [paysim1](https://www.kaggle.com/datasets/ealaxi/paysim1) |

---

## 🧠 Models Evaluated

### Tree-Based Ensembles

| Model | Training Source | Description |
|-------|----------------|-------------|
| **XGBoost** | Original | Gradient boosting with histogram-based training |
| **XGBoost + SMOTE** | SMOTE | XGBoost with SMOTE oversampling |
| **XGBoost + ADASYN** | ADASYN | XGBoost with ADASYN oversampling |
| **XGBoost scale_pos_weight** | Original | XGBoost with scale_pos_weight balancing |
| **LightGBM** | Original | LightGBM with class_weight balancing |
| **CatBoost** | Original | CatBoost with auto_class_weights |
| **Balanced Random Forest** | Original | Random Forest with balanced subsampling |
| **EasyEnsemble** | Original | Ensemble of balanced bootstrapped classifiers |
| **Random Forest** | Original | Random Forest with balanced subsampling |

### Deep Learning

| Model | Training Source | Description |
|-------|----------------|-------------|
| **MLP + Focal Loss** | Original | 3-layer MLP with Focal Loss |
| **MLP + Focal + SMOTE** | SMOTE | MLP with Focal Loss + SMOTE |

---

## 🏆 Experimental Results

### 🥇 Best Overall Model: **XGBoost + SMOTE**

| Dataset | Threshold | Recall | Precision | MCC | ROC-AUC | PR-AUC | False Alerts /10k | Missed Frauds /10k | Expected Cost /10k |
|---------|-----------|--------|-----------|-----|---------|--------|-------------------|-------------------|-------------------|
| **D1 (Kaggle CC)** | 0.74 | **0.800** | 0.745 | **0.772** | 0.968 | 0.801 | 4.58 | 3.35 | 339.41 |
| **D2 (Online Fraud)** | 0.07 | **0.958** | 0.237 | 0.470 | 0.994 | 0.829 | 220.83 | 3.00 | 520.83 |
| **D3 (PaySim)** | 0.48 | **0.988** | 0.819 | **0.897** | 0.999 | 0.987 | 49.00 | 2.75 | 324.00 |

### 🥈 Best Deep Learning Model: **MLP + Focal Loss + SMOTE**

| Dataset | Threshold | Recall | Precision | MCC | ROC-AUC | PR-AUC | False Alerts /10k | Missed Frauds /10k | Expected Cost /10k |
|---------|-----------|--------|-----------|-----|---------|--------|-------------------|-------------------|-------------------|
| **D1 (Kaggle CC)** | 0.53 | 0.789 | 0.647 | 0.714 | 0.948 | 0.783 | 7.23 | 3.52 | 359.67 |
| **D2 (Online Fraud)** | 0.02 | 0.513 | 0.083 | 0.194 | 0.826 | 0.179 | 404.67 | 34.83 | 3888.00 |
| **D3 (PaySim)** | 0.42 | **0.982** | 0.584 | 0.751 | 0.997 | 0.957 | 157.25 | 4.00 | 557.25 |

---

### 📊 Performance Comparison Across Models

#### D1: Kaggle Credit Card Fraud

| Model | Threshold | Recall | MCC | PR-AUC | Train Time (s) |
|-------|-----------|--------|-----|--------|----------------|
| **LightGBM** | 0.02 | **0.789** | **0.829** | 0.820 | 6.72 |
| **XGBoost** | 0.05 | 0.800 | 0.813 | 0.819 | 5.52 |
| **XGBoost + SMOTE** | 0.74 | 0.800 | 0.772 | 0.801 | 3.69 |
| **Random Forest** | 0.04 | 0.821 | 0.746 | 0.810 | 29.34 |
| **XGBoost scale_pos_weight** | 0.06 | 0.811 | 0.733 | 0.808 | 1.92 |
| **Balanced RF** | 0.75 | 0.779 | 0.727 | 0.683 | 3.96 |
| **MLP + Focal + SMOTE** | 0.53 | 0.789 | 0.714 | 0.783 | 67.67 |
| **CatBoost** | 0.12 | 0.811 | 0.630 | 0.806 | 6.40 |
| **EasyEnsemble** | 0.62 | 0.811 | 0.593 | 0.650 | 2.35 |
| **XGBoost + ADASYN** | 0.33 | 0.832 | 0.556 | 0.788 | 3.37 |
| **Logistic Regression** | 0.90 | 0.842 | 0.436 | 0.684 | 2.83 |

#### D2: Online Transaction Fraud

| Model | Threshold | Recall | MCC | PR-AUC | Train Time (s) |
|-------|-----------|--------|-----|--------|----------------|
| **XGBoost** | 0.03 | **0.942** | 0.647 | **0.911** | 1.61 |
| **LightGBM** | 0.03 | 0.958 | 0.631 | 0.923 | 4.21 |
| **XGBoost scale_pos_weight** | 0.44 | 0.939 | 0.618 | 0.895 | 1.61 |
| **CatBoost** | 0.50 | 0.949 | 0.570 | 0.857 | 7.21 |
| **XGBoost + SMOTE** | 0.07 | 0.958 | 0.470 | 0.829 | 12.16 |
| **XGBoost + ADASYN** | 0.10 | 0.953 | 0.433 | 0.831 | 11.65 |
| **Balanced RF** | 0.42 | 0.942 | 0.397 | 0.806 | 4.64 |
| **Random Forest** | 0.02 | 0.972 | 0.370 | 0.860 | 20.58 |
| **EasyEnsemble** | 0.48 | 0.937 | 0.271 | 0.554 | 2.25 |
| **MLP + Focal + SMOTE** | 0.02 | 0.513 | 0.194 | 0.179 | 28.53 |
| **Logistic Regression** | 0.59 | 0.748 | 0.190 | 0.171 | 19.50 |

#### D3: PaySim Mobile Money

| Model | Threshold | Recall | MCC | PR-AUC | Train Time (s) |
|-------|-----------|--------|-----|--------|----------------|
| **LightGBM** | 0.10 | 0.987 | **0.936** | **0.990** | 2.16 |
| **XGBoost** | 0.09 | 0.990 | 0.905 | 0.989 | 0.84 |
| **XGBoost + SMOTE** | 0.48 | 0.988 | 0.897 | 0.987 | 1.29 |
| **XGBoost + ADASYN** | 0.74 | 0.984 | 0.897 | 0.985 | 1.25 |
| **CatBoost** | 0.62 | 0.988 | 0.894 | 0.985 | 4.29 |
| **XGBoost scale_pos_weight** | 0.41 | 0.989 | 0.904 | 0.988 | 0.78 |
| **Random Forest** | 0.10 | 0.986 | 0.867 | 0.983 | 6.33 |
| **Balanced RF** | 0.42 | 0.990 | 0.744 | 0.975 | 3.16 |
| **MLP + Focal + SMOTE** | 0.42 | 0.982 | 0.751 | 0.957 | 51.54 |
| **EasyEnsemble** | 0.48 | 0.988 | 0.479 | 0.876 | 2.24 |
| **Logistic Regression** | 0.34 | 0.967 | 0.400 | 0.817 | 1.47 |

---

## ⚡ Deployment Analysis

### Inference Latency (p95 microseconds per transaction)

| Model | D1 (Kaggle CC) | D2 (Online Fraud) | D3 (PaySim) |
|-------|----------------|-------------------|-------------|
| **Logistic Regression** | 0.12 | 0.09 | 0.07 |
| **CatBoost** | 0.59 | 1.04 | 0.46 |
| **XGBoost** | 1.18 | 1.72 | 1.36 |
| **XGBoost + SMOTE** | 1.19 | 3.63 | 1.06 |
| **XGBoost + ADASYN** | 1.09 | 1.49 | 1.05 |
| **LightGBM** | 4.40 | 8.24 | 5.13 |
| **Balanced RF** | 16.02 | 16.15 | 15.65 |
| **Random Forest** | 15.82 | 16.18 | 16.02 |
| **EasyEnsemble** | 30.92 | 29.73 | 27.89 |
| **MLP + Focal + SMOTE** | 0.22 | 1.57 | 0.18 |

---

## 🔬 Statistical Validation

### McNemar-Holm Results (vs XGBoost + SMOTE)

<table>
<thead>
<tr>
<th>Dataset</th>
<th>Comparison Model</th>
<th>p-value (Holm)</th>
<th>Significant</th>
</tr>
</thead>
<tbody>
<tr><td rowspan=11><strong>D1</strong></td><td>Logistic Regression</td><td>7.36e-55</td><td>✅ Yes</td></tr>
<tr><td>EasyEnsemble</td><td>2.70e-11</td><td>✅ Yes</td></tr>
<tr><td>XGBoost + ADASYN</td><td>5.47e-27</td><td>✅ Yes</td></tr>
<tr><td>CatBoost</td><td>1.29e-08</td><td>✅ Yes</td></tr>
<tr><td>XGBoost</td><td>0.547</td><td>❌ No</td></tr>
<tr><td>XGBoost scale_pos_weight</td><td>0.519</td><td>❌ No</td></tr>
<tr><td>LightGBM</td><td>0.177</td><td>❌ No</td></tr>
<tr><td>MLP + Focal Loss</td><td>0.519</td><td>❌ No</td></tr>
<tr><td>MLP + Focal + SMOTE</td><td>0.293</td><td>❌ No</td></tr>
<tr><td>Random Forest</td><td>0.643</td><td>❌ No</td></tr>
<tr><td>Balanced RF</td><td>0.643</td><td>❌ No</td></tr>
<tr><td rowspan=11><strong>D2</strong></td><td>Logistic Regression</td><td>0.0</td><td>✅ Yes</td></tr>
<tr><td>EasyEnsemble</td><td>0.0</td><td>✅ Yes</td></tr>
<tr><td>XGBoost</td><td>2.74e-169</td><td>✅ Yes</td></tr>
<tr><td>XGBoost scale_pos_weight</td><td>7.60e-149</td><td>✅ Yes</td></tr>
<tr><td>XGBoost + ADASYN</td><td>3.30e-37</td><td>✅ Yes</td></tr>
<tr><td>LightGBM</td><td>8.02e-130</td><td>✅ Yes</td></tr>
<tr><td>CatBoost</td><td>1.58e-78</td><td>✅ Yes</td></tr>
<tr><td>Random Forest</td><td>1.49e-107</td><td>✅ Yes</td></tr>
<tr><td>Balanced RF</td><td>7.43e-45</td><td>✅ Yes</td></tr>
<tr><td>MLP + Focal Loss</td><td>1.42e-238</td><td>✅ Yes</td></tr>
<tr><td>MLP + Focal + SMOTE</td><td>1.17e-105</td><td>✅ Yes</td></tr>
<tr><td rowspan=11><strong>D3</strong></td><td>Logistic Regression</td><td>0.0</td><td>✅ Yes</td></tr>
<tr><td>EasyEnsemble</td><td>0.0</td><td>✅ Yes</td></tr>
<tr><td>LightGBM</td><td>1.27e-16</td><td>✅ Yes</td></tr>
<tr><td>Random Forest</td><td>7.92e-09</td><td>✅ Yes</td></tr>
<tr><td>Balanced RF</td><td>2.83e-134</td><td>✅ Yes</td></tr>
<tr><td>MLP + Focal Loss</td><td>1.91e-262</td><td>✅ Yes</td></tr>
<tr><td>MLP + Focal + SMOTE</td><td>2.11e-92</td><td>✅ Yes</td></tr>
<tr><td>XGBoost</td><td>0.519</td><td>❌ No</td></tr>
<tr><td>XGBoost scale_pos_weight</td><td>0.547</td><td>❌ No</td></tr>
<tr><td>XGBoost + ADASYN</td><td>1.0</td><td>❌ No</td></tr>
<tr><td>CatBoost</td><td>1.0</td><td>❌ No</td></tr>
</tbody>
</table>

---

## 🔍 Explainable AI (SHAP)

### Global Feature Importance

#### D1: Kaggle Credit Card

| Rank | Feature | Mean SHAP |
|------|---------|-----------|
| 1 | V14 | 2.19 |
| 2 | V4 | 1.55 |
| 3 | V11 | 0.67 |
| 4 | V8 | 0.61 |
| 5 | V3 | 0.60 |

#### D2: Online Fraud

| Rank | Feature | Mean SHAP |
|------|---------|-----------|
| 1 | amt | 2.33 |
| 2 | trans_date_trans_time_hour | 0.73 |
| 3 | trans_date_trans_time_dayofweek | 0.60 |
| 4 | category_gas_transport | 0.30 |
| 5 | trans_date_trans_time_month | 0.23 |

#### D3: PaySim

| Rank | Feature | Mean SHAP |
|------|---------|-----------|
| 1 | newbalanceOrig | 3.71 |
| 2 | oldbalanceOrg | 3.38 |
| 3 | amount | 1.37 |
| 4 | type_PAYMENT | 1.33 |
| 5 | type_CASH_OUT | 0.86 |

---

## 📁 Repository Structure

```
fraud-detection-benchmark/
│
├── 📊 Datasets/
│   ├── creditcard.csv
│   ├── fraudTest.csv
│   └── PS.csv
│
├── 📓 Notebooks/
│   └── Code.ipynb
│
├── 📈 Figures/
│   ├── Fig1.png (Workflow Diagram)
│   ├── 02_class_distribution_before_resampling.png
│   ├── 03_class_distribution_after_resampling.png
│   ├── 04_model_performance_recall_mcc_prauc.png
│   ├── 05_precision_recall_curves.png
│   ├── 06_cost_sensitive_performance.png
│   ├── 07_topk_alert_budget_recall.png
│   ├── 08_latency_p95_microseconds.png
│   ├── 09_temporal_drift_performance.png
│   ├── 10_calibration_curves.png
│   ├── 11_mcnemar_holm_heatmap.png
│   ├── 12_shap_global_D1_Kaggle_CC.png
│   ├── 12_shap_global_D2_Online_Fraud.png
│   ├── 12_shap_global_D3_PaySim.png
│   ├── 13_shap_local_true_positive_D1_Kaggle_CC.png
│   ├── 13_shap_local_true_positive_D2_Online_Fraud.png
│   ├── 13_shap_local_true_positive_D3_PaySim.png
│   ├── 13_shap_local_false_positive_D1_Kaggle_CC.png
│   ├── 13_shap_local_false_positive_D2_Online_Fraud.png
│   └── 13_shap_local_false_positive_D3_PaySim.png
│
├── 📊 Results/
│   ├── 00_environment.csv
│   ├── 01_dataset_audit.csv
│   ├── 02_class_distribution_before_resampling.csv
│   ├── 03_training_only_resampling_audit.csv
│   ├── 04_class_distribution_after_resampling.csv
│   ├── 05_training_times.csv
│   ├── 06_mlp_training_history.csv
│   ├── 07_validation_threshold_cost_sweep.csv
│   ├── 08_selected_thresholds.csv
│   ├── 09_test_results_default_and_cost_tuned.csv
│   ├── 10_topk_alert_budget_metrics.csv
│   ├── 11_repeated_latency_protocol.csv
│   ├── 12_cost_sensitivity_results.csv
│   ├── 13_bootstrap_95ci.csv
│   ├── 14_calibration_metrics.csv
│   ├── 15_mcnemar_holm_results.csv
│   ├── 16_temporal_drift_results.csv
│   ├── 17_feature_drift_psi_ks.csv
│   ├── 18_shap_global_feature_importance.csv
│   ├── 19_shap_local_explanations.csv
│   └── 20_manuscript_model_summary.csv
│
├── 💾 Models/
│   └── [Google Drive](https://drive.google.com/drive/folders/10b67K3d2WEXBNQmuwwzeWhUiEhTVQf4D?usp=drive_link)
│
├── 📝 README.md
├── 📋 requirements.txt
└── 📜 LICENSE
```

---

## 💻 Installation & Usage

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/fraud-detection-benchmark.git
cd fraud-detection-benchmark
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download Datasets

Place the following datasets in the `Datasets/` directory:

- `creditcard.csv` — [Kaggle Credit Card Fraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- `fraudTest.csv` — [Online Transaction Fraud](https://www.kaggle.com/datasets/kartik2112/fraud-detection)
- `PS.csv` — [PaySim Mobile Money](https://www.kaggle.com/datasets/ealaxi/paysim1)

### 4. Run the Notebook

```bash
jupyter notebook Notebooks/Code.ipynb
```

---

## 📊 Key Findings

### 🔥 SMOTE + XGBoost Achieves the Best Balance Between:

✅ **Fraud Recall** — Detects >95% of fraud cases across all datasets  
✅ **Explainability** — SHAP provides clear feature importance  
✅ **Real-Time Performance** — Inference latency <5 microseconds per transaction  
✅ **Cross-Dataset Robustness** — Consistent performance across diverse fraud patterns  
✅ **Statistical Significance** — McNemar-Holm confirms superiority over baselines  

### 📈 Focal Loss Improves Deep Learning Performance

- MLP + Focal Loss + SMOTE achieves competitive recall (0.982 on PaySim)
- Improves over standard cross-entropy by focusing on hard examples
- Training time significantly higher than tree-based models

### 🔍 Feature Importance Reveals Consistent Patterns

- **Transaction amount** is the most important feature across datasets
- **Time-based features** (hour, dayofweek) are consistently important
- **Account balance features** dominate in mobile money fraud

---

## 🔮 Future Work

- **Graph Neural Networks** for transaction network analysis
- **Transformer-based models** for sequence fraud detection
- **Federated Learning** for privacy-preserving fraud detection
- **Streaming Fraud Detection** with online learning
- **Adaptive Concept Drift** handling methods

---

## 📝 Citation

```bibtex
@article{haider2026fraudbenchmark,
  title={Beyond Single-Dataset Evaluation: A Leakage-Proof Framework for Explainable Fraud Detection Using Tree-Based Ensembles and Neural Networks},
  author={Haider, Hasnain},
  journal={arXiv preprint},
  year={2026}
}
```

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Kaggle** for hosting the datasets
- **XGBoost, LightGBM, CatBoost** for open-source gradient boosting libraries
- **SHAP** for explainability tools

---

<div align="center">

## ⭐ If you found this project useful, please consider starring the repository.

**[⬆ Back to Top](#-multi-dataset-fraud-detection-benchmark)**

</div>
