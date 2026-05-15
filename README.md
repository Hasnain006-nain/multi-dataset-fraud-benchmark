<div align="center">

# 🚨 Beyond Single-Dataset Evaluation
## A Multi-Dataset Benchmark of SMOTE, Focal Loss, and XGBoost for Explainable Fraud Detection

<br>

<img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python">
<img src="https://img.shields.io/badge/XGBoost-Fraud%20Detection-red?style=for-the-badge">
<img src="https://img.shields.io/badge/Deep%20Learning-Focal%20Loss-purple?style=for-the-badge">
<img src="https://img.shields.io/badge/Explainable%20AI-SHAP-green?style=for-the-badge">
<img src="https://img.shields.io/badge/Research-Multi--Dataset-orange?style=for-the-badge">
<img src="https://img.shields.io/badge/Status-Research%20Project-success?style=for-the-badge">

<br><br>

### 🔍 Explainable Fraud Detection Benchmark Across Multiple Financial Datasets

### ⚡ SMOTE + XGBoost + Focal Loss + SHAP + Statistical Validation

<br>

</div>

---

# 📌 Table of Contents

- [📖 Overview](#-overview)
- [🚀 Key Contributions](#-key-contributions)
- [📊 Datasets](#-datasets)
- [📥 Dataset Sources](#-dataset-sources)
- [🧠 Models Evaluated](#-models-evaluated)
- [⚙️ Techniques Used](#️-techniques-used)
- [📂 Repository Structure](#-repository-structure)
- [📈 Evaluation Metrics](#-evaluation-metrics)
- [🏆 Best Results](#-best-results)
- [📷 Figures Included](#-figures-included)
- [🔍 Explainability (SHAP)](#-explainability-shap)
- [⚡ Deployment Analysis](#-deployment-analysis)
- [💻 Installation](#-installation)
- [▶️ Running the Project](#️-running-the-project)
- [📦 Libraries Used](#-libraries-used)
- [📚 Research Highlights](#-research-highlights)
- [🔮 Future Work](#-future-work)
- [📄 Citation](#-citation)
- [⭐ Final Conclusion](#-final-conclusion)

---

# 📖 Overview

Financial fraud detection remains one of the most challenging problems in machine learning due to:

- Extreme class imbalance
- Evolving fraud behavior
- Poor cross-dataset generalization
- Lack of explainability in AI systems

Most existing studies evaluate models on only a **single dataset**, making it difficult to assess real-world robustness.

This project introduces a comprehensive **multi-dataset fraud detection benchmark** combining:

✅ Traditional Machine Learning  
✅ Deep Learning  
✅ SMOTE Oversampling  
✅ Focal Loss  
✅ SHAP Explainability  
✅ Statistical Validation  
✅ Real-Time Latency Benchmarking  
✅ Cross-Dataset Generalization  

---

# 🚀 Key Contributions

<div align="center">

| Contribution | Included |
|---|---|
| Multi-Dataset Benchmark | ✅ |
| SMOTE Evaluation | ✅ |
| Focal Loss Evaluation | ✅ |
| SHAP Explainability | ✅ |
| McNemar Statistical Testing | ✅ |
| ROC & PR Curve Analysis | ✅ |
| MCC Evaluation | ✅ |
| Real-Time Latency Testing | ✅ |
| Cross-Dataset Validation | ✅ |
| Deep Learning Benchmarking | ✅ |

</div>

---

# 📊 Datasets

---

## 🏦 D1 — Kaggle Credit Card Fraud Dataset

### Description

Real-world European credit card transactions with highly imbalanced fraud labels.

### Dataset Statistics

| Metric | Value |
|---|---|
| Total Transactions | 284,807 |
| Fraud Cases | 492 |
| Fraud Ratio | 0.17% |

### Main Features

- PCA-transformed features (V1–V28)
- Time
- Amount

---

## 🌐 D2 — Online Transaction Fraud Dataset

### Description

Online financial transaction dataset containing merchant, category, amount, balances, and transaction metadata.

### Dataset Statistics

| Metric | Value |
|---|---|
| Dataset Size | ~300,000 |
| Fraud Type | Online Transaction Fraud |

### Main Features

- Merchant
- Category
- Amount
- Transaction Type
- Account Balances

---

## 📱 D3 — PaySim Mobile Money Dataset

### Description

Synthetic mobile money transaction simulator based on real financial transaction patterns.

### Dataset Statistics

| Metric | Value |
|---|---|
| Dataset Size | ~200,000 |
| Domain | Mobile Money Fraud |

### Main Features

- Step
- Type
- Amount
- oldbalanceOrg
- newbalanceOrig

---

# 📥 Dataset Sources

---

## 🏦 Kaggle Credit Card Fraud Dataset

🔗 https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

---

## 🌐 Online Transaction Fraud Dataset

🔗 https://www.kaggle.com/datasets/kartik2112/fraud-detection

---

## 📱 PaySim Mobile Money Dataset

🔗 https://www.kaggle.com/datasets/ealaxi/paysim1

---

# 🧠 Models Evaluated

<div align="center">

| ID | Model |
|---|---|
| A | XGBoost (No SMOTE) |
| B | XGBoost + SMOTE |
| C | Logistic Regression + SMOTE |
| D | Random Forest + SMOTE |
| E1 | MLP + Cross Entropy |
| E2 | MLP + Focal Loss |
| E3 | MLP + Cross Entropy + SMOTE |
| E4 | MLP + Focal Loss + SMOTE |

</div>

---

# ⚙️ Techniques Used

---

## ⚖️ SMOTE — Class Imbalance Mitigation

SMOTE (Synthetic Minority Over-sampling Technique) was used to balance fraud classes.

### Benefits

✅ Improved Recall  
✅ Better Minority Learning  
✅ Reduced Majority Bias  

---

## 🎯 Focal Loss

Focal Loss was integrated into deep learning models to focus learning on hard fraud samples.

### Benefits

✅ Better Hard Example Learning  
✅ Improved Fraud Sensitivity  
✅ Stronger Minority Classification  

---

## 🔍 SHAP Explainability

SHAP explains:

- Why a transaction is classified as fraud
- Which features influence predictions
- Model decision transparency

Implemented for:

- XGBoost
- Deep Learning Models

---

## 📈 McNemar Statistical Testing

Used for rigorous pairwise model comparison.

### Advantages

✅ More reliable than t-tests  
✅ Exact classifier comparison  
✅ Statistical significance validation  

---

# 📂 Repository Structure

```bash
multi-dataset-fraud-benchmark/
│
├── 📁 datasets/
│   ├── creditcard.csv
│   ├── fraudTest.csv
│   └── PS.csv
│
├── 📁 notebooks/
│   └── Untitled.ipynb
│
├── 📁 figures/
│   ├── fig1_distribution_before_smote.png
│   ├── fig2_distribution_after_smote.png
│   ├── fig3_cm_model_B.png
│   ├── fig4_cm_model_E4.png
│   ├── fig5_roc_curves.png
│   ├── fig6_pr_curves.png
│   ├── fig7_recall_comparison.png
│   ├── fig8_training_time.png
│   ├── fig9_latency.png
│   ├── fig10_cross_dataset.png
│   ├── fig11_mcc_comparison.png
│   ├── fig12_shap_tree.png
│   ├── fig13_shap_deep.png
│   ├── fig14_deep_training_curves.png
│   └── fig15_mcnemar_heatmap.png
│
├── 📁 results/
│   ├── results_table.csv
│   ├── table1_dataset_summary.csv
│   ├── table2_full_results.csv
│   ├── table3_literature_comparison.csv
│   ├── ablation_results.csv
│   ├── drift_results.csv
│   └── mcnemar_results.csv
│
├── 📁 papers/
│   ├── paper.pdf
│   ├── OVERVIEW.pdf
│   └── Comparison_Report.pdf
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

# 📈 Evaluation Metrics

This project evaluates models using:

| Metric | Purpose |
|---|---|
| Recall | Fraud detection sensitivity |
| Precision | False alarm reduction |
| F1 Score | Precision/Recall balance |
| ROC-AUC | Class discrimination |
| PR-AUC | Imbalance-aware evaluation |
| MCC | Balanced classification quality |
| Accuracy | Overall correctness |

---

# 🏆 Best Results

<div align="center">

# 🥇 Best Overall Model
## Model B — SMOTE + XGBoost

</div>

---

| Dataset | Recall | ROC-AUC | PR-AUC |
|---|---|---|---|
| D1 (Kaggle CC) | 0.800 | 0.972 | 0.810 |
| D2 (Online Fraud) | 0.897 | 0.995 | 0.885 |
| D3 (PaySim) | 0.984 | 0.999 | 0.987 |

---

# ⚡ Deployment Analysis

| Metric | Performance |
|---|---|
| Training Time | 6–15 sec |
| Inference Latency | ~0.005 ms |
| Real-Time Ready | ✅ |
| CPU Deployable | ✅ |

---

# 📷 Figures Included

<div align="center">

| Figure | Description |
|---|---|
| Figure 1 | Class imbalance before SMOTE |
| Figure 2 | Balanced training sets after SMOTE |
| Figure 3 | Confusion matrices — XGBoost |
| Figure 4 | Confusion matrices — Deep Learning |
| Figure 5 | ROC curves |
| Figure 6 | Precision-Recall curves |
| Figure 7 | Recall comparison |
| Figure 8 | Training time comparison |
| Figure 9 | Inference latency |
| Figure 10 | Cross-dataset generalization |
| Figure 11 | MCC comparison |
| Figure 12 | SHAP explainability (XGBoost) |
| Figure 13 | SHAP explainability (Deep Learning) |
| Figure 14 | Deep training curves |
| Figure 15 | McNemar significance heatmap |

</div>

---

# 🔍 Explainability (SHAP)

---

## 🏦 Kaggle Dataset

Top Features:

- V14
- V12
- V1
- V3

---

## 🌐 Online Fraud Dataset

Top Features:

- amount
- merchant
- category

---

## 📱 PaySim Dataset

Top Features:

- oldbalanceOrg
- newbalanceOrig
- type
- amount

---

# 💻 Installation

---

## Clone Repository

```bash
git clone https://github.com/yourusername/multi-dataset-fraud-benchmark.git
```

---

## Navigate into Project

```bash
cd multi-dataset-fraud-benchmark
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```bash
notebooks/Untitled.ipynb
```

---

# 📦 Libraries Used

```python
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
matplotlib
seaborn
shap
torch
tensorflow
scipy
```

---

# 📚 Research Highlights

<div align="center">

| Feature | This Study |
|---|---|
| Multiple datasets | ✅ |
| SHAP explainability | ✅ |
| McNemar testing | ✅ |
| PR-AUC evaluation | ✅ |
| MCC evaluation | ✅ |
| Latency benchmarking | ✅ |
| Cross-dataset analysis | ✅ |
| Deep learning benchmarking | ✅ |

</div>

---

# 🔮 Future Work

Potential future extensions include:

- Graph Neural Networks (GNNs)
- Federated Fraud Learning
- Transformer-based Fraud Detection
- Real-Time Streaming Fraud Detection
- Adaptive Concept Drift Handling
- Explainable Ensemble Systems

---

# 📄 Citation

```bibtex
@article{fraudbenchmark2026,
  title={Beyond Single-Dataset Evaluation: A Multi-Dataset Benchmark of SMOTE, Focal Loss, and XGBoost for Explainable Fraud Detection},
  author={Author},
  year={2026}
}
```

---

# ⭐ Why This Repository Stands Out

✅ Multi-dataset evaluation  
✅ Explainable AI integration  
✅ Statistical validation  
✅ Deep learning benchmarking  
✅ Real-world deployment analysis  
✅ Publication-quality visualizations  
✅ Reproducible experiments  
✅ Research-grade methodology  

---

<div align="center">

# 🚀 Final Conclusion

This repository demonstrates that:

# 🔥 SMOTE + XGBoost provides the best balance between:

### Fraud Recall ⚡
### Explainability 🔍
### Speed 🚀
### Real-Time Deployment 💻
### Cross-Dataset Robustness 🌍

<br>

## ⭐ If you found this project useful, consider starring the repository.

</div>
