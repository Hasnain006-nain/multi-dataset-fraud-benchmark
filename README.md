# Beyond Single-Dataset Evaluation: A Multi-Dataset Benchmark of SMOTE, Focal Loss, and XGBoost for Explainable Fraud Detection

A comprehensive multi-dataset benchmark for explainable fraud detection using SMOTE, XGBoost, Focal Loss, SHAP explainability, and deep learning models across Kaggle Credit Card, Online Fraud, and PaySim datasets.

---

# 📌 Overview

This repository contains the complete implementation, experiments, visualizations, statistical analysis, and research artifacts for the paper:

> **Beyond Single-Dataset Evaluation: A Multi-Dataset Benchmark of SMOTE, Focal Loss, and XGBoost for Explainable Fraud Detection**

The project evaluates traditional machine learning and deep learning approaches for highly imbalanced fraud detection problems using three different financial fraud datasets.

The study emphasizes:

- Multi-dataset benchmarking
- Class imbalance mitigation
- Explainable AI (XAI)
- Statistical validation
- Real-time deployment feasibility
- Cross-dataset generalization

---

# 🚀 Key Contributions

- ✅ Evaluation across **3 heterogeneous fraud datasets**
- ✅ Comparison of **8 fraud detection models**
- ✅ Integration of **SMOTE** for imbalance handling
- ✅ Deep learning with **Focal Loss**
- ✅ Explainability using **SHAP**
- ✅ Statistical significance testing using **McNemar’s Test**
- ✅ Real-time feasibility analysis
- ✅ Training time and inference latency benchmarking
- ✅ ROC, PR, MCC, Recall, and PR-AUC analysis
- ✅ Cross-dataset validation and generalization

---

# 📂 Repository Structure

```bash
multi-dataset-fraud-benchmark/
│
├── datasets/
│   ├── creditcard.csv
│   ├── fraudTest.csv
│   ├── PS.csv
│
├── notebooks/
│   ├── Untitled.ipynb
│
├── figures/
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
│   ├── fig15_mcnemar_heatmap.png
│
├── results/
│   ├── results_table.csv
│   ├── table1_dataset_summary.csv
│   ├── table2_full_results.csv
│   ├── table3_literature_comparison.csv
│   ├── ablation_results.csv
│   ├── drift_results.csv
│   ├── mcnemar_results.csv
│
├── papers/
│   ├── paper.pdf
│   ├── OVERVIEW.pdf
│   ├── Comparison_Report.pdf
│
├── README.md
├── requirements.txt
└── LICENSE
```

---

# 📊 Datasets

## D1 — Kaggle Credit Card Fraud

- Real-world European credit card transactions
- Highly imbalanced dataset
- Fraud ratio: **0.17%**
- Features:
  - PCA-transformed features (V1–V28)
  - Time
  - Amount

Dataset size:

- 284,807 transactions
- 492 fraud cases

---

## D2 — Online Transaction Fraud Dataset

- Online financial transaction dataset
- Includes:
  - transaction type
  - merchant
  - category
  - balances
  - timestamps

Dataset size:

- ~300,000 samples

---

## D3 — PaySim Mobile Money Dataset

- Simulated mobile money fraud dataset
- Based on real mobile money transfer patterns

Features include:

- amount
- balances
- transaction type
- time step

Dataset size:

- ~200,000 samples

---

# 🧠 Models Evaluated

| Model ID | Description |
|---|---|
| A | XGBoost (No SMOTE) |
| B | XGBoost + SMOTE |
| C | Logistic Regression + SMOTE |
| D | Random Forest + SMOTE |
| E1 | MLP + Cross Entropy (No SMOTE) |
| E2 | MLP + Focal Loss (No SMOTE) |
| E3 | MLP + Cross Entropy + SMOTE |
| E4 | MLP + Focal Loss + SMOTE |

---

# ⚙️ Techniques Used

## SMOTE

Synthetic Minority Over-sampling Technique used to balance fraud classes.

Benefits:

- Improved recall
- Better minority class learning
- Reduced bias toward majority class

---

## Focal Loss

Applied to deep learning models to focus training on difficult fraud samples.

Benefits:

- Better minority learning
- Higher fraud recall
- Reduced easy-example dominance

---

## SHAP Explainability

SHAP was used to explain model predictions and identify important fraud indicators.

Implemented for:

- XGBoost models
- Deep learning models

---

## McNemar Statistical Testing

Used to validate whether differences between models are statistically significant.

Advantages:

- More reliable than simple t-tests
- Standard classifier comparison technique

---

# 📈 Evaluation Metrics

The following metrics were used:

- Recall
- Precision
- F1 Score
- ROC-AUC
- PR-AUC
- MCC (Matthews Correlation Coefficient)
- Accuracy

Primary metric:

> Recall (fraud detection sensitivity)

---

# 📷 Figures Included

| Figure | Description |
|---|---|
| Figure 1 | Class distribution before SMOTE |
| Figure 2 | Class distribution after SMOTE |
| Figure 3 | Confusion matrices — Model B |
| Figure 4 | Confusion matrices — Model E4 |
| Figure 5 | ROC curves |
| Figure 6 | Precision-Recall curves |
| Figure 7 | Recall comparison |
| Figure 8 | Training time comparison |
| Figure 9 | Inference latency |
| Figure 10 | Cross-dataset generalization |
| Figure 11 | MCC comparison |
| Figure 12 | SHAP feature importance (Model B) |
| Figure 13 | SHAP feature importance (Model E4) |
| Figure 14 | Deep learning training curves |
| Figure 15 | McNemar statistical heatmap |

---

# 🏆 Best Performing Models

## Best Overall Recall

### Model B — SMOTE + XGBoost

| Dataset | Recall |
|---|---|
| D1 | 0.800 |
| D2 | 0.897 |
| D3 | 0.984 |

Advantages:

- Excellent fraud detection
- Fast inference
- Strong generalization
- SHAP explainability support

---

## Best Deep Learning Model

### Model E4 — MLP + Focal Loss + SMOTE

| Dataset | Recall |
|---|---|
| D1 | 0.758 |
| D2 | 0.886 |
| D3 | 0.973 |

Advantages:

- Strong precision
- Good recall
- Effective focal loss integration

---

# ⏱️ Real-Time Deployment Analysis

The repository includes:

- Training time benchmarking
- Inference latency benchmarking
- Throughput feasibility
- Real-time fraud detection suitability

XGBoost models demonstrated:

- Low latency
- Fast training
- Practical deployment capability

---

# 🔍 Explainability (XAI)

SHAP analysis identified important fraud indicators.

Examples:

## Kaggle Dataset

Important features:

- V14
- V12
- V1
- V3

## Online Fraud Dataset

Important features:

- amount
- category
- merchant

## PaySim Dataset

Important features:

- oldbalanceOrg
- newbalanceOrig
- type
- amount

---

# 📉 Statistical Validation

McNemar’s Test was applied pairwise against the proposed model.

Results confirm:

- Significant performance improvements
- Robust model differences
- Reliable benchmarking conclusions

---

# 💻 Installation

Clone repository:

```bash
git clone https://github.com/yourusername/multi-dataset-fraud-benchmark.git
```

Move into project folder:

```bash
cd multi-dataset-fraud-benchmark
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Run notebook:

```bash
jupyter notebook
```

Open:

```bash
notebooks/Untitled.ipynb
```

---

# 📦 Main Libraries Used

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

# 📄 Research Papers Included

This repository includes:

- Full research paper
- Overview report
- Comparative literature analysis

Located in:

```bash
papers/
```

---

# 📚 Research Highlights

Compared with previous fraud detection studies, this work provides:

| Feature | This Study |
|---|---|
| Multi-dataset evaluation | ✅ |
| SHAP explainability | ✅ |
| Statistical testing | ✅ |
| MCC analysis | ✅ |
| PR-AUC evaluation | ✅ |
| Real-time latency analysis | ✅ |
| Cross-dataset benchmarking | ✅ |
| Deep learning + focal loss | ✅ |

---

# 🔮 Future Work

Possible extensions include:

- Graph Neural Networks (GNNs)
- Federated fraud learning
- Transformer-based fraud detection
- Streaming fraud detection
- Adaptive concept drift handling
- Explainable ensemble systems

---

# 📜 Citation

If you use this repository, please cite:

```bibtex
@article{fraudbenchmark2026,
  title={Beyond Single-Dataset Evaluation: A Multi-Dataset Benchmark of SMOTE, Focal Loss, and XGBoost for Explainable Fraud Detection},
  author={Author},
  year={2026}
}
```

---

# 📬 Contact

For questions, collaborations, or research discussions:

- GitHub Issues
- Pull Requests

---

# ⭐ Acknowledgements

Datasets sourced from:

- Kaggle Credit Card Fraud Dataset
- PaySim Dataset
- Online Transaction Fraud Dataset

Libraries and frameworks:

- Scikit-learn
- XGBoost
- PyTorch
- SHAP
- Imbalanced-learn

---

# 📖 Related Documents

- Research paper
- Comparison report
- Experimental results
- Statistical analysis
- SHAP explainability outputs

---

# 🏁 Conclusion

This repository presents a comprehensive and explainable fraud detection benchmark combining:

- SMOTE
- XGBoost
- Focal Loss
- Deep Learning
- SHAP Explainability
- Statistical Validation

across multiple financial fraud datasets.

The results demonstrate that:

> SMOTE + XGBoost provides the strongest balance between fraud recall, explainability, speed, and deployment feasibility across diverse fraud environments.
