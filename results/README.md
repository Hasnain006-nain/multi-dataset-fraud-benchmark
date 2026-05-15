
# Results

This directory contains the experimental outputs, benchmark metrics, statistical analyses, and evaluation summaries generated during the multi-dataset fraud detection study.

The results were produced from experiments involving:

- Machine Learning models
- Deep Learning architectures
- SMOTE oversampling
- Focal Loss optimization
- SHAP explainability
- Cross-dataset benchmarking

---

# 📊 Included Result Files

Typical outputs include:

```bash
results/
├── results_table.csv
├── table1_dataset_summary.csv
├── table2_full_results.csv
├── table3_literature_comparison.csv
├── ablation_results.csv
├── drift_results.csv
└── mcnemar_results.csv
```

---

# 📈 Evaluation Metrics

The experimental results include:

- Recall
- Precision
- F1 Score
- ROC-AUC
- PR-AUC
- MCC (Matthews Correlation Coefficient)
- Accuracy
- Training Time
- Inference Latency

Primary evaluation focus:

> Fraud Recall and Cross-Dataset Generalization Performance

---

# 🧪 Statistical Validation

This directory also contains statistical evaluation outputs generated using:

- McNemar significance testing
- Comparative benchmarking
- Cross-model validation

These analyses were used to verify whether performance improvements between models were statistically significant.

---

# ⚡ Performance Benchmarking

The stored results include:

- Training time analysis
- Real-time inference latency
- Model efficiency comparison
- Scalability observations

---

# 🔍 Key Findings

Major findings from the experiments include:

- SMOTE consistently improved fraud recall across datasets.
- XGBoost achieved the strongest balance between recall, latency, and explainability.
- Focal Loss improved deep learning robustness on highly imbalanced data.
- SHAP analysis identified meaningful fraud-related features across multiple datasets.
- Cross-dataset evaluation demonstrated improved model generalization.

---

# 📌 Notes

- Experimental outputs may vary slightly depending on random initialization and hardware configuration.
- Large intermediate outputs are excluded from the repository to maintain lightweight storage.
- All benchmark figures corresponding to these results are available in:

```bash
figures/
```
