
# Notebooks

This directory contains the Jupyter notebooks used to conduct the experiments, evaluations, and visualizations for the multi-dataset fraud detection benchmark study.

---

# 📓 Main Notebook

```bash
fraud_detection_benchmark.ipynb
```

The notebook provides the complete experimental workflow for evaluating machine learning and deep learning approaches for financial fraud detection across multiple datasets.

---

# 🔬 Included Experiments

The notebook includes:

- Data preprocessing and cleaning
- Exploratory Data Analysis (EDA)
- Feature engineering
- Class imbalance handling using SMOTE
- XGBoost model training
- Logistic Regression and Random Forest benchmarking
- Deep learning model implementation
- Focal Loss integration
- SHAP explainability analysis
- ROC and Precision-Recall curve generation
- MCC and Recall evaluation
- McNemar statistical significance testing
- Cross-dataset generalization analysis
- Training time benchmarking
- Inference latency evaluation

---

# 📊 Generated Outputs

The notebook generates:

- ROC Curves
- Precision-Recall Curves
- Confusion Matrices
- SHAP Feature Importance Plots
- Recall Comparison Charts
- MCC Comparison Charts
- Latency Analysis
- Statistical Heatmaps

Generated outputs are stored inside:

```bash
figures/
results/
```

---

# 🧠 Frameworks and Libraries

Main technologies used:

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- TensorFlow
- PyTorch
- SHAP
- Matplotlib
- Seaborn

---

# ▶️ Running the Notebook

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open:

```bash
fraud_detection_benchmark.ipynb
```

---

# 📌 Notes

- The notebook was designed for reproducible experimentation.
- Dataset download links are provided inside the `datasets/` directory.
- Large datasets are not directly included in the repository due to GitHub storage limitations.
- Experimental results and figures may vary slightly depending on hardware configuration and random seed initialization.
