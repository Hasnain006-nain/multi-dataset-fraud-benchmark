
# 💾 Trained Models

## 📂 Google Drive Repository

All trained models are available for download from Google Drive:

[![Google Drive](https://img.shields.io/badge/Google%20Drive-4285F4?style=for-the-badge&logo=googledrive&logoColor=white)](https://drive.google.com/drive/folders/10b67K3d2WEXBNQmuwwzeWhUiEhTVQf4D?usp=drive_link)

---

## 📁 Folder Structure

The models are organized by dataset:

```
Trained Models/
│
├── 📂 D1_Kaggle_CC/
│   ├── preprocessor.joblib
│   ├── LR_balanced.joblib
│   ├── RF.joblib
│   ├── BRF.joblib
│   ├── EasyEnsemble.joblib
│   ├── XGB.joblib
│   ├── XGB_spw.joblib
│   ├── XGB_SMOTE.joblib
│   ├── XGB_ADASYN.joblib
│   ├── LightGBM.joblib
│   └── CatBoost.joblib
│
├── 📂 D2_Online_Fraud/
│   ├── preprocessor.joblib
│   ├── LR_balanced.joblib
│   ├── RF.joblib
│   ├── BRF.joblib
│   ├── EasyEnsemble.joblib
│   ├── XGB.joblib
│   ├── XGB_spw.joblib
│   ├── XGB_SMOTE.joblib
│   ├── XGB_ADASYN.joblib
│   ├── LightGBM.joblib
│   └── CatBoost.joblib
│
└── 📂 D3_PaySim/
    ├── preprocessor.joblib
    ├── LR_balanced.joblib
    ├── RF.joblib
    ├── BRF.joblib
    ├── EasyEnsemble.joblib
    ├── XGB.joblib
    ├── XGB_spw.joblib
    ├── XGB_SMOTE.joblib
    ├── XGB_ADASYN.joblib
    ├── LightGBM.joblib
    └── CatBoost.joblib
```

---

## 🧠 Model Details

### Saved Models Per Dataset

| Model | File Name | Description |
|-------|-----------|-------------|
| **Preprocessor** | `preprocessor.joblib` | Fitted ColumnTransformer (scalers + encoders) |
| **Logistic Regression** | `LR_balanced.joblib` | Class-balanced logistic regression |
| **Random Forest** | `RF.joblib` | Random Forest with balanced subsampling |
| **Balanced RF** | `BRF.joblib` | Balanced Random Forest |
| **EasyEnsemble** | `EasyEnsemble.joblib` | Ensemble of balanced bootstrapped classifiers |
| **XGBoost** | `XGB.joblib` | Vanilla XGBoost |
| **XGBoost + SPW** | `XGB_spw.joblib` | XGBoost with scale_pos_weight |
| **XGBoost + SMOTE** | `XGB_SMOTE.joblib` | XGBoost trained on SMOTE data |
| **XGBoost + ADASYN** | `XGB_ADASYN.joblib` | XGBoost trained on ADASYN data |
| **LightGBM** | `LightGBM.joblib` | LightGBM with class_weight balancing |
| **CatBoost** | `CatBoost.joblib` | CatBoost with auto_class_weights |

---

## 📥 Download Instructions

### Option 1: Direct Download

1. Visit the [Google Drive folder](https://drive.google.com/drive/folders/10b67K3d2WEXBNQmuwwzeWhUiEhTVQf4D?usp=drive_link)
2. Navigate to the desired dataset folder
3. Right-click on the model file and select **"Download"**

### Option 2: Clone via `gdown`

```bash
# Install gdown
pip install gdown

# Download entire folder (replace FOLDER_ID with actual ID)
gdown --folder https://drive.google.com/drive/folders/10b67K3d2WEXBNQmuwwzeWhUiEhTVQf4D
```

### Option 3: Use Google Drive API

```python
import gdown

# Example: Download XGBoost + SMOTE model for D1
url = "https://drive.google.com/uc?id=FILE_ID"
output = "models/D1_Kaggle_CC/XGB_SMOTE.joblib"
gdown.download(url, output, quiet=False)
```

---

## 🔧 Loading Models in Python

```python
import joblib
import numpy as np
import pandas as pd

# Load preprocessor
preprocessor = joblib.load("models/D1_Kaggle_CC/preprocessor.joblib")

# Load model
model = joblib.load("models/D1_Kaggle_CC/XGB_SMOTE.joblib")

# Preprocess new data
X_processed = preprocessor.transform(X_raw)

# Get predictions
predictions = model.predict(X_processed)
probabilities = model.predict_proba(X_processed)[:, 1]
```

---

## 📊 Model Performance Summary

### D1: Kaggle Credit Card

| Model | Threshold | Recall | MCC | PR-AUC | Cost/10k |
|-------|-----------|--------|-----|--------|----------|
| **LightGBM** | 0.02 | **0.789** | **0.829** | 0.820 | 354.39 |
| **XGBoost** | 0.05 | 0.800 | 0.813 | 0.819 | 337.64 |
| **XGBoost + SMOTE** | 0.74 | 0.800 | 0.772 | 0.801 | 339.41 |
| **Random Forest** | 0.04 | 0.821 | 0.746 | 0.810 | 306.10 |

### D2: Online Fraud

| Model | Threshold | Recall | MCC | PR-AUC | Cost/10k |
|-------|-----------|--------|-----|--------|----------|
| **XGBoost** | 0.03 | **0.942** | **0.647** | 0.911 | 499.17 |
| **LightGBM** | 0.03 | 0.958 | 0.631 | **0.923** | **394.50** |
| **CatBoost** | 0.50 | 0.949 | 0.570 | 0.857 | 493.83 |
| **XGBoost + SMOTE** | 0.07 | 0.958 | 0.470 | 0.829 | 520.83 |

### D3: PaySim Mobile Money

| Model | Threshold | Recall | MCC | PR-AUC | Cost/10k |
|-------|-----------|--------|-----|--------|----------|
| **LightGBM** | 0.10 | 0.987 | **0.936** | **0.990** | 327.00 |
| **XGBoost** | 0.09 | **0.990** | 0.905 | 0.989 | **270.25** |
| **XGBoost + SMOTE** | 0.48 | 0.988 | 0.897 | 0.987 | 324.00 |
| **CatBoost** | 0.62 | 0.988 | 0.894 | 0.985 | 325.75 |

---

## 📝 Important Notes

### Model Storage Format
- All models are saved using **joblib** (`.joblib` extension)
- Compatible with scikit-learn, XGBoost, LightGBM, and CatBoost
- Preprocessors include fitted transformers for consistent preprocessing

### Usage Requirements
```bash
pip install scikit-learn xgboost lightgbm catboost joblib
```

### File Sizes
- Tree-based models: ~5-50 MB depending on dataset
- Preprocessor: ~1-10 MB (larger for D2 with many features)
- Total size: ~500 MB per dataset

### ⚠️ Important
- Models were trained on specific data distributions
- Ensure consistent preprocessing (use provided `preprocessor.joblib`)
- MLP models are **not included** in this folder (large file size)
- MLP models can be retrained using the provided notebook

---

## 🔗 Related Resources

| Resource | Link |
|----------|------|
| **Main Repository** | [GitHub](https://github.com/hasnain006/Multi-Dataset-Fraud-Detection-Benchmark) |
| **Datasets** | [Kaggle Credit Card](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), [Online Fraud](https://www.kaggle.com/datasets/kartik2112/fraud-detection), [PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) |
| **Results** | [CSV Files](../Results/) |
| **Notebook** | [Code.ipynb](../Notebooks/Code.ipynb) |

---

## 📄 License

These models are shared under the MIT License. Please cite the repository if you use these models in your research.

---

<div align="center">

## ⭐ If you found these models useful, please consider starring the repository.

</div>
