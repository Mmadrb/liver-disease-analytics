# 🩺 Liver Disease Analytics & Clinical ML Predictor

[![Streamlit App](https://static.streamlit.io/badge_streamlit.svg)](https://liver-disease-analytics.streamlit.app/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready clinical machine learning system for liver disease classification, featuring **TRIPOD-AI** compliant pipelines, **SHAP-based** interpretability, and a real-time **Streamlit** interface.

---

## 📋 Project Overview

This repository integrates a clinical ML pipeline designed to classify primary liver diagnoses based on baseline patient features. The system is built with a focus on **reproducibility**, **clinical utility**, and **transparency**, adhering to international reporting standards for AI in healthcare.

### Key Features
- **TRIPOD-AI Compliant Pipeline**: End-to-end ML workflow including data cleaning, three-way splitting, and automated model selection.
- **Calibrated Probabilities**: Uses **Isotonic Regression** to ensure predicted probabilities reflect true clinical risk.
- **Explainable AI (XAI)**: Integrated **SHAP (SHapley Additive exPlanations)** for local and global model interpretability.
- **Clinical Utility Analysis**: Includes **Decision Curve Analysis (DCA)** to evaluate the net benefit of model-based decisions.
- **Production-Ready Architecture**: Modular design separating training logic from inference, with serialized model loading.

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/Mmadrb/liver-disease-analytics.git
cd liver-disease-analytics
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run liver_disease.py
```

---

## 🛠️ Technical Architecture

### Data Schema
The model utilizes 19 baseline clinical features:
- **Demographics**: Age, Sex
- **Comorbidities/Habits**: DM, Smoke, Alcohol Use, Other toxins
- **Laboratory Values**: WBC, Hb, Plt, PT, INR, Total Bilirubin, Albumin, ALP, AST, ALT, CRP
- **Calculated Ratios**: AST/ALT Ratio

### ML Pipeline Details
- **Preprocessing**: Median imputation for numeric data, most-frequent for categorical; Standard scaling and One-Hot encoding.
- **Imbalance Handling**: Synthetic Minority Over-sampling Technique (**SMOTE**) integrated into the training pipeline.
- **Model Selection**: Automated comparison between **Random Forest** and **Logistic Regression** using Repeated Stratified K-Fold CV.
- **Calibration**: Post-hoc calibration via Isotonic Regression on a dedicated calibration set.

---

## 📊 Model Evaluation Summary

| Metric | Value (Test Set) |
| :--- | :--- |
| **ROC-AUC (Weighted OvR)** | 0.894 |
| **Balanced Accuracy** | 0.821 |
| **F1-Score (Weighted)** | 0.845 |

---

## 🧬 Interpretability (SHAP)

The system provides both **Global** (feature importance across the population) and **Local** (individual patient reasoning) explanations.

- **Global**: Identifies AST, Bilirubin, and Age as top predictors.
- **Local**: The Streamlit app generates a SHAP bar plot for every prediction, showing which features increased or decreased the risk for a specific patient.

---

## 📑 Citation

If you use this pipeline or app in your research, please cite:

```bibtex
@software{liver_disease_analytics_2026,
  author = {Mmadrb},
  title = {Liver Disease Analytics: A TRIPOD-AI Compliant Clinical ML System},
  year = {2026},
  url = {https://github.com/Mmadrb/liver-disease-analytics}
}
```

---

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.
