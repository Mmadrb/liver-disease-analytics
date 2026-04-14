# 🩺 Liver Disease Analytics & Clinical ML Predictor

[![Streamlit App](https://static.streamlit.io/badge_streamlit.svg)](https://liver-disease-analytics.streamlit.app/)
[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/release/python-3110/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A clinical analytics platform for liver disease classification, featuring a **Dual-Engine ML Architecture**, **SHAP-based** interpretability, and a real-time **Streamlit** interface.

---

## 📋 Project Overview

This repository provides an end-to-end clinical analytics dashboard for liver disease. It features two distinct machine learning engines:
1.  **🩺 Clinical Model**: A lightweight, 8-feature model designed for rapid "Time-Zero" assessment.
2.  **📊 Full Data Pipeline**: A comprehensive, **TRIPOD-AI** compliant model integrating 19 baseline clinical features with post-hoc **Isotonic Calibration**.

### Key Features
- **Dual-Engine ML**: Choose between a simplified clinical model or a high-dimensional research-grade pipeline.
- **TRIPOD-AI Compliance**: The full pipeline follows international reporting standards for clinical AI.
- **Calibrated Probabilities**: Uses **Isotonic Regression** to ensure predicted probabilities reflect true clinical risk.
- **Explainable AI (XAI)**: Integrated **SHAP (SHapley Additive exPlanations)** for both global and local model interpretability.
- **Clinical Utility Analysis**: Comprehensive dashboard covering mortality rates, ACLF stratification, and organ failure analysis.

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

### Full Pipeline Data Schema
The research-grade model utilizes 19 baseline clinical features:
- **Demographics**: Age, Sex
- **Comorbidities/Habits**: DM, Smoke, Alcohol Use, Other toxins
- **Laboratory Values**: WBC, Hb, Plt, PT, INR, Total Bilirubin, Albumin, ALP, AST, ALT, CRP
- **Calculated Ratios**: AST/ALT Ratio

### ML Pipeline Details
- **Preprocessing**: Median imputation for numeric data, most-frequent for categorical; Standard scaling and One-Hot encoding.
- **Imbalance Handling**: Synthetic Minority Over-sampling Technique (**SMOTE**) integrated into the training pipeline.
- **Calibration**: Post-hoc calibration via Isotonic Regression on a dedicated calibration set (25% of development data).

---

## 🧬 Interpretability (SHAP)

The system provides both **Global** (feature importance across the population) and **Local** (individual patient reasoning) explanations.

- **Global**: Identifies key predictors like AST, Bilirubin, and Age.
- **Local**: The Streamlit app generates a SHAP bar plot for every prediction, showing which features increased or decreased the risk for a specific patient.

---

## 📑 Citation

If you use this pipeline or app in your research, please cite:

```bibtex
@software{liver_disease_analytics_2026,
  author = {Mmadrb},
  title = {Liver Disease Analytics: A Dual-Engine Clinical ML System},
  year = {2026},
  url = {https://github.com/Mmadrb/liver-disease-analytics}
}
```

---

## ⚖️ License
Distributed under the MIT License. See `LICENSE` for more information.
