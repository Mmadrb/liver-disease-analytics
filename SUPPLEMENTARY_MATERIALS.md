# Supplementary Materials: Clinical Prediction Model for Liver Disease Classification

**Reporting Standard:** TRIPOD-AI (Transparent Reporting of a multivariable prediction model for Individual Prognosis Or Diagnosis - Artificial Intelligence)

---

## 1. Dataset Description

The dataset utilized for the development and validation of this clinical prediction model was sourced from clinical records at a tertiary care liver center. The primary objective was to classify the underlying liver disease based on baseline clinical features.

To ensure the robustness of the model, specific inclusion and exclusion criteria were applied. Patients were included if they had a confirmed primary diagnosis of liver disease and complete baseline clinical data available. Conversely, patients were excluded if their primary diagnosis was missing or if they presented with rare diagnoses that lacked sufficient sample size for effective model training, such as Autoimmune Hepatitis (AIH) and Hepatitis B (HepB). Following these exclusions, the final dataset comprised a substantial cohort of patients, providing a solid foundation for model development.

## 2. Feature Definitions

The model incorporates a comprehensive set of 19 baseline clinical features, categorized into several key domains to capture the multifaceted nature of liver disease.

| Feature Category | Variables Included |
| :--- | :--- |
| **Demographics** | Age, Sex |
| **Comorbidities** | Diabetes Mellitus (DM) |
| **Habits** | Smoking Status, Alcohol Use, Other toxins |
| **Hematology** | WBC, Hemoglobin (Hb), Platelets (Plt) |
| **Coagulation** | Prothrombin Time (PT), INR |
| **Biochemistry** | Total Bilirubin, Albumin, ALP, AST, ALT, CRP |
| **Calculated** | AST/ALT Ratio |

## 3. Model Development Pipeline

### 3.1 Preprocessing

Data preprocessing is a critical step in the machine learning pipeline to ensure data quality and model stability. For continuous variables, missing values were addressed using median imputation, which is robust to outliers. Categorical variables with missing entries were imputed using the most frequent value (mode). Following imputation, all numeric features underwent standard scaling (Z-score normalization) to ensure they contributed equally to the model's distance calculations. Categorical variables were transformed using one-hot encoding, with the "drop first" parameter enabled to mitigate the risk of multicollinearity.

### 3.2 Model Selection & Tuning

The model selection process involved a rigorous comparison between two established algorithms: Random Forest (RF) and Logistic Regression (LR). To evaluate their performance reliably, a Repeated Stratified 5-Fold Cross-Validation strategy was employed, repeated 10 times to account for variance in the splits. The statistical significance of the performance difference between the two models was assessed using the Wilcoxon signed-rank test.

Following the selection of the optimal algorithm, hyperparameter tuning was conducted using Randomized Search CV. This process involved 15 iterations, optimizing for the Weighted One-vs-Rest (OvR) Area Under the Receiver Operating Characteristic Curve (ROC-AUC), ensuring the model was finely tuned for the specific characteristics of the dataset.

### 3.3 Class Imbalance

Clinical datasets frequently exhibit class imbalance, which can bias the model towards the majority class. To address this, the Synthetic Minority Over-sampling Technique (SMOTE) was implemented. Crucially, SMOTE was applied exclusively to the training folds within the cross-validation loop. This approach prevents data leakage, ensuring that the validation folds remain representative of the true, imbalanced clinical population.

## 4. Calibration Strategy

Model calibration is essential for clinical prediction models, as it ensures that the predicted probabilities accurately reflect the true likelihood of the event occurring. In this study, Isotonic Regression was selected as the calibration method.

The implementation involved setting aside a dedicated calibration set, comprising 25% of the initial development set. This separate dataset was used to fit the isotonic calibrators for each diagnostic class independently. The outcome of this calibration process was a significant improvement in the alignment between the model's predicted probabilities and the observed frequencies of the diagnoses, thereby enhancing the clinical reliability of the predictions.

## 5. Validation & Performance

The validation strategy was designed to rigorously assess the model's performance and generalizability. Internal validation was conducted using bootstrap confidence intervals (95% CI) with 2,000 iterations, providing a robust estimate of the model's stability.

For external validation, a held-out test set was utilized. This set, representing 20% of the original dataset, was strictly isolated during the entire training and tuning process, serving as an unbiased evaluator of the final model's performance on unseen data.

Furthermore, to evaluate the practical value of the model in a clinical setting, Decision Curve Analysis (DCA) was performed. DCA assesses the net benefit of using the model across a range of threshold probabilities, demonstrating its clinical utility compared to default strategies such as "treat-all" or "treat-none".

## 6. Explainability (SHAP)

To foster trust and transparency in the model's predictions, the SHapley Additive exPlanations (SHAP) framework was integrated into the pipeline. SHAP provides a unified approach to interpreting model output by assigning an importance value to each feature for a particular prediction.

The system offers both local and global explanations. For local explanations, individual patient predictions are accompanied by SHAP bar plots, which visually detail the specific contribution of each feature to that patient's predicted diagnosis. Global explanations are derived from the mean absolute SHAP values across the entire test set, providing a comprehensive ranking of feature importance across the population.

## 7. Failure Mode Analysis

A systematic Failure Mode Analysis was conducted to understand the limitations of the model and identify patterns in its errors. This involved a detailed review of all misclassified cases within the test set.

The findings revealed that misclassifications predominantly occurred in instances where the model exhibited lower confidence, specifically when the maximum predicted probability was below 0.6. Additionally, these errors were frequently observed between diagnoses that are clinically similar, highlighting areas where the model struggles to differentiate subtle clinical presentations.

## 8. Reproducibility Metadata

To ensure the reproducibility of this research, the following metadata details the computational environment and key dependencies used during the development of the model:

- **Environment**: Python 3.11
- **Key Libraries**: 
  - `scikit-learn` (v1.3.0)
  - `imbalanced-learn` (v0.11.0)
  - `shap` (v0.42.1)
- **Random Seed**: 42 (This seed was fixed for all data splits and stochastic processes to guarantee consistent results across runs).
- **Code Repository**: The complete source code, including the training pipeline and the Streamlit application, is available at [https://github.com/Mmadrb/liver-disease-analytics](https://github.com/Mmadrb/liver-disease-analytics).
