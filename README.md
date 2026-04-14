# Liver Disease Analytics

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://liver-disease-analytics.streamlit.app/)

## Project Description

This project provides an interactive Streamlit application for analyzing liver disease data. It includes data cleaning, exploratory data analysis, machine learning model training (Random Forest with SMOTE for imbalanced data), and an explainable AI (SHAP) component for live patient predictions. The application aims to offer insights into liver disease progression and risk factors based on various clinical parameters.

**Live App:** [https://liver-disease-analytics.streamlit.app/](https://liver-disease-analytics.streamlit.app/)

## Features

*   **Interactive Data Exploration:** Visualize key demographic and clinical features.
*   **Automated Data Cleaning:** Handles missing values, renames columns, and converts data types.
*   **ACLF Grade Calculation:** Dynamically identifies and processes Acute-on-Chronic Liver Failure (ACLF) grades.
*   **Machine Learning Model:** Predicts liver disease outcomes using a Random Forest Classifier.
*   **Explainable AI (SHAP):** Provides insights into feature importance for global model predictions and individual patient predictions.
*   **User-Friendly Interface:** Built with Streamlit for easy interaction and visualization.

## Installation & Local Run

To set up the project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/Mmadrb/liver-disease-analytics.git
    cd liver-disease-analytics
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the Streamlit application:**

    ```bash
    streamlit run liver_disease.py
    ```

## Project Structure

```
liver-disease-analytics/
├── liver_disease.py            # Main Streamlit application script
├── liver disease dataset.xlsx  # Sample dataset
├── requirements.txt            # Python dependencies
├── DEPLOYMENT_GUIDE.md         # Deployment instructions
└── README.md                   # Project documentation
```

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License.
