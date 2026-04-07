# Liver Disease Analytics

## Project Description

This project provides an interactive Streamlit application for analyzing liver disease data. It includes data cleaning, exploratory data analysis, machine learning model training (Random Forest with SMOTE for imbalanced data), and an explainable AI (SHAP) component for live patient predictions. The application aims to offer insights into liver disease progression and risk factors based on various clinical parameters.

## Features

*   **Interactive Data Exploration:** Visualize key demographic and clinical features.
*   **Automated Data Cleaning:** Handles missing values, renames columns, and converts data types.
*   **ACLF Grade Calculation:** Dynamically identifies and processes Acute-on-Chronic Liver Failure (ACLF) grades.
*   **Machine Learning Model:** Predicts liver disease outcomes using a Random Forest Classifier.
*   **Explainable AI (SHAP):** Provides insights into feature importance for global model predictions and individual patient predictions.
*   **User-Friendly Interface:** Built with Streamlit for easy interaction and visualization.

## Installation

To set up the project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/YOUR_USERNAME/liver-disease-analytics.git
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

## Usage

To run the Streamlit application:

```bash
streamlit run liver_disease.py
```

The application will open in your web browser. You can then interact with the various sections to explore data, view model performance, and make predictions.

## Project Structure

```
liver-disease-analytics/
├── liver_disease.py            # Main Streamlit application script
├── liver disease dataset.xlsx  # Sample dataset
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Deployment on Streamlit Cloud

1.  **Push your repository to GitHub.**
2.  **Go to [Streamlit Cloud](https://share.streamlit.io/) and log in.**
3.  **Click on "New app" and select your GitHub repository.**
4.  **Specify the main file path as `liver_disease.py` and the Python version.**
5.  **Click "Deploy!"**

## Contributing

Contributions are welcome! Please feel free to open issues or submit pull requests.

## License

This project is licensed under the MIT License - see the LICENSE file for details (if applicable).

## Contact

For any questions or suggestions, please contact [Your Name/Email/GitHub Profile].
