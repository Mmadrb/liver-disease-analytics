import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.isotonic import IsotonicRegression
from pipeline import LiverDiseaseML

def train_and_save_model(data_path, model_save_path):
    # Load data
    df = pd.read_excel(data_path).drop(['Unnamed: 0', 'Key'], axis=1, errors='ignore')
    df_clean = df[df["Primary diagnosis"].notna()].copy()
    EXCLUDED_CLASSES = ["AIH", "HepB"]
    df_clean = df_clean[~df_clean["Primary diagnosis"].isin(EXCLUDED_CLASSES)]
    
    # Define features and target
    BASELINE_FEATURES = [
        'Age', 'Sex', 'DM', 'Smoke', 'Alcohol Use', 'Other toxins',
        'WBC-1', 'Hb', 'Plt', 'PT', 'INR-1',
        'TB: Total bilirubin', 'Albumin',
        'ALP: Alkaline phosphatase',
        'AST: Aspartate amino transferase',
        'ALT: Alamine amino transferase',
        'AST/ALT Ratio', 'Creatinine unadjusted', 'CRP level'
    ]
    TARGET_COL = "Primary diagnosis"
    
    available_features = [f for f in BASELINE_FEATURES if f in df_clean.columns]
    df_clean = df_clean[available_features + [TARGET_COL]].copy()
    
    X = df_clean.drop(columns=[TARGET_COL]).copy()
    y = df_clean[TARGET_COL]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # Split data
    X_dev, X_test, y_dev, y_test = train_test_split(
        X, y_encoded, test_size=0.20, stratify=y_encoded, random_state=42
    )
    X_train, X_calib, y_train, y_calib = train_test_split(
        X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=42
    )
    
    # Initialize and train
    ml = LiverDiseaseML()
    # Use RF as it was significantly better in the original script
    pipeline = ml.build_pipeline(classifier_type='rf', params={
        'n_estimators': 500,
        'max_depth': None,
        'min_samples_split': 2,
        'min_samples_leaf': 1
    })
    
    pipeline.fit(X_train, y_train)
    
    # Calibration
    y_prob_calib = pipeline.predict_proba(X_calib)
    iso_calibrators = []
    for i in range(len(le.classes_)):
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(y_prob_calib[:, i], (y_calib == i).astype(int))
        iso_calibrators.append(iso)
        
    # Store in object
    ml.pipeline = pipeline
    ml.calibrators = iso_calibrators
    ml.label_encoder = le
    ml.feature_names = available_features
    
    # Save
    ml.save_model(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == "__main__":
    data_path = "/home/ubuntu/liver-disease-analytics/liver disease dataset.xlsx"
    model_save_path = "/home/ubuntu/liver-disease-analytics/ml_module/liver_model.joblib"
    train_and_save_model(data_path, model_save_path)
