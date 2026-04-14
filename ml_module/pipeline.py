import os
import joblib
import pandas as pd
import numpy as np
import shap
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

class LiverDiseaseML:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.pipeline = None
        self.calibrators = None
        self.label_encoder = None
        self.feature_names = None
        self.numeric_features = [
            'Age', 'WBC-1', 'Hb', 'Plt', 'PT', 'INR-1',
            'TB: Total bilirubin', 'Albumin',
            'ALP: Alkaline phosphatase',
            'AST: Aspartate amino transferase',
            'ALT: Alamine amino transferase',
            'AST/ALT Ratio', 'Creatinine unadjusted', 'CRP level'
        ]
        self.categorical_features = ['Sex', 'DM', 'Smoke', 'Alcohol Use', 'Other toxins']
        
    def build_pipeline(self, classifier_type='rf', params=None):
        numeric_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler())
        ])
        categorical_transformer = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot",  OneHotEncoder(
                drop="first", handle_unknown="ignore", sparse_output=False
            ))
        ])
        preprocessor = ColumnTransformer([
            ("num", numeric_transformer,     self.numeric_features),
            ("cat", categorical_transformer, self.categorical_features)
        ])
        
        if classifier_type == 'rf':
            classifier = RandomForestClassifier(random_state=42, **(params or {}))
        else:
            classifier = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42, **(params or {}))
            
        return ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote",        SMOTE(random_state=42)),
            ("classifier",   classifier)
        ])

    def save_model(self, path):
        model_data = {
            'pipeline': self.pipeline,
            'calibrators': self.calibrators,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'numeric_features': self.numeric_features,
            'categorical_features': self.categorical_features
        }
        joblib.dump(model_data, path)

    def load_model(self, path):
        if not os.path.exists(path):
            return False
        model_data = joblib.load(path)
        self.pipeline = model_data['pipeline']
        self.calibrators = model_data['calibrators']
        self.label_encoder = model_data['label_encoder']
        self.feature_names = model_data['feature_names']
        self.numeric_features = model_data['numeric_features']
        self.categorical_features = model_data['categorical_features']
        return True

    def predict_proba(self, X):
        # Get raw probabilities
        probs = self.pipeline.predict_proba(X)
        
        # Apply calibration if available
        if self.calibrators:
            calibrated_probs = np.zeros_like(probs)
            for i, iso in enumerate(self.calibrators):
                calibrated_probs[:, i] = iso.transform(probs[:, i])
            # Re-normalize
            row_sums = calibrated_probs.sum(axis=1, keepdims=True)
            calibrated_probs = np.divide(calibrated_probs, row_sums, 
                                         out=np.zeros_like(calibrated_probs), 
                                         where=row_sums!=0)
            return calibrated_probs
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.label_encoder.inverse_transform(np.argmax(probs, axis=1))

    def get_shap_explainer(self):
        # Preprocess data for SHAP
        # SHAP works best on the output of the preprocessor
        classifier = self.pipeline.named_steps['classifier']
        return shap.TreeExplainer(classifier) if isinstance(classifier, RandomForestClassifier) else shap.LinearExplainer(classifier)

    def transform_for_shap(self, X):
        return self.pipeline.named_steps['preprocessor'].transform(X)

    def get_feature_names_out(self):
        return self.pipeline.named_steps['preprocessor'].get_feature_names_out()
