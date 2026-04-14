import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap
import os
import warnings
import joblib
from ml_module.pipeline import LiverDiseaseML

# ==========================================
# TARGET SPECIFIC WARNINGS ONLY
# ==========================================
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ==========================================
# 1. PAGE CONFIGURATION & UI CLEANUP
# ==========================================
st.set_page_config(page_title="Liver Disease Analytics", page_icon="🩺", layout="wide")

hide_streamlit_style = """
<style>
    .stAppDeployButton {display:none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .block-container {padding-top: 2rem;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ==========================================
# 2. DATA CLEANING ENGINE (Cached)
# ==========================================
@st.cache_data
def process_raw_data(df):
    """Heavy data cleaning logic, safely cached."""
    drop_cols = ["Unnamed: 0", "Key", "HEAD"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    
    # RENAME COLUMNS WITH VALIDATION
    rename_mapping = {
        "Primary diagnosis": "Diagnosis", "LOS: length of stay in hospital": "Length of Stay",
        "Living Status: 1= alive": "Alive", "TB: Total bilirubin": "Bilirubin",
        "AST: Aspartate amino transferase": "AST", "ALT: Alamine amino transferase": "ALT",
        "MELD Score": "MELD Score", "CTP Score": "CTP Score", "Age": "Age", "Sex": "Sex",
    }
    for old_name, new_name in rename_mapping.items():
        if old_name in df.columns:
            df = df.rename(columns={old_name: new_name})
    
    # SAFE NUMERIC CONVERSION
    numeric_cols = ['Age', 'MELD Score', 'CTP Score', 'Length of Stay', 'AST', 'ALT', 'Bilirubin']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].replace('*', np.nan)
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

# ==========================================
# 3. FILE INGESTION
# ==========================================
try:
    current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    current_dir = os.getcwd()
    
file_path = os.path.join(current_dir, "liver disease dataset.xlsx")

if os.path.exists(file_path):
    df_raw = process_raw_data(pd.read_excel(file_path))
else:
    uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx', 'xls'])
    if uploaded_file is not None:
        df_raw = process_raw_data(pd.read_excel(uploaded_file))
    else:
        st.info("Waiting for file upload...")
        st.stop()

# ==========================================
# 4. ML MODEL LOADING
# ==========================================
@st.cache_resource
def load_ml_model():
    model_path = os.path.join(current_dir, "ml_module", "liver_model.joblib")
    ml = LiverDiseaseML()
    if ml.load_model(model_path):
        return ml
    return None

ml_model = load_ml_model()

# ==========================================
# 5. APP NAVIGATION
# ==========================================
st.sidebar.title("🩺 Liver Analytics")
page = st.sidebar.radio("Go to", ["📊 Data Overview", "🤖 ML & SHAP Predictor"])

if page == "📊 Data Overview":
    st.title("📊 Clinical Data Overview")
    st.write("This section provides a high-level overview of the liver disease dataset.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients", len(df_raw))
    col2.metric("Features", len(df_raw.columns))
    col3.metric("Diagnoses", df_raw['Diagnosis'].nunique() if 'Diagnosis' in df_raw.columns else "N/A")
    
    if 'Diagnosis' in df_raw.columns:
        fig = px.pie(df_raw, names='Diagnosis', title='Distribution of Primary Diagnoses')
        st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(df_raw.head(10), use_container_width=True)

elif page == "🤖 ML & SHAP Predictor":
    st.title("🤖 ML & SHAP Predictor")
    st.markdown("### Clinical Prediction Interface")
    
    if ml_model is None:
        st.error("Model file not found. Please ensure `ml_module/liver_model.joblib` exists.")
        st.stop()
        
    with st.expander("ℹ️ About this Model", expanded=False):
        st.write("""
        This model is a **Random Forest Classifier** trained on clinical baseline features.
        It follows **TRIPOD-AI** guidelines for clinical prediction models and includes 
        **Isotonic Regression** for probability calibration.
        """)

    # Input Form
    with st.form("prediction_form"):
        st.subheader("Patient Baseline Features")
        c1, c2, c3 = st.columns(3)
        
        with c1:
            age = st.number_input("Age", 1, 120, 45)
            sex = st.selectbox("Sex", ["Male", "Female"])
            dm = st.selectbox("Diabetes Mellitus (DM)", ["No", "Yes"])
            smoke = st.selectbox("Smoking Status", ["No", "Yes"])
            
        with c2:
            alcohol = st.selectbox("Alcohol Use", ["No", "Yes"])
            toxins = st.selectbox("Other Toxins", ["No", "Yes"])
            wbc = st.number_input("WBC-1", 0.0, 100.0, 7.5)
            hb = st.number_input("Hb", 0.0, 20.0, 13.0)
            plt = st.number_input("Plt", 0.0, 1000.0, 200.0)
            
        with c3:
            pt = st.number_input("PT", 0.0, 100.0, 12.0)
            inr = st.number_input("INR-1", 0.0, 10.0, 1.0)
            tb = st.number_input("Total Bilirubin", 0.0, 50.0, 1.2)
            alb = st.number_input("Albumin", 0.0, 10.0, 4.0)
            alp = st.number_input("ALP", 0.0, 2000.0, 100.0)

        st.markdown("---")
        c4, c5, c6 = st.columns(3)
        with c4:
            ast = st.number_input("AST", 0.0, 2000.0, 35.0)
        with c5:
            alt = st.number_input("ALT", 0.0, 2000.0, 40.0)
        with c6:
            crp = st.number_input("CRP level", 0.0, 500.0, 5.0)
            
        submit = st.form_submit_button("Generate Prediction & SHAP Analysis", type="primary")

    if submit:
        # Prepare input data
        input_dict = {
            'Age': [age], 'Sex': [sex], 'DM': [dm], 'Smoke': [smoke],
            'Alcohol Use': [alcohol], 'Other toxins': [toxins],
            'WBC-1': [wbc], 'Hb': [hb], 'Plt': [plt], 'PT': [pt], 'INR-1': [inr],
            'TB: Total bilirubin': [tb], 'Albumin': [alb],
            'ALP: Alkaline phosphatase': [alp],
            'AST: Aspartate amino transferase': [ast],
            'ALT: Alamine amino transferase': [alt],
            'AST/ALT Ratio': [ast/alt if alt != 0 else 0],
            'Creatinine unadjusted': [1.0], # Default value as it wasn't in form but in features
            'CRP level': [crp]
        }
        input_df = pd.DataFrame(input_dict)
        
        # Prediction
        probs = ml_model.predict_proba(input_df)[0]
        pred_class = ml_model.predict(input_df)[0]
        
        # Results Display
        st.markdown("---")
        res_c1, res_c2 = st.columns([1, 1.5])
        
        with res_c1:
            st.success(f"### Predicted Diagnosis: **{pred_class}**")
            
            prob_df = pd.DataFrame({
                'Diagnosis': ml_model.label_encoder.classes_,
                'Probability': probs
            }).sort_values('Probability', ascending=False)
            
            fig_prob = px.bar(prob_df, x='Probability', y='Diagnosis', 
                             orientation='h', title="Class Probabilities (Calibrated)",
                             color='Probability', color_continuous_scale='Blues')
            st.plotly_chart(fig_prob, use_container_width=True)
            
        with res_c2:
            st.markdown("### 🧬 SHAP Interpretability")
            try:
                # SHAP Analysis
                explainer = ml_model.get_shap_explainer()
                X_transformed = ml_model.transform_for_shap(input_df)
                feature_names = ml_model.get_feature_names_out()
                
                shap_values = explainer.shap_values(X_transformed)
                
                # Handle multi-class SHAP output
                class_idx = list(ml_model.label_encoder.classes_).index(pred_class)
                if isinstance(shap_values, list):
                    # For some versions of SHAP/RF
                    sv = shap_values[class_idx][0]
                else:
                    # For newer SHAP versions
                    sv = shap_values[0, :, class_idx]
                
                shap_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Impact': sv
                }).sort_values('Impact', key=abs, ascending=False).head(10)
                
                shap_df['Direction'] = shap_df['Impact'].apply(lambda x: 'Positive' if x > 0 else 'Negative')
                
                fig_shap = px.bar(shap_df, x='Impact', y='Feature', color='Direction',
                                 orientation='h', title=f"Top 10 Features Influencing '{pred_class}'",
                                 color_discrete_map={'Positive': '#ef553b', 'Negative': '#636efa'})
                st.plotly_chart(fig_shap, use_container_width=True)
                
            except Exception as e:
                st.error(f"SHAP Analysis Error: {e}")
                st.info("This might be due to feature mismatch or SHAP versioning.")

st.markdown("---")
st.caption("© 2026 Liver Disease Analytics | Production-Ready Clinical ML System")
