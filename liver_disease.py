
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import shap
import os
import warnings

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
from sklearn.exceptions import DataConversionWarning
from sklearn.isotonic import IsotonicRegression
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DataConversionWarning)

st.set_page_config(page_title="Liver Disease Analytics", page_icon="🩺", layout="wide")

hide_streamlit_style = """<style>.stAppDeployButton {display:none;} #MainMenu {visibility: hidden;} footer {visibility: hidden;} .block-container {padding-top: 2rem;}</style>"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# ==========================================
# 2. DATA CLEANING ENGINE (Cached)
# ==========================================
@st.cache_data
def process_raw_data(df):
    drop_cols = ["Unnamed: 0", "Key", "HEAD"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    
    aclf_col = None
    for col in df.columns:
        col_upper = str(col).upper().strip()
        if 'ACLF-EASL' in col_upper and ('ORGAN' in col_upper or '0=' in col_upper): aclf_col = col; break
    if aclf_col is None:
        for col in df.columns:
            if 'ACLF' in str(col).upper() and 'GRADE' not in str(col).upper(): aclf_col = col; break
    
    raw_unique = []
    if aclf_col:
        raw_unique = df[aclf_col].unique()
        aclf_text_map = {'0': 0, '0.0': 0, 'no': 0, 'No': 0, 'NO': 0, 'none': 0, 'None': 0, '1': 1, '1.0': 1, 'yes': 1, 'Yes': 1, 'YES': 1, 'one organ': 1, 'One organ': 1, 'ONE ORGAN': 1, 'one': 1, 'One': 1, '2': 2, '2.0': 2, 'two organs': 2, 'Two organs': 2, 'TWO ORGANS': 2, 'two': 2, 'Two': 2, '3': 3, '3.0': 3, 'three organs': 3, 'Three organs': 3, 'THREE ORGANS': 3, 'three': 3, '>= 3 organs': 3, '>=3 organs': 3, '>=3': 3, '3+': 3}
        df['ACLF Grade'] = df[aclf_col].astype(str).str.strip().map(aclf_text_map)
        df['ACLF Grade'] = pd.to_numeric(df['ACLF Grade'], errors='coerce')
        df.loc[df[aclf_col].astype(str).str.strip().isin(['*', 'NA', 'N/A', 'nan', 'NaN', '']), 'ACLF Grade'] = np.nan
        df['ACLF Grade'] = df['ACLF Grade'].fillna(0).astype(int)
    else: df['ACLF Grade'] = 0

    organ_failure_map = {'liver_failure': None, 'kidney_failure': None, 'brain_failure': None, 'circulatory_failure': None, 'respiratory_failure': None, 'coagulation_failure': None}
    for col in df.columns:
        col_lower = str(col).lower()
        if 'liver failure' in col_lower and 'apasl' in col_lower: organ_failure_map['liver_failure'] = col
        elif 'kf' in col_lower and 'easl' in col_lower and 'liver' not in col_lower: organ_failure_map['kidney_failure'] = col
        elif 'brain failure' in col_lower: organ_failure_map['brain_failure'] = col
        elif 'circulatory' in col_lower: organ_failure_map['circulatory_failure'] = col
        elif 'resp' in col_lower and 'failure' in col_lower: organ_failure_map['respiratory_failure'] = col
        elif 'coagulation' in col_lower: organ_failure_map['coagulation_failure'] = col
    
    for key, col in organ_failure_map.items():
        if col and col in df.columns: df[key] = df[col].astype(str).str.strip().isin(['yes', 'Yes', 'YES', '1', '1.0', 'true', 'True']).astype(int)
        else: df[key] = 0
    organ_cols = [k for k in organ_failure_map.keys() if k in df.columns]
    df['Total Organ Failures'] = df[organ_cols].sum(axis=1) if organ_cols else 0
    
    rename_mapping = {"Primary diagnosis": "Diagnosis", "LOS: length of stay in hospital": "Length of Stay", "Living Status: 1= alive": "Alive", "TB: Total bilirubin": "Bilirubin", "AST: Aspartate amino transferase": "AST", "ALT: Alamine amino transferase": "ALT", "MELD Score": "MELD Score", "CTP Score": "CTP Score", "Age": "Age", "Sex": "Sex"}
    successful_renames = {}
    for old_name, new_name in rename_mapping.items():
        if old_name in df.columns: df = df.rename(columns={old_name: new_name}); successful_renames[old_name] = new_name
        else:
            similar = [c for c in df.columns if old_name.lower().replace(" ", "") in c.lower().replace(" ", "")]
            if similar and len(similar) == 1: df = df.rename(columns={similar[0]: new_name}); successful_renames[f"{old_name} (fuzzy: {similar[0]})"] = new_name
    
    if 'Alive' in df.columns:
        alive_map = {'alive': 1, 'Alive': 1, 'ALIVE': 1, '1': 1, '1.0': 1, 'survived': 1, 'yes': 1, 'Yes': 1, 'death': 0, 'Death': 0, 'DEATH': 0, 'dead': 0, 'Dead': 0, 'deceased': 0, 'Deceased': 0, '0': 0, '0.0': 0, 'no': 0, 'No': 0, 'expired': 0, 'died': 0}
        df['Alive_Numeric'] = df['Alive'].astype(str).str.strip().map(alive_map)
        mask_failed = df['Alive_Numeric'].isna()
        if mask_failed.any(): df.loc[mask_failed, 'Alive_Numeric'] = pd.to_numeric(df.loc[mask_failed, 'Alive'], errors='coerce')
        df['Patient Outcome'] = df['Alive_Numeric'].apply(lambda x: 'Survived' if x == 1 else ('Deceased' if x == 0 else 'Unknown'))
    else: df['Alive_Numeric'] = np.nan; df['Patient Outcome'] = 'Unknown'
    
    for col in ['Age', 'MELD Score', 'CTP Score', 'Length of Stay', 'AST', 'ALT', 'Bilirubin']:
        if col in df.columns: df[col] = df[col].replace('*', np.nan); df[col] = pd.to_numeric(df[col], errors='coerce')
    
    if 'Age' in df.columns: df['Age Group'] = pd.cut(df['Age'], bins=[0, 34, 49, 64, 120], labels=['18-34', '35-49', '50-64', '65+'], right=True)
    if 'MELD Score' in df.columns: df['MELD Risk Tier'] = pd.cut(df['MELD Score'], bins=[0, 9.9, 19.9, 29.9, 100], labels=['Low (<10)', 'Moderate (10-19)', 'High (20-29)', 'Critical (30+)'])
    if 'CTP Score' in df.columns: df['CTP Class'] = pd.cut(df['CTP Score'], bins=[0, 6, 9, 15], labels=['Class A', 'Class B', 'Class C'])
    
    for cat_col in ['Age Group', 'MELD Risk Tier', 'CTP Class']:
        if cat_col in df.columns:
            if df[cat_col].dtype.name == 'category': df[cat_col] = df[cat_col].cat.add_categories('Unknown').fillna('Unknown')
            else: df[cat_col] = df[cat_col].fillna('Unknown')
            
    df.attrs['validation_report'] = {'total_rows': len(df), 'aclf_column_found': aclf_col is not None, 'aclf_column_name': aclf_col, 'aclf_raw_unique_sample': list(raw_unique[:10]) if aclf_col else [], 'successful_renames': successful_renames, 'organ_failure_columns_found': {k: v for k, v in organ_failure_map.items() if v}, 'missing_critical': [c for c in ['Diagnosis', 'MELD Score', 'CTP Score', 'Alive'] if c not in df.columns], 'aclf_distribution': df['ACLF Grade'].value_counts().to_dict()}
    return df

# ==========================================
# 3. FILE INGESTION
# ==========================================
try: current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError: current_dir = os.getcwd()
file_path = os.path.join(current_dir, "liver disease dataset.xlsx")

if os.path.exists(file_path): df_raw = process_raw_data(pd.read_excel(file_path))
else:
    st.warning("Default dataset not found. Please upload a dataset to continue.")
    uploaded_file = st.file_uploader("Upload Excel File", type=['xlsx', 'xls'], key="file_uploader")
    if uploaded_file is not None: df_raw = process_raw_data(pd.read_excel(uploaded_file)); st.success("Dataset successfully uploaded and processed!")
    else: st.info("Waiting for file upload..."); st.stop()

# ==========================================
# 4. DATA QUALITY DEBUG PANEL
# ==========================================
with st.expander("🔧 Data Quality & Validation Report", expanded=False):
    report = df_raw.attrs.get('validation_report', {})
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows Loaded", report.get('total_rows', 'N/A'))
    col2.metric("ACLF Column Found", "✅ Yes" if report.get('aclf_column_found') else "❌ No")
    col3.metric("Missing Critical Cols", len(report.get('missing_critical', [])))
    if report.get('missing_critical'): st.error(f"Missing critical columns: {report['missing_critical']}")
    aclf_dist = report.get('aclf_distribution', {})
    if aclf_dist:
        aclf_display = pd.DataFrame([{'Grade': str(k), 'Count': v, 'Description': 'No ACLF' if k == 0 else f'{k} Organ Failure(s)' if k < 3 else '≥3 Organ Failures'} for k, v in sorted(aclf_dist.items())])
        st.dataframe(aclf_display, use_container_width=True, hide_index=True)
        total_aclf = sum(v for k, v in aclf_dist.items() if k > 0)
        st.success(f"✅ Calculated ACLF Rate: {(total_aclf / report['total_rows'] * 100):.1f}% ({total_aclf} patients)")

# ==========================================
# 5. DUAL ML ENGINES
# ==========================================
@st.cache_resource
def train_clinical_model(df):
    """Strict 8-feature Time-Zero Model"""
    if 'Diagnosis' not in df.columns: return None
    df_clean = df.dropna(subset=["Diagnosis"]).copy()
    class_counts = df_clean['Diagnosis'].value_counts()
    valid_classes = class_counts[class_counts >= 6].index
    if len(valid_classes) < 2: return None
    df_clean = df_clean[df_clean['Diagnosis'].isin(valid_classes)].copy()
    
    features = ['Age', 'Sex', 'MELD Score', 'CTP Score', 'ACLF Grade', 'AST', 'ALT', 'Bilirubin']
    features = [f for f in features if f in df_clean.columns]
    X, y = df_clean[features].copy(), df_clean['Diagnosis'].copy()
    le = LabelEncoder(); y_encoded = le.fit_transform(y)
    
    num_cols = X.select_dtypes(include=['int64', 'float64', 'int32', 'Int64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    transformers = []
    if num_cols: transformers.append(("num", SimpleImputer(strategy="median"), num_cols))
    if cat_cols: transformers.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat_cols))
    if not transformers: return None
    
    preprocessor = ColumnTransformer(transformers)
    rf_model = RandomForestClassifier(n_estimators=300, min_samples_split=5, class_weight="balanced", random_state=42, n_jobs=-1)
    pipeline = ImbPipeline([("prep", preprocessor), ("smote", SMOTE(random_state=42, k_neighbors=min(5, len(valid_classes) - 1))), ("clf", rf_model)])
    
    try: cv_score = cross_val_score(pipeline, X, y_encoded, cv=5, scoring='accuracy').mean()
    except: cv_score = np.nan
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred, y_prob = pipeline.predict(X_test), pipeline.predict_proba(X_test)
    
    ohe = pipeline.named_steps['prep'].named_transformers_['cat'].named_steps['ohe'] if 'cat' in pipeline.named_steps['prep'].named_transformers_ else None
    cat_features_list = ohe.get_feature_names_out(cat_cols) if ohe else []
    all_features = np.concatenate([num_cols, cat_features_list])
    
    return {'pipeline': pipeline, 'label_encoder': le, 'test_accuracy': accuracy_score(y_test, y_pred), 'cv_accuracy': cv_score, 'roc_auc': roc_auc_score(y_test, y_prob, multi_class="ovr"), 'confusion_matrix': confusion_matrix(y_test, y_pred), 'classification_report': classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0), 'feature_importances': pd.DataFrame({"Feature": all_features, "Importance": pipeline.named_steps['clf'].feature_importances_}).sort_values(by="Importance", ascending=False).reset_index(drop=True), 'clean_feature_names': all_features.tolist(), 'valid_features': features, 'explainer': shap.TreeExplainer(pipeline.named_steps['clf'])}

@st.cache_resource
def train_full_pipeline_model(df):
    """
    TRIPOD-AI Compliant Liver Disease Classifier
    Integrates liver_baseline_classifier.py logic
    """
    if 'Diagnosis' not in df.columns: return None
    
    # 1. Data Cleaning & Exclusion (from liver_baseline_classifier.py)
    df_clean = df.dropna(subset=["Diagnosis"]).copy()
    EXCLUDED_CLASSES = ["AIH", "HepB"]
    df_clean = df_clean[~df_clean["Diagnosis"].isin(EXCLUDED_CLASSES)]
    
    # 2. Define Baseline Features
    BASELINE_FEATURES = [
        'Age', 'Sex', 'DM', 'Smoke', 'Alcohol Use', 'Other toxins',
        'WBC-1', 'Hb', 'Plt', 'PT', 'INR-1',
        'TB: Total bilirubin', 'Albumin',
        'ALP: Alkaline phosphatase',
        'AST: Aspartate amino transferase',
        'ALT: Alamine amino transferase',
        'AST/ALT Ratio', 'Creatinine unadjusted', 'CRP level'
    ]
    
    available_features = [f for f in BASELINE_FEATURES if f in df_clean.columns]
    X = df_clean[available_features].copy()
    y = df_clean["Diagnosis"]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 3. Three-Way Split (60/20/20)
    X_dev, X_test, y_dev, y_test = train_test_split(X, y_encoded, test_size=0.20, stratify=y_encoded, random_state=42)
    X_train, X_calib, y_train, y_calib = train_test_split(X_dev, y_dev, test_size=0.25, stratify=y_dev, random_state=42)
    
    # 4. Preprocessing Pipeline
    numeric_features = X.select_dtypes(include=["int64", "float64", "int32", "float32"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()
    
    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_transformer = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))])
    
    preprocessor = ColumnTransformer([("num", numeric_transformer, numeric_features), ("cat", categorical_transformer, categorical_features)])
    
    # 5. Model Pipeline (RF + SMOTE)
    rf_model = RandomForestClassifier(n_estimators=500, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42, n_jobs=-1)
    pipeline = ImbPipeline([("preprocessor", preprocessor), ("smote", SMOTE(random_state=42)), ("classifier", rf_model)])
    
    # 6. Fit & Calibrate
    pipeline.fit(X_train, y_train)
    
    # Isotonic Calibration
    y_prob_calib = pipeline.predict_proba(X_calib)
    iso_calibrators = []
    for i in range(len(le.classes_)):
        iso = IsotonicRegression(out_of_bounds='clip')
        iso.fit(y_prob_calib[:, i], (y_calib == i).astype(int))
        iso_calibrators.append(iso)
    
    # 7. Evaluation on Test Set
    y_prob_raw = pipeline.predict_proba(X_test)
    y_prob_calibrated = np.zeros_like(y_prob_raw)
    for i, iso in enumerate(iso_calibrators):
        y_prob_calibrated[:, i] = iso.transform(y_prob_raw[:, i])
    
    # Re-normalize calibrated probabilities
    row_sums = y_prob_calibrated.sum(axis=1, keepdims=True)
    y_prob_calibrated = np.divide(y_prob_calibrated, row_sums, out=np.zeros_like(y_prob_calibrated), where=row_sums!=0)
    y_pred = np.argmax(y_prob_calibrated, axis=1)
    
    # Feature Names
    ohe = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'] if categorical_features else None
    cat_features_list = ohe.get_feature_names_out(categorical_features) if ohe else []
    all_features = np.concatenate([numeric_features, cat_features_list])
    
    return {
        'pipeline': pipeline, 'calibrators': iso_calibrators, 'label_encoder': le, 
        'test_accuracy': accuracy_score(y_test, y_pred), 'cv_accuracy': np.nan,
        'roc_auc': roc_auc_score(y_test, y_prob_calibrated, multi_class="ovr"),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, target_names=le.classes_, output_dict=True, zero_division=0),
        'feature_importances': pd.DataFrame({"Feature": all_features, "Importance": pipeline.named_steps['classifier'].feature_importances_}).sort_values(by="Importance", ascending=False).reset_index(drop=True),
        'clean_feature_names': all_features.tolist(), 'valid_features': available_features, 
        'explainer': shap.TreeExplainer(pipeline.named_steps['classifier']),
        'numeric_features': numeric_features, 'categorical_features': categorical_features, 'raw_df': X
    }

# Train both models on load
with st.spinner("Training Dual ML Engines..."):
    clinical_results = train_clinical_model(df_raw)
    full_results = train_full_pipeline_model(df_raw)

# ==========================================
# 6. SIDEBAR & GLOBAL FILTERS
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3024/3024996.png", width=60)
st.sidebar.title("DataMed Analytics")
st.sidebar.download_button(label="📥 Download Cleaned Dataset", data=df_raw.to_csv(index=False).encode('utf-8'), file_name='liver_disease_cleaned.csv', mime='text/csv')
st.sidebar.markdown("---")
st.sidebar.markdown("### Global Filters")

sex_options = df_raw['Sex'].dropna().unique().tolist() if 'Sex' in df_raw.columns else []
age_options = [a for a in df_raw['Age Group'].unique() if a not in ['Unknown', None]] if 'Age Group' in df_raw.columns else []
diag_options = df_raw['Diagnosis'].dropna().unique().tolist() if 'Diagnosis' in df_raw.columns else []

sel_sex = st.sidebar.multiselect("Sex", sex_options, default=sex_options)
sel_age = st.sidebar.multiselect("Age Group", age_options, default=age_options)
sel_diag = st.sidebar.multiselect("Diagnosis", diag_options, default=diag_options)

df = df_raw.copy()
if sel_sex and 'Sex' in df.columns: df = df[df['Sex'].isin(sel_sex)]
if sel_age and 'Age Group' in df.columns: df = df[df['Age Group'].isin(sel_age)]
if sel_diag and 'Diagnosis' in df.columns: df = df[df['Diagnosis'].isin(sel_diag)]

# ==========================================
# 7. HEADER
# ==========================================
st.title("📊 Liver Disease Analytics Dashboard")
st.markdown("**End-to-end clinical analytics platform** featuring a Dual-Engine ML Architecture.")
if len(df) > 0:
    aclf_rate = (df['ACLF Grade'] > 0).sum() / len(df) * 100 if len(df) > 0 else 0
    mortality_rate = (df['Patient Outcome'] == 'Deceased').sum() / len(df) * 100 if len(df) > 0 else 0
    st.info(f"📊 **Current Filter:** {len(df)} patients | ACLF Rate: {aclf_rate:.1f}% | Mortality: {mortality_rate:.1f}%")
else: st.warning("No data matches the selected filters."); st.stop()

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["Executive Summary", "Risk Stratification", "Clinical Indicators", "Organ Failure & ACLF", "Length of Stay", "🤖 ML & SHAP Predictor"])

# ----------------- TABS 1-5 (Unchanged) -----------------
with tab1:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Patients", f"{len(df):,}"); c2.metric("Mortality Rate", f"{mortality_rate:.1f}%")
    avg_meld = df['MELD Score'].mean() if 'MELD Score' in df.columns else np.nan
    c3.metric("Avg MELD Score", f"{avg_meld:.1f}" if not np.isnan(avg_meld) else "N/A")
    avg_los = df['Length of Stay'].mean() if 'Length of Stay' in df.columns else np.nan
    c4.metric("Avg LOS", f"{avg_los:.1f} days" if not np.isnan(avg_los) else "N/A"); c5.metric("ACLF Rate", f"{aclf_rate:.1f}%")
    st.markdown("---"); col1, col2 = st.columns(2)
    with col1:
        if 'Diagnosis' in df.columns: st.plotly_chart(px.pie(df['Diagnosis'].value_counts().reset_index(), values='count', names='Diagnosis', hole=0.5, title="Etiology Breakdown").update_traces(textposition='inside', textinfo='percent+label'), use_container_width=True)
    with col2:
        if not np.isnan(avg_meld): st.plotly_chart(go.Figure(go.Indicator(mode="gauge+number", value=avg_meld, title={'text': "Average MELD Score"}, gauge={'axis': {'range': [None, 40]}, 'bar': {'color': "darkred"}, 'steps': [{'range': [0, 15], 'color': "lightgreen"}, {'range': [15, 25], 'color': "gold"}, {'range': [25, 40], 'color': "salmon"}], 'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': 30}})), use_container_width=True)
    st.markdown("---"); st.markdown("### ACLF Grade Summary")
    aclf_summary = df['ACLF Grade'].value_counts().sort_index().reset_index(); aclf_summary.columns = ['ACLF Grade', 'Count']
    aclf_summary['Percentage'] = (aclf_summary['Count'] / len(df) * 100).round(1)
    aclf_summary['Description'] = aclf_summary['ACLF Grade'].apply(lambda x: 'No ACLF' if x == 0 else f'{x} Organ Failure(s)' if x < 3 else '≥3 Organ Failures')
    st.dataframe(aclf_summary, use_container_width=True, hide_index=True)

with tab2:
    col1, col2 = st.columns(2)
    with col1:
        if 'MELD Risk Tier' in df.columns:
            tier_outcome = df.groupby(['MELD Risk Tier', 'Patient Outcome'], observed=False).size().reset_index(name='Count')
            tier_outcome = tier_outcome[tier_outcome['MELD Risk Tier'] != 'Unknown']
            if len(tier_outcome) > 0: st.plotly_chart(px.bar(tier_outcome, x='MELD Risk Tier', y='Count', color='Patient Outcome', barmode='stack', title="MELD Risk Tier vs Outcome", color_discrete_map={'Survived': '#2E86C1', 'Deceased': '#E74C3C', 'Unknown': '#95A5A6'}), use_container_width=True)
    with col2:
        if 'Age Group' in df.columns and 'Alive_Numeric' in df.columns:
            heat_df = df[(df['Age Group'] != 'Unknown') & (df['CTP Class'] != 'Unknown')]
            if len(heat_df) > 0:
                pivot = heat_df.pivot_table(index='Age Group', columns='CTP Class', values='Alive_Numeric', aggfunc=lambda x: (1 - np.nanmean(x)) * 100, observed=False).fillna(0)
                if 'Unknown' in pivot.columns: pivot = pivot.drop(columns=['Unknown'])
                st.plotly_chart(px.imshow(pivot, text_auto=".1f", color_continuous_scale="Reds", title="Mortality Rate (%) by Age & CTP Class"), use_container_width=True)
    st.markdown("---")
    if 'ACLF Grade' in df.columns and 'Patient Outcome' in df.columns:
        aclf_mort = df.groupby(['ACLF Grade', 'Patient Outcome'], observed=False).size().reset_index(name='Count')
        aclf_mort['Total'] = aclf_mort.groupby('ACLF Grade')['Count'].transform('sum')
        aclf_mort['Rate'] = (aclf_mort['Count'] / aclf_mort['Total'] * 100).round(1)
        aclf_mort_display = aclf_mort[aclf_mort['Patient Outcome'] == 'Deceased'][['ACLF Grade', 'Count', 'Total', 'Rate']]
        aclf_mort_display.columns = ['ACLF Grade', 'Deaths', 'Total in Grade', 'Mortality Rate (%)']
        st.markdown("### Mortality Rate by ACLF Grade"); st.dataframe(aclf_mort_display, use_container_width=True, hide_index=True)

with tab3:
    col1, col2 = st.columns(2)
    with col1:
        if 'AST' in df.columns: st.plotly_chart(px.box(df.dropna(subset=['AST']), x="Patient Outcome", y="AST", color="Patient Outcome", title="AST Level by Outcome", color_discrete_map={'Survived': '#2E86C1', 'Deceased': '#E74C3C', 'Unknown': '#95A5A6'}).update_layout(showlegend=False), use_container_width=True)
    with col2:
        if 'ALT' in df.columns: st.plotly_chart(px.box(df.dropna(subset=['ALT']), x="Patient Outcome", y="ALT", color="Patient Outcome", title="ALT Level by Outcome", color_discrete_map={'Survived': '#2E86C1', 'Deceased': '#E74C3C', 'Unknown': '#95A5A6'}).update_layout(showlegend=False), use_container_width=True)
    st.markdown("---"); col3, col4 = st.columns(2)
    with col3:
        if 'Bilirubin' in df.columns: st.plotly_chart(px.box(df.dropna(subset=['Bilirubin']), x="Patient Outcome", y="Bilirubin", color="Patient Outcome", title="Bilirubin Level by Outcome", color_discrete_map={'Survived': '#2E86C1', 'Deceased': '#E74C3C', 'Unknown': '#95A5A6'}).update_layout(showlegend=False), use_container_width=True)
    with col4:
        if 'MELD Score' in df.columns: st.plotly_chart(px.box(df.dropna(subset=['MELD Score']), x="Patient Outcome", y="MELD Score", color="Patient Outcome", title="MELD Score by Outcome", color_discrete_map={'Survived': '#2E86C1', 'Deceased': '#E74C3C', 'Unknown': '#95A5A6'}).update_layout(showlegend=False), use_container_width=True)

with tab4:
    st.markdown("### Organ Failure & ACLF Analysis"); aclf_positive = (df['ACLF Grade'] > 0).sum() if 'ACLF Grade' in df.columns else 0
    if aclf_positive == 0: st.warning("⚠️ No ACLF cases detected in the current filtered data.")
    else:
        col1, col2 = st.columns([2, 1])
        with col1:
            aclf_df = df[df['ACLF Grade'] > 0].copy()
            if 'Diagnosis' in aclf_df.columns and len(aclf_df) > 0:
                aclf_diag = aclf_df.groupby(['Diagnosis', 'ACLF Grade']).size().reset_index(name='Count')
                aclf_diag['ACLF Grade'] = aclf_diag['ACLF Grade'].apply(lambda x: f"{x} Organ Failure(s)" if x < 3 else "≥3 Organ Failures")
                st.plotly_chart(px.bar(aclf_diag, x='Diagnosis', y='Count', color='ACLF Grade', barmode='stack', title=f"ACLF Distribution by Diagnosis (n={len(aclf_df)})", color_discrete_sequence=px.colors.sequential.YlOrRd), use_container_width=True)
        with col2:
            st.plotly_chart(px.funnel(dict(number=[len(df), aclf_positive, (df['ACLF Grade'] >= 2).sum(), (df['Patient Outcome'] == 'Deceased').sum()], stage=[f"Admitted (n={len(df)})", f"Any ACLF (n={aclf_positive})", f"Severe ACLF ≥2 (n={(df['ACLF Grade'] >= 2).sum()})", f"Deceased (n={(df['Patient Outcome'] == 'Deceased').sum()})"]), x='number', y='stage', title="Patient Flow"), use_container_width=True)
    st.markdown("---"); st.markdown("### Individual Organ Failure Analysis")
    organ_cols_check = ['liver_failure', 'kidney_failure', 'brain_failure', 'circulatory_failure', 'respiratory_failure', 'coagulation_failure']
    existing_organs = [c for c in organ_cols_check if c in df.columns and df[c].sum() > 0]
    if existing_organs:
        organ_summary = pd.DataFrame({'Organ System': [c.replace('_', ' ').title() for c in existing_organs], 'Failure Count': [df[c].sum() for c in existing_organs], 'Failure Rate (%)': [(df[c].sum() / len(df) * 100).round(1) for c in existing_organs]}).sort_values('Failure Count', ascending=False)
        col1, col2 = st.columns([1, 1])
        with col1: st.plotly_chart(px.bar(organ_summary, x='Organ System', y='Failure Count', title="Organ Failure Frequency", color='Failure Rate (%)', color_continuous_scale="Reds"), use_container_width=True)
        with col2: st.dataframe(organ_summary, use_container_width=True, hide_index=True)

with tab5:
    col1, col2 = st.columns(2)
    with col1:
        if 'Length of Stay' in df.columns:
            los_df = df.dropna(subset=['Length of Stay'])
            if len(los_df) > 0: st.plotly_chart(px.histogram(los_df, x="Length of Stay", nbins=30, color="Patient Outcome", title="LOS Distribution", color_discrete_map={'Survived': '#2E86C1', 'Deceased': '#E74C3C', 'Unknown': '#95A5A6'}, marginal="box"), use_container_width=True)
    with col2:
        if 'MELD Score' in df.columns and 'Length of Stay' in df.columns:
            meld_los = df.dropna(subset=['MELD Score', 'Length of Stay'])
            if len(meld_los) > 0: st.plotly_chart(px.scatter(meld_los, x="MELD Score", y="Length of Stay", color="Patient Outcome", trendline="ols", title="MELD Score vs Length of Stay", color_discrete_map={'Survived': '#2E86C1', 'Deceased': '#E74C3C', 'Unknown': '#95A5A6'}, opacity=0.6), use_container_width=True)
    st.markdown("---")
    if 'Length of Stay' in df.columns and 'ACLF Grade' in df.columns:
        los_by_aclf = df.groupby('ACLF Grade')['Length of Stay'].agg(['mean', 'median', 'count']).round(1); los_by_aclf.columns = ['Mean LOS', 'Median LOS', 'Count']; los_by_aclf = los_by_aclf.reset_index()
        st.markdown("### Length of Stay by ACLF Grade"); st.dataframe(los_by_aclf, use_container_width=True, hide_index=True)

# ----------------- TAB 6: DUAL ML ENGINE -----------------
with tab6:
    st.markdown("### 🧠 Machine Learning & Explainable AI")
    
    engine_choice = st.segmented_control("Select Inference Engine", options=["🩺 Clinical Model (8 Features)", "📊 Full Data Pipeline (All Features)"], default="🩺 Clinical Model (8 Features)")
    
    is_full_model = "Full Data" in engine_choice
    active_results = full_results if is_full_model else clinical_results
    
    if active_results is None: st.error(f"Could not train the selected model. Check data constraints."); st.stop()
        
    model_pipeline = active_results['pipeline']
    label_encoder = active_results['label_encoder']
    test_acc = active_results['test_accuracy']
    cv_acc = active_results['cv_accuracy']
    roc_auc = active_results['roc_auc']
    conf_matrix = active_results['confusion_matrix']
    class_report = active_results['classification_report']
    feature_importances = active_results['feature_importances']
    clean_feat_names = active_results['clean_feature_names']
    valid_features = active_results['valid_features']
    shap_explainer = active_results['explainer']

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Algorithm", "RF + SMOTE (500 Trees)" if is_full_model else "RF + SMOTE (300 Trees)")
    m2.metric("5-Fold CV", f"{cv_acc * 100:.1f}%" if not np.isnan(cv_acc) else "Skipped (High Dim)")
    m3.metric("Test Accuracy", f"{test_acc * 100:.1f}%")
    m4.metric("ROC-AUC", f"{roc_auc:.3f}" if not np.isnan(roc_auc) else "N/A")
    
    if is_full_model: st.caption("💡 *Note: High accuracy in the Full Data Pipeline is driven by administrative/post-admission variables (e.g., DILI status, Precipitant Factors) that act as proxies for the target. See Feature Importance below.*")

    with st.expander("🔮 Live Patient Predictor & AI Explanation", expanded=True):
        if is_full_model:
            st.markdown("**Dynamic Feature Input** (Auto-generated from dataset columns)")
            user_dict = {}
            lab_keywords = ['AST', 'ALT', 'TB', 'PT', 'INR', 'Albumin', 'ALP', 'WBC', 'Plt', 'Hb', 'Creatinine', 'CRP', 'Lactate', 'Bili', 'MELD', 'CTP', 'APRI', 'Fib']
            demo_keywords = ['Age', 'Sex', 'DM', 'Smoke']
            prec_keywords = ['Precip', 'DILI', 'Alcohol', 'GIB', 'Toxin', 'Suppl']
            lab_cols = [c for c in valid_features if any(k in c for k in lab_keywords)]
            demo_cols = [c for c in valid_features if any(k in c for k in demo_keywords)]
            prec_cols = [c for c in valid_features if any(k in c for k in prec_keywords)]
            other_cols = [c for c in valid_features if c not in lab_cols + demo_cols + prec_cols]
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                st.markdown("**Demographics & Precipitants**")
                for col in demo_cols + prec_cols:
                    if df_raw[col].dtype in ['int64', 'float64', 'Int64']: user_dict[col] = [st.number_input(col, value=float(df_raw[col].median()), key=f"num_{col}")]
                    else:
                        opts = df_raw[col].dropna().unique().tolist()
                        user_dict[col] = [st.selectbox(col, opts, key=f"cat_{col}")] if opts else [np.nan]
            with c2:
                st.markdown("**Core Labs**")
                for col in lab_cols[:10]:
                    if df_raw[col].dtype in ['int64', 'float64', 'Int64']: user_dict[col] = [st.number_input(col, value=float(df_raw[col].median()), key=f"num_{col}")]
                    else:
                        opts = df_raw[col].dropna().unique().tolist()
                        user_dict[col] = [st.selectbox(col, opts, key=f"cat_{col}")] if opts else [np.nan]
            with c3:
                st.markdown("**Advanced Labs & Scores**")
                for col in lab_cols[10:]:
                    if df_raw[col].dtype in ['int64', 'float64', 'Int64']: user_dict[col] = [st.number_input(col, value=float(df_raw[col].median()), key=f"num_{col}")]
                    else:
                        opts = df_raw[col].dropna().unique().tolist()
                        user_dict[col] = [st.selectbox(col, opts, key=f"cat_{col}")] if opts else [np.nan]
            with c4:
                st.markdown("**Miscellaneous Features**")
                for col in other_cols:
                    if df_raw[col].dtype in ['int64', 'float64', 'Int64']: user_dict[col] = [st.number_input(col, value=float(df_raw[col].median()), key=f"num_{col}")]
                    else:
                        opts = df_raw[col].dropna().unique().tolist()
                        user_dict[col] = [st.selectbox(col, opts, key=f"cat_{col}")] if opts else [np.nan]
        else:
            user_dict = {}
            form_c1, form_c2, form_c3 = st.columns(3)
            with form_c1:
                user_dict['Age'] = [st.slider("Age", int(df_raw['Age'].min()), int(df_raw['Age'].max()), int(df_raw['Age'].median()))]
                user_dict['Sex'] = [st.selectbox("Sex", df_raw['Sex'].dropna().unique())]
                if 'AST' in valid_features: user_dict['AST'] = [st.slider("AST", int(df_raw['AST'].min()), int(df_raw['AST'].max()), int(df_raw['AST'].median()))]
            with form_c2:
                user_dict['MELD Score'] = [st.slider("MELD Score", int(df_raw['MELD Score'].min()), int(df_raw['MELD Score'].max()), int(df_raw['MELD Score'].median()))]
                if 'CTP Score' in valid_features: user_dict['CTP Score'] = [st.slider("CTP Score", int(df_raw['CTP Score'].min()), int(df_raw['CTP Score'].max()), int(df_raw['CTP Score'].median()))]
                if 'ALT' in valid_features: user_dict['ALT'] = [st.slider("ALT", int(df_raw['ALT'].min()), int(df_raw['ALT'].max()), int(df_raw['ALT'].median()))]
            with form_c3:
                if 'Bilirubin' in valid_features: user_dict['Bilirubin'] = [st.slider("Bilirubin", float(df_raw['Bilirubin'].min()), float(df_raw['Bilirubin'].max()), float(df_raw['Bilirubin'].median()))]
                if 'ACLF Grade' in valid_features: user_dict['ACLF Grade'] = [st.selectbox("ACLF Grade", sorted(df_raw['ACLF Grade'].unique()))]
                st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Predict & Explain 🚀", type="primary", use_container_width=True): st.session_state.run_prediction = True

        if st.session_state.get('run_prediction', False):
            user_data = pd.DataFrame(user_dict)
            
            # Prediction Logic (Calibrated for Full Model)
            if is_full_model:
                y_prob_raw = model_pipeline.predict_proba(user_data)
                y_prob_calibrated = np.zeros_like(y_prob_raw)
                for i, iso in enumerate(active_results['calibrators']): y_prob_calibrated[:, i] = iso.transform(y_prob_raw[:, i])
                row_sums = y_prob_calibrated.sum(axis=1, keepdims=True)
                y_prob_calibrated = np.divide(y_prob_calibrated, row_sums, out=np.zeros_like(y_prob_calibrated), where=row_sums!=0)
                pred_proba = y_prob_calibrated[0]
                pred_encoded = [np.argmax(pred_proba)]
            else:
                pred_encoded = model_pipeline.predict(user_data)
                pred_proba = model_pipeline.predict_proba(user_data)[0]
            
            pred_class = label_encoder.inverse_transform(pred_encoded)[0]
            proba_df = pd.DataFrame({'Diagnosis': label_encoder.classes_, 'Probability': pred_proba * 100}).sort_values('Probability', ascending=False).round(2)
            
            col_r1, col_r2 = st.columns(2)
            with col_r1:
                st.success(f"### Predicted: **{pred_class}**")
                st.dataframe(proba_df.head(5), use_container_width=True, hide_index=True)
            with col_r2:
                st.markdown("#### AI Reasoning (SHAP)")
                try:
                    user_transformed = model_pipeline.named_steps['preprocessor'].transform(user_data) if is_full_model else model_pipeline.named_steps['prep'].transform(user_data)
                    if shap_explainer is not None: shap_values = shap_explainer.shap_values(user_transformed)
                    else: shap_values = shap.TreeExplainer(model_pipeline.named_steps['classifier'] if is_full_model else model_pipeline.named_steps['clf']).shap_values(user_transformed)
                    
                    if isinstance(shap_values, list): local_shap = shap_values[pred_encoded[0]][0]
                    elif len(np.array(shap_values).shape) == 3: local_shap = shap_values[0, :, pred_encoded[0]]
                    else: local_shap = shap_values[0]
                        
                    shap_df = pd.DataFrame({'Feature': clean_feat_names, 'SHAP Impact': local_shap})
                    shap_df['Direction'] = shap_df['SHAP Impact'].apply(lambda x: '↑ Increases' if x > 0 else '↓ Decreases')
                    shap_df = shap_df.reindex(shap_df['SHAP Impact'].abs().sort_values(ascending=False).index)
                    st.plotly_chart(px.bar(shap_df.head(10), x='SHAP Impact', y='Feature', color='Direction', orientation='h', color_discrete_map={'↑ Increases': '#E74C3C', '↓ Decreases': '#2E86C1'}, title=f"Top Factors for '{pred_class}'").update_layout(yaxis={'categoryorder': 'total ascending'}), use_container_width=True)
                except Exception as e: st.warning(f"SHAP calculation failed: {e}")
    
    st.markdown("---")
    col1, col2 = st.columns([1.2, 1])
    with col1:
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        cm_norm = np.divide(conf_matrix, row_sums, out=np.zeros_like(conf_matrix, dtype=float), where=row_sums != 0)
        st.plotly_chart(px.imshow(cm_norm, text_auto=".2f", color_continuous_scale="Blues", x=label_encoder.classes_, y=label_encoder.classes_, labels=dict(x="Predicted", y="Actual"), title="Normalized Confusion Matrix"), use_container_width=True)
    with col2:
        top_n = 20 if is_full_model else 15
        st.plotly_chart(px.bar(feature_importances.head(top_n), x="Importance", y="Feature", orientation='h', title=f"Global Feature Importance (Top {top_n})", color="Importance", color_continuous_scale="Viridis").update_layout(yaxis={'categoryorder': 'total ascending'}), use_container_width=True)
    with st.expander("📋 Full Classification Report"):
        report_df = pd.DataFrame(class_report).T.round(3)
        st.dataframe(report_df, use_container_width=True)

st.markdown("---")
st.caption("**DataMed Liver Disease Analytics** | Dual-Engine ML Architecture | Streamlit + Plotly + Scikit-learn + SHAP")
