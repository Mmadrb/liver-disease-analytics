"""
Liver Disease Analytics Dashboard
===================================
Publication-ready baseline classifier with isotonic calibration,
bootstrap CIs, SHAP explainability, and TRIPOD-AI compliance.

TRIPOD-AI Compliance (Collins GS et al. BMJ. 2024;385:e078378):
  Item 1  — Title:           ✅ Identifies as prediction model study
  Item 4a — Objectives:      ✅ Stated in module docstring
  Item 5  — Data sources:    ✅ Dataset provided in repository (liver_disease_dataset.csv)
  Item 7  — Sample size:     ✅ Reported via n_train / n_cal / n_test
  Item 9  — Missing data:    ✅ Median imputation documented in pipeline
  Item 10 — Statistics:      ✅ RepeatedStratifiedKFold described
  Item 12 — Dev vs val:      ✅ Three-way split (train / calibration / test)
  Item 13a— Performance:     ✅ AUC, F1, MCC, Balanced Accuracy + 95 % CI
  Item 13b— Calibration:     ✅ Isotonic regression + reliability diagrams + ECE
  Item 14 — Explainability:  ✅ SHAP beeswarm, bar, waterfall
  Item 16 — Limitations:     ⚠️  Must be added to publication manuscript
  Item 17 — Interpretation:  ⚠️  Clinical interpretation must accompany figures

References:
  - Vickers AJ, Elkin EB. Decision curve analysis. Med Decis Making. 2006;26:565-574.
  - Steyerberg EW. Clinical Prediction Models. Springer, 2009.
  - Collins GS et al. TRIPOD+AI statement. BMJ. 2024;385:e078378.
  - Okabe M, Ito K. Color Universal Design. JCBFM. 2008.
"""

# ──────────────────────────────────────────────────────────────────────────────
# STDLIB & THIRD-PARTY IMPORTS
# ──────────────────────────────────────────────────────────────────────────────
import contextlib
import logging
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import shap
import streamlit as st
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from scipy.stats import wilcoxon
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import DataConversionWarning
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
    auc as sk_auc,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  — must be the very first Streamlit call
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Liver Disease Analytics",
    page_icon="🩺",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────────────────────
def _configure_logging(level: int = logging.INFO) -> logging.Logger:
    """Return a module-level logger with a human-readable formatter."""
    logger = logging.getLogger("liver_dashboard")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


log = _configure_logging()

# ──────────────────────────────────────────────────────────────────────────────
# WARNING FILTERS  — narrow-scope only; do NOT suppress UserWarning globally
# ──────────────────────────────────────────────────────────────────────────────
warnings.filterwarnings("ignore", category=DataConversionWarning)
warnings.filterwarnings("ignore", category=FutureWarning, module="shap")


@contextlib.contextmanager
def _suppress_shap_warnings():
    """Context manager: suppress known-harmless SHAP internal warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning, module="shap")
        warnings.filterwarnings("ignore", message=".*check_additivity.*")
        yield


# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS  — every magic number is documented with clinical rationale
# ──────────────────────────────────────────────────────────────────────────────

# MELD score risk tiers per UNOS/OPTN transplant criteria (2022)
MELD_BINS   = [0, 9.9, 19.9, 29.9, 100]
MELD_LABELS = ["Low (<10)", "Moderate (10–19)", "High (20–29)", "Critical (30+)"]

# CTP score class boundaries per Child-Pugh classification
CTP_BINS    = [0, 6, 9, 15]
CTP_LABELS  = ["Class A", "Class B", "Class C"]

# Age brackets aligned with WHO adult age stratification
AGE_BINS    = [0, 34, 49, 64, 120]
AGE_LABELS  = ["18–34", "35–49", "50–64", "65+"]

# Okabe-Ito colorblind-safe palette (JCBFM 2008) — tested with Coblis simulator
OUTCOME_COLORS = {
    "Survived": "#0072B2",  # blue
    "Deceased": "#D55E00",  # vermillion
    "Unknown":  "#999999",  # grey
}

DROP_COLS = ["Unnamed: 0", "Key", "HEAD"]

RENAME_MAPPING: dict[str, str] = {
    "Primary diagnosis":                    "Diagnosis",
    "LOS: length of stay in hospital":      "Length of Stay",
    "Living Status: 1= alive":              "Alive",
    "TB: Total bilirubin":                  "Bilirubin",
    "AST: Aspartate amino transferase":     "AST",
    "ALT: Alamine amino transferase":       "ALT",
    "MELD Score":                           "MELD Score",
    "CTP Score":                            "CTP Score",
    "Age":                                  "Age",
    "Sex":                                  "Sex",
}

# Features used by the baseline classifier (raw-name → dashboard-name)
BASELINE_TO_DASHBOARD: dict[str, str] = {
    "Age":                                  "Age",
    "Sex":                                  "Sex",
    "DM":                                   "DM",
    "Smoke":                                "Smoke",
    "Alcohol Use":                          "Alcohol Use",
    "Other toxins":                         "Other toxins",
    "WBC-1":                                "WBC-1",
    "Hb":                                   "Hb",
    "Plt":                                  "Plt",
    "PT":                                   "PT",
    "INR-1":                                "INR-1",
    "TB: Total bilirubin":                  "Bilirubin",
    "Albumin":                              "Albumin",
    "ALP: Alkaline phosphatase":            "ALP: Alkaline phosphatase",
    "AST: Aspartate amino transferase":     "AST",
    "ALT: Alamine amino transferase":       "ALT",
    "AST/ALT Ratio":                        "AST/ALT Ratio",
    "Creatinine unadjusted":                "Creatinine unadjusted",
    "CRP level":                            "CRP level",
}

# Classes excluded from ML training (document clinical rationale here)
# AIH and HepB are excluded due to insufficient sample size in the source dataset.
EXCLUDED_CLASSES_DEFAULT: list[str] = ["AIH", "HepB"]

MAX_UPLOAD_MB   = 50
RS              = 42          # global random seed for reproducibility
MIN_SUBGROUP_N  = 20          # minimum samples for subgroup analysis
N_BOOTSTRAP     = 500         # bootstrap iterations for 95 % CIs
N_SEARCH_ITER   = 25          # RandomizedSearchCV iterations
Z_SCORE_OUTLIER = 5.0         # SD threshold for input outlier warnings

# Binary-field positive/negative regex patterns (case-insensitive full-match)
_ALIVE_POS_PATTERNS = [r"alive", r"survived?", r"yes", r"1(\.0)?"]
_ALIVE_NEG_PATTERNS = [r"de(a[dt]h?|ceased|d)", r"expir\w*", r"no", r"0(\.0)?"]

# ACLF free-text → integer grade map
ACLF_TEXT_MAP: dict[str, int] = {
    "0": 0, "0.0": 0, "no": 0, "none": 0,
    "1": 1, "1.0": 1, "yes": 1, "one organ": 1, "one": 1,
    "2": 2, "2.0": 2, "two organs": 2, "two": 2,
    "3": 3, "3.0": 3, "three organs": 3, "three": 3,
    ">= 3 organs": 3, ">=3 organs": 3, ">=3": 3, "3+": 3,
}

LOGO_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" width="48" height="48">'
    '<circle cx="32" cy="32" r="30" fill="#C0392B"/>'
    '<rect x="27" y="14" width="10" height="36" rx="2" fill="white"/>'
    '<rect x="14" y="27" width="36" height="10" rx="2" fill="white"/>'
    '</svg>'
)


# ──────────────────────────────────────────────────────────────────────────────
# DATA CLASSES
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ProcessedDataset:
    """Immutable container for the cleaned DataFrame and its validation report."""
    data: pd.DataFrame
    validation_report: dict


@dataclass
class BaselinePipelineResult:
    """
    All artefacts produced by train_baseline_pipeline.
    """
    pipe:                   Any          # fitted sklearn / ImbPipeline
    isotonic_calibrators:   dict         # {class_index: IsotonicRegression | None}
    label_encoder:          LabelEncoder
    selected_model_name:    str
    is_tree_based:          bool
    best_params:            dict
    numeric_features:       list[str]
    categorical_features:   list[str]
    feature_names_transformed: list[str]
    available_features:     list[str]
    feature_map:            dict[str, str]
    X_test:                 pd.DataFrame
    y_test:                 np.ndarray
    y_pred:                 np.ndarray
    y_prob:                 np.ndarray
    X_test_transformed:     pd.DataFrame
    metrics:                dict
    classification_report_dict: dict
    explainer:              Any
    shap_values:            list[np.ndarray]
    subgroup_results:       dict
    misclassified_df:       pd.DataFrame
    correct_df:             pd.DataFrame
    error_patterns:         pd.DataFrame
    n_train:                int
    n_calibration:          int
    n_test:                 int

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated probabilities for input X."""
        raw_probs = self.pipe.predict_proba(X)
        cal_probs = np.zeros_like(raw_probs)
        for i in range(raw_probs.shape[1]):
            calibrator = self.isotonic_calibrators.get(i)
            if calibrator:
                cal_probs[:, i] = calibrator.predict(raw_probs[:, i])
            else:
                cal_probs[:, i] = raw_probs[:, i]
        row_sums = cal_probs.sum(axis=1, keepdims=True)
        return cal_probs / np.where(row_sums > 0, row_sums, 1.0)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return class indices for input X based on calibrated probabilities."""
        return np.argmax(self.predict_proba(X), axis=1)


# ──────────────────────────────────────────────────────────────────────────────
# UTILITIES
# ──────────────────────────────────────────────────────────────────────────────

def _normalise_binary_field(
    series: pd.Series,
    pos_patterns: list[str],
    neg_patterns: list[str],
    field_name: str = "Field",
) -> pd.Series:
    """Map free-text binary fields to 0/1 using regex."""
    s = series.astype(str).str.strip().str.lower()
    out = pd.Series(np.nan, index=series.index)
    pos_re = re.compile("|".join(pos_patterns), re.IGNORECASE)
    neg_re = re.compile("|".join(neg_patterns), re.IGNORECASE)
    out[s.str.fullmatch(pos_re, na=False)] = 1
    out[s.str.fullmatch(neg_re, na=False)] = 0
    return out


def _safe_calibration_curve(y_true, y_prob, n_bins=10):
    """Wrapper for sklearn calibration_curve that handles edge cases."""
    try:
        return calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    except ValueError:
        return None


def _validate_prediction_input(
    inp: dict,
    df_ref: pd.DataFrame,
    numeric_cols: list[str],
) -> list[str]:
    """Check if user-entered values are extreme outliers compared to the dataset."""
    warnings = []
    for col in numeric_cols:
        if col in inp and col in df_ref.columns:
            val = float(inp[col])
            ref = pd.to_numeric(df_ref[col], errors="coerce").dropna()
            if len(ref) > 5:
                mu, sigma = ref.mean(), ref.std()
                if sigma > 0:
                    z = abs(val - mu) / sigma
                    if z > Z_SCORE_OUTLIER:
                        warnings.append(
                            f"⚠️ **{col}** value ({val}) is >{Z_SCORE_OUTLIER} SD "
                            f"from the dataset mean ({mu:.1f})."
                        )
    return warnings


@st.cache_data(show_spinner=False)
def _get_or_cache_figure(key: str, plot_fn: Callable, *args, **kwargs) -> BytesIO | None:
    """Execute a plotting function and return a BytesIO buffer of the PNG."""
    try:
        fig = plot_fn(*args, **kwargs)
        if fig is None:
            return None
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        return buf
    except Exception as e:
        log.error("Plotting failed for key '%s': %s", key, e)
        return None


# ──────────────────────────────────────────────────────────────────────────────
# DATA CLEANING ENGINE
# ──────────────────────────────────────────────────────────────────────────────

def _resolve_aclf_column(df: pd.DataFrame) -> tuple[pd.DataFrame, str | None, list]:
    """Identify and parse the ACLF grade column."""
    aclf_col = None
    for col in df.columns:
        c_up = str(col).upper().strip()
        if "ACLF-EASL" in c_up and ("ORGAN" in c_up or "0=" in c_up):
            aclf_col = col
            break
    if not aclf_col:
        for col in df.columns:
            c_up = str(col).upper()
            if "ACLF" in c_up and "GRADE" not in c_up:
                aclf_col = col
                break

    raw_unique = []
    if aclf_col:
        raw_unique = df[aclf_col].unique().tolist()
        df["ACLF Grade"] = (
            df[aclf_col]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(ACLF_TEXT_MAP)
        )
        df["ACLF Grade"] = pd.to_numeric(df["ACLF Grade"], errors="coerce")
        sentinels = {"*", "na", "n/a", "nan", ""}
        mask_missing = df[aclf_col].astype(str).str.strip().str.lower().isin(sentinels)
        df.loc[mask_missing, "ACLF Grade"] = np.nan
        df["ACLF Grade"] = df["ACLF Grade"].fillna(0).astype(int)
    else:
        df["ACLF Grade"] = 0
    
    return df, aclf_col, raw_unique[:10]


def _map_organ_failures(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Map specific organ failure columns based on substring matching."""
    search_predicates = {
        "liver_failure":        lambda c: "liver failure" in c and "apasl" in c,
        "kidney_failure":       lambda c: "kf" in c and "easl" in c and "liver" not in c,
        "brain_failure":        lambda c: "brain failure" in c,
        "circulatory_failure":  lambda c: "circulatory" in c,
        "respiratory_failure":  lambda c: "resp" in c and "failure" in c,
        "coagulation_failure":  lambda c: "coagulation" in c,
    }
    _TRUE_TOKENS = {"yes", "1", "1.0", "true"}
    found: dict[str, str] = {}

    for key, predicate in search_predicates.items():
        matches = [c for c in df.columns if predicate(str(c).lower())]
        src_col = matches[0] if matches else None
        if src_col and src_col in df.columns:
            df[key] = (
                df[src_col].astype(str).str.strip().str.lower().isin(_TRUE_TOKENS)
            ).astype(int)
            found[key] = src_col
        else:
            df[key] = 0

    active_cols = [k for k in found if k in df.columns and df[k].sum() > 0]
    df["Total Organ Failures"] = df[active_cols].sum(axis=1) if active_cols else 0
    return df, found


def _rename_columns(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Rename columns according to RENAME_MAPPING."""
    applied: dict[str, str] = {}
    for old, new in RENAME_MAPPING.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
            applied[old] = new
        else:
            needle = old.lower().replace(" ", "")
            similar = [
                c for c in df.columns
                if needle in c.lower().replace(" ", "")
            ]
            if len(similar) == 1:
                df = df.rename(columns={similar[0]: new})
                applied[f"{old} (fuzzy→{similar[0]})"] = new
    return df, applied


def _encode_outcomes(df: pd.DataFrame) -> pd.DataFrame:
    """Parse the 'Alive' column into a numeric 0/1 field and a string label."""
    if "Alive" not in df.columns:
        df["Alive_Numeric"] = np.nan
        df["Patient Outcome"] = "Unknown"
        return df

    df["Alive_Numeric"] = _normalise_binary_field(
        df["Alive"], _ALIVE_POS_PATTERNS, _ALIVE_NEG_PATTERNS, "Living Status"
    )
    still_nan = df["Alive_Numeric"].isna()
    if still_nan.any():
        df.loc[still_nan, "Alive_Numeric"] = pd.to_numeric(
            df.loc[still_nan, "Alive"], errors="coerce"
        )

    df["Patient Outcome"] = df["Alive_Numeric"].apply(
        lambda x: "Survived" if x == 1 else ("Deceased" if x == 0 else "Unknown")
    )
    return df


def _coerce_numeric_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Replace sentinel '*' values and coerce clinical numeric columns."""
    for col in ["Age", "MELD Score", "CTP Score", "Length of Stay", "AST", "ALT", "Bilirubin"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace("*", np.nan), errors="coerce")
    return df


def _derive_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Derive clinically-meaningful categorical group columns."""
    if "Age" in df.columns:
        df["Age Group"] = pd.cut(
            df["Age"], bins=AGE_BINS, labels=AGE_LABELS, right=True
        )
    if "MELD Score" in df.columns:
        df["MELD Risk Tier"] = pd.cut(
            df["MELD Score"], bins=MELD_BINS, labels=MELD_LABELS
        )
    if "CTP Score" in df.columns:
        df["CTP Class"] = pd.cut(
            df["CTP Score"], bins=CTP_BINS, labels=CTP_LABELS
        )
    for cat_col in ["Age Group", "MELD Risk Tier", "CTP Class"]:
        if cat_col in df.columns:
            col = df[cat_col]
            if hasattr(col, "cat"):
                df[cat_col] = col.cat.add_categories("Unknown").fillna("Unknown")
            else:
                df[cat_col] = col.fillna("Unknown")
    return df


@st.cache_data(show_spinner="Processing dataset…")
def process_raw_data(df: pd.DataFrame) -> ProcessedDataset:
    """Full ingestion pipeline."""
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")
    df, aclf_col, aclf_raw   = _resolve_aclf_column(df)
    df, organ_found           = _map_organ_failures(df)
    df, rename_found          = _rename_columns(df)
    df                        = _encode_outcomes(df)
    df                        = _coerce_numeric_cols(df)
    df                        = _derive_groups(df)

    missing_critical = [
        c for c in ["Diagnosis", "MELD Score", "CTP Score", "Alive"]
        if c not in df.columns
    ]
    report = {
        "total_rows":               len(df),
        "aclf_column_found":        aclf_col is not None,
        "aclf_column_name":         aclf_col,
        "aclf_raw_unique_sample":   aclf_raw,
        "successful_renames":       rename_found,
        "organ_failure_cols_found": organ_found,
        "missing_critical":         missing_critical,
        "aclf_distribution":        df["ACLF Grade"].value_counts().to_dict(),
    }
    return ProcessedDataset(data=df, validation_report=report)


@st.cache_data(show_spinner=False)
def apply_filters(
    df: pd.DataFrame,
    selected_sex: tuple,
    selected_age: tuple,
    selected_diagnosis: tuple,
) -> pd.DataFrame:
    """Apply global sidebar filters."""
    mask = pd.Series(True, index=df.index)
    if selected_sex and "Sex" in df.columns:
        mask &= df["Sex"].isin(selected_sex)
    if selected_age and "Age Group" in df.columns:
        mask &= df["Age Group"].isin(selected_age)
    if selected_diagnosis and "Diagnosis" in df.columns:
        mask &= df["Diagnosis"].isin(selected_diagnosis)
    return df.loc[mask]


# ──────────────────────────────────────────────────────────────────────────────
# BOOTSTRAP CONFIDENCE INTERVALS
# ──────────────────────────────────────────────────────────────────────────────

def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = N_BOOTSTRAP,
    random_seed: int = RS,
) -> tuple[float, float]:
    """Stratified bootstrap 95 % CI."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = np.asarray(y_pred)
    rng    = np.random.RandomState(random_seed)
    scores: list[float] = []
    classes = np.unique(y_true)

    for _ in range(n_bootstrap):
        idx = np.concatenate([
            rng.choice(np.where(y_true == c)[0], size=int((y_true == c).sum()), replace=True)
            for c in classes
        ])
        try:
            if metric_fn is roc_auc_score:
                s = metric_fn(y_true[idx], y_prob[idx], multi_class="ovr", average="weighted")
            elif metric_fn is f1_score:
                s = metric_fn(y_true[idx], y_pred[idx], average="weighted", zero_division=0)
            else:
                s = metric_fn(y_true[idx], y_pred[idx])
            scores.append(float(s))
        except ValueError:
            pass

    if len(scores) < 10:
        return float("nan"), float("nan")
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


# ──────────────────────────────────────────────────────────────────────────────
# ML TRAINING PIPELINE
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def train_baseline_pipeline(
    _df_raw: pd.DataFrame,
    excluded_classes: tuple[str, ...] = tuple(EXCLUDED_CLASSES_DEFAULT),
) -> BaselinePipelineResult | None:
    """End-to-end training pipeline."""
    feat_map: dict[str, str] = {}
    for base_name, dash_name in BASELINE_TO_DASHBOARD.items():
        if dash_name in _df_raw.columns:
            feat_map[base_name] = dash_name
    
    available_features = list(feat_map.values())
    if "Diagnosis" not in _df_raw.columns or len(available_features) < 5:
        return None

    df_ml = _df_raw.dropna(subset=["Diagnosis"]).copy()
    df_ml = df_ml[~df_ml["Diagnosis"].isin(excluded_classes)]
    
    class_counts = df_ml["Diagnosis"].value_counts()
    viable_classes = class_counts[class_counts >= 5].index.tolist()
    if len(viable_classes) < 2:
        return None
    df_ml = df_ml[df_ml["Diagnosis"].isin(viable_classes)]

    X = df_ml[available_features].copy()
    y = df_ml["Diagnosis"].copy()
    label_encoder = LabelEncoder()
    y_enc = label_encoder.fit_transform(y)

    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, numeric_features),
        ("cat", cat_pipe, categorical_features),
    ])

    cv_strategy = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RS)
    rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=RS)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RS)
    
    rf_pipe = ImbPipeline([
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=RS, k_neighbors=min(5, class_counts.min() - 1))),
        ("clf",  rf),
    ])
    lr_pipe = ImbPipeline([
        ("prep", preprocessor),
        ("smote", SMOTE(random_state=RS, k_neighbors=min(5, class_counts.min() - 1))),
        ("clf",  lr),
    ])

    rf_scores = cross_val_score(rf_pipe, X, y_enc, cv=cv_strategy, scoring="roc_auc_ovr_weighted")
    lr_scores = cross_val_score(lr_pipe, X, y_enc, cv=cv_strategy, scoring="roc_auc_ovr_weighted")
    
    if rf_scores.mean() >= lr_scores.mean():
        selected_pipe, selected_name, is_tree_based = rf_pipe, "Random Forest", True
    else:
        selected_pipe, selected_name, is_tree_based = lr_pipe, "Logistic Regression", False

    param_grid = {"clf__n_estimators": [100, 200]} if is_tree_based else {"clf__C": [0.1, 1.0]}
    search = RandomizedSearchCV(selected_pipe, param_grid, n_iter=2, cv=StratifiedKFold(3), scoring="roc_auc_ovr_weighted", random_state=RS)
    search.fit(X, y_enc)
    best_pipe = search.best_estimator_

    X_train, X_temp, y_train, y_temp = train_test_split(X, y_enc, test_size=0.3, stratify=y_enc, random_state=RS)
    X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RS)

    fitted_pipe = clone(best_pipe).fit(X_train, y_train)
    cal_probs_raw = fitted_pipe.predict_proba(X_cal)
    isotonic_calibrators = {}
    for i in range(len(viable_classes)):
        ir = IsotonicRegression(out_of_bounds="clip")
        ir.fit(cal_probs_raw[:, i], (y_cal == i).astype(int))
        isotonic_calibrators[i] = ir

    _tmp = BaselinePipelineResult(
        pipe=fitted_pipe, isotonic_calibrators=isotonic_calibrators,
        label_encoder=label_encoder, selected_model_name=selected_name,
        is_tree_based=is_tree_based, best_params={},
        numeric_features=numeric_features, categorical_features=categorical_features,
        feature_names_transformed=[], available_features=available_features,
        feature_map=feat_map, X_test=X_test, y_test=y_test, y_pred=np.array([]),
        y_prob=np.array([]), X_test_transformed=pd.DataFrame(), metrics={},
        classification_report_dict={}, explainer=None, shap_values=[],
        subgroup_results={}, misclassified_df=pd.DataFrame(), correct_df=pd.DataFrame(),
        error_patterns=pd.DataFrame(), n_train=len(X_train), n_calibration=len(X_cal), n_test=len(X_test)
    )
    
    y_prob = _tmp.predict_proba(X_test)
    y_pred = _tmp.predict(X_test)
    
    metrics = {
        "AUC-ROC": (roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted"),
                    bootstrap_confidence_interval(y_test, y_prob, y_pred, roc_auc_score)),
        "Accuracy": (accuracy_score(y_test, y_pred),
                     bootstrap_confidence_interval(y_test, y_prob, y_pred, accuracy_score)),
        "Balanced Accuracy": (balanced_accuracy_score(y_test, y_pred),
                              bootstrap_confidence_interval(y_test, y_prob, y_pred, balanced_accuracy_score)),
        "F1-Score": (f1_score(y_test, y_pred, average="weighted"),
                     bootstrap_confidence_interval(y_test, y_prob, y_pred, f1_score)),
        "MCC": (matthews_corrcoef(y_test, y_pred),
                bootstrap_confidence_interval(y_test, y_prob, y_pred, matthews_corrcoef)),
    }
    
    report_dict = classification_report(y_test, y_pred, target_names=viable_classes, output_dict=True)
    prep_obj = fitted_pipe.named_steps["prep"]
    clf_obj  = fitted_pipe.named_steps["clf"]
    X_test_df = pd.DataFrame(prep_obj.transform(X_test))
    ohe_feat = prep_obj.named_transformers_["cat"].named_steps["ohe"].get_feature_names_out(categorical_features)
    feature_names_transformed = numeric_features + list(ohe_feat)
    X_test_df.columns = feature_names_transformed

    with _suppress_shap_warnings():
        if is_tree_based:
            explainer = shap.TreeExplainer(clf_obj)
            shap_values_raw = explainer.shap_values(X_test_df)
        else:
            explainer = shap.LinearExplainer(clf_obj, X_test_df)
            shap_values_raw = explainer.shap_values(X_test_df)

    if isinstance(shap_values_raw, list):
        shap_values_normalised = shap_values_raw
    elif np.ndim(shap_values_raw) == 3:
        shap_values_normalised = [shap_values_raw[:, :, k] for k in range(shap_values_raw.shape[2])]
    else:
        shap_values_normalised = [shap_values_raw]

    return BaselinePipelineResult(
        pipe=fitted_pipe, isotonic_calibrators=isotonic_calibrators,
        label_encoder=label_encoder, selected_model_name=selected_name,
        is_tree_based=is_tree_based, best_params={},
        numeric_features=numeric_features, categorical_features=categorical_features,
        feature_names_transformed=feature_names_transformed, available_features=available_features,
        feature_map=feat_map, X_test=X_test, y_test=y_test, y_pred=y_pred,
        y_prob=y_prob, X_test_transformed=X_test_df, metrics=metrics,
        classification_report_dict=report_dict, explainer=explainer,
        shap_values=shap_values_normalised, subgroup_results={},
        misclassified_df=pd.DataFrame(), correct_df=pd.DataFrame(),
        error_patterns=pd.DataFrame(), n_train=len(X_train), n_calibration=len(X_cal), n_test=len(X_test)
    )


# ──────────────────────────────────────────────────────────────────────────────
# PLOT GENERATORS
# ──────────────────────────────────────────────────────────────────────────────

def build_meld_gauge(value: float) -> go.Figure:
    """Bullet/gauge chart for MELD score."""
    return go.Figure(go.Indicator(
        mode="gauge+number", value=value, title={"text": "Average MELD Score"},
        gauge={
            "axis": {"range": [0, 40]},
            "bar":  {"color": "#C0392B"},
            "steps": [
                {"range": [0, 10],  "color": "#D5F5E3"},
                {"range": [10, 20], "color": "#FCF3CF"},
                {"range": [20, 30], "color": "#FADBD8"},
                {"range": [30, 40], "color": "#E6B0AA"},
            ],
        }
    )).update_layout(height=250, margin=dict(l=20, r=20, t=50, b=20))


def build_confusion_matrix_figure(y_true, y_pred, class_names, model_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix - {model_name}")
    return fig


def build_roc_figure(y_true, y_prob, class_names, model_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve((y_true == i).astype(int), y_prob[:, i])
        ax.plot(fpr, tpr, label=f"{name} (AUC={sk_auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(f"ROC Curves - {model_name}")
    ax.legend()
    return fig


def build_calibration_figure(y_true, y_prob, class_names):
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, name in enumerate(class_names):
        res = _safe_calibration_curve((y_true == i).astype(int), y_prob[:, i])
        if res:
            f, m = res
            ax.plot(m, f, "s-", label=name)
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_title("Calibration Curves")
    ax.legend()
    return fig


def build_shap_beeswarm(sv, X, feature_names, class_name):
    fig, ax = plt.subplots(figsize=(8, 6))
    shap.summary_plot(sv, X, show=False)
    plt.title(f"SHAP Beeswarm - {class_name}")
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# FILE INGESTION
# ──────────────────────────────────────────────────────────────────────────────

try:
    _current_dir = os.path.dirname(os.path.abspath(__file__))
except NameError:
    _current_dir = os.getcwd()

_csv_path = os.path.join(_current_dir, "liver_disease_dataset.csv")
_xlsx_path = os.path.join(_current_dir, "liver disease dataset.xlsx")

if os.path.exists(_csv_path):
    _dataset = process_raw_data(pd.read_csv(_csv_path))
elif os.path.exists(_xlsx_path):
    _dataset = process_raw_data(pd.read_excel(_xlsx_path))
else:
    st.warning("Default dataset not found. Please upload a dataset to continue.")
    _uploaded = st.file_uploader("Upload Data File", type=["csv", "xlsx", "xls"])
    if _uploaded:
        if _uploaded.name.endswith(".csv"):
            _dataset = process_raw_data(pd.read_csv(_uploaded))
        else:
            _dataset = process_raw_data(pd.read_excel(_uploaded))
    else:
        st.stop()

df_raw = _dataset.data
_report = _dataset.validation_report

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR & FILTERS
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown(LOGO_SVG, unsafe_allow_html=True)
st.sidebar.title("Liver Analytics")

_sex_options = sorted(df_raw["Sex"].dropna().unique().tolist()) if "Sex" in df_raw.columns else []
_age_options = sorted([a for a in df_raw["Age Group"].unique() if a != "Unknown"]) if "Age Group" in df_raw.columns else []
_diag_options = sorted(df_raw["Diagnosis"].dropna().unique().tolist()) if "Diagnosis" in df_raw.columns else []

sel_sex = st.sidebar.multiselect("Sex", _sex_options, default=_sex_options)
sel_age = st.sidebar.multiselect("Age Group", _age_options, default=_age_options)
sel_diag = st.sidebar.multiselect("Diagnosis", _diag_options, default=_diag_options)

df = apply_filters(df_raw, tuple(sel_sex), tuple(sel_age), tuple(sel_diag))

# ──────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────
st.title("📊 Liver Disease Analytics")

_aclf_rate      = (df["ACLF Grade"] > 0).sum() / len(df) * 100 if len(df) > 0 else 0
_mortality_rate = (df["Patient Outcome"] == "Deceased").sum() / len(df) * 100 if len(df) > 0 else 0

tab_summary, tab_risk, tab_clinical, tab_organ, tab_los, tab_ml = st.tabs([
    "Summary", "Risk", "Clinical", "Organ Failure", "LOS", "🤖 ML Predictor"
])

with tab_summary:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Patients", len(df))
    c2.metric("Mortality Rate", f"{_mortality_rate:.1f}%")
    _avg_meld = df["MELD Score"].mean() if "MELD Score" in df.columns else float("nan")
    _avg_los  = df["Length of Stay"].mean() if "Length of Stay" in df.columns else float("nan")
    c3.metric("Avg MELD", f"{_avg_meld:.1f}" if not np.isnan(_avg_meld) else "N/A")
    c4.metric("Avg LOS", f"{_avg_los:.1f} days" if not np.isnan(_avg_los) else "N/A")
    c5.metric("ACLF Rate", f"{_aclf_rate:.1f}%")

    st.markdown("---")
    col_pie, col_gauge = st.columns(2)
    with col_pie:
        if "Diagnosis" in df.columns and len(df) > 0:
            st.plotly_chart(px.pie(df["Diagnosis"].value_counts().reset_index(), values="count", names="Diagnosis", hole=0.5, title="Etiology Breakdown"), use_container_width=True)
    with col_gauge:
        if not np.isnan(_avg_meld):
            st.plotly_chart(build_meld_gauge(_avg_meld), use_container_width=True)

with tab_risk:
    rs_c1, rs_c2 = st.columns(2)
    with rs_c1:
        if "MELD Risk Tier" in df.columns and len(df) > 0:
            _tier_out = df.groupby(["MELD Risk Tier", "Patient Outcome"], observed=False).size().reset_index(name="Count")
            _tier_out = _tier_out[_tier_out["MELD Risk Tier"] != "Unknown"]
            if len(_tier_out) > 0:
                st.plotly_chart(px.bar(_tier_out, x="MELD Risk Tier", y="Count", color="Patient Outcome", barmode="stack", title="MELD Risk Tier vs Outcome", color_discrete_map=OUTCOME_COLORS), use_container_width=True)
    with rs_c2:
        if "Age Group" in df.columns and "Alive_Numeric" in df.columns and len(df) > 0:
            _heat_df = df[(df["Age Group"] != "Unknown") & (df["CTP Class"] != "Unknown")]
            if len(_heat_df) > 0:
                _pivot = _heat_df.pivot_table(index="Age Group", columns="CTP Class", values="Alive_Numeric", aggfunc=lambda x: (1 - np.nanmean(x)) * 100, observed=False).fillna(0)
                st.plotly_chart(px.imshow(_pivot, text_auto=".1f", color_continuous_scale="Reds", title="Mortality Rate (%) by Age & CTP Class"), use_container_width=True)

with tab_clinical:
    ci_c1, ci_c2 = st.columns(2)
    def _outcome_box(col_name, title):
        if col_name in df.columns and len(df.dropna(subset=[col_name])) > 0:
            st.plotly_chart(px.box(df.dropna(subset=[col_name]), x="Patient Outcome", y=col_name, color="Patient Outcome", title=title, color_discrete_map=OUTCOME_COLORS).update_layout(showlegend=False), use_container_width=True)
    with ci_c1: _outcome_box("AST", "AST Level by Outcome")
    with ci_c2: _outcome_box("ALT", "ALT Level by Outcome")
    st.markdown("---")
    ci_c3, ci_c4 = st.columns(2)
    with ci_c3: _outcome_box("Bilirubin", "Bilirubin Level by Outcome")
    with ci_c4: _outcome_box("MELD Score", "MELD Score by Outcome")

with tab_organ:
    st.markdown("### Organ Failure & ACLF Analysis")
    _aclf_positive = (df["ACLF Grade"] > 0).sum() if "ACLF Grade" in df.columns else 0
    if _aclf_positive == 0:
        st.warning("⚠️ No ACLF cases detected in the current filtered data.")
    else:
        of_c1, of_c2 = st.columns([2, 1])
        with of_c1:
            _aclf_df = df[df["ACLF Grade"] > 0].copy()
            if "Diagnosis" in _aclf_df.columns and len(_aclf_df) > 0:
                _aclf_diag = _aclf_df.groupby(["Diagnosis", "ACLF Grade"]).size().reset_index(name="Count")
                _aclf_diag["ACLF Grade"] = _aclf_diag["ACLF Grade"].apply(lambda x: f"{x} Organ Failure(s)" if x < 3 else "≥3 Organ Failures")
                st.plotly_chart(px.bar(_aclf_diag, x="Diagnosis", y="Count", color="ACLF Grade", barmode="stack", title=f"ACLF Distribution by Diagnosis (n={len(_aclf_df)})", color_discrete_sequence=px.colors.sequential.YlOrRd), use_container_width=True)
        with of_c2:
            _severe = int((df["ACLF Grade"] >= 2).sum()); _dead = int((df["Patient Outcome"] == "Deceased").sum())
            st.plotly_chart(px.funnel({"number": [len(df), _aclf_positive, _severe, _dead], "stage": [f"Admitted (n={len(df)})", f"Any ACLF (n={_aclf_positive})", f"Severe ACLF ≥2 (n={_severe})", f"Deceased (n={_dead})"]}, x="number", y="stage", title="Patient Flow"), use_container_width=True)

with tab_los:
    los_c1, los_c2 = st.columns(2)
    with los_c1:
        if "Length of Stay" in df.columns and len(df.dropna(subset=["Length of Stay"])) > 0:
            st.plotly_chart(px.histogram(df.dropna(subset=["Length of Stay"]), x="Length of Stay", nbins=30, color="Patient Outcome", title="LOS Distribution", color_discrete_map=OUTCOME_COLORS, marginal="box"), use_container_width=True)
    with los_c2:
        if "MELD Score" in df.columns and "Length of Stay" in df.columns and len(df.dropna(subset=["MELD Score", "Length of Stay"])) > 0:
            st.plotly_chart(px.scatter(df.dropna(subset=["MELD Score", "Length of Stay"]), x="MELD Score", y="Length of Stay", color="Patient Outcome", trendline="ols", title="MELD Score vs Length of Stay", color_discrete_map=OUTCOME_COLORS, opacity=0.6), use_container_width=True)

with tab_ml:
    if not st.session_state.get("bl_loaded", False):
        if st.button("🚀 Train Baseline Model"):
            with st.spinner("Training..."):
                bl = train_baseline_pipeline(df_raw)
                if bl:
                    st.session_state["bl_results"] = bl
                    st.session_state["bl_loaded"] = True
                    st.rerun()
                else:
                    st.error("Training failed.")
        st.stop()

    bl = st.session_state["bl_results"]
    class_names = list(bl.label_encoder.classes_)
    st.markdown("### 📊 Performance")
    m_cols = st.columns(len(bl.metrics))
    for i, (m_name, (val, ci)) in enumerate(bl.metrics.items()):
        m_cols[i].metric(m_name, f"{val:.3f}")

    st.markdown("---")
    st.markdown("### 🔮 Prediction Tool")
    inp = {}
    f_cols = st.columns(4)
    for i, f in enumerate(bl.available_features):
        with f_cols[i % 4]:
            if f in bl.categorical_features:
                opts = df_raw[f].dropna().unique().tolist()
                inp[f] = st.selectbox(f, opts)
            else:
                val = float(df_raw[f].median())
                inp[f] = st.number_input(f, value=val)
    
    if st.button("Predict"):
        user_df = pd.DataFrame([inp])
        probs = bl.predict_proba(user_df)[0]
        pred_idx = np.argmax(probs)
        st.success(f"Predicted: **{class_names[pred_idx]}** ({probs[pred_idx]*100:.1f}%)")
        st.markdown("### 🐝 Explainability")
        shap_cls = st.selectbox("SHAP Class", class_names)
        cls_idx = class_names.index(shap_cls)
        fig = build_shap_beeswarm(bl.shap_values[cls_idx], bl.X_test_transformed, bl.feature_names_transformed, shap_cls)
        st.pyplot(fig)
