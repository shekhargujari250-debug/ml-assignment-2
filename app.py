"""
Student Outcome Predictor: Streamlit front-end for ML Assignment 2.

Loads the five pre-trained pipelines from ./model and evaluates them on a
user-uploaded test CSV. No training happens here.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import classification_report, confusion_matrix

from metrics_utils import compute_metrics

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
SAMPLE_CSV = ROOT / "test_data.csv"

st.set_page_config(page_title="Student Outcome Predictor", page_icon="🎓", layout="wide")


@st.cache_resource
def load_artifacts():
    meta = json.loads((MODEL_DIR / "metadata.json").read_text())
    models = {name: joblib.load(MODEL_DIR / fname) for name, fname in meta["model_files"].items()}
    encoder = joblib.load(MODEL_DIR / "label_encoder.joblib")
    return meta, models, encoder


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().replace("\t", "") for c in df.columns]
    return df


def read_upload(file) -> pd.DataFrame:
    # Accept both the comma-separated test file and the raw ';'-separated UCI file
    df = pd.read_csv(file, sep=None, engine="python")
    return clean_columns(df)


def evaluate(model, X, y_true, n_classes):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)
    return y_pred, compute_metrics(y_true, y_pred, y_proba, labels=np.arange(n_classes))


def plot_confusion(cm, classes, title):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    fig.tight_layout()
    return fig


meta, models, encoder = load_artifacts()
features, target, classes = meta["features"], meta["target"], meta["classes"]

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Controls")
    uploaded = st.file_uploader("Upload test CSV", type=["csv"],
                                help=f"Must contain the {len(features)} feature columns and '{target}'.")
    use_sample = st.checkbox("Use bundled test_data.csv", value=uploaded is None)
    model_name = st.selectbox("Choose a model", list(models.keys()))
    st.divider()
    with open(SAMPLE_CSV, "rb") as fh:
        st.download_button("⬇️ Download sample test data", fh, file_name="test_data.csv",
                           mime="text/csv")
    st.caption("Models were trained offline (see train_models.py); this app only runs inference.")

# ---------------- Header ----------------
st.title("🎓 Student Dropout & Academic Success Predictor")
st.markdown(
    "Predicts whether a higher-education student will **Dropout**, stay **Enrolled**, "
    "or **Graduate**, using demographic, socio-economic and 1st/2nd semester academic data "
    "([UCI dataset #697](https://archive.ics.uci.edu/dataset/697))."
)

if uploaded is not None:
    data = read_upload(uploaded)
    source = uploaded.name
elif use_sample:
    data = clean_columns(pd.read_csv(SAMPLE_CSV))
    source = "test_data.csv (bundled)"
else:
    st.info("👈 Upload a test CSV or tick *Use bundled test_data.csv* in the sidebar to begin.")
    st.stop()

missing = [c for c in features + [target] if c not in data.columns]
if missing:
    st.error(f"The uploaded file is missing {len(missing)} required column(s):")
    st.code("\n".join(missing))
    st.stop()

unknown = sorted(set(data[target].astype(str)) - set(classes))
if unknown:
    st.error(f"Unknown target labels {unknown}. Expected one of {classes}.")
    st.stop()

X = data[features]
y_true = encoder.transform(data[target].astype(str))

tab_eval, tab_compare, tab_data = st.tabs(["📊 Model Evaluation", "🏆 Compare All Models", "🗂️ Data Preview"])

# ---------------- Selected model ----------------
with tab_eval:
    st.subheader(f"{model_name} on *{source}* ({len(data)} rows)")
    y_pred, m = evaluate(models[model_name], X, y_true, len(classes))

    cols = st.columns(6)
    for col, (k, v) in zip(cols, m.items()):
        col.metric(k, "n/a" if np.isnan(v) else f"{v:.4f}")
    st.caption("Precision, Recall and F1 are macro-averaged across the 3 classes; AUC is one-vs-rest macro.")

    left, right = st.columns([1, 1.2])
    with left:
        cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(classes)))
        st.pyplot(plot_confusion(cm, classes, f"Confusion Matrix: {model_name}"))
    with right:
        st.markdown("**Classification Report**")
        report = classification_report(y_true, y_pred, labels=np.arange(len(classes)),
                                       target_names=classes, output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

    pred_df = data.copy()
    pred_df["Predicted"] = encoder.inverse_transform(y_pred)
    st.download_button("⬇️ Download predictions", pred_df.to_csv(index=False).encode(),
                       file_name=f"predictions_{model_name.lower().replace(' ', '_')}.csv",
                       mime="text/csv")

# ---------------- All models ----------------
with tab_compare:
    st.subheader("All five models on the same test data")
    rows = {name: evaluate(mdl, X, y_true, len(classes))[1] for name, mdl in models.items()}
    comp = pd.DataFrame(rows).T
    comp.index.name = "ML Model Name"
    st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0, color="#c6efce"),
                 use_container_width=True)

    best = comp["MCC"].idxmax()
    st.success(f"Best model by MCC on this data: **{best}** (MCC = {comp.loc[best, 'MCC']:.4f})")

    metric_pick = st.radio("Metric to chart", list(comp.columns), horizontal=True, index=5)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ordered = comp[metric_pick].sort_values()
    ax.barh(ordered.index, ordered.values, color="#4C72B0")
    for i, v in enumerate(ordered.values):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel(metric_pick)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)

# ---------------- Data preview ----------------
with tab_data:
    st.subheader("Uploaded data")
    st.dataframe(data.head(50), use_container_width=True)
    c1, c2 = st.columns(2)
    c1.metric("Rows", len(data))
    c2.metric("Feature columns", len(features))
    st.markdown("**Target distribution**")
    st.bar_chart(data[target].value_counts())
