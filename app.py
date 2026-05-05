"""
app.py — Streamlit dashboard for the AI-Powered Fraud Risk Detection Platform.

Run with:  streamlit run app.py

Pages:
  1. Overview       — KPIs, fraud distribution, risk breakdown
  2. Predictions    — Filterable risk-scored transaction table
  3. Model Perf.    — Confusion matrix, ROC, PR curves, classification report
  4. Explainability — SHAP feature importance + plain-English explanations
  5. PII Scanner    — Interactive PII detection & masking
  6. Downloads      — Export predictions.csv and privacy_report.txt
"""

import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.preprocess import run_preprocessing
from src.model import train_model, load_model, predict, evaluate_model, MODEL_PATH
from src.anomaly import run_anomaly_detection
from src.risk import build_risk_dataframe, save_predictions
from src.pii import detect_pii, mask_pii, generate_privacy_report
from src.explainability import compute_shap_values, get_top_features, explain_top_features

OUTPUTS = os.path.join(ROOT, "outputs")
PRED_PATH = os.path.join(OUTPUTS, "predictions.csv")
REPORT_PATH = os.path.join(OUTPUTS, "privacy_report.txt")

# ---------------------------------------------------------------------------
# Streamlit config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Risk Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Dark-themed custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* Global light theme */
.stApp { background: #f8fafc; font-family: 'Inter', sans-serif; color: #0f172a; }

/* KPI cards */
.kpi-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 16px; padding: 24px;
    text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    transition: transform 0.2s; margin-bottom: 8px;
}
.kpi-card:hover { transform: translateY(-4px); box-shadow: 0 10px 15px rgba(0,0,0,0.1); }
.kpi-value { font-size: 2.2rem; font-weight: 700; color: #1e293b; }
.kpi-label { font-size: 0.85rem; color: #64748b; text-transform: uppercase;
             letter-spacing: 1px; margin-top: 4px; font-weight: 600; }

/* Risk badges */
.badge-safe { background: #d1fae5; color: #065f46; padding: 3px 10px; border-radius: 8px; font-weight: 600; }
.badge-atrisk { background: #fef3c7; color: #92400e; padding: 3px 10px; border-radius: 8px; font-weight: 600; }
.badge-critical { background: #fee2e2; color: #991b1b; padding: 3px 10px; border-radius: 8px; font-weight: 600; }

/* Section headers */
.section-header {
    font-size: 1.3rem; font-weight: 600; color: #0f172a;
    border-bottom: 2px solid #cbd5e1; padding-bottom: 8px; margin: 24px 0 16px 0;
}

/* Sidebar */
section[data-testid="stSidebar"] { background: #f1f5f9 !important; border-right: 1px solid #e2e8f0; }
section[data-testid="stSidebar"] .stRadio label { color: #334155 !important; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached pipeline
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading & preprocessing data …")
def load_pipeline():
    """Run full pipeline: preprocess → train → anomaly → risk score."""
    df_eng, X_train, X_test, y_train, y_test = run_preprocessing(
        os.path.join(ROOT, "data", "creditcard.csv")
    )

    # Train or load model
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
    else:
        model = train_model(X_train, y_train)

    # Predict on test set
    y_pred, y_proba = predict(model, X_test)
    metrics = evaluate_model(y_test, y_pred, y_proba)

    # Anomaly detection on test set
    _, anomaly_flags = run_anomaly_detection(X_test)

    # Risk scoring
    risk_df = build_risk_dataframe(
        amounts=X_test["Amount"],
        fraud_proba=y_proba,
        anomaly_flags=anomaly_flags,
        true_labels=y_test,
    )

    # Save predictions
    save_predictions(risk_df, PRED_PATH)

    return model, X_test, y_test, y_pred, y_proba, metrics, risk_df


@st.cache_data(show_spinner="Computing SHAP values …")
def load_shap(_model, _X):
    """Compute SHAP values (cached)."""
    X_sample = _X.sample(n=min(2000, len(_X)), random_state=42)
    _, shap_vals = compute_shap_values(_model, X_sample)
    top_feats = get_top_features(shap_vals, list(X_sample.columns), top_n=10)
    explanation = explain_top_features(top_feats)
    return shap_vals, X_sample, top_feats, explanation


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
model, X_test, y_test, y_pred, y_proba, metrics, risk_df = load_pipeline()


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.markdown("## 🛡️ Fraud Risk Platform")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigate",
    ["📊 Overview", "📋 Predictions", "📈 Model Performance",
     "🔍 Explainability", "🔐 PII Scanner"],
    label_visibility="collapsed",
)


# ===================================================================
# PAGE 1 — OVERVIEW
# ===================================================================
if page == "📊 Overview":
    st.markdown("# 📊 Dashboard Overview")

    total = len(risk_df)
    fraud_count = int((risk_df["true_label"] == 1).sum()) if "true_label" in risk_df.columns else 0
    fraud_pct = fraud_count / total * 100 if total > 0 else 0
    anomaly_count = int(risk_df["anomaly_flag"].sum())

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label, color in [
        (c1, f"{total:,}", "Total Transactions", "#60a5fa"),
        (c2, f"{fraud_count:,}", "Fraud Detected", "#f87171"),
        (c3, f"{fraud_pct:.3f}%", "Fraud Rate", "#fbbf24"),
        (c4, f"{anomaly_count:,}", "Anomalies Flagged", "#a78bfa"),
    ]:
        col.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-value" style="color:{color}">{val}</div>
            <div class="kpi-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Fraud vs Normal Distribution</div>', unsafe_allow_html=True)
        labels_pie = ["Normal", "Fraud"]
        values_pie = [total - fraud_count, fraud_count]
        fig_pie = px.pie(
            names=labels_pie, values=values_pie,
            color_discrete_sequence=["#60a5fa", "#f87171"],
            hole=0.45,
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#1e293b", legend=dict(font=dict(color="#1e293b")),
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.info("💡 **Understanding this chart:** Credit card datasets are highly imbalanced. The vast majority of transactions are normal, making the tiny fraction of fraudulent transactions difficult to detect. Our AI models are specifically weighted to catch this rare minority without aggressively blocking normal users.")

    with col_b:
        st.markdown('<div class="section-header">Risk Distribution</div>', unsafe_allow_html=True)
        risk_counts = risk_df["risk_label"].value_counts().reindex(["SAFE", "AT RISK", "CRITICAL"], fill_value=0)
        fig_bar = px.bar(
            x=risk_counts.index, y=risk_counts.values,
            color=risk_counts.index,
            color_discrete_map={"SAFE": "#10b981", "AT RISK": "#f59e0b", "CRITICAL": "#ef4444"},
            labels={"x": "Risk Level", "y": "Count"},
        )
        fig_bar.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#1e293b", showlegend=False,
            xaxis=dict(gridcolor="#e2e8f0"),
            yaxis=dict(gridcolor="#e2e8f0"),
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
        st.info("💡 **Understanding this chart:** Every transaction is scored from 0–100. Transactions in the 'CRITICAL' category are flagged for immediate manual review or outright rejection, while 'AT RISK' transactions might require additional authentication (like a 2FA prompt).")


# ===================================================================
# PAGE 2 — PREDICTIONS TABLE
# ===================================================================
elif page == "📋 Predictions":
    st.markdown("# 📋 Predictions Table")

    filter_label = st.selectbox("Filter by Risk Label", ["ALL", "SAFE", "AT RISK", "CRITICAL"])
    display_df = risk_df if filter_label == "ALL" else risk_df[risk_df["risk_label"] == filter_label]

    def style_row(row):
        color_map = {"SAFE": "#d1fae5", "AT RISK": "#fef3c7", "CRITICAL": "#fee2e2"}
        bg = color_map.get(row["risk_label"], "")
        text_color = "#0f172a"
        return [f"background-color: {bg}; color: {text_color}"] * len(row)

    st.dataframe(
        display_df.head(500).style.apply(style_row, axis=1).format({"risk_score": "{:.2f}", "amount": "{:.4f}"}),
        use_container_width=True, height=600,
    )
    st.caption(f"Showing {min(500, len(display_df)):,} of {len(display_df):,} rows")


# ===================================================================
# PAGE 3 — MODEL PERFORMANCE
# ===================================================================
elif page == "📈 Model Performance":
    st.markdown("# 📈 Model Performance")

    col1, col2 = st.columns(2)

    # Confusion Matrix
    with col1:
        st.markdown('<div class="section-header">Confusion Matrix</div>', unsafe_allow_html=True)
        cm = metrics["confusion_matrix"]
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual"),
            x=["Normal", "Fraud"], y=["Normal", "Fraud"],
        )
        fig_cm.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#1e293b", margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        st.info("💡 **Confusion Matrix:** Shows accurate predictions vs errors. Look at the bottom-left quadrant (False Negatives) — these are the fraudulent transactions the AI missed. The top-right (False Positives) are normal transactions falsely flagged as fraud.")

    # ROC Curve
    with col2:
        st.markdown('<div class="section-header">ROC Curve</div>', unsafe_allow_html=True)
        fpr = metrics["roc_curve"]["fpr"]
        tpr = metrics["roc_curve"]["tpr"]
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={metrics['roc_auc']:.4f}",
                                     line=dict(color="#3b82f6", width=2)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random",
                                     line=dict(color="#94a3b8", dash="dash")))
        fig_roc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#1e293b", xaxis_title="False Positive Rate (FPR)", yaxis_title="True Positive Rate (TPR)",
            xaxis=dict(gridcolor="#e2e8f0"),
            yaxis=dict(gridcolor="#e2e8f0"),
            legend=dict(font=dict(color="#1e293b")),
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_roc, use_container_width=True)
        st.info("💡 **ROC Curve:** This curve plots the True Positive Rate against the False Positive Rate. An AUC (Area Under Curve) closer to 1.0 means the model is excellent at separating normal from fraudulent transactions.")

    col3, col4 = st.columns(2)

    # PR Curve
    with col3:
        st.markdown('<div class="section-header">Precision–Recall Curve</div>', unsafe_allow_html=True)
        prec = metrics["pr_curve"]["precision"]
        rec = metrics["pr_curve"]["recall"]
        fig_pr = go.Figure()
        fig_pr.add_trace(go.Scatter(x=rec, y=prec, mode="lines",
                                    name=f"PR-AUC={metrics['pr_auc']:.4f}",
                                    line=dict(color="#a78bfa", width=2)))
        fig_pr.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#1e293b", xaxis_title="Recall", yaxis_title="Precision",
            xaxis=dict(gridcolor="#e2e8f0"),
            yaxis=dict(gridcolor="#e2e8f0"),
            legend=dict(font=dict(color="#1e293b")),
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_pr, use_container_width=True)
        st.info("💡 **PR Curve:** Because fraud is so rare, the Precision-Recall curve is often more informative than ROC. A higher PR-AUC means when the model flags fraud, it is usually correct (Precision), and it catches most of the fraud that exists (Recall).")

    # Classification Report
    with col4:
        st.markdown('<div class="section-header">Classification Report</div>', unsafe_allow_html=True)
        cr = metrics["classification_report"]
        cr_df = pd.DataFrame(cr).T
        cr_df = cr_df.round(4)
        st.dataframe(cr_df.style.format("{:.4f}"), use_container_width=True)
        st.info("💡 **Classification Report:** Precision tells you 'out of all predicted frauds, how many were actually fraud?' Recall tells you 'out of all actual frauds, how many did we find?'.")

    st.markdown("---")
    mc1, mc2 = st.columns(2)
    mc1.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
    mc2.metric("PR-AUC", f"{metrics['pr_auc']:.4f}")


# ===================================================================
# PAGE 4 — EXPLAINABILITY
# ===================================================================
elif page == "🔍 Explainability":
    st.markdown("# 🔍 Model Explainability (SHAP)")

    with st.spinner("Computing SHAP values (this may take a minute) …"):
        shap_vals, X_sample, top_feats, explanation = load_shap(model, X_test)

    st.markdown('<div class="section-header">Top 10 Feature Importance</div>', unsafe_allow_html=True)
    fig_shap = px.bar(
        top_feats, x="importance", y="feature", orientation="h",
        color="importance", color_continuous_scale="Viridis",
    )
    fig_shap.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#1e293b", yaxis=dict(autorange="reversed"),
        xaxis=dict(gridcolor="#e2e8f0", title="Mean |SHAP Value| (Impact on Model Output)"),
        yaxis_title="Feature Name", coloraxis_showscale=False,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_shap, use_container_width=True)
    st.info("💡 **Understanding Feature Importance:** Machine Learning models can sometimes be 'black boxes'. SHAP (SHapley Additive exPlanations) values break down exactly *why* the model made its decision. The features listed at the top are the strongest indicators of fraud according to the AI. This transparency is crucial for auditors and risk teams to trust the AI's predictions.")

    st.markdown('<div class="section-header">Detailed Breakdown</div>', unsafe_allow_html=True)
    st.markdown(explanation)


# ===================================================================
# PAGE 5 — PII SCANNER
# ===================================================================
elif page == "🔐 PII Scanner":
    st.markdown("# 🔐 PII Detection & Masking")

    default_text = ("Contact john.doe@example.com or call +91-9876543210. "
                    "Card: 4111 1111 1111 1111. Aadhaar: 1234 5678 9012.")
    user_text = st.text_area("Paste text to scan for PII:", value=default_text, height=180)

    if st.button("🔍 Scan for PII", type="primary"):
        findings = detect_pii(user_text)
        masked = mask_pii(user_text)

        if findings:
            st.markdown(f'<div class="section-header">Found {len(findings)} PII Instance(s)</div>',
                        unsafe_allow_html=True)
            for f in findings:
                badge = f"<span class='badge-critical'>{f['type']}</span>"
                st.markdown(f"{badge} &nbsp; `{f['match']}`", unsafe_allow_html=True)
        else:
            st.success("No PII detected.")

        st.markdown('<div class="section-header">Masked Output</div>', unsafe_allow_html=True)
        st.code(masked)
        
        # Generate report to local disk and display it
        report = generate_privacy_report(user_text, REPORT_PATH)
        
        st.markdown('<div class="section-header">Privacy Report Summary</div>', unsafe_allow_html=True)
        st.text(report)
        st.info("💡 The full report has also been saved to `outputs/privacy_report.txt` on your local machine.")





# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='text-align:center;color:#4b5563;font-size:0.75rem;'>"
    "AI-Powered Fraud Risk Platform<br>© 2026</div>",
    unsafe_allow_html=True,
)
