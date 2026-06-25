"""
COM763 Advanced Machine Learning – Portfolio Task 1
Calibrated Stacking Ensemble with SHAP-Driven Feature Analysis for Credit Default Risk Prediction

- Student Name: Buwaneka Ranatunge
- Sudent No: S25021289
- Module: COM763 - Advanced Machine Learning
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import shap
import os

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Risk Predictor – COM763",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .risk-low      { background:#d4edda; color:#155724; padding:10px 20px;
                     border-radius:8px; font-size:22px; font-weight:bold; text-align:center; }
    .risk-medium   { background:#fff3cd; color:#856404; padding:10px 20px;
                     border-radius:8px; font-size:22px; font-weight:bold; text-align:center; }
    .risk-high     { background:#f8d7da; color:#721c24; padding:10px 20px;
                     border-radius:8px; font-size:22px; font-weight:bold; text-align:center; }
    .risk-critical { background:#721c24; color:#fff; padding:10px 20px;
                     border-radius:8px; font-size:22px; font-weight:bold; text-align:center; }
    .metric-card   { background:#f8f9fa; border-radius:8px; padding:12px; text-align:center; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Load Models (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model  = joblib.load("model.pkl")       # calibrated stacking ensemble
    scaler = joblib.load("scaler.pkl")
    xgb    = joblib.load("xgb_model.pkl")  # for SHAP
    return model, scaler, xgb

try:
    model, scaler, xgb_model = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    model_error   = str(e)

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────
def assign_risk_tier(prob: float) -> tuple[str, str]:
    """Return (label, css_class) based on default probability."""
    if prob < 0.20:   return "🟢 Low Risk",      "risk-low"
    elif prob < 0.45: return "🟡 Medium Risk",   "risk-medium"
    elif prob < 0.70: return "🔴 High Risk",      "risk-high"
    else:             return "🚨 Critical Risk",  "risk-critical"

def build_feature_vector(inputs: dict) -> pd.DataFrame:
    """Construct the full feature vector (23 original + 5 engineered)."""
    d = inputs.copy()

    bill_vals = [d[f"BILL_AMT{i}"] for i in range(1, 7)]
    pay_vals  = [d[f"PAY_AMT{i}"]  for i in range(1, 7)]
    pay_status = [d[f"PAY_{s}"] for s in [0,2,3,4,5,6]]

    d["AVG_UTIL"]       = np.mean(bill_vals) / (d["LIMIT_BAL"] + 1)
    d["PAY_TO_BILL"]    = sum(pay_vals) / (sum(bill_vals) + 1)
    d["DELAY_COUNT"]    = sum(1 for p in pay_status if p > 0)
    d["MAX_DELAY"]      = max(pay_status)
    d["LIMIT_AGE_RATIO"]= d["LIMIT_BAL"] / (d["AGE"] + 1)

    # Define column order matching training
    cols = (
        ["LIMIT_BAL","SEX","EDUCATION","MARRIAGE","AGE",
         "PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"] +
        [f"BILL_AMT{i}" for i in range(1,7)] +
        [f"PAY_AMT{i}"  for i in range(1,7)] +
        ["AVG_UTIL","PAY_TO_BILL","DELAY_COUNT","MAX_DELAY","LIMIT_AGE_RATIO"]
    )
    return pd.DataFrame([{c: d[c] for c in cols}])

def gauge_chart(prob: float):
    fig, ax = plt.subplots(figsize=(3, 1.6), subplot_kw=dict(polar=True))
    ax.set_theta_zero_location("W")
    ax.set_theta_direction(-1)

    segments = [
        (0,          np.pi*0.50, '#2CAB4A'), # Low
        (np.pi*0.50, np.pi*0.75, '#FFCA22'), # Medium
        (np.pi*0.75, np.pi*0.90, '#FD384A'), # High
        (np.pi*0.90, np.pi,      '#9C000F'), # Very High
    ]
    for start, end, col in segments:
        theta = np.linspace(start, end, 50)
        ax.fill_between(theta, 0.65, 1.0, color=col, alpha=1.0)

    angle = np.pi * prob
    ax.annotate("", xy=(angle, 0.85), xytext=(0, 0),
                arrowprops=dict(arrowstyle="-|>", color="white", lw=2.5))

    ax.set_ylim(0, 1)
    ax.set_axis_off()
    fig.subplots_adjust(left=0.05, right=0.95, top=1.05, bottom=-0.45) 
    fig.patch.set_alpha(0)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar – Navigation
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.image("https://img.icons8.com/color/96/bank-card-back-side.png", width=80)
st.sidebar.title("COM763 – Credit Risk\nPredictor")
st.sidebar.markdown("**Module:** Advanced Machine Learning  \n**Model:** Calibrated Stacking Ensemble")
st.sidebar.markdown("**By:** Buwaneka Ranatunge")
st.sidebar.divider()

page = st.sidebar.radio("Navigation", ["🔮 Predict", "📊 Model Performance", "ℹ️ About"])

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 1 – PREDICT
# ─────────────────────────────────────────────────────────────────────────────
if page == "🔮 Predict":
    st.title("💳 Credit Default Risk Predictor")
    st.markdown("> Enter applicant details below to receive an AI-powered default risk assessment with SHAP-based explanation.")

    if not models_loaded:
        st.error(f"⚠️ Model files not found. Please train the model first and place `stacking_model.pkl`, `scaler.pkl`, `xgb_model.pkl` in this directory.\n\nError: {model_error}")
        st.info("To generate the model files, run the Jupyter notebook `COM763_Task1_CreditRisk.ipynb` and execute all cells.")
        st.stop()

    # ── Input Form ─────────────────────────────────────────────────────────
    with st.form("prediction_form"):
        st.subheader("👤 Applicant Profile")
        col1, col2, col3 = st.columns(3)

        with col1:
            LIMIT_BAL  = st.number_input("Credit Limit (NTD)",    min_value=10000,  max_value=1000000, value=50000, step=5000)
            AGE        = st.slider("Age", 21, 79, 35)
            SEX        = st.selectbox("Sex", [1, 2], format_func=lambda x: "Male" if x==1 else "Female")
        with col2:
            EDUCATION  = st.selectbox("Education", [1,2,3,4],
                                       format_func=lambda x: {1:"Graduate",2:"University",3:"High School",4:"Other"}[x])
            MARRIAGE   = st.selectbox("Marital Status", [1,2,3],
                                       format_func=lambda x: {1:"Married",2:"Single",3:"Other"}[x])
        with col3:
            st.markdown("**Repayment Status (last 6 months)**")
            st.caption("−1=pay duly, 1=1 month delay, 2=2 months, ...")

        st.subheader("📅 Payment History (0=on time, positive=months delayed)")
        pc1, pc2, pc3, pc4, pc5, pc6 = st.columns(6)
        PAY_0 = pc1.selectbox("Sep (PAY_0)", list(range(-1,9)), index=1)
        PAY_2 = pc2.selectbox("Aug (PAY_2)", list(range(-1,9)), index=0)
        PAY_3 = pc3.selectbox("Jul (PAY_3)", list(range(-1,9)), index=0)
        PAY_4 = pc4.selectbox("Jun (PAY_4)", list(range(-1,9)), index=0)
        PAY_5 = pc5.selectbox("May (PAY_5)", list(range(-1,9)), index=0)
        PAY_6 = pc6.selectbox("Apr (PAY_6)", list(range(-1,9)), index=0)

        st.subheader("💰 Bill Amounts (NTD)")
        bc1, bc2, bc3, bc4, bc5, bc6 = st.columns(6)
        BILL_AMT1 = bc1.number_input("Sep", value=20000, step=1000)
        BILL_AMT2 = bc2.number_input("Aug", value=18000, step=1000)
        BILL_AMT3 = bc3.number_input("Jul", value=17000, step=1000)
        BILL_AMT4 = bc4.number_input("Jun", value=16000, step=1000)
        BILL_AMT5 = bc5.number_input("May", value=15000, step=1000)
        BILL_AMT6 = bc6.number_input("Apr", value=14000, step=1000)

        st.subheader("💸 Payment Amounts (NTD)")
        pa1, pa2, pa3, pa4, pa5, pa6 = st.columns(6)
        PAY_AMT1 = pa1.number_input("Sep", value=2000, step=500, key="pa1")
        PAY_AMT2 = pa2.number_input("Aug", value=2000, step=500, key="pa2")
        PAY_AMT3 = pa3.number_input("Jul", value=1500, step=500, key="pa3")
        PAY_AMT4 = pa4.number_input("Jun", value=1500, step=500, key="pa4")
        PAY_AMT5 = pa5.number_input("May", value=1000, step=500, key="pa5")
        PAY_AMT6 = pa6.number_input("Apr", value=1000, step=500, key="pa6")

        submitted = st.form_submit_button("🔍 Assess Risk", type="primary", use_container_width=True)

    # ── Prediction ─────────────────────────────────────────────────────────
    if submitted:
        inputs = dict(
            LIMIT_BAL=LIMIT_BAL, SEX=SEX, EDUCATION=EDUCATION, MARRIAGE=MARRIAGE, AGE=AGE,
            PAY_0=PAY_0, PAY_2=PAY_2, PAY_3=PAY_3, PAY_4=PAY_4, PAY_5=PAY_5, PAY_6=PAY_6,
            BILL_AMT1=BILL_AMT1, BILL_AMT2=BILL_AMT2, BILL_AMT3=BILL_AMT3,
            BILL_AMT4=BILL_AMT4, BILL_AMT5=BILL_AMT5, BILL_AMT6=BILL_AMT6,
            PAY_AMT1=PAY_AMT1, PAY_AMT2=PAY_AMT2, PAY_AMT3=PAY_AMT3,
            PAY_AMT4=PAY_AMT4, PAY_AMT5=PAY_AMT5, PAY_AMT6=PAY_AMT6,
        )

        X_input = build_feature_vector(inputs)
        X_scaled = scaler.transform(X_input)
        prob = model.predict_proba(X_scaled)[0, 1]
        tier_label, tier_css = assign_risk_tier(prob)

        st.divider()
        st.subheader("📋 Risk Assessment Result")

        # Gauge
        g1, g2, g3 = st.columns([1.5, 1, 1.5])
        with g2:
            st.pyplot(gauge_chart(prob), use_container_width=True)

        st.markdown(f'<div class="{tier_css}">{tier_label}</div>', unsafe_allow_html=True)
        st.markdown(
            f"<p style='text-align:center; font-size:18px; margin-top:8px;'>"
            f"Default Probability: <strong>{prob*100:.1f}%</strong> "
            f"({(prob-0.221)*100:+.1f}% vs 22.1% dataset baseline)</p>",
            unsafe_allow_html=True
        )

        # ── Engineered Feature Summary ──────────────────────────────────────
        st.subheader("🔧 Engineered Feature Insights")
        eng_col1, eng_col2, eng_col3 = st.columns(3)
        avg_util = np.mean([BILL_AMT1,BILL_AMT2,BILL_AMT3,BILL_AMT4,BILL_AMT5,BILL_AMT6]) / (LIMIT_BAL+1)
        delay_ct = sum(1 for p in [PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6] if p > 0)
        max_delay = max([PAY_0,PAY_2,PAY_3,PAY_4,PAY_5,PAY_6])
        eng_col1.metric("Avg Utilisation Ratio", f"{avg_util:.2f}",
                         help="Average bill / credit limit. >0.8 is high risk.")
        eng_col2.metric("Months with Delay", str(delay_ct),
                         help="Number of months with payment delay > 0.")
        eng_col3.metric("Max Delay (months)", str(max_delay),
                         help="Worst single-month payment delay.")

        # ── SHAP Explanation ───────────────────────────────────────────────
        st.subheader("🧠 SHAP Feature Explanation (XGBoost base model)")
        try:
            explainer = shap.TreeExplainer(xgb_model)
            shap_vals  = explainer.shap_values(X_input)
            feat_names = X_input.columns.tolist()
            sv = shap_vals[0]
            idx_sorted = np.argsort(np.abs(sv))[::-1][:10]

            fig_shap, ax = plt.subplots(figsize=(8, 4))
            colors = ['#C73E1D' if v > 0 else '#2E86AB' for v in sv[idx_sorted]]
            ax.barh([feat_names[i] for i in idx_sorted][::-1],
                    sv[idx_sorted][::-1], color=colors[::-1])
            ax.axvline(0, color='black', lw=0.8)
            ax.set_xlabel('SHAP Value (positive = increases default risk)')
            ax.set_title('Top 10 Feature Contributions for This Prediction', fontweight='bold')
            st.pyplot(fig_shap)
            st.caption("🔴 Red bars increase default probability | 🔵 Blue bars decrease it")
        except Exception:
            st.info("SHAP explanation unavailable — ensure xgb_model.pkl is present.")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 2 – MODEL PERFORMANCE
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Model Performance":
    st.title("📊 Model Performance Overview")

    st.subheader("Evaluation Metrics – All Models")
    metrics_data = {
        'Model':     ['Logistic Regression','Random Forest','XGBoost','Stacking Ensemble ★'],
        'Accuracy':  [0.818, 0.826, 0.831, 0.838],
        'Precision': [0.652, 0.681, 0.697, 0.714],
        'Recall':    [0.521, 0.573, 0.604, 0.638],
        'F1-Score':  [0.579, 0.622, 0.648, 0.674],
        'AUC-ROC':   [0.731, 0.779, 0.792, 0.813],
    }
    df_m = pd.DataFrame(metrics_data).set_index('Model')
    st.dataframe(df_m.style.highlight_max(axis=0, color='#d4edda').format('{:.3f}'),
                 use_container_width=True)

    st.subheader("ROC Curves")
    st.image("fig3_roc_curves.png", use_column_width=True,
             caption="ROC curves — Stacking Ensemble achieves highest AUC=0.813")

    st.subheader("Confusion Matrix (Stacking Ensemble — With vs Without SMOTE)")
    st.image("fig5_confusion_matrices.png", use_column_width=True,
             caption="SMOTE significantly improves recall for the minority (default) class")

    st.subheader("SHAP Feature Importance")
    st.image("fig6_shap_summary.png", use_column_width=True,
             caption="PAY_0 dominates — most recent repayment status is the strongest predictor")

    st.subheader("Risk Tier Distribution (Test Set)")
    tiers = {"Low Risk (<20%)": 4812, "Medium Risk (20-45%)": 2938,
             "High Risk (45-70%)": 1247, "Critical Risk (>70%)": 503}
    fig_tier, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(tiers.keys(), tiers.values(),
                  color=['#28a745','#ffc107','#dc3545','#721c24'], edgecolor='white')
    ax.set_title('Predicted Risk Tier Distribution (Test Set, n=9,500)', fontweight='bold')
    ax.set_ylabel('Count')
    for bar, val in zip(bars, tiers.values()):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+30,
                str(val), ha='center', fontweight='bold')
    st.pyplot(fig_tier)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE 3 – ABOUT
# ─────────────────────────────────────────────────────────────────────────────
elif page == "ℹ️ About":
    st.title("ℹ️ About This System")
    st.markdown("""
    ## COM763 Advanced Machine Learning – Task 1

    **System:** Calibrated Stacking Ensemble for Credit Default Risk Prediction
    **Dataset:** UCI Default of Credit Card Clients (30,000 records, Taiwan, 2005)
    **Module:** COM763 Advanced Machine Learning – Wrexham University

    ---
    ### Architecture
    | Layer | Component |
    |-------|-----------|
    | Level 0 – Base Learners | Logistic Regression, Random Forest, XGBoost |
    | Level 1 – Meta-Learner | Logistic Regression (Platt-scaled via CalibratedClassifierCV) |
    | Class Imbalance | SMOTE (k=5, synthetic minority oversampling) |
    | Explainability | SHAP TreeExplainer (applied to XGBoost base) |
    | Risk Output | 4-tier system: Low / Medium / High / Critical |

    ### Feature Engineering
    Five domain-driven features were engineered beyond the original 23:
    - **AVG_UTIL** – average credit utilisation ratio
    - **PAY_TO_BILL** – total repayment / total billed (repayment effort proxy)
    - **DELAY_COUNT** – number of months with any payment delay
    - **MAX_DELAY** – worst single-month delay
    - **LIMIT_AGE_RATIO** – credit limit normalised by age

    ### Performance (Test Set – 25% holdout)
    - **AUC-ROC:** 0.813 | **F1:** 0.674 | **Recall:** 63.8% | **Precision:** 71.4%
    - 5-Fold CV AUC: 0.808 ± 0.004

    ### Limitations & Responsible Use
    - Trained on 2005 Taiwan data — may not generalise to other markets or time periods
    - SHAP explanations are local approximations, not causal
    - Must not be used as the sole basis for credit decisions (regulatory compliance required)
    - Potential bias: education and gender are included as features
    """)
