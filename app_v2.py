import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import os

# ─────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="HealthGuard — Obesity Diagnostic AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# GOOGLE FONTS + GLOBAL CSS
# ─────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Outfit:wght@300;400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
/* ── Root ── */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif !important;
}

/* ── Hide sidebar toggle & default header ── */
[data-testid="stSidebar"]          { display: none !important; }
[data-testid="collapsedControl"]   { display: none !important; }
#MainMenu                          { visibility: hidden; }
header[data-testid="stHeader"]     { display: none !important; }
footer                             { visibility: hidden; }
[data-testid="stDeployButton"]     { display: none !important; }

/* ── Remove default top padding so our nav sits at very top ── */
.block-container {
    padding-top: 0rem !important;
    padding-bottom: 2rem !important;
    max-width: 1200px !important;
}

/* ── TOP NAVBAR ── */
.top-nav {
    position: sticky;
    top: 0;
    z-index: 999;
    background: rgba(8, 15, 30, 0.92);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border-bottom: 1px solid rgba(59,130,246,0.18);
    padding: 0 32px;
    height: 60px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin: 0 -4rem;               /* bleed to edges */
}

/* ── Streamlit Tabs — styled as top nav links ── */
[data-testid="stTabs"] {
    margin-top: 0 !important;
}

/* Tab bar strip */
[data-testid="stTabs"] > div:first-child {
    position: sticky !important;
    top: 0 !important;
    z-index: 998 !important;
    background: rgba(8, 15, 30, 0.95) !important;
    backdrop-filter: blur(16px) !important;
    border-bottom: 1px solid rgba(59,130,246,0.18) !important;
    padding: 0 2rem !important;
    margin: 0 -4rem !important;
    gap: 0 !important;
}

/* Individual tab buttons */
[data-testid="stTabs"] button[role="tab"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: #64748b !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 18px 22px !important;
    transition: all 0.2s !important;
    white-space: nowrap !important;
}

[data-testid="stTabs"] button[role="tab"]:hover {
    color: #e2e8f0 !important;
    background: rgba(59,130,246,0.06) !important;
}

[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #60a5fa !important;
    border-bottom: 2px solid #3b82f6 !important;
    background: transparent !important;
}

/* Hide the default underline indicator Streamlit adds */
[data-testid="stTabs"] > div:first-child > div {
    background: transparent !important;
}

/* Tab content area */
[data-testid="stTabsContent"] {
    padding-top: 2rem !important;
}

/* ── Brand in tab bar (first pseudo element) ── */
/* We inject a brand via st.markdown inside the tab bar row */

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: rgba(30,106,220,0.08) !important;
    border: 1px solid rgba(59,130,246,0.25) !important;
    border-radius: 14px !important;
    padding: 16px 20px !important;
}
[data-testid="stMetricLabel"] {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}
[data-testid="stMetricValue"] {
    font-family: 'Bebas Neue', sans-serif !important;
    font-size: 2rem !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 0.1em !important;
    font-size: 1rem !important;
    padding: 12px 28px !important;
    border-radius: 10px !important;
    background: linear-gradient(135deg, #1e6adc, #0f3a9e) !important;
    border: none !important;
    color: white !important;
    box-shadow: 0 4px 20px rgba(30,106,220,0.35) !important;
    transition: all 0.25s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(30,106,220,0.55) !important;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 1px solid rgba(59,130,246,0.2) !important;
    border-radius: 12px !important;
    overflow: hidden;
}
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    font-size: 0.95rem !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] label {
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}

/* ── Divider ── */
hr { border-color: rgba(59,130,246,0.15) !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PATHS & LOADING
# ─────────────────────────────────────────────
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_PATH, "ObesityDataSet.csv")

@st.cache_resource
def load_resources():
    rf_model  = joblib.load(os.path.join(BASE_PATH, "rf_model.pkl"))
    try:
        xgb_model = joblib.load(os.path.join(BASE_PATH, "xgb_model.pkl"))
    except:
        xgb_model = None
    encoder = joblib.load(os.path.join(BASE_PATH, "target_encoder.pkl"))
    scaler  = joblib.load(os.path.join(BASE_PATH, "scaler.pkl"))
    return rf_model, xgb_model, encoder, scaler

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

rf_model, xgb_model, target_encoder, scaler = load_resources()
df = load_data()

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
binary_map  = {"Yes": 1, "No": 0}
gender_map  = {"Male": 1, "Female": 0}
ordinal_map = {"No": 0, "Sometimes": 1, "Frequently": 2, "Always": 3}
mtrans_map  = {"Automobile": 0, "Bike": 1, "Motorbike": 2, "Public Transportation": 3, "Walking": 4}
numeric_cols = ["Age", "Height", "Weight", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
col_order = [
    "Age", "Gender", "Height", "Weight", "CALC", "FAVC", "FCVC", "NCP",
    "SCC", "SMOKE", "CH2O", "family_history_with_overweight", "FAF", "TUE",
    "CAEC", "MTRANS"
]

CLASS_COLORS = {
    "Insufficient_Weight": "#3b97e3",
    "Normal_Weight":       "#00c97a",
    "Overweight_Level_I":  "#d4a017",
    "Overweight_Level_II": "#c2701e",
    "Obesity_Type_I":      "#e74c3c",
    "Obesity_Type_II":     "#c0392b",
    "Obesity_Type_III":    "#7b241c",
}
HEALTH_ADVICE = {
    "Insufficient_Weight": "⚠️ Weight is below the healthy range. Consult a nutritionist for a calorie-dense, balanced meal plan and strength training program.",
    "Normal_Weight":       "✅ Excellent! Weight is in the healthy range. Maintain your current balanced diet and regular physical activity.",
    "Overweight_Level_I":  "🟡 Mild overweight detected. Reduce high-caloric food, increase vegetables, and add 3+ days of moderate exercise per week.",
    "Overweight_Level_II": "🟠 Moderate overweight. Consider a healthcare consultation. Dietary control and daily 30-min walks are recommended.",
    "Obesity_Type_I":      "🔴 Obesity Class I. Medical consultation advised. A structured weight-loss program with dietary changes and supervised exercise is recommended.",
    "Obesity_Type_II":     "🔴 Obesity Class II requires medical intervention. Consult a specialist for a comprehensive weight management plan.",
    "Obesity_Type_III":    "🚨 Severe Obesity Class III. Immediate medical attention strongly recommended. Clinical evaluation and possible surgical consultation may be needed.",
}

# ─────────────────────────────────────────────
# TOP NAVIGATION using st.tabs()
# ─────────────────────────────────────────────
tab_home, tab_eda, tab_predict = st.tabs([
    "🛡️  HealthGuard — Home",
    "📊  Data Analysis (EDA)",
    "⚕️  Obesity Predictor",
])

# Status check for model
_model_ok = xgb_model is not None


# ═══════════════════════════════════════════════
#  PAGE 1 — HOME & OVERVIEW
# ═══════════════════════════════════════════════
with tab_home:

    # ── Hero Banner ──
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(30,106,220,0.15) 0%, rgba(0,212,255,0.08) 100%);
                border: 1px solid rgba(30,106,220,0.25); border-radius: 20px;
                padding: 40px 48px; margin-bottom: 32px; position: relative; overflow: hidden;">
        <div style="position:absolute; top:0; left:0; right:0; height:3px;
                    background: linear-gradient(90deg, #1e6adc, #00d4ff, #00c97a);"></div>
        <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#60a5fa;
                    letter-spacing:0.15em; text-transform:uppercase; margin-bottom:10px;">
            ■ BCA Major Project · NCSC Bharuch · 2025–26
        </div>
        <div style="font-family:'Bebas Neue',sans-serif; font-size:clamp(2.5rem,5vw,4rem);
                    letter-spacing:0.04em; color:white; line-height:1; margin-bottom:14px;">
            DIGITAL OBESITY<br>DIAGNOSTIC SYSTEM
        </div>
        <div style="font-size:1rem; color:#94a3b8; max-width:680px; line-height:1.75;">
            An AI-powered clinical classification system that diagnoses obesity levels across 
            <strong style="color:white;">7 WHO-standard categories</strong> using dual ensemble ML models — 
            <strong style="color:#60a5fa;">Random Forest</strong> &amp; 
            <strong style="color:#fbbf24;">XGBoost</strong> — trained on 2,111 real-world health records 
            from Mexico, Peru, and Colombia.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Key Metrics ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Model Accuracy", "95.27%", "Optimal")
    m2.metric("Dataset Records", "2,111", "SMOTE Balanced")
    m3.metric("Obesity Classes", "7", "WHO Standard")
    m4.metric("Macro F1 Score", "0.95", "RF + XGBoost")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Model Cards ──
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#60a5fa;
                letter-spacing:0.15em; text-transform:uppercase; margin-bottom:6px;">
        ■ MACHINE LEARNING ENGINES
    </div>
    <div style="font-family:'Bebas Neue',sans-serif; font-size:2rem; letter-spacing:0.04em; 
                margin-bottom:24px;">DUAL-ENGINE ARCHITECTURE</div>
    """, unsafe_allow_html=True)

    mc1, mc2 = st.columns(2)

    with mc1:
        st.markdown("""
        <div style="background:rgba(30,106,220,0.1); border:1px solid rgba(30,106,220,0.3);
                    border-radius:16px; padding:24px; height:100%; position:relative; overflow:hidden;">
            <div style="position:absolute;top:0;left:0;right:0;height:3px;background:#1e6adc;"></div>
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:#60a5fa;
                             background:rgba(30,106,220,0.2); padding:3px 10px; border-radius:20px;">
                    RF · BAGGING
                </span>
                <span style="font-size:1.5rem;">🌲</span>
            </div>
            <div style="font-family:'Bebas Neue',sans-serif; font-size:1.8rem; letter-spacing:0.04em; 
                        color:white; margin-bottom:4px;">Random Forest</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#64748b; 
                        margin-bottom:16px;">SKLEARN · PARALLEL ENSEMBLE</div>
            <div style="font-family:'Bebas Neue',sans-serif; font-size:3rem; color:#60a5fa; 
                        line-height:1; margin-bottom:4px;">95.27%</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#64748b; 
                        margin-bottom:16px;">TEST ACCURACY · MACRO F1: 0.95</div>
            <ul style="color:#94a3b8; font-size:0.85rem; line-height:1.8; padding-left:16px;">
                <li>Builds N decision trees independently in parallel</li>
                <li>Final class by majority vote across all trees</li>
                <li>Averages noise — highly resistant to overfitting</li>
                <li>Best: Obesity Type II &amp; III — F1: <strong style="color:white;">0.99</strong></li>
                <li>Top feature: <strong style="color:white;">Weight (importance=0.33)</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with mc2:
        st.markdown("""
        <div style="background:rgba(251,191,36,0.08); border:1px solid rgba(251,191,36,0.25);
                    border-radius:16px; padding:24px; height:100%; position:relative; overflow:hidden;">
            <div style="position:absolute;top:0;left:0;right:0;height:3px;background:#fbbf24;"></div>
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:12px;">
                <span style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:#fbbf24;
                             background:rgba(251,191,36,0.15); padding:3px 10px; border-radius:20px;">
                    XGB · BOOSTING
                </span>
                <span style="font-size:1.5rem;">⚡</span>
            </div>
            <div style="font-family:'Bebas Neue',sans-serif; font-size:1.8rem; letter-spacing:0.04em; 
                        color:white; margin-bottom:4px;">XGBoost</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#64748b; 
                        margin-bottom:16px;">EXTREME GRADIENT BOOSTING</div>
            <div style="font-family:'Bebas Neue',sans-serif; font-size:3rem; color:#fbbf24; 
                        line-height:1; margin-bottom:4px;">95.27%</div>
            <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#64748b; 
                        margin-bottom:16px;">TEST ACCURACY · MACRO F1: 0.95</div>
            <ul style="color:#94a3b8; font-size:0.85rem; line-height:1.8; padding-left:16px;">
                <li>Builds trees sequentially — each corrects prior errors</li>
                <li>Iterative gradient descent optimization</li>
                <li>Superior at distinguishing Overweight Level I vs II</li>
                <li>Best: Obesity Type II — F1: <strong style="color:#00c97a;">1.00 PERFECT</strong></li>
                <li>Top feature: <strong style="color:white;">Gender (importance=0.26)</strong></li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── ML Pipeline ──
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#60a5fa;
                letter-spacing:0.15em; text-transform:uppercase; margin-bottom:6px;">
        ■ WORKFLOW
    </div>
    <div style="font-family:'Bebas Neue',sans-serif; font-size:2rem; letter-spacing:0.04em; 
                margin-bottom:24px;">ML PIPELINE</div>
    """, unsafe_allow_html=True)

    steps = [
        ("1", "Data Collection", "UCI Kaggle · 2,111 records · 0 nulls"),
        ("2", "EDA", "Distribution · Correlation · Patterns"),
        ("3", "Preprocessing", "Encoding · Scaling · Label mapping"),
        ("4", "Modeling", "RF + XGBoost · 80/20 · Stratified"),
        ("5", "Evaluation", "Accuracy · F1 · Confusion Matrix"),
        ("6", "Deployment", "Streamlit Cloud · Live diagnosis"),
    ]
    cols = st.columns(6)
    for col, (num, name, desc) in zip(cols, steps):
        with col:
            st.markdown(f"""
            <div style="text-align:center; padding:16px 8px;">
                <div style="width:52px; height:52px; border-radius:50%; margin:0 auto 12px;
                            background:rgba(30,106,220,0.15); border:2px solid #1e6adc;
                            display:flex; align-items:center; justify-content:center;
                            font-family:'Bebas Neue',sans-serif; font-size:1.3rem; color:#60a5fa;
                            box-shadow:0 0 16px rgba(30,106,220,0.3);">
                    {num}
                </div>
                <div style="font-weight:700; font-size:0.85rem; color:white; margin-bottom:4px;">{name}</div>
                <div style="font-size:0.72rem; color:#64748b; line-height:1.5;">{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Feature Guide ──
    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#60a5fa;
                letter-spacing:0.15em; text-transform:uppercase; margin-bottom:6px;">
        ■ INPUT REFERENCE
    </div>
    <div style="font-family:'Bebas Neue',sans-serif; font-size:2rem; letter-spacing:0.04em; 
                margin-bottom:24px;">FEATURE FIELD GUIDE</div>
    """, unsafe_allow_html=True)

    fg1, fg2 = st.columns(2)
    with fg1:
        with st.expander("🍎  Nutrition & Diet Indicators", expanded=True):
            features_diet = [
                ("FAVC", "High Caloric Food", "Frequent consumption of greasy/processed food — Yes/No"),
                ("FCVC", "Vegetable Frequency", "Servings per meal — 1: Rarely, 2: Sometimes, 3: Always"),
                ("NCP",  "Main Meals",          "Daily count of main meals — 1 to 4"),
                ("CAEC", "Snacking",             "Eating between meals — No/Sometimes/Frequently/Always"),
                ("CH2O", "Hydration",            "Daily water in litres — 1: <1L, 2: 1–2L, 3: >2L"),
                ("CALC", "Alcohol",              "Frequency of alcoholic beverage consumption"),
            ]
            for code, name, desc in features_diet:
                st.markdown(f"""
                <div style="display:flex; gap:12px; padding:8px 0; 
                            border-bottom:1px solid rgba(59,130,246,0.1); align-items:flex-start;">
                    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:#00d4ff;
                                 background:rgba(0,212,255,0.08); padding:2px 8px; border-radius:4px;
                                 flex-shrink:0; margin-top:1px;">{code}</span>
                    <div>
                        <div style="font-weight:600; font-size:0.85rem; color:white; margin-bottom:2px;">{name}</div>
                        <div style="font-size:0.78rem; color:#7a8ea8;">{desc}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with fg2:
        with st.expander("🏃  Lifestyle & Biological Factors", expanded=True):
            features_life = [
                ("family_history", "Family History",    "Genetic history of overweight — near-perfect predictor"),
                ("FAF",            "Exercise",           "Days/week of 30+ min moderate activity — 0 to 3 scale"),
                ("TUE",            "Screen Time",        "Hours daily on devices — 0: <2h, 1: 3–5h, 2: >5h"),
                ("SCC",            "Calorie Monitoring", "Active tracking of calorie intake — Yes/No"),
                ("MTRANS",         "Transport",          "Mode of transport — Walking/Bike (active) vs Auto (passive)"),
                ("SMOKE",          "Smoking",            "Current smoking status — near-zero model importance"),
            ]
            for code, name, desc in features_life:
                st.markdown(f"""
                <div style="display:flex; gap:12px; padding:8px 0; 
                            border-bottom:1px solid rgba(59,130,246,0.1); align-items:flex-start;">
                    <span style="font-family:'IBM Plex Mono',monospace; font-size:0.7rem; color:#00d4ff;
                                 background:rgba(0,212,255,0.08); padding:2px 8px; border-radius:4px;
                                 flex-shrink:0; margin-top:1px;">{code}</span>
                    <div>
                        <div style="font-weight:600; font-size:0.85rem; color:white; margin-bottom:2px;">{name}</div>
                        <div style="font-size:0.78rem; color:#7a8ea8;">{desc}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════
#  PAGE 2 — EDA
# ═══════════════════════════════════════════════
with tab_eda:

    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#60a5fa;
                letter-spacing:0.15em; text-transform:uppercase; margin-bottom:6px;">
        ■ EXPLORATORY DATA ANALYSIS
    </div>
    <div style="font-family:'Bebas Neue',sans-serif; font-size:2.5rem; letter-spacing:0.04em; 
                margin-bottom:8px;">DATA INSIGHTS</div>
    <div style="color:#94a3b8; margin-bottom:32px; font-size:0.95rem;">
        Scientific analysis of how biometrics, nutrition, lifestyle, and genetics correlate with obesity progression.
    </div>
    """, unsafe_allow_html=True)

    # Dataset stats
    ds1, ds2, ds3, ds4 = st.columns(4)
    ds1.metric("Total Records", "2,111")
    ds2.metric("Input Features", "16")
    ds3.metric("Missing Values", "0")
    ds4.metric("Target Classes", "7")

    st.markdown("<br>", unsafe_allow_html=True)

    def styled_chart(fig, title, insight):
        """Wrap a matplotlib figure in a styled card."""
        st.markdown(f"""
        <div style="background:rgba(30,106,220,0.06); border:1px solid rgba(30,106,220,0.2);
                    border-radius:14px; padding:0; overflow:hidden; margin-bottom:4px;">
            <div style="padding:14px 18px; border-bottom:1px solid rgba(30,106,220,0.15);
                        display:flex; align-items:flex-start; justify-content:space-between; gap:12px;">
                <div>
                    <div style="font-weight:700; font-size:0.95rem; color:white;">{title}</div>
                    <div style="font-size:0.8rem; color:#64748b; margin-top:3px; line-height:1.5;">{insight}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    # Chart style helper
    def apply_style(ax, fig):
        fig.patch.set_facecolor('#0a1628')
        ax.set_facecolor('#0c1e35')
        ax.tick_params(colors='#94a3b8', labelsize=9)
        ax.xaxis.label.set_color('#94a3b8')
        ax.yaxis.label.set_color('#94a3b8')
        ax.title.set_color('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#3b82f633')
        return fig, ax

    # ── Row 1 ──
    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(8, 5))
        apply_style(ax, fig)
        counts = df["NObeyesdad"].value_counts()
        colors = [CLASS_COLORS.get(c, "#64748b") for c in counts.index]
        bars = ax.barh(
            [x.replace("_", " ") for x in counts.index],
            counts.values, color=colors, edgecolor='none'
        )
        for bar, val in zip(bars, counts.values):
            ax.text(val + 5, bar.get_y() + bar.get_height()/2,
                    str(val), va='center', color='#94a3b8', fontsize=9)
        ax.set_xlabel("Count", color='#64748b')
        ax.set_title("")
        fig.tight_layout()
        styled_chart(fig, "Target Class Distribution",
                     "Dataset balanced across all 7 categories via SMOTE — equal AI priority for every tier.")

    with c2:
        fig, ax = plt.subplots(figsize=(8, 5))
        apply_style(ax, fig)
        order = list(CLASS_COLORS.keys())
        sns.boxplot(data=df, x="NObeyesdad", y="Weight",
                    order=order, palette=list(CLASS_COLORS.values()), ax=ax)
        ax.set_xticklabels([x.replace("_", "\n") for x in order], fontsize=7.5)
        ax.set_xlabel("")
        ax.set_ylabel("Weight (kg)", color='#94a3b8')
        fig.tight_layout()
        styled_chart(fig, "Weight Distribution by Obesity Class",
                     "Overlapping weight ranges between tiers prove lifestyle features are critical tie-breakers.")

    # ── Row 2 ──
    c3, c4 = st.columns(2)

    with c3:
        fig, ax = plt.subplots(figsize=(8, 5))
        apply_style(ax, fig)
        sns.histplot(data=df, x="Age", hue="Gender",
                     multiple="stack", palette=["#60a5fa", "#f472b6"], ax=ax)
        ax.set_xlabel("Age", color='#94a3b8')
        ax.set_ylabel("Count", color='#94a3b8')
        if ax.get_legend():
            ax.get_legend().get_frame().set_facecolor('#0a1628')
            for t in ax.get_legend().get_texts():
                t.set_color('#94a3b8')
        fig.tight_layout()
        styled_chart(fig, "Age & Gender Distribution",
                     "Majority of subjects in 18–35 range — typical for modern preventive health research.")

    with c4:
        fig, ax = plt.subplots(figsize=(8, 5))
        apply_style(ax, fig)
        sns.countplot(data=df, x="NObeyesdad", hue="family_history_with_overweight",
                      order=order, palette=["#e74c3c", "#00c97a"], ax=ax)
        ax.set_xticklabels([x.replace("_", "\n") for x in order], fontsize=7.5)
        ax.set_xlabel("")
        if ax.get_legend():
            ax.get_legend().set_title("Family History")
            ax.get_legend().get_frame().set_facecolor('#0a1628')
            for t in ax.get_legend().get_texts():
                t.set_color('#94a3b8')
        fig.tight_layout()
        styled_chart(fig, "Family History Impact",
                     "Family history is a near-perfect predictor of obesity progression in higher tiers.")

    # ── Row 3 ──
    c5, c6 = st.columns(2)

    with c5:
        fig, ax = plt.subplots(figsize=(8, 5))
        apply_style(ax, fig)
        sns.barplot(data=df, x="NObeyesdad", y="FAF",
                    order=order, palette=list(CLASS_COLORS.values()), ax=ax)
        ax.set_xticklabels([x.replace("_", "\n") for x in order], fontsize=7.5)
        ax.set_xlabel("")
        ax.set_ylabel("Avg Physical Activity (FAF)", color='#94a3b8')
        fig.tight_layout()
        styled_chart(fig, "Physical Activity vs Obesity Class",
                     "Drastic drop in FAF across obesity tiers confirms the sedentary lifestyle risk factor.")

    with c6:
        fig, ax = plt.subplots(figsize=(8, 5))
        apply_style(ax, fig)
        numeric_df = df.select_dtypes(include=[np.number])
        mask = np.zeros_like(numeric_df.corr())
        mask[np.triu_indices_from(mask)] = True
        sns.heatmap(numeric_df.corr(), annot=True, cmap="YlOrRd",
                    fmt=".2f", ax=ax, mask=mask,
                    cbar_kws={"shrink": 0.8},
                    annot_kws={"size": 7})
        ax.tick_params(labelsize=7)
        fig.tight_layout()
        styled_chart(fig, "Clinical Correlation Heatmap",
                     "Hidden inter-variable connections empower the AI to reach 95%+ accuracy.")


# ═══════════════════════════════════════════════
#  PAGE 3 — PREDICTOR
# ═══════════════════════════════════════════════
with tab_predict:

    st.markdown("""
    <div style="font-family:'IBM Plex Mono',monospace; font-size:0.72rem; color:#60a5fa;
                letter-spacing:0.15em; text-transform:uppercase; margin-bottom:6px;">
        ■ PREDICTION ENGINE
    </div>
    <div style="font-family:'Bebas Neue',sans-serif; font-size:2.5rem; letter-spacing:0.04em; 
                margin-bottom:8px;">RUN DIAGNOSIS</div>
    <div style="color:#94a3b8; margin-bottom:28px; font-size:0.95rem;">
        Enter individual measurements to receive real-time dual-model obesity classification with confidence scores.
    </div>
    """, unsafe_allow_html=True)

    # ── Form Sections ──
    with st.expander("👤  Section 1 — Biometric Profile", expanded=True):
        b1, b2, b3 = st.columns(3)
        with b1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            fam    = st.selectbox("Family History of Overweight", ["Yes", "No"])
        with b2:
            age    = st.slider("Age (years)", 10, 80, 25)
            weight = st.slider("Weight (kg)", 30, 180, 70)
        with b3:
            h_unit = st.radio("Height Unit", ["Meters", "Feet & Inches"], horizontal=True)
            if h_unit == "Meters":
                h_m = st.number_input("Height (m)", 1.40, 2.30, 1.70, step=0.01,
                                      format="%.2f")
            else:
                hc1, hc2 = st.columns(2)
                f   = hc1.number_input("Feet", 4, 7, 5)
                inc = hc2.number_input("Inches", 0, 11, 7)
                h_m = (f * 0.3048) + (inc * 0.0254)
                st.caption(f"Converted: {h_m:.2f} m")

            bmi = weight / (h_m ** 2)
            st.markdown(f"""
            <div style="background:rgba(30,106,220,0.12); border:1px solid rgba(30,106,220,0.25);
                        border-radius:10px; padding:12px 16px; margin-top:8px;">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.65rem; color:#60a5fa;
                            text-transform:uppercase; letter-spacing:0.1em;">Calculated BMI</div>
                <div style="font-family:'Bebas Neue',sans-serif; font-size:1.8rem; color:white;">
                    {bmi:.1f}
                </div>
                <div style="font-size:0.75rem; color:#94a3b8;">kg/m²</div>
            </div>
            """, unsafe_allow_html=True)

    with st.expander("🍎  Section 2 — Nutrition & Diet", expanded=True):
        n1, n2, n3 = st.columns(3)
        with n1:
            favc = st.selectbox("High-Caloric Food (FAVC)", ["Yes", "No"])
            caec = st.selectbox("Eating Between Meals (CAEC)",
                                ["No", "Sometimes", "Frequently", "Always"])
        with n2:
            fcvc = st.slider("Vegetable Frequency/Meal (FCVC)", 1, 3, 2,
                             help="1=Rarely, 2=Sometimes, 3=Always")
            ncp  = st.slider("Main Meals per Day (NCP)", 1, 4, 3)
        with n3:
            ch2o = st.slider("Daily Water Intake — L (CH2O)", 1, 3, 2,
                             help="1=<1L, 2=1–2L, 3=>2L")
            calc = st.selectbox("Alcohol Consumption (CALC)",
                                ["No", "Sometimes", "Frequently", "Always"])

    with st.expander("🏋️  Section 3 — Lifestyle & Habits", expanded=True):
        l1, l2, l3 = st.columns(3)
        with l1:
            faf    = st.slider("Physical Activity Freq (FAF)", 0, 3, 1,
                               help="0=None, 1=1–2d, 2=3–4d, 3=4–5d")
            tue    = st.slider("Screen/Device Usage (TUE)", 0, 2, 1,
                               help="0=<2h, 1=3–5h, 2=>5h")
        with l2:
            smoke  = st.selectbox("Smoking (SMOKE)", ["No", "Yes"])
            scc    = st.selectbox("Calorie Monitoring (SCC)", ["No", "Yes"])
        with l3:
            mtrans = st.selectbox("Primary Transport (MTRANS)",
                                  ["Automobile", "Bike", "Motorbike",
                                   "Public Transportation", "Walking"])

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("⚕️  EXECUTE DUAL MODEL DIAGNOSIS", use_container_width=True):

        input_data = {
            "Age": age, "Gender": gender_map[gender],
            "Height": h_m, "Weight": weight,
            "CALC": ordinal_map[calc], "FAVC": binary_map[favc],
            "FCVC": fcvc, "NCP": ncp,
            "SCC": binary_map[scc], "SMOKE": binary_map[smoke],
            "CH2O": ch2o, "family_history_with_overweight": binary_map[fam],
            "FAF": faf, "TUE": tue,
            "CAEC": ordinal_map[caec], "MTRANS": mtrans_map[mtrans]
        }
        input_df = pd.DataFrame([input_data])[col_order]
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

        # ── Predict ──
        rf_p     = rf_model.predict(input_df)[0]
        rf_lbl   = target_encoder.inverse_transform([rf_p])[0]
        rf_probs = rf_model.predict_proba(input_df)[0]
        rf_conf  = round(float(np.max(rf_probs)) * 100, 2)

        if xgb_model:
            xgb_p     = xgb_model.predict(input_df)[0]
            xgb_lbl   = target_encoder.inverse_transform([xgb_p])[0]
            xgb_probs = xgb_model.predict_proba(input_df)[0]
            xgb_conf  = round(float(np.max(xgb_probs)) * 100, 2)
        else:
            xgb_lbl, xgb_conf, xgb_probs = "Unavailable", 0, np.zeros(7)

        rf_color  = CLASS_COLORS.get(rf_lbl,  "#64748b")
        xgb_color = CLASS_COLORS.get(xgb_lbl, "#64748b")

        st.markdown("""
        <div style="font-family:'Bebas Neue',sans-serif; font-size:1.6rem; letter-spacing:0.04em;
                    color:white; margin-bottom:20px; padding-top:8px;">
            ■ DIAGNOSTIC RESULTS
        </div>
        """, unsafe_allow_html=True)

        r1, r2 = st.columns(2)

        # RF Result Card
        with r1:
            st.markdown(f"""
            <div style="background:{rf_color}22; border:1px solid {rf_color}55;
                        border-radius:16px; padding:28px; text-align:center; position:relative; overflow:hidden;">
                <div style="position:absolute;top:0;left:0;right:0;height:3px;background:{rf_color};"></div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:{rf_color};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">
                    ENGINE 01 · RANDOM FOREST
                </div>
                <div style="font-family:'Bebas Neue',sans-serif; font-size:2rem; letter-spacing:0.04em;
                            color:white; margin-bottom:12px; line-height:1.1;">
                    {rf_lbl.replace('_', ' ')}
                </div>
                <div style="background:rgba(255,255,255,0.15); border-radius:30px; padding:6px 18px;
                            display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:0.85rem; color:white;">
                    Confidence: {rf_conf}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("RF Probability Distribution"):
                classes_sorted = target_encoder.classes_
                rf_df = pd.DataFrame({
                    "Class": [c.replace("_", " ") for c in classes_sorted],
                    "Probability (%)": (rf_probs * 100).round(2)
                }).sort_values("Probability (%)", ascending=False)
                st.dataframe(rf_df, hide_index=True, use_container_width=True)

                # Bar chart
                fig, ax = plt.subplots(figsize=(7, 3))
                fig.patch.set_facecolor('#0a1628')
                ax.set_facecolor('#0c1e35')
                colors = [CLASS_COLORS.get(c, "#64748b") for c in classes_sorted]
                ax.barh([c.replace("_", " ") for c in classes_sorted],
                        rf_probs * 100, color=colors, edgecolor='none')
                ax.tick_params(colors='#94a3b8', labelsize=8)
                for spine in ax.spines.values():
                    spine.set_edgecolor('#3b82f626')
                ax.set_xlabel("Probability (%)", color='#94a3b8', fontsize=8)
                fig.tight_layout()
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

        # XGB Result Card
        with r2:
            st.markdown(f"""
            <div style="background:{xgb_color}22; border:1px solid {xgb_color}55;
                        border-radius:16px; padding:28px; text-align:center; position:relative; overflow:hidden;">
                <div style="position:absolute;top:0;left:0;right:0;height:3px;background:{xgb_color};"></div>
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:{xgb_color};
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:8px;">
                    ENGINE 02 · XGBOOST
                </div>
                <div style="font-family:'Bebas Neue',sans-serif; font-size:2rem; letter-spacing:0.04em;
                            color:white; margin-bottom:12px; line-height:1.1;">
                    {xgb_lbl.replace('_', ' ')}
                </div>
                <div style="background:rgba(255,255,255,0.15); border-radius:30px; padding:6px 18px;
                            display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:0.85rem; color:white;">
                    Confidence: {xgb_conf}%
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander("XGB Probability Distribution"):
                if xgb_model:
                    xgb_df = pd.DataFrame({
                        "Class": [c.replace("_", " ") for c in classes_sorted],
                        "Probability (%)": (xgb_probs * 100).round(2)
                    }).sort_values("Probability (%)", ascending=False)
                    st.dataframe(xgb_df, hide_index=True, use_container_width=True)

                    fig, ax = plt.subplots(figsize=(7, 3))
                    fig.patch.set_facecolor('#0a1628')
                    ax.set_facecolor('#0c1e35')
                    ax.barh([c.replace("_", " ") for c in classes_sorted],
                            xgb_probs * 100, color=colors, edgecolor='none')
                    ax.tick_params(colors='#94a3b8', labelsize=8)
                    for spine in ax.spines.values():
                        spine.set_edgecolor('#3b82f626')
                    ax.set_xlabel("Probability (%)", color='#94a3b8', fontsize=8)
                    fig.tight_layout()
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

        # ── Verdict + Advisory ──
        agree = rf_lbl == xgb_lbl
        verdict_color = "#00c97a" if agree else "#fbbf24"
        verdict_text  = "✅ Both models agree on the diagnosis" if agree else "⚠️ Models give different results — consider consulting a physician"

        st.markdown(f"""
        <div style="margin-top:24px; background:rgba(30,106,220,0.08);
                    border:1px solid rgba(30,106,220,0.2); border-radius:16px; padding:24px;">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;">
                <div style="font-family:'Bebas Neue',sans-serif; font-size:1.3rem; letter-spacing:0.04em;
                            color:white;">MODEL CONSENSUS</div>
                <span style="font-size:0.85rem; font-weight:600; color:{verdict_color};">{verdict_text}</span>
            </div>
            <div style="background:rgba(0,0,0,0.2); border-radius:10px; padding:16px;">
                <div style="font-family:'IBM Plex Mono',monospace; font-size:0.68rem; color:#60a5fa;
                            text-transform:uppercase; letter-spacing:0.1em; margin-bottom:6px;">
                    ■ HEALTH ADVISORY
                </div>
                <div style="font-size:0.9rem; color:#94a3b8; line-height:1.7;">
                    {HEALTH_ADVICE.get(rf_lbl, "Please consult a medical professional for personalised advice.")}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Input Summary ──
        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📋 View Input Summary"):
            s1, s2, s3 = st.columns(3)
            with s1:
                st.markdown("**Biometrics**")
                st.write(f"Gender: {gender} | Age: {age}y")
                st.write(f"Height: {h_m:.2f}m | Weight: {weight}kg")
                st.write(f"BMI: **{bmi:.1f}** kg/m²")
                st.write(f"Family History: {fam}")
            with s2:
                st.markdown("**Nutrition**")
                st.write(f"High-Cal Food: {favc}")
                st.write(f"Veg Freq: {fcvc}/3 | Meals: {ncp}/day")
                st.write(f"Snacking: {caec}")
                st.write(f"Water: {ch2o}/3 | Alcohol: {calc}")
            with s3:
                st.markdown("**Lifestyle**")
                st.write(f"Exercise: {faf}/3 days/week")
                st.write(f"Screen Time: {tue}/2")
                st.write(f"Smoking: {smoke} | Cal Monitor: {scc}")
                st.write(f"Transport: {mtrans}")


# ── Global Footer ──
st.markdown("---")
st.markdown("""
<p style="text-align:center; font-family:'IBM Plex Mono',monospace; font-size:0.75rem; 
           color:#334155; letter-spacing:0.08em;">
    🛡️ HEALTHGUARD: DIGITAL OBESITY DIAGNOSTIC SYSTEM &nbsp;·&nbsp; 
    PATHAN MOHAMMADMAHIR I. &nbsp;·&nbsp; SEAT NO. 1254 &nbsp;·&nbsp; 
    NCSC BHARUCH · 2025–26 &nbsp;·&nbsp; DATASET: UCI/KAGGLE (PALECHOR & MANOTAS, 2019)
</p>
""", unsafe_allow_html=True)
