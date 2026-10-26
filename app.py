"""
🧩 ASD Screening Analysis Dashboard
A comprehensive Streamlit dashboard for the ML-Powered Autism Screening project.
Includes live predictions, data exploration, and model performance metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve
)
from sklearn.model_selection import train_test_split
try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL STYLES
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ASD Screening Dashboard",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for premium aesthetics
st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Root Variables ── */
    :root {
        --violet-400: #A78BFA;
        --violet-500: #8B5CF6;
        --violet-600: #7C3AED;
        --violet-700: #6D28D9;
        --emerald-400: #34D399;
        --emerald-500: #10B981;
        --rose-400: #FB7185;
        --rose-500: #F43F5E;
        --amber-400: #FBBF24;
        --sky-400: #38BDF8;
        --slate-100: #F1F5F9;
        --slate-200: #E2E8F0;
        --slate-300: #CBD5E1;
        --slate-400: #94A3B8;
        --slate-700: #334155;
        --slate-800: #1E293B;
        --slate-900: #0F172A;
        --bg-dark: #0F0F1A;
        --bg-card: #1A1A2E;
        --bg-card-hover: #232340;
    }

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1A1A2E 0%, #16162A 50%, #0F0F1A 100%);
        border-right: 1px solid rgba(124, 58, 237, 0.15);
    }
    section[data-testid="stSidebar"] .stRadio label {
        padding: 0.5rem 0.75rem;
        border-radius: 8px;
        transition: all 0.2s ease;
        margin-bottom: 2px;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(124, 58, 237, 0.12);
    }

    /* ── Metric Cards ── */
    .metric-card {
        background: linear-gradient(145deg, #1A1A2E 0%, #232340 100%);
        border: 1px solid rgba(124, 58, 237, 0.2);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, var(--violet-500), var(--emerald-400));
        border-radius: 16px 16px 0 0;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(124, 58, 237, 0.4);
        box-shadow: 0 8px 30px rgba(124, 58, 237, 0.15);
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--violet-400), var(--emerald-400));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0.25rem 0;
    }
    .metric-label {
        font-size: 0.85rem;
        color: var(--slate-400);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-weight: 600;
    }

    /* ── Hero Banner ── */
    .hero {
        background: linear-gradient(135deg, rgba(124, 58, 237, 0.15) 0%, rgba(16, 185, 129, 0.1) 50%, rgba(56, 189, 248, 0.08) 100%);
        border: 1px solid rgba(124, 58, 237, 0.2);
        border-radius: 20px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        position: relative;
        overflow: hidden;
    }
    .hero::after {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(124, 58, 237, 0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .hero h1 {
        font-size: 2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(135deg, #E2E8F0, #A78BFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }
    .hero p {
        color: var(--slate-400);
        font-size: 1rem;
        line-height: 1.6;
        max-width: 700px;
    }

    /* ── Prediction Result Cards ── */
    .prediction-positive {
        background: linear-gradient(145deg, rgba(244, 63, 94, 0.12) 0%, rgba(244, 63, 94, 0.05) 100%);
        border: 1px solid rgba(244, 63, 94, 0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .prediction-negative {
        background: linear-gradient(145deg, rgba(16, 185, 129, 0.12) 0%, rgba(16, 185, 129, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
    }
    .result-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .result-text {
        font-size: 1.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .result-prob {
        font-size: 0.95rem;
        color: var(--slate-400);
    }

    /* ── Section Headers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.75rem;
        border-bottom: 1px solid rgba(124, 58, 237, 0.15);
    }
    .section-header h2 {
        font-size: 1.3rem;
        font-weight: 700;
        color: var(--slate-200);
        margin: 0;
    }

    /* ── Question Cards ── */
    .question-card {
        background: linear-gradient(145deg, #1A1A2E 0%, #1E1E35 100%);
        border: 1px solid rgba(124, 58, 237, 0.12);
        border-radius: 12px;
        padding: 1rem 1.25rem;
        margin-bottom: 0.5rem;
        transition: border-color 0.2s ease;
    }
    .question-card:hover {
        border-color: rgba(124, 58, 237, 0.3);
    }
    .question-number {
        display: inline-block;
        background: linear-gradient(135deg, var(--violet-600), var(--violet-500));
        color: white;
        font-weight: 700;
        font-size: 0.75rem;
        padding: 0.15rem 0.55rem;
        border-radius: 6px;
        margin-right: 0.5rem;
    }
    .question-text {
        color: var(--slate-200);
        font-size: 0.95rem;
        font-weight: 500;
    }

    /* ── Info Callout ── */
    .info-callout {
        background: linear-gradient(145deg, rgba(56, 189, 248, 0.08) 0%, rgba(56, 189, 248, 0.03) 100%);
        border-left: 3px solid var(--sky-400);
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }
    .info-callout p {
        color: var(--slate-300);
        font-size: 0.88rem;
        margin: 0;
    }

    /* ── Warning Callout ── */
    .warning-callout {
        background: linear-gradient(145deg, rgba(251, 191, 36, 0.08) 0%, rgba(251, 191, 36, 0.03) 100%);
        border-left: 3px solid var(--amber-400);
        border-radius: 0 12px 12px 0;
        padding: 1rem 1.25rem;
        margin: 1rem 0;
    }
    .warning-callout p {
        color: var(--slate-300);
        font-size: 0.88rem;
        margin: 0;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 0.5rem 1.25rem;
        font-weight: 600;
        color: var(--slate-400);
        background: rgba(124, 58, 237, 0.05);
        border: 1px solid rgba(124, 58, 237, 0.1);
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--slate-200);
        background: rgba(124, 58, 237, 0.12);
    }
    .stTabs [aria-selected="true"] {
        background: rgba(124, 58, 237, 0.2) !important;
        color: var(--violet-400) !important;
        border-color: rgba(124, 58, 237, 0.4) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        background: linear-gradient(135deg, var(--violet-600), var(--violet-500));
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.65rem 2rem;
        font-weight: 600;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(124, 58, 237, 0.25);
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.35);
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        background: var(--bg-card);
        border-radius: 12px;
        font-weight: 600;
    }

    /* ── Divider ── */
    hr {
        border: none;
        border-top: 1px solid rgba(124, 58, 237, 0.12);
        margin: 1.5rem 0;
    }

    /* ── Footer ── */
    .footer {
        text-align: center;
        color: var(--slate-400);
        font-size: 0.8rem;
        padding: 2rem 0 1rem 0;
        border-top: 1px solid rgba(124, 58, 237, 0.1);
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# DATA & MODEL LOADING (cached)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("best_model.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_resource
def load_encoders():
    with open("encoders.pkl", "rb") as f:
        return pickle.load(f)

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    return df

@st.cache_data
def prepare_evaluation_data():
    """Reproduce the notebook pipeline to get test-set metrics."""
    df = pd.read_csv("train.csv")

    # Clean data (same steps as notebook)
    df["ethnicity"] = df["ethnicity"].replace({"?": "Others", "others": "Others"})
    df["relation"] = df["relation"].replace({
        "?": "Others", "Relative": "Others",
        "Parent": "Others", "Health care professional": "Others"
    })
    df.drop(columns=["age_desc", "ID"], inplace=True)
    df["age"] = df["age"].astype(int)

    # Encode categorical columns
    object_columns = df.select_dtypes(include=["object"]).columns
    local_encoders = {}
    for column in object_columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column])
        local_encoders[column] = le

    # Outlier replacement
    def replace_outliers(df, col):
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        med = df[col].median()
        df[col] = df[col].apply(lambda x: med if x < lo or x > hi else x)
        return df

    df = replace_outliers(df, "age")
    df = replace_outliers(df, "result")

    X = df.drop(columns=["Class/ASD"])
    y = df["Class/ASD"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if HAS_SMOTE:
        smote = SMOTE(random_state=42)
        X_train_s, y_train_s = smote.fit_resample(X_train, y_train)
    else:
        X_train_s, y_train_s = X_train, y_train

    return X_train_s, X_test, y_train_s, y_test, df, local_encoders


model = load_model()
encoders = load_encoders()
raw_df = load_data()
X_train_s, X_test, y_train_s, y_test, clean_df, local_encoders = prepare_evaluation_data()

# Feature importance
feature_importance = pd.DataFrame({
    "Feature": model.feature_names_in_,
    "Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1rem 0 0.5rem 0;">
        <span style="font-size:2.5rem;">🧩</span>
        <h2 style="margin:0.25rem 0 0 0; font-weight:800;
            background: linear-gradient(135deg, #A78BFA, #34D399);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
            background-clip: text; font-size:1.3rem;">
            ASD Screening
        </h2>
        <p style="color:#94A3B8; font-size:0.78rem; margin-top:0.2rem;">
            ML-Powered Analysis Dashboard
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠  Overview", "🔮  Predict", "📊  Data Explorer", "🎯  Model Performance", "ℹ️  About"],
        label_visibility="collapsed",
    )

    st.markdown("---")

    st.markdown("""
    <div style="padding: 0.75rem; background: rgba(124,58,237,0.08);
        border-radius: 12px; border: 1px solid rgba(124,58,237,0.15);">
        <p style="color:#94A3B8; font-size:0.75rem; margin:0;">
            <strong style="color:#A78BFA;">Model:</strong> Random Forest<br>
            <strong style="color:#A78BFA;">CV Accuracy:</strong> 93%<br>
            <strong style="color:#A78BFA;">Test Accuracy:</strong> 81.9%<br>
            <strong style="color:#A78BFA;">Dataset:</strong> 800 records
        </p>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────────────────
def metric_card(label, value, icon=""):
    return f"""
    <div class="metric-card">
        <div style="font-size:1.5rem; margin-bottom:0.25rem;">{icon}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """

def plotly_theme(fig, height=450):
    """Apply consistent dark theme to Plotly figures."""
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", color="#CBD5E1"),
        height=height,
        margin=dict(l=40, r=40, t=50, b=40),
        title_font=dict(size=16, color="#E2E8F0"),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(124,58,237,0.2)",
            borderwidth=1,
            font=dict(size=11)
        ),
        xaxis=dict(gridcolor="rgba(124,58,237,0.08)", zerolinecolor="rgba(124,58,237,0.08)"),
        yaxis=dict(gridcolor="rgba(124,58,237,0.08)", zerolinecolor="rgba(124,58,237,0.08)"),
    )
    return fig


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: OVERVIEW
# ──────────────────────────────────────────────────────────────────────────────
if page == "🏠  Overview":

    # Hero
    st.markdown("""
    <div class="hero">
        <h1>🧩 ASD Screening Analysis Dashboard</h1>
        <p>
            Machine-learning powered autism screening prediction based on the AQ-10
            questionnaire and demographic features. Explore the data, understand the
            model, and run live predictions — all in one place.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # KPI row
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    recall_asd = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    precision_asd = cm[1, 1] / (cm[0, 1] + cm[1, 1])

    cols = st.columns(4)
    cards = [
        ("Test Accuracy",  f"{acc:.1%}", "🎯"),
        ("ASD Recall",     f"{recall_asd:.1%}", "🔍"),
        ("ASD Precision",  f"{precision_asd:.1%}", "✅"),
        ("Dataset Size",   f"{len(raw_df):,}", "📦"),
    ]
    for col, (label, value, icon) in zip(cols, cards):
        col.markdown(metric_card(label, value, icon), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Two-column layout
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown('<div class="section-header"><h2>📊 Feature Importance (Top 10)</h2></div>', unsafe_allow_html=True)
        top10 = feature_importance.head(10).sort_values("Importance", ascending=True)
        fig = px.bar(
            top10, x="Importance", y="Feature", orientation="h",
            color="Importance",
            color_continuous_scale=["#7C3AED", "#A78BFA", "#34D399"],
        )
        fig.update_coloraxes(showscale=False)
        fig = plotly_theme(fig, height=400)
        fig.update_layout(title="")
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown('<div class="section-header"><h2>🎯 Test Set Confusion Matrix</h2></div>', unsafe_allow_html=True)
        labels = ["No ASD", "ASD"]
        fig = go.Figure(data=go.Heatmap(
            z=cm[::-1],
            x=labels, y=labels[::-1],
            text=cm[::-1], texttemplate="%{text}",
            textfont=dict(size=20, color="white"),
            colorscale=[[0, "#1A1A2E"], [0.5, "#6D28D9"], [1, "#A78BFA"]],
            showscale=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
        ))
        fig = plotly_theme(fig, height=400)
        fig.update_layout(
            xaxis_title="Predicted", yaxis_title="Actual",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Class distribution & Age distribution
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header"><h2>⚖️ Class Distribution</h2></div>', unsafe_allow_html=True)
        class_counts = raw_df["Class/ASD"].value_counts()
        fig = px.pie(
            names=["No ASD (0)", "ASD (1)"],
            values=class_counts.values,
            color_discrete_sequence=["#6D28D9", "#34D399"],
            hole=0.55,
        )
        fig.update_traces(textinfo="percent+value", textfont_size=14)
        fig = plotly_theme(fig, height=350)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header"><h2>📈 Age Distribution</h2></div>', unsafe_allow_html=True)
        fig = px.histogram(
            raw_df, x="age", nbins=30,
            color_discrete_sequence=["#7C3AED"],
            opacity=0.8,
        )
        fig.update_traces(marker_line_width=0.5, marker_line_color="#A78BFA")
        fig = plotly_theme(fig, height=350)
        fig.update_layout(xaxis_title="Age", yaxis_title="Count")
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: PREDICT
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🔮  Predict":

    st.markdown("""
    <div class="hero">
        <h1>🔮 ASD Screening Prediction</h1>
        <p>
            Answer the 10 AQ-10 screening questions and provide demographic details
            to receive a machine-learning powered screening result. Results appear
            instantly after submission.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="warning-callout">
        <p>⚠️ <strong>Disclaimer:</strong> This tool is for educational and research
        purposes only. It is <strong>not</strong> a medical diagnostic tool. A proper
        ASD diagnosis requires clinical evaluation by qualified professionals.</p>
    </div>
    """, unsafe_allow_html=True)

    # AQ-10 Questions mapping
    aq10_questions = {
        "A1_Score": "I often notice small sounds when others do not.",
        "A2_Score": "I usually concentrate more on the whole picture, rather than the small details.",
        "A3_Score": "I find it easy to do more than one thing at once.",
        "A4_Score": "If there is an interruption, I can switch back to what I was doing very quickly.",
        "A5_Score": "I find it easy to 'read between the lines' when someone is talking to me.",
        "A6_Score": "I know how to tell if someone listening to me is getting bored.",
        "A7_Score": "When I'm reading a story, I find it difficult to work out the characters' intentions.",
        "A8_Score": "I like to collect information about categories of things.",
        "A9_Score": "I find it easy to work out what someone is thinking or feeling just by looking at their face.",
        "A10_Score": "I find it difficult to work out people's intentions.",
    }

    st.markdown('<div class="section-header"><h2>📝 AQ-10 Screening Questions</h2></div>', unsafe_allow_html=True)

    aq_scores = {}
    cols_q = st.columns(2)
    for idx, (key, question) in enumerate(aq10_questions.items()):
        with cols_q[idx % 2]:
            st.markdown(f"""
            <div class="question-card">
                <span class="question-number">Q{idx+1}</span>
                <span class="question-text">{question}</span>
            </div>
            """, unsafe_allow_html=True)
            aq_scores[key] = st.selectbox(
                f"Q{idx+1}", options=[0, 1],
                format_func=lambda x: "✅ Agree" if x == 1 else "❌ Disagree",
                key=key, label_visibility="collapsed"
            )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header"><h2>👤 Demographic Information</h2></div>', unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)
    with d1:
        age = st.number_input("Age", min_value=1, max_value=100, value=25, step=1)
        gender = st.selectbox("Gender", options=["f", "m"], format_func=lambda x: "Female" if x == "f" else "Male")
        ethnicity = st.selectbox("Ethnicity", options=list(encoders["ethnicity"].classes_))

    with d2:
        jaundice = st.selectbox("Born with Jaundice?", options=["no", "yes"], format_func=str.capitalize)
        austim = st.selectbox("Family member with ASD?", options=["no", "yes"], format_func=str.capitalize)
        country = st.selectbox("Country of Residence", options=list(encoders["contry_of_res"].classes_))

    with d3:
        used_app = st.selectbox("Used screening app before?", options=["no", "yes"], format_func=str.capitalize)
        relation = st.selectbox("Who is completing this?", options=["Self", "Others"])
        result = st.number_input("Screening Result Score", min_value=-10.0, max_value=20.0, value=5.0, step=0.5,
                                 help="Combined screening result score from the AQ-10 test")

    st.markdown("<br>", unsafe_allow_html=True)

    # Predict
    predict_col = st.columns([1, 2, 1])
    with predict_col[1]:
        predict_btn = st.button("🔮  Run Prediction", use_container_width=True)

    if predict_btn:
        # Encode inputs
        try:
            input_data = {}
            for k, v in aq_scores.items():
                input_data[k] = v

            input_data["age"] = int(age)
            input_data["gender"] = int(encoders["gender"].transform([gender])[0])
            input_data["ethnicity"] = int(encoders["ethnicity"].transform([ethnicity])[0])
            input_data["jaundice"] = int(encoders["jaundice"].transform([jaundice])[0])
            input_data["austim"] = int(encoders["austim"].transform([austim])[0])
            input_data["contry_of_res"] = int(encoders["contry_of_res"].transform([country])[0])
            input_data["used_app_before"] = int(encoders["used_app_before"].transform([used_app])[0])
            input_data["result"] = float(result)
            input_data["relation"] = int(encoders["relation"].transform([relation])[0])

            # Build DataFrame in feature order
            feature_order = list(model.feature_names_in_)
            input_df = pd.DataFrame([input_data])[feature_order]

            prediction = model.predict(input_df)[0]
            probabilities = model.predict_proba(input_df)[0]

            st.markdown("<br>", unsafe_allow_html=True)

            r1, r2 = st.columns([2, 1])
            with r1:
                if prediction == 1:
                    st.markdown(f"""
                    <div class="prediction-positive">
                        <div class="result-icon">⚠️</div>
                        <div class="result-text" style="color: #FB7185;">ASD Traits Detected</div>
                        <div class="result-prob">
                            Confidence: {probabilities[1]:.1%} probability of ASD traits<br>
                            <span style="font-size:0.8rem; color:#94A3B8;">
                                The model suggests this screening profile is consistent with ASD traits.
                                Please consult a healthcare professional for proper evaluation.
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="prediction-negative">
                        <div class="result-icon">✅</div>
                        <div class="result-text" style="color: #34D399;">No ASD Traits Detected</div>
                        <div class="result-prob">
                            Confidence: {probabilities[0]:.1%} probability of no ASD traits<br>
                            <span style="font-size:0.8rem; color:#94A3B8;">
                                The model does not detect ASD traits in this screening profile.
                                If concerns persist, consider consulting a professional.
                            </span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with r2:
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=probabilities[1] * 100,
                    title={"text": "ASD Probability", "font": {"size": 14, "color": "#CBD5E1"}},
                    number={"suffix": "%", "font": {"size": 28, "color": "#E2E8F0"}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#94A3B8"},
                        "bar": {"color": "#7C3AED"},
                        "bgcolor": "#1A1A2E",
                        "borderwidth": 0,
                        "steps": [
                            {"range": [0, 40], "color": "rgba(16,185,129,0.15)"},
                            {"range": [40, 65], "color": "rgba(251,191,36,0.15)"},
                            {"range": [65, 100], "color": "rgba(244,63,94,0.15)"},
                        ],
                        "threshold": {
                            "line": {"color": "#F43F5E", "width": 3},
                            "thickness": 0.8, "value": 50
                        }
                    }
                ))
                fig = plotly_theme(fig, height=280)
                st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Prediction error: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: DATA EXPLORER
# ──────────────────────────────────────────────────────────────────────────────
elif page == "📊  Data Explorer":

    st.markdown("""
    <div class="hero">
        <h1>📊 Data Explorer</h1>
        <p>
            Explore the ASD screening dataset interactively — distributions,
            correlations, and patterns across all features.
        </p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📋 Dataset Overview", "📈 Distributions", "🔗 Correlations", "🔬 Feature Analysis"
    ])

    # ── Tab 1: Dataset Overview ──
    with tab1:
        st.markdown('<div class="section-header"><h2>📋 Raw Dataset</h2></div>', unsafe_allow_html=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.markdown(metric_card("Rows", str(raw_df.shape[0]), "📄"), unsafe_allow_html=True)
        c2.markdown(metric_card("Columns", str(raw_df.shape[1]), "📐"), unsafe_allow_html=True)
        c3.markdown(metric_card("ASD Cases", str(raw_df["Class/ASD"].sum()), "🧩"), unsafe_allow_html=True)
        c4.markdown(metric_card("Non-ASD", str((raw_df["Class/ASD"] == 0).sum()), "👥"), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(raw_df, use_container_width=True, height=400)

        st.markdown("<br>", unsafe_allow_html=True)
        with st.expander("📊 Statistical Summary"):
            st.dataframe(raw_df.describe(), use_container_width=True)

    # ── Tab 2: Distributions ──
    with tab2:
        st.markdown('<div class="section-header"><h2>📈 Feature Distributions</h2></div>', unsafe_allow_html=True)

        col_sel = st.selectbox(
            "Select feature to visualize",
            ["age", "result", "gender", "ethnicity", "jaundice", "austim",
             "contry_of_res", "used_app_before", "relation", "Class/ASD"]
        )

        left, right = st.columns(2)

        with left:
            if raw_df[col_sel].dtype in [np.float64, np.int64, float, int] and raw_df[col_sel].nunique() > 10:
                fig = px.histogram(
                    raw_df, x=col_sel, nbins=30,
                    color_discrete_sequence=["#7C3AED"], opacity=0.85,
                    marginal="box",
                    title=f"Distribution of {col_sel}"
                )
            else:
                counts = raw_df[col_sel].value_counts().reset_index()
                counts.columns = [col_sel, "count"]
                fig = px.bar(
                    counts, x=col_sel, y="count",
                    color_discrete_sequence=["#7C3AED"],
                    title=f"Value Counts — {col_sel}"
                )
            fig = plotly_theme(fig, 400)
            st.plotly_chart(fig, use_container_width=True)

        with right:
            # Distribution split by ASD class
            if raw_df[col_sel].dtype in [np.float64, np.int64, float, int] and raw_df[col_sel].nunique() > 10:
                fig = px.histogram(
                    raw_df, x=col_sel, color="Class/ASD", barmode="overlay",
                    color_discrete_map={0: "#6D28D9", 1: "#34D399"},
                    opacity=0.7, nbins=30,
                    title=f"{col_sel} by ASD Class"
                )
            else:
                fig = px.histogram(
                    raw_df, x=col_sel, color="Class/ASD", barmode="group",
                    color_discrete_map={0: "#6D28D9", 1: "#34D399"},
                    title=f"{col_sel} by ASD Class"
                )
            fig = plotly_theme(fig, 400)
            st.plotly_chart(fig, use_container_width=True)

    # ── Tab 3: Correlations ──
    with tab3:
        st.markdown('<div class="section-header"><h2>🔗 Correlation Heatmap</h2></div>', unsafe_allow_html=True)

        numeric_df = clean_df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()

        fig = go.Figure(data=go.Heatmap(
            z=corr.values,
            x=corr.columns, y=corr.columns,
            text=np.round(corr.values, 2), texttemplate="%{text}",
            textfont=dict(size=8),
            colorscale=[[0, "#1A1A2E"], [0.25, "#6D28D9"], [0.5, "#A78BFA"],
                         [0.75, "#34D399"], [1, "#10B981"]],
            hovertemplate="X: %{x}<br>Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>"
        ))
        fig = plotly_theme(fig, height=700)
        fig.update_layout(
            title="Feature Correlation Matrix",
            xaxis=dict(tickangle=-45)
        )
        st.plotly_chart(fig, use_container_width=True)

        # Correlation with target
        st.markdown('<div class="section-header"><h2>🎯 Correlation with ASD Target</h2></div>', unsafe_allow_html=True)
        target_corr = corr["Class/ASD"].drop("Class/ASD").sort_values(ascending=True)
        fig = px.bar(
            x=target_corr.values, y=target_corr.index,
            orientation="h",
            color=target_corr.values,
            color_continuous_scale=["#F43F5E", "#FBBF24", "#10B981"],
            labels={"x": "Correlation", "y": "Feature"},
            title="Correlation with Class/ASD"
        )
        fig.update_coloraxes(showscale=False)
        fig = plotly_theme(fig, height=500)
        st.plotly_chart(fig, use_container_width=True)

    # ── Tab 4: Feature Analysis ──
    with tab4:
        st.markdown('<div class="section-header"><h2>🔬 AQ-10 Score Analysis</h2></div>', unsafe_allow_html=True)

        aq_cols = [f"A{i}_Score" for i in range(1, 11)]

        # AQ scores by ASD class
        aq_means = raw_df.groupby("Class/ASD")[aq_cols].mean().T.reset_index()
        aq_means.columns = ["Question", "No ASD", "ASD"]
        aq_means_melted = aq_means.melt(id_vars="Question", var_name="Class", value_name="Mean Score")

        fig = px.bar(
            aq_means_melted, x="Question", y="Mean Score", color="Class",
            barmode="group",
            color_discrete_map={"No ASD": "#6D28D9", "ASD": "#34D399"},
            title="Mean AQ-10 Scores by ASD Class"
        )
        fig = plotly_theme(fig, 420)
        st.plotly_chart(fig, use_container_width=True)

        # Gender & Jaundice breakdown
        col_a, col_b = st.columns(2)
        with col_a:
            gender_asd = raw_df.groupby(["gender", "Class/ASD"]).size().reset_index(name="Count")
            gender_asd["Class/ASD"] = gender_asd["Class/ASD"].map({0: "No ASD", 1: "ASD"})
            fig = px.bar(
                gender_asd, x="gender", y="Count", color="Class/ASD",
                barmode="group",
                color_discrete_map={"No ASD": "#6D28D9", "ASD": "#34D399"},
                title="Gender vs ASD"
            )
            fig = plotly_theme(fig, 350)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            jaundice_asd = raw_df.groupby(["jaundice", "Class/ASD"]).size().reset_index(name="Count")
            jaundice_asd["Class/ASD"] = jaundice_asd["Class/ASD"].map({0: "No ASD", 1: "ASD"})
            fig = px.bar(
                jaundice_asd, x="jaundice", y="Count", color="Class/ASD",
                barmode="group",
                color_discrete_map={"No ASD": "#6D28D9", "ASD": "#34D399"},
                title="Jaundice History vs ASD"
            )
            fig = plotly_theme(fig, 350)
            st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ──────────────────────────────────────────────────────────────────────────────
elif page == "🎯  Model Performance":

    st.markdown("""
    <div class="hero">
        <h1>🎯 Model Performance</h1>
        <p>
            Detailed evaluation metrics, ROC curves, and model comparison for the
            tuned Random Forest classifier on the held-out test set.
        </p>
    </div>
    """, unsafe_allow_html=True)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["No ASD", "ASD"], output_dict=True)

    # KPI row
    cols = st.columns(5)
    kpis = [
        ("Accuracy",       f"{acc:.1%}",                          "🎯"),
        ("Precision (ASD)", f"{report['ASD']['precision']:.1%}",  "✅"),
        ("Recall (ASD)",    f"{report['ASD']['recall']:.1%}",     "🔍"),
        ("F1 Score (ASD)",  f"{report['ASD']['f1-score']:.1%}",   "⚡"),
        ("Support",         f"{int(report['ASD']['support'])}",   "📊"),
    ]
    for col, (label, value, icon) in zip(cols, kpis):
        col.markdown(metric_card(label, value, icon), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Charts row
    c_left, c_right = st.columns(2)

    with c_left:
        st.markdown('<div class="section-header"><h2>🎯 Confusion Matrix</h2></div>', unsafe_allow_html=True)
        labels = ["No ASD", "ASD"]
        fig = go.Figure(data=go.Heatmap(
            z=cm[::-1], x=labels, y=labels[::-1],
            text=cm[::-1], texttemplate="%{text}",
            textfont=dict(size=24, color="white"),
            colorscale=[[0, "#1A1A2E"], [0.5, "#6D28D9"], [1, "#A78BFA"]],
            showscale=False,
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
        ))
        fig = plotly_theme(fig, height=400)
        fig.update_layout(xaxis_title="Predicted Label", yaxis_title="Actual Label")
        st.plotly_chart(fig, use_container_width=True)

    with c_right:
        st.markdown('<div class="section-header"><h2>📈 ROC Curve</h2></div>', unsafe_allow_html=True)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines",
            name=f"ROC (AUC = {roc_auc:.3f})",
            line=dict(color="#7C3AED", width=3),
            fill="tozeroy",
            fillcolor="rgba(124,58,237,0.1)"
        ))
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random (AUC = 0.5)",
            line=dict(color="#94A3B8", dash="dash", width=1.5)
        ))
        fig = plotly_theme(fig, height=400)
        fig.update_layout(
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Precision-Recall Curve & Classification Report
    c_left2, c_right2 = st.columns(2)

    with c_left2:
        st.markdown('<div class="section-header"><h2>📊 Precision-Recall Curve</h2></div>', unsafe_allow_html=True)
        precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recall_vals, y=precision_vals, mode="lines",
            name="Precision-Recall",
            line=dict(color="#34D399", width=3),
            fill="tozeroy",
            fillcolor="rgba(16,185,129,0.1)"
        ))
        fig = plotly_theme(fig, height=400)
        fig.update_layout(
            xaxis_title="Recall", yaxis_title="Precision",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c_right2:
        st.markdown('<div class="section-header"><h2>📋 Classification Report</h2></div>', unsafe_allow_html=True)
        report_df = pd.DataFrame(report).T
        report_df = report_df.round(3)
        st.dataframe(report_df, use_container_width=True, height=220)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
        <div class="info-callout">
            <p>
                📌 <strong>Key insight:</strong> ASD recall is 64% — the model catches ~2 out of every 3
                actual ASD cases. In a screening context, improving recall (at the cost of precision) would
                be preferable since missing a real case is worse than a false alarm.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Feature importance
    st.markdown('<div class="section-header"><h2>🏆 Feature Importance</h2></div>', unsafe_allow_html=True)
    fig = px.bar(
        feature_importance, x="Feature", y="Importance",
        color="Importance",
        color_continuous_scale=["#1A1A2E", "#6D28D9", "#A78BFA", "#34D399"],
        title="All Features by Importance"
    )
    fig.update_coloraxes(showscale=False)
    fig = plotly_theme(fig, height=400)
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)

    # Cross-validation comparison
    st.markdown('<div class="section-header"><h2>🏅 Model Comparison (5-Fold CV)</h2></div>', unsafe_allow_html=True)
    cv_data = pd.DataFrame({
        "Model": ["Decision Tree", "Random Forest", "XGBoost", "Random Forest (Tuned)"],
        "CV Accuracy (%)": [86, 92, 90, 93],
        "Status": ["Baseline", "Strong", "Good", "✅ Best"]
    })
    fig = px.bar(
        cv_data, x="Model", y="CV Accuracy (%)",
        color="CV Accuracy (%)",
        color_continuous_scale=["#6D28D9", "#A78BFA", "#34D399"],
        text="CV Accuracy (%)",
        title=""
    )
    fig.update_traces(textposition="outside", textfont_size=14)
    fig.update_coloraxes(showscale=False)
    fig = plotly_theme(fig, height=380)
    fig.update_layout(yaxis_range=[80, 97])
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────────────────────────────────────
# PAGE: ABOUT
# ──────────────────────────────────────────────────────────────────────────────
elif page == "ℹ️  About":

    st.markdown("""
    <div class="hero">
        <h1>ℹ️ About This Project</h1>
        <p>
            Learn about the methodology, dataset, and the machine learning pipeline
            behind this ASD screening prediction tool.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
        ### 🧩 What This Is

        This project builds a machine learning model that predicts whether someone is likely
        to have **Autism Spectrum Disorder (ASD)** based on a short behavioral screening
        questionnaire (AQ-10) plus a few demographic fields.

        The AQ-10 is a widely used 10-question screening tool. It's **not** a diagnostic
        tool — real ASD diagnosis requires clinical evaluation — but it demonstrates a
        complete ML workflow around a real healthcare dataset.

        ---

        ### 🔧 ML Pipeline

        | Step | Details |
        |------|---------|
        | **Data Cleaning** | Fixed inconsistent values, handled `?` placeholders, converted types |
        | **EDA** | Distribution plots, box plots, correlation heatmap |
        | **Outlier Handling** | IQR method → replaced with median |
        | **Encoding** | LabelEncoder for all categorical columns |
        | **Class Imbalance** | SMOTE oversampling on training set only |
        | **Model Comparison** | Decision Tree, Random Forest, XGBoost (5-fold CV) |
        | **Hyperparameter Tuning** | RandomizedSearchCV across all 3 models |
        | **Final Evaluation** | Accuracy, confusion matrix, classification report on test set |

        ---

        ### 📊 Results Summary

        - **Best Model:** Random Forest (tuned) — **93% CV accuracy**
        - **Test Accuracy:** 81.9% on unseen data
        - **ASD Recall:** 64% — catches ~2 out of 3 actual cases
        - The gap between CV and test scores is expected — CV after SMOTE is optimistic

        ---

        ### 🛠️ Tech Stack

        `Python` · `pandas` · `scikit-learn` · `XGBoost` · `imbalanced-learn (SMOTE)` ·
        `Streamlit` · `Plotly`
        """)

    with col2:
        st.markdown("""
        ### 📁 Project Files

        | File | Purpose |
        |------|---------|
        | `app.py` | Streamlit dashboard |
        | `best_model.pkl` | Trained Random Forest |
        | `encoders.pkl` | Label encoders |
        | `train.csv` | Dataset (800 records) |
        | `requirements.txt` | Dependencies |
        | `Procfile` | Deployment config |
        """)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        ### 💡 Future Improvements

        - **SHAP values** for explainable predictions
        - **Push ASD recall higher** at some cost to precision
        - **Additional models** (SVM, Neural Net)
        - **Cloud deployment** (Streamlit Cloud / Heroku)
        """)

        st.markdown("<br>", unsafe_allow_html=True)

        st.markdown("""
        ### 👤 Built By

        **Gurvinder** — Mining engineering background,
        building out ML and data science skills.
        """)

    st.markdown("""
    <div class="warning-callout">
        <p>
            ⚠️ <strong>Important:</strong> This project is strictly for educational and
            research purposes. It should never be used as a substitute for professional
            medical evaluation or diagnosis.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🧩 ASD Screening Dashboard · Built with Streamlit & Plotly ·
    Model: Random Forest (tuned) · Dataset: AQ-10 Screening (800 records)
</div>
""", unsafe_allow_html=True)
