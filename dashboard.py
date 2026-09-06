
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score
)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="CreditWise AI | Loan Intelligence",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PREMIUM UI
# =========================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f5f7fb;
}

[data-testid="stHeader"] {
    background: rgba(245,247,251,0.90);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0b1f3a 0%, #102d52 100%);
    border-right: 1px solid rgba(255,255,255,0.08);
}

[data-testid="stSidebar"] * {
    color: #eef5ff !important;
}

/* ── Sidebar file uploader — kill white background ── */
[data-testid="stSidebar"] [data-testid="stFileUploader"],
[data-testid="stSidebar"] [data-testid="stFileUploader"] > div,
[data-testid="stSidebar"] [data-testid="stFileUploader"] section,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px dashed rgba(255,255,255,0.15) !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] *,
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * {
    color: #a9bdd6 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] button {
    background: rgba(255,255,255,0.10) !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
    color: #e2e8f0 !important;
    border-radius: 7px !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploader"] small {
    color: #6b8aaa !important;
}

/* ── Sidebar success / info / warning boxes ── */
[data-testid="stSidebar"] [data-testid="stAlert"],
[data-testid="stSidebar"] .stAlert,
[data-testid="stSidebar"] [data-testid="stNotification"] {
    background: rgba(22,163,106,0.15) !important;
    border: 1px solid rgba(22,163,106,0.30) !important;
    border-radius: 10px !important;
    color: #6ee7b7 !important;
}
[data-testid="stSidebar"] [data-testid="stAlert"] *,
[data-testid="stSidebar"] .stAlert * {
    color: #6ee7b7 !important;
}

/* ── System status chips ── */
.info-chip {
    display: inline-block;
    background: rgba(255,255,255,0.08) !important;
    color: #6ee7b7 !important;
    border: 1px solid rgba(110,231,183,0.25) !important;
    border-radius: 999px;
    padding: 5px 11px;
    font-size: 11px;
    font-weight: 700;
    margin: 3px 3px 3px 0;
}

/* ── Sidebar caption text ── */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #6b8aaa !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}

/* ── Sidebar radio label (WORKSPACE heading) ── */
[data-testid="stSidebar"] .stRadio > label {
    color: #6b8aaa !important;
    font-size: 10px !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}

/* ── Sidebar divider ── */
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.08) !important;
}


.brand {
    padding: 8px 0 20px 0;
}

.brand-title {
    font-size: 25px;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.5px;
}

.brand-sub {
    color: #a9bdd6;
    font-size: 12px;
    margin-top: 2px;
}

.hero {
    background: linear-gradient(135deg, #0b1f3a 0%, #123d68 55%, #0e7490 100%);
    border-radius: 20px;
    padding: 30px 34px;
    color: white;
    margin-bottom: 22px;
    box-shadow: 0 12px 35px rgba(16,42,67,.16);
}

.hero h1 {
    font-size: 34px;
    line-height: 1.15;
    margin: 0;
    font-weight: 800;
}

.hero p {
    margin: 9px 0 0;
    color: #d7e7f8;
    font-size: 14px;
}

.hero-badge {
    display: inline-block;
    background: rgba(255,255,255,.12);
    border: 1px solid rgba(255,255,255,.18);
    padding: 6px 11px;
    border-radius: 999px;
    font-size: 11px;
    font-weight: 700;
    margin-bottom: 13px;
}

.section-title {
    font-size: 19px;
    font-weight: 800;
    color: #102a43;
    margin: 22px 0 10px;
}

.section-sub {
    color: #718096;
    font-size: 12px;
    margin-top: -6px;
    margin-bottom: 14px;
}

.kpi {
    background: white;
    border: 1px solid #e7edf5;
    border-radius: 16px;
    padding: 18px 19px;
    min-height: 112px;
    box-shadow: 0 5px 18px rgba(16,42,67,.06);
}

.kpi-label {
    color: #718096;
    font-size: 12px;
    font-weight: 600;
}

.kpi-value {
    color: #102a43;
    font-size: 28px;
    font-weight: 800;
    margin-top: 5px;
}

.kpi-note {
    color: #718096;
    font-size: 11px;
    margin-top: 4px;
}

.card {
    background: white;
    border: 1px solid #e7edf5;
    border-radius: 16px;
    padding: 18px;
    box-shadow: 0 5px 18px rgba(16,42,67,.05);
}

.result-approved {
    background: linear-gradient(135deg, #ecfdf5, #f5fffa);
    border: 1px solid #b7ebd0;
    border-left: 6px solid #16a36a;
    border-radius: 16px;
    padding: 22px;
}

.result-rejected {
    background: linear-gradient(135deg, #fff3f3, #fffafa);
    border: 1px solid #f4c3c3;
    border-left: 6px solid #d64545;
    border-radius: 16px;
    padding: 22px;
}

.result-title {
    font-size: 24px;
    font-weight: 800;
    color: #102a43;
}

.result-text {
    color: #52677d;
    font-size: 13px;
    margin-top: 4px;
}

.info-chip {
    display: inline-block;
    background: #eef5ff;
    color: #24527a;
    border-radius: 999px;
    padding: 6px 10px;
    font-size: 11px;
    font-weight: 700;
    margin: 3px;
}

.footer {
    text-align: center;
    color: #8796a8;
    font-size: 11px;
    padding: 30px 0 10px;
}

div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e7edf5;
    border-radius: 14px;
    padding: 12px;
}

button[kind="primary"] {
    border-radius: 10px;
}

.stProgress > div > div > div > div {
    border-radius: 999px;
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# HELPERS
# =========================================================
def money(value):
    if pd.isna(value):
        return "—"
    value = float(value)
    if abs(value) >= 1_000_000:
        return f"₹{value/1_000_000:.2f}M"
    if abs(value) >= 100_000:
        return f"₹{value/100_000:.1f}L"
    return f"₹{value:,.0f}"


def clean_columns(df):
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.strip()
        .str.replace(" ", "_", regex=False)
    )
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def encode_target(df):
    df = df.copy()

    if "Loan_Approved" not in df.columns:
        return None

    raw = df["Loan_Approved"].astype(str).str.strip().str.lower()

    mapping = {
        "yes": 1, "approved": 1, "approve": 1,
        "true": 1, "1": 1, "y": 1,
        "no": 0, "rejected": 0, "reject": 0,
        "false": 0, "0": 0, "n": 0
    }

    encoded = raw.map(mapping)

    if encoded.isna().any():
        numeric = pd.to_numeric(df["Loan_Approved"], errors="coerce")
        if numeric.notna().all() and numeric.isin([0, 1]).all():
            encoded = numeric

    if encoded.isna().any():
        return None

    df["Approved"] = encoded.astype(int)
    return df


@st.cache_data
def prepare_data(raw):
    df = clean_columns(raw)

    numeric = df.select_dtypes(include=np.number).columns
    categorical = df.select_dtypes(exclude=np.number).columns

    if len(numeric):
        df[numeric] = SimpleImputer(strategy="median").fit_transform(df[numeric])

    if len(categorical):
        df[categorical] = SimpleImputer(strategy="most_frequent").fit_transform(
            df[categorical]
        )

    return df


@st.cache_resource
def train_models(df):
    data = df.copy()

    y = data["Approved"].astype(int)
    X = data.drop(columns=["Loan_Approved", "Approved"], errors="ignore")
    X = X.drop(columns=["Applicant_ID"], errors="ignore")

    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
    categorical_cols = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocess = ColumnTransformer([
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols)
    ], remainder="drop")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1500, random_state=42
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Gaussian Naive Bayes": GaussianNB()
    }

    results = {}

    for name, estimator in models.items():
        pipe = Pipeline([
            ("preprocess", preprocess),
            ("model", estimator)
        ])

        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)

        if hasattr(pipe, "predict_proba"):
            prob = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, prob)
        else:
            auc = np.nan

        results[name] = {
            "model": pipe,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "auc": auc,
            "cm": confusion_matrix(y_test, pred),
            "y_test": y_test,
            "pred": pred
        }

    return results


def default_value(df, col, fallback):
    if col in df.columns:
        try:
            return float(df[col].median())
        except Exception:
            return fallback
    return fallback


def category_options(df, col, fallback):
    if col in df.columns:
        values = (
            df[col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )
        if values:
            return sorted(values)
    return fallback


def predict_applicant(row, model_name, results):
    x = pd.DataFrame([row])
    model = results[model_name]["model"]

    prediction = int(model.predict(x)[0])

    probability = None
    if hasattr(model, "predict_proba"):
        probability = float(model.predict_proba(x)[0][prediction])

    return prediction, probability


def chart_layout(fig, height=340):
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=40, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#334e68"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,
            xanchor="left",
            x=0
        )
    )
    return fig


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-title">💳 CreditWise AI</div>
        <div class="brand-sub">Loan Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload loan dataset",
        type=["csv", "pdf"],
        help="Upload a CSV for analysis, or drag a PDF report to preview it."
    )

    if uploaded is not None:
        if uploaded.name.endswith(".pdf"):
            import base64
            pdf_bytes = uploaded.read()
            b64 = base64.b64encode(pdf_bytes).decode("utf-8")
            st.markdown(
                f'<iframe src="data:application/pdf;base64,{b64}" '
                f'width="100%" height="400px" style="border:none;border-radius:10px;'
                f'background:#0b1f3a"></iframe>',
                unsafe_allow_html=True
            )
            raw_df = None
        else:
            raw_df = pd.read_csv(uploaded)
            st.success("Dataset uploaded")
    else:
        try:
            raw_df = pd.read_csv("loan_approval_data.csv")
            st.success("Local dataset loaded")
        except FileNotFoundError:
            raw_df = None

    st.markdown("---")

    page = st.radio(
        "WORKSPACE",
        [
            "🏠 Executive Dashboard",
            "🔮 AI Loan Predictor",
            "📊 Model Intelligence",
            "🔎 Applicant Explorer"
        ],
        label_visibility="visible"
    )

    st.markdown("---")

    if raw_df is not None:
        st.caption("SYSTEM STATUS")
        st.markdown(
            '<span class="info-chip">● Data connected</span>'
            '<span class="info-chip">● ML ready</span>',
            unsafe_allow_html=True
        )


# =========================================================
# EMPTY STATE
# =========================================================
if raw_df is None and (uploaded is None or not uploaded.name.endswith(".pdf")):
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">CREDITWISE AI</div>
        <h1>Loan Intelligence Dashboard</h1>
        <p>Analyze applications, compare machine-learning models and predict loan approval.</p>
    </div>
    """, unsafe_allow_html=True)

    st.info(
        "Upload your loan approval CSV from the sidebar, or place "
        "`loan_approval_data.csv` in the same folder as this Streamlit app."
    )

    st.markdown("### Expected dataset")
    st.code("Loan_Approved", language="text")
    st.markdown("""
    Your dataset can contain numerical and categorical applicant features.
    The dashboard automatically handles missing values and categorical variables.
    """)
    st.stop()


# =========================================================
# PREPARE + TRAIN
# =========================================================
df = prepare_data(raw_df)
df = encode_target(df)

if df is None:
    st.error(
        "The dataset must contain a valid `Loan_Approved` column "
        "with values such as Yes/No, Approved/Rejected or 1/0."
    )
    st.stop()

if df["Approved"].nunique() < 2:
    st.error("The target column must contain both approved and rejected applications.")
    st.stop()

try:
    model_results = train_models(df)
except Exception as e:
    st.error(f"Model training failed: {e}")
    st.stop()

best_model = max(
    model_results,
    key=lambda name: model_results[name]["f1"]
)

# =========================================================
# EXECUTIVE DASHBOARD
# =========================================================
if page == "🏠 Executive Dashboard":

    st.markdown("""
    <div class="hero">
        <div class="hero-badge">AI-POWERED CREDIT ANALYTICS</div>
        <h1>Loan Intelligence Overview</h1>
        <p>Monitor application volume, approval behavior, credit quality and portfolio patterns.</p>
    </div>
    """, unsafe_allow_html=True)

    # ---------------- FILTERS ----------------
    with st.expander("⚙️ Refine Dashboard", expanded=False):
        f1, f2, f3 = st.columns(3)

        if "Employment_Status" in df.columns:
            emp_values = sorted(df["Employment_Status"].astype(str).unique())
            emp_filter = f1.multiselect(
                "Employment Status",
                emp_values,
                default=emp_values
            )
        else:
            emp_filter = None

        if "Loan_Purpose" in df.columns:
            purpose_values = sorted(df["Loan_Purpose"].astype(str).unique())
            purpose_filter = f2.multiselect(
                "Loan Purpose",
                purpose_values,
                default=purpose_values
            )
        else:
            purpose_filter = None

        if "Credit_Score" in df.columns:
            lo = int(df["Credit_Score"].min())
            hi = int(df["Credit_Score"].max())
            score_filter = f3.slider(
                "Credit Score",
                lo, hi, (lo, hi)
            )
        else:
            score_filter = None

    view = df.copy()

    if emp_filter is not None:
        view = view[view["Employment_Status"].astype(str).isin(emp_filter)]

    if purpose_filter is not None:
        view = view[view["Loan_Purpose"].astype(str).isin(purpose_filter)]

    if score_filter is not None:
        view = view[
            view["Credit_Score"].between(
                score_filter[0], score_filter[1]
            )
        ]

    total = len(view)
    approved = int(view["Approved"].sum())
    rejected = total - approved
    approval_rate = (approved / total * 100) if total else 0

    avg_score = (
        view["Credit_Score"].mean()
        if "Credit_Score" in view.columns and total
        else np.nan
    )

    avg_loan = (
        view["Loan_Amount"].mean()
        if "Loan_Amount" in view.columns and total
        else np.nan
    )

    # ---------------- KPI ROW ----------------
    st.markdown('<div class="section-title">Portfolio Snapshot</div>', unsafe_allow_html=True)

    k1, k2, k3, k4, k5 = st.columns(5)

    with k1:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">TOTAL APPLICATIONS</div>
            <div class="kpi-value">{total:,}</div>
            <div class="kpi-note">Filtered portfolio</div>
        </div>
        """, unsafe_allow_html=True)

    with k2:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">APPROVED</div>
            <div class="kpi-value">{approved:,}</div>
            <div class="kpi-note">Successful applications</div>
        </div>
        """, unsafe_allow_html=True)

    with k3:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">APPROVAL RATE</div>
            <div class="kpi-value">{approval_rate:.1f}%</div>
            <div class="kpi-note">Portfolio approval ratio</div>
        </div>
        """, unsafe_allow_html=True)

    with k4:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">AVG CREDIT SCORE</div>
            <div class="kpi-value">{avg_score:.0f}</div>
            <div class="kpi-note">Applicant quality</div>
        </div>
        """, unsafe_allow_html=True)

    with k5:
        st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">AVG LOAN AMOUNT</div>
            <div class="kpi-value">{money(avg_loan)}</div>
            <div class="kpi-note">Average requested amount</div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------- ROW 1 ----------------
    st.markdown('<div class="section-title">Approval Analytics</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        status_df = pd.DataFrame({
            "Status": ["Approved", "Rejected"],
            "Applications": [approved, rejected]
        })

        fig = px.pie(
            status_df,
            names="Status",
            values="Applications",
            hole=0.65,
            title="Application Outcome"
        )
        fig.update_traces(
            textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>%{value:,} applications<extra></extra>"
        )
        fig.update_layout(showlegend=False)
        chart_layout(fig, 350)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "Credit_Score" in view.columns:
            fig = px.histogram(
                view,
                x="Credit_Score",
                color="Loan_Approved",
                nbins=24,
                opacity=0.80,
                title="Credit Score Distribution",
                labels={"Loan_Approved": "Outcome"}
            )
            fig.update_layout(bargap=0.06)
            chart_layout(fig, 350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Credit_Score is not available in this dataset.")

    # ---------------- ROW 2 ----------------
    c1, c2 = st.columns(2)

    with c1:
        if "Loan_Purpose" in view.columns:
            rate = (
                view.groupby("Loan_Purpose")["Approved"]
                .mean()
                .mul(100)
                .reset_index(name="Approval Rate")
                .sort_values("Approval Rate", ascending=False)
            )

            fig = px.bar(
                rate,
                x="Loan_Purpose",
                y="Approval Rate",
                text="Approval Rate",
                title="Approval Rate by Loan Purpose"
            )
            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )
            fig.update_yaxes(range=[0, 100], ticksuffix="%")
            chart_layout(fig, 350)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "Employment_Status" in view.columns:
            rate = (
                view.groupby("Employment_Status")["Approved"]
                .mean()
                .mul(100)
                .reset_index(name="Approval Rate")
                .sort_values("Approval Rate", ascending=False)
            )

            fig = px.bar(
                rate,
                x="Employment_Status",
                y="Approval Rate",
                text="Approval Rate",
                title="Approval Rate by Employment"
            )
            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )
            fig.update_yaxes(range=[0, 100], ticksuffix="%")
            chart_layout(fig, 350)
            st.plotly_chart(fig, use_container_width=True)

    # ---------------- ROW 3 ----------------
    c1, c2 = st.columns(2)

    with c1:
        if "Loan_Amount" in view.columns:
            fig = px.box(
                view,
                x="Loan_Approved",
                y="Loan_Amount",
                color="Loan_Approved",
                title="Loan Amount vs Approval"
            )
            chart_layout(fig, 350)
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "Credit_Score" in view.columns and "Loan_Amount" in view.columns:
            fig = px.scatter(
                view,
                x="Credit_Score",
                y="Loan_Amount",
                color="Loan_Approved",
                size="Loan_Amount",
                hover_data=[
                    c for c in ["Applicant_ID", "Employment_Status", "Loan_Purpose"]
                    if c in view.columns
                ],
                title="Credit Score vs Loan Amount"
            )
            chart_layout(fig, 350)
            st.plotly_chart(fig, use_container_width=True)


# =========================================================
# AI LOAN PREDICTOR
# =========================================================
elif page == "🔮 AI Loan Predictor":

    st.markdown("""
    <div class="hero">
        <div class="hero-badge">PREDICTIVE AI</div>
        <h1>AI Loan Approval Predictor</h1>
        <p>Enter applicant information and evaluate the approval probability using trained ML models.</p>
    </div>
    """, unsafe_allow_html=True)

    top1, top2 = st.columns([2, 1])

    with top1:
        model_name = st.selectbox(
            "Select prediction model",
            list(model_results.keys()),
            index=list(model_results.keys()).index(best_model)
        )

    with top2:
        st.metric(
            "Selected Model F1",
            f"{model_results[model_name]['f1']*100:.1f}%"
        )

    st.markdown('<div class="section-title">💰 Financial Profile</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        income = st.number_input(
            "Applicant Income",
            min_value=0.0,
            value=default_value(df, "Applicant_Income", 8000),
            step=500.0
        )
        co_income = st.number_input(
            "Coapplicant Income",
            min_value=0.0,
            value=default_value(df, "Coapplicant_Income", 2000),
            step=500.0
        )

    with c2:
        credit_score = st.number_input(
            "Credit Score",
            min_value=300.0,
            max_value=900.0,
            value=default_value(df, "Credit_Score", 680),
            step=1.0
        )
        dti = st.number_input(
            "DTI Ratio",
            min_value=0.0,
            max_value=1.0,
            value=default_value(df, "DTI_Ratio", 0.35),
            step=0.01,
            format="%.2f"
        )

    with c3:
        savings = st.number_input(
            "Savings",
            min_value=0.0,
            value=default_value(df, "Savings", 5000),
            step=500.0
        )
        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=default_value(df, "Loan_Amount", 20000),
            step=500.0
        )

    st.markdown('<div class="section-title">🏦 Loan Profile</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        loan_term = st.number_input(
            "Loan Term (months)",
            min_value=1,
            value=int(default_value(df, "Loan_Term", 48)),
            step=1
        )

    with c2:
        existing_loans = st.number_input(
            "Existing Loans",
            min_value=0,
            value=int(default_value(df, "Existing_Loans", 1)),
            step=1
        )

    with c3:
        collateral = st.number_input(
            "Collateral Value",
            min_value=0.0,
            value=default_value(df, "Collateral_Value", 15000),
            step=500.0
        )

    st.markdown('<div class="section-title">👤 Applicant Profile</div>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=int(default_value(df, "Age", 35))
        )
        dependents = st.number_input(
            "Dependents",
            min_value=0,
            max_value=15,
            value=int(default_value(df, "Dependents", 1))
        )

    with c2:
        gender = st.selectbox(
            "Gender",
            category_options(df, "Gender", ["Male", "Female"])
        )
        marital = st.selectbox(
            "Marital Status",
            category_options(df, "Marital_Status", ["Single", "Married"])
        )

    with c3:
        education = st.selectbox(
            "Education Level",
            category_options(df, "Education_Level", ["Graduate", "Undergraduate"])
        )
        employment = st.selectbox(
            "Employment Status",
            category_options(
                df,
                "Employment_Status",
                ["Salaried", "Self-employed", "Unemployed"]
            )
        )

    c1, c2 = st.columns(2)

    with c1:
        purpose = st.selectbox(
            "Loan Purpose",
            category_options(
                df,
                "Loan_Purpose",
                ["Car", "Education", "Home", "Personal"]
            )
        )

    with c2:
        property_area = st.selectbox(
            "Property Area",
            category_options(
                df,
                "Property_Area",
                ["Rural", "Semiurban", "Urban"]
            )
        )

    employer = None
    if "Employer_Category" in df.columns:
        employer = st.selectbox(
            "Employer Category",
            category_options(
                df,
                "Employer_Category",
                ["MNC", "Government", "Private", "Unemployed"]
            )
        )

    # ---------------- LIVE RISK ----------------
    st.markdown('<div class="section-title">⚡ Live Risk Indicators</div>', unsafe_allow_html=True)

    lti = loan_amount / income if income else 0
    savings_cover = savings / loan_amount if loan_amount else 0
    credit_tier = (
        "Prime" if credit_score >= 720
        else "Near-Prime" if credit_score >= 620
        else "Subprime"
    )

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Credit Tier", credit_tier)
    r2.metric("DTI", f"{dti*100:.1f}%")
    r3.metric("Loan / Income", f"{lti:.2f}×")
    r4.metric("Savings Coverage", f"{savings_cover*100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    if st.button(
        "🔮  ANALYZE LOAN APPLICATION",
        use_container_width=True,
        type="primary"
    ):
        row = {
            "Age": age,
            "Applicant_Income": income,
            "Coapplicant_Income": co_income,
            "Credit_Score": credit_score,
            "DTI_Ratio": dti,
            "Loan_Amount": loan_amount,
            "Loan_Term": loan_term,
            "Existing_Loans": existing_loans,
            "Savings": savings,
            "Collateral_Value": collateral,
            "Dependents": dependents,
            "Gender": gender,
            "Marital_Status": marital,
            "Education_Level": education,
            "Employment_Status": employment,
            "Loan_Purpose": purpose,
            "Property_Area": property_area
        }

        if employer is not None:
            row["Employer_Category"] = employer

        # Fill any additional dataset features automatically.
        for col in df.drop(
            columns=["Loan_Approved", "Approved"],
            errors="ignore"
        ).columns:
            if col not in row and col != "Applicant_ID":
                if pd.api.types.is_numeric_dtype(df[col]):
                    row[col] = float(df[col].median())
                else:
                    row[col] = str(df[col].mode().iloc[0])

        pred, probability = predict_applicant(
            row, model_name, model_results
        )

        confidence = probability * 100 if probability is not None else 0

        if pred == 1:
            st.markdown(f"""
            <div class="result-approved">
                <div class="result-title">✅ Loan Likely Approved</div>
                <div class="result-text">
                    The selected model predicts an approval outcome.
                </div>
                <br>
                <span class="info-chip">Model: {model_name}</span>
                <span class="info-chip">Confidence: {confidence:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-rejected">
                <div class="result-title">❌ Loan Likely Rejected</div>
                <div class="result-text">
                    The selected model predicts a rejection outcome.
                </div>
                <br>
                <span class="info-chip">Model: {model_name}</span>
                <span class="info-chip">Confidence: {confidence:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

        if probability is not None:
            st.markdown("#### Prediction confidence")
            st.progress(min(max(confidence / 100, 0), 1))

        st.caption(
            "This prediction is a machine-learning estimate for analytical/demo use "
            "and should not be treated as a final lending decision."
        )


# =========================================================
# MODEL INTELLIGENCE
# =========================================================
elif page == "📊 Model Intelligence":

    st.markdown("""
    <div class="hero">
        <div class="hero-badge">MODEL MONITORING</div>
        <h1>Model Intelligence</h1>
        <p>Compare classification models across accuracy, precision, recall, F1 and ROC-AUC.</p>
    </div>
    """, unsafe_allow_html=True)

    rows = []

    for name, result in model_results.items():
        rows.append({
            "Model": name,
            "Accuracy": result["accuracy"] * 100,
            "Precision": result["precision"] * 100,
            "Recall": result["recall"] * 100,
            "F1 Score": result["f1"] * 100,
            "ROC-AUC": result["auc"] * 100
        })

    perf = pd.DataFrame(rows)

    k1, k2, k3, k4 = st.columns(4)

    k1.metric(
        "Best Model",
        best_model
    )
    k2.metric(
        "Best Accuracy",
        f"{perf['Accuracy'].max():.1f}%"
    )
    k3.metric(
        "Best F1",
        f"{perf['F1 Score'].max():.1f}%"
    )
    k4.metric(
        "Best ROC-AUC",
        f"{perf['ROC-AUC'].max():.1f}%"
    )

    st.markdown('<div class="section-title">Model Comparison</div>', unsafe_allow_html=True)

    metric_cols = [
        "Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"
    ]

    melted = perf.melt(
        id_vars="Model",
        value_vars=metric_cols,
        var_name="Metric",
        value_name="Score"
    )

    fig = px.bar(
        melted,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text_auto=".1f",
        title="Classification Performance"
    )

    fig.update_yaxes(range=[0, 100], ticksuffix="%")
    chart_layout(fig, 430)
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(
        perf.style.format({
            "Accuracy": "{:.2f}%",
            "Precision": "{:.2f}%",
            "Recall": "{:.2f}%",
            "F1 Score": "{:.2f}%",
            "ROC-AUC": "{:.2f}%"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.markdown('<div class="section-title">Confusion Matrix</div>', unsafe_allow_html=True)

    selected = st.selectbox(
        "Select model",
        list(model_results.keys())
    )

    cm = model_results[selected]["cm"]

    fig = px.imshow(
        cm,
        text_auto=True,
        x=["Rejected", "Approved"],
        y=["Rejected", "Approved"],
        labels={
            "x": "Predicted",
            "y": "Actual",
            "color": "Count"
        },
        title=f"{selected} — Confusion Matrix"
    )

    chart_layout(fig, 420)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""
    <div class="card">
        <b>How to read the matrix</b><br>
        <span style="color:#718096;font-size:12px;">
        Rows represent the actual outcome and columns represent the model prediction.
        A strong classifier aims for high values on the diagonal and low values off the diagonal.
        </span>
    </div>
    """, unsafe_allow_html=True)


# =========================================================
# APPLICANT EXPLORER
# =========================================================
else:

    st.markdown("""
    <div class="hero">
        <div class="hero-badge">DATA EXPLORER</div>
        <h1>Applicant Explorer</h1>
        <p>Search, filter, inspect and export loan application records.</p>
    </div>
    """, unsafe_allow_html=True)

    view = df.drop(columns=["Approved"], errors="ignore").copy()

    c1, c2, c3 = st.columns(3)

    search = c1.text_input(
        "🔍 Search applicant data",
        placeholder="Search any value..."
    )

    status_values = (
        ["All"]
        + sorted(view["Loan_Approved"].astype(str).unique().tolist())
        if "Loan_Approved" in view.columns
        else ["All"]
    )

    status = c2.selectbox(
        "Approval Status",
        status_values
    )

    if "Credit_Score" in view.columns:
        lo = int(view["Credit_Score"].min())
        hi = int(view["Credit_Score"].max())
        score = c3.slider(
            "Credit Score",
            lo, hi, (lo, hi)
        )
        view = view[view["Credit_Score"].between(score[0], score[1])]

    if status != "All" and "Loan_Approved" in view.columns:
        view = view[
            view["Loan_Approved"].astype(str) == status
        ]

    if search:
        mask = view.astype(str).apply(
            lambda col: col.str.contains(
                search,
                case=False,
                na=False,
                regex=False
            )
        ).any(axis=1)
        view = view[mask]

    k1, k2, k3 = st.columns(3)

    k1.metric("Visible Records", f"{len(view):,}")

    if "Credit_Score" in view.columns and len(view):
        k2.metric(
            "Average Credit Score",
            f"{view['Credit_Score'].mean():.0f}"
        )
    else:
        k2.metric("Average Credit Score", "—")

    if "Loan_Amount" in view.columns and len(view):
        k3.metric(
            "Average Loan Amount",
            money(view["Loan_Amount"].mean())
        )
    else:
        k3.metric("Average Loan Amount", "—")

    st.markdown("<br>", unsafe_allow_html=True)

    st.dataframe(
        view,
        use_container_width=True,
        height=500,
        hide_index=True
    )

    csv_data = view.to_csv(index=False).encode("utf-8")

    st.download_button(
        "⬇️ Download Filtered CSV",
        data=csv_data,
        file_name="creditwise_filtered.csv",
        mime="text/csv",
        use_container_width=False
    )


# =========================================================
# FOOTER
# =========================================================
st.markdown("""
<div class="footer">
    CreditWise AI • Loan Approval Analytics • Built with Streamlit + Plotly + Scikit-learn
</div>
""", unsafe_allow_html=True)
