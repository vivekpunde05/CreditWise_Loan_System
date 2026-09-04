import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

# ── Config ─────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="CreditWise Pro", layout="wide",
                   page_icon="💳", initial_sidebar_state="expanded")

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* { font-family: 'Inter', sans-serif !important; }

/* ── App background ── */
.stApp { background: #050a14; }
[data-testid="stAppViewContainer"] { background: #050a14; }
[data-testid="stHeader"] { background: transparent; }
section[data-testid="stSidebar"] { background: #070d1a !important; border-right: 1px solid #1e2d45; }

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem 3rem; max-width: 1600px; }

/* ── Sidebar ── */
.sidebar-brand {
  background: linear-gradient(135deg, #1a56db 0%, #06b6d4 100%);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  font-size: 1.6rem; font-weight: 800; letter-spacing: -0.03em;
}
.sidebar-sub { color: #475569; font-size: .72rem; text-transform: uppercase; letter-spacing: .1em; margin-top: -4px; }
.nav-item {
  display: flex; align-items: center; gap: 10px;
  padding: 10px 14px; border-radius: 10px; margin-bottom: 4px;
  cursor: pointer; transition: all .2s; color: #64748b; font-size: .875rem; font-weight: 500;
}
.nav-item:hover { background: #0f1f35; color: #e2e8f0; }
.nav-item.active { background: linear-gradient(135deg,#1a56db18,#06b6d418); color: #60a5fa; border-left: 3px solid #3b82f6; }
[data-testid="stSidebar"] [data-testid="stRadio"] label { display:none; }
[data-testid="stSidebar"] [data-testid="stRadio"] > div { gap: 2px !important; }
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radio"] {
  background: transparent; border: none; padding: 10px 14px;
  border-radius: 10px; color: #64748b; font-weight: 500; font-size: .875rem;
  transition: all .2s;
}
[data-testid="stSidebar"] [data-testid="stRadio"] div[role="radio"]:hover { background: #0f1f35; color: #e2e8f0; }
[data-testid="stSidebar"] [data-testid="stRadio"] div[aria-checked="true"] {
  background: linear-gradient(135deg,#1a56db22,#06b6d422) !important;
  color: #60a5fa !important; border-left: 3px solid #3b82f6;
}

/* ── Page title ── */
.page-title {
  font-size: 1.75rem; font-weight: 800; color: #f1f5f9;
  letter-spacing: -0.03em; margin-bottom: 0;
}
.page-sub { font-size: .875rem; color: #475569; margin-top: 2px; }

/* ── KPI glass cards ── */
.kpi-card {
  background: linear-gradient(135deg, #0d1b2e 0%, #0a1628 100%);
  border: 1px solid #1e3a5f; border-radius: 16px; padding: 22px 24px;
  position: relative; overflow: hidden; transition: transform .2s, box-shadow .2s;
}
.kpi-card::before {
  content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
  background: linear-gradient(90deg, #3b82f6, #06b6d4);
}
.kpi-card:hover { transform: translateY(-2px); box-shadow: 0 8px 32px #3b82f620; }
.kpi-icon { font-size: 1.5rem; margin-bottom: 10px; }
.kpi-label { font-size: .72rem; color: #64748b; text-transform: uppercase; letter-spacing: .08em; font-weight: 600; }
.kpi-value { font-size: 2.1rem; font-weight: 800; color: #f1f5f9; letter-spacing: -0.03em; line-height: 1.1; margin-top: 4px; }
.kpi-delta { font-size: .78rem; margin-top: 6px; font-weight: 500; }
.kpi-delta.up   { color: #22c55e; }
.kpi-delta.down { color: #ef4444; }
.kpi-delta.neutral { color: #94a3b8; }

/* ── Chart cards ── */
.chart-card {
  background: #0a1628; border: 1px solid #1e3a5f;
  border-radius: 16px; padding: 20px 20px 8px;
}
.chart-title {
  font-size: .95rem; font-weight: 700; color: #e2e8f0;
  letter-spacing: -0.01em; margin-bottom: 14px;
}

/* ── Filter bar ── */
.filter-bar {
  background: #0a1628; border: 1px solid #1e3a5f;
  border-radius: 12px; padding: 16px 20px; margin-bottom: 20px;
}

/* ── Risk boxes ── */
.risk-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 14px; margin-bottom: 18px; }
.risk-box {
  background: #0a1628; border: 1px solid #1e3a5f; border-radius: 14px;
  padding: 18px 16px; text-align: center; position: relative; overflow: hidden;
}
.risk-box::after {
  content: ''; position: absolute; bottom: 0; left: 0; right: 0; height: 3px;
}
.risk-box.green::after  { background: #22c55e; }
.risk-box.yellow::after { background: #f59e0b; }
.risk-box.red::after    { background: #ef4444; }
.risk-label { font-size: .68rem; color: #64748b; text-transform: uppercase; letter-spacing: .08em; font-weight: 600; }
.risk-value { font-size: 1.55rem; font-weight: 800; margin-top: 8px; line-height: 1; }
.risk-desc  { font-size: .72rem; color: #475569; margin-top: 6px; }

/* ── Result banner ── */
.result-banner {
  border-radius: 16px; padding: 28px 32px;
  display: flex; align-items: center; justify-content: space-between;
  margin: 18px 0;
}
.result-banner.approved { background: linear-gradient(135deg, #052e16, #064e3b); border: 1px solid #059669; }
.result-banner.rejected { background: linear-gradient(135deg, #450a0a, #7f1d1d); border: 1px solid #dc2626; }
.result-verdict { font-size: 2rem; font-weight: 800; color: #fff; }
.result-conf    { font-size: 1rem; color: #a3e6cb; margin-top: 4px; }
.result-badge   { font-size: 3rem; }

/* ── Prediction form section headers ── */
.section-pill {
  display: inline-block; background: #1e3a5f22; border: 1px solid #1e3a5f;
  color: #60a5fa; font-size: .72rem; font-weight: 700; text-transform: uppercase;
  letter-spacing: .08em; padding: 4px 12px; border-radius: 20px; margin-bottom: 12px;
}

/* ── Table styling ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ── Tabs ── */
[data-baseweb="tab-list"] { background: #0a1628 !important; border-radius: 12px; padding: 4px; gap: 4px; }
[data-baseweb="tab"] { border-radius: 8px !important; color: #64748b !important; font-weight: 600 !important; }
[aria-selected="true"][data-baseweb="tab"] { background: #1e3a5f !important; color: #60a5fa !important; }

/* ── All widget labels ── */
label, .stSlider label, .stSelectbox label, .stMultiSelect label,
.stNumberInput label, .stTextInput label, .stRadio label,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] span,
div[data-testid="stSlider"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stTextInput"] label {
  color: #cbd5e1 !important; font-size: .82rem !important;
  font-weight: 600 !important; letter-spacing: .01em !important;
}

/* ── Selectbox / multiselect displayed value ── */
[data-baseweb="select"] [data-testid="stMarkdownContainer"] p,
[data-baseweb="select"] span,
[data-baseweb="select"] div { color: #e2e8f0 !important; }
[data-baseweb="select"] { background: #0d1b2e !important; border-color: #1e3a5f !important; }
[data-baseweb="select"] > div { background: #0d1b2e !important; border-color: #1e3a5f !important; }

/* ── Dropdown menu items ── */
[data-baseweb="popover"] li, [data-baseweb="menu"] li,
[role="option"] { color: #e2e8f0 !important; background: #0d1b2e !important; }
[role="option"]:hover { background: #1e3a5f !important; }

/* ── Multiselect tags ── */
[data-baseweb="tag"] { background: #1e3a5f !important; }
[data-baseweb="tag"] span { color: #93c5fd !important; }

/* ── Number input ── */
[data-testid="stNumberInput"] input { background: #0d1b2e !important;
  border-color: #1e3a5f !important; color: #e2e8f0 !important; }

/* ── Text input ── */
[data-testid="stTextInput"] input { background: #0d1b2e !important;
  border-color: #1e3a5f !important; color: #e2e8f0 !important; }

/* ── Slider track & thumb ── */
[data-testid="stSlider"] > div > div > div { background: #1a56db !important; }
[data-testid="stSlider"] [data-testid="stTickBarMin"],
[data-testid="stSlider"] [data-testid="stTickBarMax"] { color: #64748b !important; }

/* ── Slider current value text ── */
[data-testid="stSlider"] p { color: #94a3b8 !important; }

/* ── Radio button labels ── */
[data-testid="stRadio"] div[role="radio"] p,
[data-testid="stRadio"] label p { color: #cbd5e1 !important; }

/* ── Toggle label ── */
[data-testid="stToggle"] p,
[data-testid="stToggle"] label { color: #cbd5e1 !important; }

/* ── Expander header ── */
[data-testid="stExpander"] summary p,
[data-testid="stExpander"] summary svg { color: #cbd5e1 !important; fill: #cbd5e1 !important; }
[data-testid="stExpander"] { border-color: #1e3a5f !important;
  background: #070d1a !important; border-radius: 12px !important; }

/* ── st.metric ── */
[data-testid="stMetric"] label { color: #94a3b8 !important; font-size: .75rem !important; }
[data-testid="stMetricValue"] { color: #f1f5f9 !important; }
[data-testid="stMetricDelta"] { color: #22c55e !important; }

/* ── Tabs ── */
[data-baseweb="tab-list"] { background: #0a1628 !important; border-radius: 12px; padding: 4px; gap: 4px; }
[data-baseweb="tab"] { border-radius: 8px !important; color: #94a3b8 !important; font-weight: 600 !important; }
[aria-selected="true"][data-baseweb="tab"] { background: #1e3a5f !important; color: #60a5fa !important; }
[data-baseweb="tab"] p { color: inherit !important; }

/* ── Dataframe text ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ── Download button ── */
[data-testid="stDownloadButton"] button { background: #0d1b2e !important;
  border: 1px solid #1e3a5f !important; color: #93c5fd !important; border-radius: 8px !important; }

/* ── Caption / markdown text ── */
[data-testid="stCaptionContainer"] p { color: #64748b !important; }
p { color: #cbd5e1; }

/* ── Divider ── */
hr { border-color: #1e3a5f !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #050a14; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly dark theme defaults ─────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#94a3b8", family="Inter"),
    margin=dict(t=24, b=32, l=8, r=8),
    xaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="#1e3a5f", zerolinecolor="#1e3a5f", tickfont=dict(size=11)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    hoverlabel=dict(bgcolor="#0d1b2e", bordercolor="#1e3a5f", font_color="#e2e8f0"),
)
C_GREEN  = "#22c55e"
C_RED    = "#ef4444"
C_BLUE   = "#3b82f6"
C_CYAN   = "#06b6d4"
C_AMBER  = "#f59e0b"
C_PURPLE = "#a855f7"

# ── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("loan_approval_data.csv")
    num = df.select_dtypes("number").columns
    cat = df.select_dtypes("object").columns
    df[num] = SimpleImputer(strategy="mean").fit_transform(df[num])
    df[cat] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat])
    df = df.drop(columns=["Applicant_ID"], errors="ignore")
    df["Approved"] = LabelEncoder().fit_transform(df["Loan_Approved"])
    return df

@st.cache_resource
def build_models(_df):
    cat_cols = ["Employment_Status","Marital_Status","Loan_Purpose",
                "Property_Area","Gender","Employer_Category"]
    d = _df.copy()
    d["Education_Level"] = LabelEncoder().fit_transform(d["Education_Level"])
    d = pd.get_dummies(d, columns=cat_cols, drop_first=True)
    fcols = [c for c in d.columns if c not in ("Loan_Approved","Approved")]
    X, y = d[fcols].values, d["Approved"].values
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42)
    di = fcols.index("DTI_Ratio"); ci = fcols.index("Credit_Score")
    Xfe = np.hstack([Xs,(Xs[:,di]**2).reshape(-1,1),(Xs[:,ci]**2).reshape(-1,1)])
    Xfe_tr, Xfe_te, _, _ = train_test_split(Xfe, y, test_size=0.2, random_state=42)
    out = {}
    for nm, b, f in [
        ("Logistic Regression",
         LogisticRegression(max_iter=1000,random_state=42),
         LogisticRegression(max_iter=1000,random_state=42)),
        ("K-Nearest Neighbors", KNeighborsClassifier(5), KNeighborsClassifier(5)),
        ("Gaussian Naive Bayes", GaussianNB(), GaussianNB()),
    ]:
        for var, clf, tr, te in [("Baseline",b,Xtr,Xte),("Feature Engineered",f,Xfe_tr,Xfe_te)]:
            clf.fit(tr, ytr); p = clf.predict(te)
            out[(nm,var)] = dict(clf=clf,
                accuracy =accuracy_score(yte,p),
                precision=precision_score(yte,p,zero_division=0),
                recall   =recall_score(yte,p,zero_division=0),
                f1       =f1_score(yte,p,zero_division=0),
                cm       =confusion_matrix(yte,p))
    return out, sc, fcols

def run_predict(row, model_name, variant, res, sc, fcols):
    inp = pd.DataFrame([row])
    inp["Education_Level"] = inp["Education_Level"].map({"Graduate":1,"Undergraduate":0}).fillna(0)
    cats = ["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Gender","Employer_Category"]
    inp = pd.get_dummies(inp, columns=cats, drop_first=True)
    for c in fcols:
        if c not in inp.columns: inp[c] = 0
    X = sc.transform(inp[fcols].values)
    if variant == "Feature Engineered":
        di = fcols.index("DTI_Ratio"); ci = fcols.index("Credit_Score")
        X = np.hstack([X,(X[:,di]**2).reshape(-1,1),(X[:,ci]**2).reshape(-1,1)])
    clf = res[(model_name, variant)]["clf"]
    pred = int(clf.predict(X)[0])
    conf = float(clf.predict_proba(X)[0][pred]*100) if hasattr(clf,"predict_proba") else None
    return pred, conf

df = load_data()
model_res, scaler, feat_cols = build_models(df)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-brand">CreditWise</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-sub">Pro Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    page = st.radio("nav", [
        "📊  Overview",
        "📈  Performance",
        "🔍  Predict",
        "🗂  Explorer",
    ], label_visibility="collapsed")

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")

    total_r = len(df); appr_r = int(df["Approved"].sum())
    st.markdown(f"""
    <div style="padding:0 4px">
      <div style="font-size:.68rem;color:#475569;text-transform:uppercase;letter-spacing:.08em;font-weight:600">Dataset</div>
      <div style="color:#94a3b8;font-size:.82rem;margin-top:6px">{total_r:,} applicants</div>
      <div style="color:#94a3b8;font-size:.82rem">{appr_r/total_r*100:.1f}% approval rate</div>
      <div style="color:#94a3b8;font-size:.82rem">3 ML models · 2 variants</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:.68rem;color:#334155;text-align:center">Built by Vivek Punde</div>',
                unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in page:
    st.markdown('<div class="page-title">Portfolio Overview</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Real-time analytics across all loan applications</div>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # ── Filters ──
    with st.expander("⚙️  Filters & Segments", expanded=False):
        f1,f2,f3,f4 = st.columns(4)
        emp_f  = f1.multiselect("Employment", df["Employment_Status"].unique().tolist(),
                                 default=df["Employment_Status"].unique().tolist(), key="ov_emp")
        purp_f = f2.multiselect("Loan Purpose", df["Loan_Purpose"].unique().tolist(),
                                 default=df["Loan_Purpose"].unique().tolist(), key="ov_purp")
        area_f = f3.multiselect("Property Area", df["Property_Area"].unique().tolist(),
                                 default=df["Property_Area"].unique().tolist(), key="ov_area")
        cs_f   = f4.slider("Credit Score", int(df["Credit_Score"].min()),
                            int(df["Credit_Score"].max()),
                            (int(df["Credit_Score"].min()), int(df["Credit_Score"].max())), key="ov_cs")

    fd = df[df["Employment_Status"].isin(emp_f) &
            df["Loan_Purpose"].isin(purp_f) &
            df["Property_Area"].isin(area_f) &
            df["Credit_Score"].between(*cs_f)]

    n, na = len(fd), int(fd["Approved"].sum())
    ar = na/n*100 if n else 0

    # ── KPI cards ──
    k1,k2,k3,k4,k5 = st.columns(5)
    def kcard(col, icon, label, value, delta, ok):
        arrow = "↑" if ok else "↓"
        cls   = "up" if ok else "down"
        col.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-icon">{icon}</div>
          <div class="kpi-label">{label}</div>
          <div class="kpi-value">{value}</div>
          <div class="kpi-delta {cls}">{arrow} {delta}</div>
        </div>""", unsafe_allow_html=True)

    kcard(k1,"📋","Total Applications",f"{n:,}","Full dataset" if n==len(df) else f"{n/len(df)*100:.0f}% of total", True)
    kcard(k2,"✅","Approval Rate",f"{ar:.1f}%","Above 50%" if ar>50 else "Below 50%", ar>50)
    avg_cs = fd["Credit_Score"].mean()
    kcard(k3,"💳","Avg Credit Score",f"{avg_cs:.0f}",
          "Prime" if avg_cs>=720 else "Near-Prime" if avg_cs>=620 else "Subprime", avg_cs>=620)
    avg_dti = fd["DTI_Ratio"].mean()*100
    kcard(k4,"📉","Avg DTI Ratio",f"{avg_dti:.1f}%","Healthy" if avg_dti<=43 else "High Risk", avg_dti<=43)
    avg_inc = fd["Applicant_Income"].mean()
    kcard(k5,"💰","Avg Income",f"${avg_inc:,.0f}","Portfolio avg", True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1 ──
    rc1, rc2 = st.columns([1,2])

    with rc1:
        st.markdown('<div class="chart-card"><div class="chart-title">Approval Split</div>', unsafe_allow_html=True)
        counts = fd["Loan_Approved"].value_counts().reset_index()
        counts.columns = ["Status","Count"]
        fig = px.pie(counts, names="Status", values="Count", hole=0.6,
                     color_discrete_sequence=[C_GREEN, C_RED])
        fig.update_traces(textinfo="percent", pull=[0.04,0], textfont_size=13,
                          marker=dict(line=dict(color="#050a14", width=3)))
        fig.update_layout(**PLOT_LAYOUT, showlegend=True,
                          annotations=[dict(text=f"<b>{ar:.0f}%</b><br>Approved",
                                            x=0.5, y=0.5, font_size=14, font_color="#f1f5f9",
                                            showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with rc2:
        st.markdown('<div class="chart-card"><div class="chart-title">Credit Score Distribution</div>', unsafe_allow_html=True)
        colour_tog = st.toggle("Split by Approval", value=True, key="cs_tog")
        if colour_tog:
            fig = px.histogram(fd, x="Credit_Score", color="Loan_Approved", nbins=40,
                               barmode="overlay", opacity=0.8,
                               color_discrete_sequence=[C_GREEN, C_RED])
        else:
            fig = px.histogram(fd, x="Credit_Score", nbins=40,
                               color_discrete_sequence=[C_BLUE])
        fig.add_vline(x=620, line_dash="dot", line_color=C_AMBER,
                      annotation_text="Near-Prime", annotation_font_color=C_AMBER)
        fig.add_vline(x=720, line_dash="dot", line_color=C_CYAN,
                      annotation_text="Prime", annotation_font_color=C_CYAN)
        fig.update_layout(**PLOT_LAYOUT, legend_title_text="", bargap=0.05)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 2 ──
    rc3, rc4 = st.columns(2)

    with rc3:
        st.markdown('<div class="chart-card"><div class="chart-title">Income vs Loan Amount</div>', unsafe_allow_html=True)
        x_sel = st.selectbox("X axis", ["Applicant_Income","Coapplicant_Income","Savings"], key="sc_x")
        fig = px.scatter(fd, x=x_sel, y="Loan_Amount", color="Loan_Approved",
                         opacity=0.55, size_max=7,
                         color_discrete_sequence=[C_GREEN, C_RED],
                         hover_data=["Credit_Score","DTI_Ratio","Loan_Purpose"],
                         trendline="lowess")
        fig.update_layout(**PLOT_LAYOUT, legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with rc4:
        st.markdown('<div class="chart-card"><div class="chart-title">Feature Distribution by Approval</div>', unsafe_allow_html=True)
        feat_b = st.selectbox("Feature", ["DTI_Ratio","Credit_Score","Applicant_Income",
                                           "Loan_Amount","Savings","Collateral_Value"], key="bx_f")
        chart_t = st.radio("Chart type", ["Box","Violin"], horizontal=True, key="bx_t")
        if chart_t == "Box":
            fig = px.box(fd, x="Loan_Approved", y=feat_b, color="Loan_Approved",
                         color_discrete_sequence=[C_GREEN,C_RED], points="outliers")
        else:
            fig = px.violin(fd, x="Loan_Approved", y=feat_b, color="Loan_Approved",
                            color_discrete_sequence=[C_GREEN,C_RED], box=True, points=False)
        fig.update_layout(**PLOT_LAYOUT, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ── Row 3 ──
    rc5, rc6 = st.columns([2,1])

    with rc5:
        st.markdown('<div class="chart-card"><div class="chart-title">Approval Rate by Category</div>', unsafe_allow_html=True)
        cat_sel = st.selectbox("Group by",
                               ["Employment_Status","Loan_Purpose","Property_Area",
                                "Marital_Status","Gender","Education_Level"], key="cat_g")
        grp = fd.groupby(cat_sel)["Approved"].agg(["sum","count"]).reset_index()
        grp.columns = [cat_sel, "Approved","Total"]
        grp["Rate"] = grp["Approved"]/grp["Total"]*100
        grp = grp.sort_values("Rate", ascending=False)
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Approved", x=grp[cat_sel], y=grp["Approved"],
                             marker_color=C_GREEN, opacity=0.9))
        fig.add_trace(go.Bar(name="Rejected", x=grp[cat_sel],
                             y=grp["Total"]-grp["Approved"],
                             marker_color=C_RED, opacity=0.9))
        fig.add_trace(go.Scatter(name="Rate %", x=grp[cat_sel], y=grp["Rate"],
                                 mode="lines+markers", yaxis="y2",
                                 line=dict(color=C_AMBER, width=2.5),
                                 marker=dict(size=7, color=C_AMBER)))
        fig.update_layout(**PLOT_LAYOUT, barmode="stack",
                          yaxis2=dict(overlaying="y", side="right", range=[0,100],
                                      ticksuffix="%", gridcolor="transparent",
                                      tickfont=dict(color=C_AMBER)))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with rc6:
        st.markdown('<div class="chart-card"><div class="chart-title">Correlation Matrix</div>', unsafe_allow_html=True)
        num_only = fd.select_dtypes("number").drop(columns=["Approved"], errors="ignore")
        corr = num_only.corr()
        fig = px.imshow(corr, color_continuous_scale=[[0,C_RED],[0.5,"#0a1628"],[1,C_BLUE]],
                        zmin=-1, zmax=1, text_auto=".1f", aspect="auto")
        fig.update_traces(textfont_size=9)
        fig.update_layout(**PLOT_LAYOUT, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Performance
# ══════════════════════════════════════════════════════════════════════════════
elif "Performance" in page:
    st.markdown('<div class="page-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Benchmark all three classifiers · Baseline vs Feature Engineered</div>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Build metrics df
    rows = []
    for (nm,var),m in model_res.items():
        rows.append({"Model":nm,"Variant":var,
                     "Accuracy" :round(m["accuracy"] *100,2),
                     "Precision":round(m["precision"]*100,2),
                     "Recall"   :round(m["recall"]   *100,2),
                     "F1-Score" :round(m["f1"]        *100,2)})
    perf = pd.DataFrame(rows)

    # ── Top metric cards for best model ──
    best = perf.loc[perf["F1-Score"].idxmax()]
    b1,b2,b3,b4 = st.columns(4)
    for col, lbl, val in [(b1,"Best Accuracy",f'{perf["Accuracy"].max():.2f}%'),
                           (b2,"Best Precision",f'{perf["Precision"].max():.2f}%'),
                           (b3,"Best Recall",f'{perf["Recall"].max():.2f}%'),
                           (b4,"Best F1-Score",f'{perf["F1-Score"].max():.2f}%')]:
        col.markdown(f"""
        <div class="kpi-card">
          <div class="kpi-label">{lbl}</div>
          <div class="kpi-value">{val}</div>
          <div class="kpi-delta up">↑ {best['Model']} ({best['Variant']})</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ──
    t1, t2, t3 = st.tabs(["📊  Metrics Table", "🕸  Radar Chart", "🔢  Confusion Matrix"])

    with t1:
        def hl(col):
            return ["background:#14532d;color:#bbf7d0;font-weight:700"
                    if v==col.max() else "" for v in col]
        st.dataframe(
            perf.style.apply(hl, subset=["Accuracy","Precision","Recall","F1-Score"])
                      .format({c:"{:.2f}%" for c in ["Accuracy","Precision","Recall","F1-Score"]}),
            use_container_width=True, height=280)

        vt = st.radio("Variant filter", ["Baseline","Feature Engineered","Both"],
                      horizontal=True, key="perf_vt")
        plt_df = perf if vt=="Both" else perf[perf["Variant"]==vt]
        melt = plt_df.melt(id_vars=["Model","Variant"],
                            value_vars=["Accuracy","Precision","Recall","F1-Score"],
                            var_name="Metric", value_name="Score")
        fig = px.bar(melt, x="Metric", y="Score", color="Model", barmode="group",
                     facet_col="Variant" if vt=="Both" else None,
                     text_auto=".1f",
                     color_discrete_sequence=[C_BLUE, C_CYAN, C_AMBER])
        fig.update_layout(**PLOT_LAYOUT, yaxis_range=[0,100])
        fig.update_traces(textposition="outside", textfont_size=10)
        st.plotly_chart(fig, use_container_width=True)

    with t2:
        metrics = ["Accuracy","Precision","Recall","F1-Score"]
        fe = perf[perf["Variant"]=="Feature Engineered"]
        bl = perf[perf["Variant"]=="Baseline"]
        v_sel = st.radio("Show variant", ["Feature Engineered","Baseline","Both"],
                         horizontal=True, key="radar_v")
        src = fe if v_sel=="Feature Engineered" else bl if v_sel=="Baseline" else pd.concat([fe,bl])
        fig = go.Figure()
        palettes = {
            "Logistic Regression":[C_BLUE,C_BLUE+"88"],
            "K-Nearest Neighbors":[C_CYAN,C_CYAN+"88"],
            "Gaussian Naive Bayes":[C_AMBER,C_AMBER+"88"],
        }
        for _, row in src.iterrows():
            vals = [row[m] for m in metrics]+[row[metrics[0]]]
            lc = palettes[row["Model"]][0]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=metrics+[metrics[0]],
                fill="toself", name=f'{row["Model"]} ({row["Variant"]})',
                line=dict(color=lc, width=2),
                fillcolor=lc+"33", opacity=0.9))
        fig.update_layout(
            polar=dict(radialaxis=dict(range=[50,100], gridcolor="#1e3a5f",
                                       tickfont=dict(color="#64748b")),
                        angularaxis=dict(gridcolor="#1e3a5f",
                                         tickfont=dict(color="#94a3b8", size=12))),
            paper_bgcolor="rgba(0,0,0,0)", font_color="#94a3b8",
            legend=dict(bgcolor="rgba(0,0,0,0)", font_size=11),
            margin=dict(t=30,b=30,l=60,r=60), height=440)
        st.plotly_chart(fig, use_container_width=True)

    with t3:
        cc1, cc2 = st.columns([1,1])
        with cc1:
            cm_model = st.selectbox("Model",
                ["Logistic Regression","K-Nearest Neighbors","Gaussian Naive Bayes"], key="cm_m")
            cm_var   = st.selectbox("Variant", ["Baseline","Feature Engineered"], key="cm_v")
        cm = model_res[(cm_model,cm_var)]["cm"]
        labels = ["Rejected","Approved"]
        fig = go.Figure(go.Heatmap(
            z=cm, x=labels, y=labels, text=cm,
            texttemplate="<b>%{text}</b>",
            colorscale=[[0,"#0a1628"],[0.5,"#1e3a5f"],[1,C_BLUE]],
            showscale=False, hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>"
        ))
        fig.update_layout(**PLOT_LAYOUT, height=340,
                          xaxis=dict(title="Predicted", side="bottom"),
                          yaxis=dict(title="Actual"))
        tn,fp,fn,tp = cm.ravel()
        with cc2:
            st.plotly_chart(fig, use_container_width=True)
        # stats below
        s1,s2,s3,s4 = st.columns(4)
        for col,lbl,val,clr in [(s1,"True Positive",tp,C_GREEN),(s2,"True Negative",tn,C_BLUE),
                                  (s3,"False Positive",fp,C_AMBER),(s4,"False Negative",fn,C_RED)]:
            col.markdown(f"""
            <div class="kpi-card" style="border-left-color:{clr}">
              <div class="kpi-label">{lbl}</div>
              <div class="kpi-value" style="color:{clr}">{val}</div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Predict
# ══════════════════════════════════════════════════════════════════════════════
elif "Predict" in page:
    st.markdown('<div class="page-title">Loan Approval Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Real-time risk assessment & ML prediction</div>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Model selector row
    ms1, ms2, ms3 = st.columns([2,1,2])
    with ms1:
        sel_model   = st.selectbox("ML Model",
            ["Logistic Regression","K-Nearest Neighbors","Gaussian Naive Bayes"])
    with ms2:
        sel_variant = st.selectbox("Variant", ["Feature Engineered","Baseline"])

    acc_here = model_res[(sel_model,sel_variant)]["accuracy"]*100
    f1_here  = model_res[(sel_model,sel_variant)]["f1"]*100
    with ms3:
        st.markdown(f"""
        <div style="display:flex;gap:16px;margin-top:28px">
          <div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:10px;
               padding:8px 18px;text-align:center">
            <div style="font-size:.65rem;color:#64748b;text-transform:uppercase">Accuracy</div>
            <div style="font-size:1.15rem;font-weight:700;color:{C_CYAN}">{acc_here:.1f}%</div>
          </div>
          <div style="background:#0a1628;border:1px solid #1e3a5f;border-radius:10px;
               padding:8px 18px;text-align:center">
            <div style="font-size:.65rem;color:#64748b;text-transform:uppercase">F1 Score</div>
            <div style="font-size:1.15rem;font-weight:700;color:{C_BLUE}">{f1_here:.1f}%</div>
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Input form ──
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown('<div class="section-pill">💰 Financial Profile</div>', unsafe_allow_html=True)
        income       = st.slider("Applicant Income ($)", 2000, 20000, 8000, 500)
        co_income    = st.slider("Coapplicant Income ($)", 0, 10000, 2000, 250)
        savings      = st.slider("Savings ($)", 0, 20000, 5000, 500)
        credit_score = st.slider("Credit Score", 550, 850, 680)
        dti_ratio    = st.slider("DTI Ratio", 0.10, 0.60, 0.35, 0.01,
                                  format="%.2f")

    with c2:
        st.markdown('<div class="section-pill">🏦 Loan Details</div>', unsafe_allow_html=True)
        loan_amount    = st.slider("Loan Amount ($)", 1000, 40000, 20000, 500)
        loan_term      = st.select_slider("Loan Term (months)", [12,24,36,48,60,72,84], 48)
        collateral     = st.slider("Collateral Value ($)", 0, 50000, 15000, 1000)
        existing_loans = st.number_input("Existing Loans", 0, 4, 1)
        loan_purpose   = st.selectbox("Loan Purpose", ["Car","Education","Home","Personal"])

    with c3:
        st.markdown('<div class="section-pill">👤 Personal Info</div>', unsafe_allow_html=True)
        age            = st.number_input("Age", 21, 59, 35)
        dependents     = st.selectbox("Dependents", [0,1,2,3])
        gender         = st.selectbox("Gender", ["Male","Female"])
        marital_status = st.selectbox("Marital Status", ["Single","Married"])
        education      = st.selectbox("Education Level", ["Graduate","Undergraduate"])
        employment     = st.selectbox("Employment Status", ["Salaried","Self-employed","Unemployed"])
        property_area  = st.selectbox("Property Area", ["Rural","Semiurban","Urban"])
        employer_cat   = st.selectbox("Employer Category", ["MNC","Government","Private","Unemployed"])

    # ── Live risk signals ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="chart-title" style="margin-bottom:12px">⚡ Live Risk Signals</div>',
                unsafe_allow_html=True)

    dti_pct = dti_ratio * 100
    lti     = loan_amount / income if income > 0 else 0
    stl     = savings / loan_amount if loan_amount > 0 else 0
    net_inc = income + co_income - (dti_ratio * income)

    def tier_info(cs):
        if cs>=720: return "Prime","green",C_GREEN
        if cs>=620: return "Near-Prime","yellow",C_AMBER
        return "Subprime","red",C_RED

    def dti_info(d):
        if d>43: return "red",C_RED,"High Risk"
        if d>30: return "yellow",C_AMBER,"Moderate"
        return "green",C_GREEN,"Healthy"

    def lti_info(l):
        if l>4: return "red",C_RED,"Very High"
        if l>2: return "yellow",C_AMBER,"Moderate"
        return "green",C_GREEN,"Low"

    def stl_info(s):
        if s>=0.2: return "green",C_GREEN,"Strong"
        if s>=0.1: return "yellow",C_AMBER,"Moderate"
        return "red",C_RED,"Weak"

    t_name,t_cls,t_col  = tier_info(credit_score)
    d_cls,d_col,d_lbl   = dti_info(dti_pct)
    l_cls,l_col,l_lbl   = lti_info(lti)
    s_cls,s_col,s_lbl   = stl_info(stl)

    lv1,lv2,lv3,lv4 = st.columns(4)
    for col, cls, label, val, desc in [
        (lv1, t_cls, "Credit Tier",      t_name,       f"Score: {credit_score}"),
        (lv2, d_cls, "DTI Ratio",        f"{dti_pct:.0f}%", d_lbl),
        (lv3, l_cls, "LTI Ratio",        f"{lti:.2f}×" if income>0 else "N/A", l_lbl),
        (lv4, s_cls, "Savings Coverage", f"{stl*100:.0f}%",  s_lbl),
    ]:
        _c = t_col if col==lv1 else d_col if col==lv2 else l_col if col==lv3 else s_col
        col.markdown(f"""
        <div class="risk-box {cls}">
          <div class="risk-label">{label}</div>
          <div class="risk-value" style="color:{_c}">{val}</div>
          <div class="risk-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

    # ── Live gauges row ──
    st.markdown("<br>", unsafe_allow_html=True)
    g1, g2, g3 = st.columns(3)

    def make_gauge(val, title, suffix, rng, zones, bar_color, threshold=None):
        steps = [{"range":z[0],"color":z[1]} for z in zones]
        gauge_dict = {
            "axis": {"range":rng, "tickcolor":"#334155", "tickfont":{"color":"#64748b","size":10}},
            "bar":  {"color": bar_color, "thickness": 0.25},
            "bgcolor": "#0a1628",
            "bordercolor": "#1e3a5f",
            "steps": steps,
        }
        if threshold:
            gauge_dict["threshold"] = {"line":{"color":"white","width":2},"value":threshold}
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=val,
            title={"text":title,"font":{"color":"#94a3b8","size":13}},
            number={"suffix":suffix,"font":{"color":"#f1f5f9","size":26}},
            gauge=gauge_dict
        ))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=200,
                          margin=dict(t=50,b=10,l=20,r=20))
        return fig

    with g1:
        st.plotly_chart(make_gauge(dti_pct,"DTI Ratio","%",[0,60],
            [([0,30],"#052e16"),([30,43],"#451a03"),([43,60],"#450a0a")],
            d_col, 43), use_container_width=True)

    with g2:
        st.plotly_chart(make_gauge(credit_score,"Credit Score","",[550,850],
            [([550,620],"#450a0a"),([620,720],"#451a03"),([720,850],"#052e16")],
            t_col, 720), use_container_width=True)

    with g3:
        lti_disp = min(lti, 8) if income > 0 else 0
        st.plotly_chart(make_gauge(lti_disp,"LTI Ratio","×",[0,8],
            [([0,2],"#052e16"),([2,4],"#451a03"),([4,8],"#450a0a")],
            l_col, 4), use_container_width=True)

    # ── Predict button ──
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔮  Run Prediction", use_container_width=True, type="primary"):
        with st.spinner("Analysing applicant profile…"):
            row_dict = dict(
                Age=age, Applicant_Income=income, Coapplicant_Income=co_income,
                Credit_Score=credit_score, DTI_Ratio=dti_ratio,
                Loan_Amount=loan_amount, Loan_Term=loan_term,
                Existing_Loans=existing_loans, Savings=savings,
                Collateral_Value=collateral, Dependents=dependents,
                Gender=gender, Marital_Status=marital_status,
                Education_Level=education, Employment_Status=employment,
                Loan_Purpose=loan_purpose, Property_Area=property_area,
                Employer_Category=employer_cat,
            )
            pred, conf = run_predict(row_dict, sel_model, sel_variant,
                                     model_res, scaler, feat_cols)

        res_cls = "approved" if pred==1 else "rejected"
        verdict = "✅  APPROVED" if pred==1 else "❌  REJECTED"
        conf_txt = f"Confidence: {conf:.1f}%" if conf else ""
        badge    = "🟢" if pred==1 else "🔴"

        st.markdown(f"""
        <div class="result-banner {res_cls}">
          <div>
            <div class="result-verdict">{verdict}</div>
            <div class="result-conf">{conf_txt} · {sel_model} ({sel_variant})</div>
          </div>
          <div class="result-badge">{badge}</div>
        </div>""", unsafe_allow_html=True)

        if conf:
            fig_c = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=conf,
                delta={"reference":50,"increasing":{"color":C_GREEN},"decreasing":{"color":C_RED}},
                title={"text":"Prediction Confidence","font":{"color":"#94a3b8","size":13}},
                number={"suffix":"%","font":{"color":"#f1f5f9","size":32}},
                gauge={
                    "axis":{"range":[0,100],"tickcolor":"#334155"},
                    "bar":{"color":C_GREEN if pred==1 else C_RED,"thickness":0.25},
                    "bgcolor":"#0a1628", "bordercolor":"#1e3a5f",
                    "steps":[{"range":[0,50],"color":"#0d1b2e"},
                              {"range":[50,100],"color":"#0d1b2e"}],
                    "threshold":{"line":{"color":"white","width":2},"value":50}
                }
            ))
            fig_c.update_layout(paper_bgcolor="rgba(0,0,0,0)", height=220,
                                 margin=dict(t=50,b=10,l=20,r=20))
            st.plotly_chart(fig_c, use_container_width=True)

        # Compare all models
        st.markdown("**All model predictions for this applicant**")
        comp_rows = []
        for nm in ["Logistic Regression","K-Nearest Neighbors","Gaussian Naive Bayes"]:
            for vr in ["Baseline","Feature Engineered"]:
                p, c = run_predict(row_dict, nm, vr, model_res, scaler, feat_cols)
                comp_rows.append({"Model":nm,"Variant":vr,
                                   "Verdict":"✅ Approved" if p==1 else "❌ Rejected",
                                   "Confidence":f"{c:.1f}%" if c else "—"})
        st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif "Explorer" in page:
    st.markdown('<div class="page-title">Data Explorer</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Browse, filter, and export applicant records</div>',
                unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    ef1,ef2,ef3,ef4,ef5 = st.columns(5)
    search   = ef1.text_input("🔍 Search", placeholder="Any field…")
    status_f = ef2.selectbox("Status", ["All"]+df["Loan_Approved"].unique().tolist())
    emp_f2   = ef3.multiselect("Employment", df["Employment_Status"].unique().tolist(),
                                default=df["Employment_Status"].unique().tolist(), key="ex_emp")
    purp_f2  = ef4.multiselect("Purpose", df["Loan_Purpose"].unique().tolist(),
                                default=df["Loan_Purpose"].unique().tolist(), key="ex_purp")
    cs_f2    = ef5.slider("Credit Score", int(df["Credit_Score"].min()),
                           int(df["Credit_Score"].max()),
                           (int(df["Credit_Score"].min()), int(df["Credit_Score"].max())), key="ex_cs")

    view = df.copy()
    if status_f != "All": view = view[view["Loan_Approved"]==status_f]
    view = view[view["Employment_Status"].isin(emp_f2) &
                view["Loan_Purpose"].isin(purp_f2) &
                view["Credit_Score"].between(*cs_f2)]
    if search:
        mask = view.apply(lambda c: c.astype(str).str.contains(search, case=False)).any(axis=1)
        view = view[mask]

    # Stats row
    sv1,sv2,sv3,sv4 = st.columns(4)
    sv1.metric("Records", f"{len(view):,}")
    sv2.metric("Approved", f"{int(view['Approved'].sum()):,}")
    sv3.metric("Avg Credit Score", f"{view['Credit_Score'].mean():.0f}" if len(view) else "—")
    sv4.metric("Avg Loan Amount", f"${view['Loan_Amount'].mean():,.0f}" if len(view) else "—")

    st.markdown("<br>", unsafe_allow_html=True)

    # Visualise filtered data
    ev1, ev2 = st.columns(2)
    with ev1:
        if len(view):
            fig = px.histogram(view, x="Credit_Score", color="Loan_Approved",
                               nbins=30, barmode="overlay", opacity=0.8,
                               color_discrete_sequence=[C_GREEN, C_RED],
                               title="Credit Score — Filtered Set")
            fig.update_layout(**PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)

    with ev2:
        if len(view):
            fig = px.scatter(view, x="Applicant_Income", y="Loan_Amount",
                             color="Loan_Approved", opacity=0.55,
                             color_discrete_sequence=[C_GREEN, C_RED],
                             title="Income vs Loan — Filtered Set",
                             hover_data=["Credit_Score","DTI_Ratio"])
            fig.update_layout(**PLOT_LAYOUT, legend_title_text="")
            st.plotly_chart(fig, use_container_width=True)

    st.dataframe(view.drop(columns=["Approved"], errors="ignore").reset_index(drop=True),
                 use_container_width=True, height=380)

    dl1, dl2 = st.columns(2)
    csv = view.drop(columns=["Approved"],errors="ignore").to_csv(index=False).encode()
    dl1.download_button("⬇️  Download CSV", csv, "creditwise_filtered.csv",
                        "text/csv", use_container_width=True)
    dl2.download_button("⬇️  Download JSON",
                        view.drop(columns=["Approved"],errors="ignore").to_json(orient="records").encode(),
                        "creditwise_filtered.json", "application/json", use_container_width=True)
