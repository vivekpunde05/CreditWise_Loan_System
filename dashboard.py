import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

# ── Page config & global CSS ───────────────────────────────────────────────────
st.set_page_config(page_title="CreditWise", layout="wide", page_icon="💳")

st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0f172a; }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  .metric-card {
    background: #1e293b; border-radius: 12px; padding: 20px 24px;
    border-left: 4px solid #3b82f6;
  }
  .metric-card .label { font-size: 0.78rem; color: #94a3b8; text-transform: uppercase; letter-spacing: .05em; }
  .metric-card .value { font-size: 2rem; font-weight: 700; color: #f1f5f9; margin-top: 4px; }
  .metric-card .delta { font-size: 0.82rem; margin-top: 4px; }
  .result-approved { background:#065f46; border-radius:12px; padding:20px; text-align:center; }
  .result-rejected { background:#7f1d1d; border-radius:12px; padding:20px; text-align:center; }
  .result-title { font-size:2.2rem; font-weight:800; color:#fff; }
  .result-sub   { font-size:1rem; color:#d1fae5; margin-top:4px; }
  .risk-box { background:#1e293b; border-radius:10px; padding:16px; text-align:center; }
  .risk-label { font-size:.75rem; color:#94a3b8; text-transform:uppercase; letter-spacing:.05em; }
  .risk-value { font-size:1.6rem; font-weight:700; margin-top:6px; }
  .stTabs [data-baseweb="tab"] { font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Data & model helpers ───────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare():
    df = pd.read_csv("loan_approval_data.csv")
    num_cols = df.select_dtypes(include="number").columns
    cat_cols = df.select_dtypes(include="object").columns
    df[num_cols] = SimpleImputer(strategy="mean").fit_transform(df[num_cols])
    df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])
    df = df.drop(columns=["Applicant_ID"], errors="ignore")
    df["Loan_Approved_enc"] = LabelEncoder().fit_transform(df["Loan_Approved"])
    return df

@st.cache_resource
def train_models(_df):
    encode_cols = ["Employment_Status", "Marital_Status", "Loan_Purpose",
                   "Property_Area", "Gender", "Employer_Category"]
    df = _df.copy()
    df["Education_Level"] = LabelEncoder().fit_transform(df["Education_Level"])
    df_enc = pd.get_dummies(df, columns=encode_cols, drop_first=True)
    feat_cols = [c for c in df_enc.columns if c not in ("Loan_Approved", "Loan_Approved_enc")]
    X = df_enc[feat_cols]; y = df_enc["Loan_Approved_enc"]
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    Xtr, Xte, ytr, yte = train_test_split(Xs, y, test_size=0.2, random_state=42)
    dti_i = feat_cols.index("DTI_Ratio"); cs_i = feat_cols.index("Credit_Score")
    Xs_fe = np.hstack([Xs, (Xs[:,dti_i]**2).reshape(-1,1), (Xs[:,cs_i]**2).reshape(-1,1)])
    Xtr_fe, Xte_fe, _, _ = train_test_split(Xs_fe, y, test_size=0.2, random_state=42)
    results = {}
    for name, cb, cf in [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42),
                                LogisticRegression(max_iter=1000, random_state=42)),
        ("K-Nearest Neighbors", KNeighborsClassifier(5), KNeighborsClassifier(5)),
        ("Gaussian Naive Bayes", GaussianNB(), GaussianNB()),
    ]:
        for variant, clf, Xtr_, Xte_ in [("Baseline", cb, Xtr, Xte),
                                          ("Feature Engineered", cf, Xtr_fe, Xte_fe)]:
            clf.fit(Xtr_, ytr); pred = clf.predict(Xte_)
            results[(name, variant)] = dict(
                clf=clf,
                accuracy =accuracy_score(yte, pred),
                precision=precision_score(yte, pred, zero_division=0),
                recall   =recall_score(yte, pred, zero_division=0),
                f1       =f1_score(yte, pred, zero_division=0),
                cm       =confusion_matrix(yte, pred),
            )
    return results, scaler, feat_cols

def predict_applicant(row_dict, model_name, variant, model_results, scaler, feat_cols):
    inp = pd.DataFrame([row_dict])
    inp["Education_Level"] = inp["Education_Level"].map({"Graduate":1,"Undergraduate":0}).fillna(0)
    enc_cols = ["Employment_Status","Marital_Status","Loan_Purpose",
                "Property_Area","Gender","Employer_Category"]
    inp = pd.get_dummies(inp, columns=enc_cols, drop_first=True)
    for c in feat_cols:
        if c not in inp.columns: inp[c] = 0
    inp = inp[feat_cols]
    X = scaler.transform(inp)
    if variant == "Feature Engineered":
        di = feat_cols.index("DTI_Ratio"); ci = feat_cols.index("Credit_Score")
        X = np.hstack([X, (X[:,di]**2).reshape(-1,1), (X[:,ci]**2).reshape(-1,1)])
    clf = model_results[(model_name, variant)]["clf"]
    pred = clf.predict(X)[0]
    conf = clf.predict_proba(X)[0][pred] * 100 if hasattr(clf, "predict_proba") else None
    return int(pred), conf

# ── Load data ──────────────────────────────────────────────────────────────────
df = load_and_prepare()
model_results, scaler, feat_cols = train_models(df)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 💳 CreditWise")
    st.markdown("---")
    page = st.radio("", ["📊 Analytics", "📈 Model Performance",
                          "🔍 Predict", "🗂 Data Explorer"])
    st.markdown("---")
    st.caption("Dataset · 1 000 applicants")
    st.caption("Models · LR · KNN · GNB")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Analytics
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Analytics":
    st.title("📊 Portfolio Analytics")

    # ── Interactive filters ──
    with st.expander("🔧 Filters", expanded=False):
        fc1, fc2, fc3 = st.columns(3)
        emp_filter  = fc1.multiselect("Employment Status",
                                      df["Employment_Status"].unique().tolist(),
                                      default=df["Employment_Status"].unique().tolist())
        purp_filter = fc2.multiselect("Loan Purpose",
                                      df["Loan_Purpose"].unique().tolist(),
                                      default=df["Loan_Purpose"].unique().tolist())
        cs_range    = fc3.slider("Credit Score Range",
                                 int(df["Credit_Score"].min()),
                                 int(df["Credit_Score"].max()),
                                 (int(df["Credit_Score"].min()), int(df["Credit_Score"].max())))

    fdf = df[
        df["Employment_Status"].isin(emp_filter) &
        df["Loan_Purpose"].isin(purp_filter) &
        df["Credit_Score"].between(*cs_range)
    ]

    approved_mask = fdf["Loan_Approved_enc"] == 1
    total   = len(fdf)
    n_appr  = approved_mask.sum()

    # ── KPI cards ──
    k1, k2, k3, k4 = st.columns(4)
    def kpi(col, label, value, delta=None, delta_ok=True):
        color = "#22c55e" if delta_ok else "#ef4444"
        delta_html = f'<div class="delta" style="color:{color}">{delta}</div>' if delta else ""
        col.markdown(f"""
        <div class="metric-card">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          {delta_html}
        </div>""", unsafe_allow_html=True)

    kpi(k1, "Total Applications", f"{total:,}")
    kpi(k2, "Approval Rate", f"{n_appr/total*100:.1f}%" if total else "—",
        f"{'▲' if n_appr/total>.5 else '▼'} vs 50% benchmark", n_appr/total>.5)
    kpi(k3, "Avg Credit Score", f"{fdf['Credit_Score'].mean():.0f}",
        "Prime tier" if fdf['Credit_Score'].mean()>=720 else
        "Near-Prime" if fdf['Credit_Score'].mean()>=620 else "Subprime")
    kpi(k4, "Avg DTI Ratio", f"{fdf['DTI_Ratio'].mean():.3f}",
        "High risk" if fdf['DTI_Ratio'].mean()>.43 else "Healthy", fdf['DTI_Ratio'].mean()<=.43)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row 1: pie + credit score histogram ──
    r1c1, r1c2 = st.columns(2)

    with r1c1:
        st.subheader("Approval Distribution")
        counts = fdf["Loan_Approved"].value_counts().reset_index()
        counts.columns = ["Status", "Count"]
        fig = px.pie(counts, names="Status", values="Count",
                     color="Status",
                     color_discrete_map={counts["Status"].iloc[0]: "#22c55e",
                                          counts["Status"].iloc[-1]: "#ef4444"},
                     hole=0.45)
        fig.update_traces(textinfo="percent+label", pull=[0.03]*len(counts))
        fig.update_layout(showlegend=False, margin=dict(t=20,b=20,l=0,r=0),
                          paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0")
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        st.subheader("Credit Score Distribution")
        color_by = st.toggle("Colour by Approval", value=True, key="cs_colour")
        if color_by:
            fig = px.histogram(fdf, x="Credit_Score", color="Loan_Approved",
                               nbins=35, barmode="overlay", opacity=0.75,
                               color_discrete_sequence=["#22c55e","#ef4444"])
        else:
            fig = px.histogram(fdf, x="Credit_Score", nbins=35,
                               color_discrete_sequence=["#3b82f6"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0", legend_title_text="",
                          margin=dict(t=20,b=40,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 2: scatter + DTI box ──
    r2c1, r2c2 = st.columns(2)

    with r2c1:
        st.subheader("Income vs Loan Amount")
        x_axis = st.selectbox("X-axis", ["Applicant_Income","Coapplicant_Income","Savings"],
                              key="scatter_x")
        fig = px.scatter(fdf, x=x_axis, y="Loan_Amount",
                         color="Loan_Approved", opacity=0.55, size_max=6,
                         color_discrete_sequence=["#22c55e","#ef4444"],
                         hover_data=["Credit_Score","DTI_Ratio"])
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0", legend_title_text="",
                          margin=dict(t=20,b=40,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        st.subheader("Feature Distribution by Approval")
        feat_sel = st.selectbox("Feature",
                                ["DTI_Ratio","Credit_Score","Applicant_Income",
                                 "Loan_Amount","Savings","Collateral_Value"],
                                key="box_feat")
        fig = px.box(fdf, x="Loan_Approved", y=feat_sel, color="Loan_Approved",
                     color_discrete_sequence=["#22c55e","#ef4444"], points="outliers")
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          font_color="#e2e8f0", showlegend=False,
                          margin=dict(t=20,b=40,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

    # ── Row 3: correlation heatmap ──
    st.subheader("Correlation Heatmap")
    num_df = fdf.select_dtypes(include="number").drop(columns=["Loan_Approved_enc"], errors="ignore")
    corr = num_df.corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                      margin=dict(t=20,b=20,l=0,r=0))
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Model Performance
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Model Performance":
    st.title("📈 Model Performance")

    rows = []
    for (name, variant), m in model_results.items():
        rows.append({"Model": name, "Variant": variant,
                     "Accuracy":  round(m["accuracy"]  * 100, 2),
                     "Precision": round(m["precision"] * 100, 2),
                     "Recall":    round(m["recall"]    * 100, 2),
                     "F1-Score":  round(m["f1"]        * 100, 2)})
    perf_df = pd.DataFrame(rows)

    # ── Metric table with highlight ──
    def highlight_best(col):
        is_best = col == col.max()
        return ["background-color:#14532d;color:#bbf7d0;font-weight:700"
                if v else "" for v in is_best]

    st.dataframe(
        perf_df.style
               .apply(highlight_best, subset=["Accuracy","Precision","Recall","F1-Score"])
               .format({c: "{:.2f}%" for c in ["Accuracy","Precision","Recall","F1-Score"]}),
        use_container_width=True, height=260
    )

    st.divider()
    pc1, pc2 = st.columns([3, 2])

    # ── Radar chart ──
    with pc1:
        st.subheader("Model Radar — Feature Engineered")
        metrics  = ["Accuracy","Precision","Recall","F1-Score"]
        fe_sub   = perf_df[perf_df["Variant"] == "Feature Engineered"]
        fig = go.Figure()
        colors_r = ["#3b82f6","#f59e0b","#22c55e"]
        for (_, row), col in zip(fe_sub.iterrows(), colors_r):
            vals = [row[m] for m in metrics] + [row[metrics[0]]]
            fig.add_trace(go.Scatterpolar(
                r=vals, theta=metrics + [metrics[0]],
                fill="toself", name=row["Model"],
                line_color=col, opacity=0.75
            ))
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[50,100],
                                                      gridcolor="#334155",
                                                      linecolor="#334155"),
                                      angularaxis=dict(gridcolor="#334155")),
                          paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                          legend=dict(bgcolor="rgba(0,0,0,0)"),
                          margin=dict(t=30,b=30,l=30,r=30))
        st.plotly_chart(fig, use_container_width=True)

    # ── Confusion matrix ──
    with pc2:
        st.subheader("Confusion Matrix")
        sel_model   = st.selectbox("Model",   ["Logistic Regression","K-Nearest Neighbors","Gaussian Naive Bayes"])
        sel_variant = st.selectbox("Variant", ["Baseline","Feature Engineered"])
        cm = model_results[(sel_model, sel_variant)]["cm"]
        labels = ["Rejected","Approved"]
        fig = px.imshow(cm, text_auto=True, x=labels, y=labels,
                        color_continuous_scale="Blues",
                        labels=dict(x="Predicted", y="Actual", color="Count"))
        fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                          margin=dict(t=20,b=40,l=60,r=0),
                          coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

        m = model_results[(sel_model, sel_variant)]
        mc1, mc2 = st.columns(2)
        mc1.metric("Accuracy",  f"{m['accuracy']*100:.2f}%")
        mc2.metric("F1-Score",  f"{m['f1']*100:.2f}%")
        mc1.metric("Precision", f"{m['precision']*100:.2f}%")
        mc2.metric("Recall",    f"{m['recall']*100:.2f}%")

    # ── Grouped bar comparison ──
    st.subheader("Side-by-Side Metric Comparison")
    variant_toggle = st.radio("Variant", ["Baseline","Feature Engineered","Both"],
                              horizontal=True, key="bar_variant")
    plot_df = perf_df if variant_toggle == "Both" else perf_df[perf_df["Variant"]==variant_toggle]
    plot_df = plot_df.melt(id_vars=["Model","Variant"], value_vars=["Accuracy","Precision","Recall","F1-Score"],
                            var_name="Metric", value_name="Score")
    fig = px.bar(plot_df, x="Metric", y="Score", color="Model",
                 barmode="group", facet_col="Variant" if variant_toggle=="Both" else None,
                 color_discrete_sequence=px.colors.qualitative.Bold,
                 text_auto=".1f")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font_color="#e2e8f0", yaxis_range=[0,100],
                      margin=dict(t=30,b=40,l=0,r=0))
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — Predict
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔍 Predict":
    st.title("🔍 Loan Approval Prediction")

    top1, top2 = st.columns([2, 1])
    with top1:
        selected_model   = st.selectbox("Model",
            ["Logistic Regression","K-Nearest Neighbors","Gaussian Naive Bayes"])
    with top2:
        selected_variant = st.selectbox("Variant", ["Baseline","Feature Engineered"])

    st.markdown("---")
    st.subheader("Applicant Details")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**Financial**")
        income       = st.slider("Applicant Income ($)",  2000, 20000, 8000, step=500)
        co_income    = st.slider("Coapplicant Income ($)", 0, 10000, 2000, step=250)
        credit_score = st.slider("Credit Score", 550, 850, 680)
        dti_ratio    = st.slider("DTI Ratio", 0.10, 0.60, 0.35, step=0.01)
        savings      = st.slider("Savings ($)", 0, 20000, 5000, step=500)

    with c2:
        st.markdown("**Loan**")
        loan_amount    = st.slider("Loan Amount ($)", 1000, 40000, 20000, step=1000)
        loan_term      = st.select_slider("Loan Term (months)", [12,24,36,48,60,72,84], 48)
        collateral     = st.slider("Collateral Value ($)", 0, 50000, 15000, step=1000)
        existing_loans = st.number_input("Existing Loans", 0, 4, 1)
        loan_purpose   = st.selectbox("Loan Purpose", ["Car","Education","Home","Personal"])

    with c3:
        st.markdown("**Personal**")
        age            = st.number_input("Age", 21, 59, 35)
        dependents     = st.selectbox("Dependents", [0, 1, 2, 3])
        gender         = st.selectbox("Gender", ["Male","Female"])
        marital_status = st.selectbox("Marital Status", ["Single","Married"])
        education      = st.selectbox("Education Level", ["Graduate","Undergraduate"])
        employment     = st.selectbox("Employment Status", ["Salaried","Self-employed","Unemployed"])
        property_area  = st.selectbox("Property Area", ["Rural","Semiurban","Urban"])
        employer_cat   = st.selectbox("Employer Category", ["MNC","Government","Private","Unemployed"])

    # ── Live risk panel (updates without pressing Predict) ──
    st.markdown("---")
    st.subheader("⚡ Live Risk Signals")
    lv1, lv2, lv3, lv4 = st.columns(4)

    # Credit tier
    if credit_score >= 720:   tier, tier_col = "Prime 🟢", "#22c55e"
    elif credit_score >= 620: tier, tier_col = "Near-Prime 🟡", "#f59e0b"
    else:                     tier, tier_col = "Subprime 🔴", "#ef4444"
    lv1.markdown(f'<div class="risk-box"><div class="risk-label">Credit Tier</div>'
                 f'<div class="risk-value" style="color:{tier_col}">{tier}</div></div>',
                 unsafe_allow_html=True)

    # DTI gauge text
    dti_pct = dti_ratio * 100
    dti_col = "#ef4444" if dti_pct > 43 else "#f59e0b" if dti_pct > 30 else "#22c55e"
    lv2.markdown(f'<div class="risk-box"><div class="risk-label">DTI Ratio</div>'
                 f'<div class="risk-value" style="color:{dti_col}">{dti_pct:.0f}%</div></div>',
                 unsafe_allow_html=True)

    # LTI ratio
    lti_val = f"{loan_amount/income:.2f}×" if income > 0 else "N/A"
    lti_num = loan_amount/income if income > 0 else 0
    lti_col = "#ef4444" if lti_num > 4 else "#f59e0b" if lti_num > 2 else "#22c55e"
    lv3.markdown(f'<div class="risk-box"><div class="risk-label">LTI Ratio</div>'
                 f'<div class="risk-value" style="color:{lti_col}">{lti_val}</div></div>',
                 unsafe_allow_html=True)

    # Savings-to-loan coverage
    stl = savings / loan_amount if loan_amount > 0 else 0
    stl_col = "#22c55e" if stl >= 0.2 else "#f59e0b" if stl >= 0.1 else "#ef4444"
    lv4.markdown(f'<div class="risk-box"><div class="risk-label">Savings Coverage</div>'
                 f'<div class="risk-value" style="color:{stl_col}">{stl*100:.0f}%</div></div>',
                 unsafe_allow_html=True)

    # ── DTI gauge chart ──
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=dti_pct,
        title={"text": "DTI Gauge", "font": {"color": "#e2e8f0"}},
        number={"suffix": "%", "font": {"color": "#e2e8f0"}},
        gauge={
            "axis": {"range": [0, 60], "tickcolor": "#64748b"},
            "bar": {"color": dti_col},
            "steps": [{"range": [0,30],"color":"#14532d"},
                      {"range": [30,43],"color":"#713f12"},
                      {"range": [43,60],"color":"#7f1d1d"}],
            "threshold": {"line": {"color": "white","width": 2}, "value": 43}
        }
    ))
    fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                             height=220, margin=dict(t=40,b=10,l=20,r=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # ── Predict button ──
    st.markdown("---")
    if st.button("🔮 Run Prediction", use_container_width=True, type="primary"):
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
        pred, conf = predict_applicant(row_dict, selected_model, selected_variant,
                                        model_results, scaler, feat_cols)

        res_class = "result-approved" if pred == 1 else "result-rejected"
        res_icon  = "✅ APPROVED" if pred == 1 else "❌ REJECTED"
        res_sub   = f"Confidence: {conf:.1f}%" if conf is not None else ""
        st.markdown(f'<div class="{res_class}"><div class="result-title">{res_icon}</div>'
                    f'<div class="result-sub">{res_sub}</div></div>', unsafe_allow_html=True)

        if conf is not None:
            st.markdown("<br>", unsafe_allow_html=True)
            fig_conf = go.Figure(go.Indicator(
                mode="gauge+number", value=conf,
                title={"text": "Confidence Score", "font": {"color": "#e2e8f0"}},
                number={"suffix": "%", "font": {"color": "#e2e8f0", "size": 36}},
                gauge={"axis": {"range": [0,100], "tickcolor":"#64748b"},
                       "bar": {"color": "#22c55e" if pred==1 else "#ef4444"},
                       "steps":[{"range":[0,50],"color":"#1e293b"},
                                 {"range":[50,100],"color":"#1e293b"}],
                       "threshold":{"line":{"color":"white","width":2},"value":50}}
            ))
            fig_conf.update_layout(paper_bgcolor="rgba(0,0,0,0)", font_color="#e2e8f0",
                                   height=220, margin=dict(t=40,b=10,l=20,r=20))
            st.plotly_chart(fig_conf, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — Data Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🗂 Data Explorer":
    st.title("🗂 Data Explorer")

    dc1, dc2, dc3, dc4 = st.columns(4)
    search      = dc1.text_input("🔍 Search any column")
    status_f    = dc2.selectbox("Approval Status", ["All"] + df["Loan_Approved"].unique().tolist())
    employ_f    = dc3.multiselect("Employment", df["Employment_Status"].unique().tolist(),
                                  default=df["Employment_Status"].unique().tolist())
    cs_range_d  = dc4.slider("Credit Score", int(df["Credit_Score"].min()),
                              int(df["Credit_Score"].max()),
                              (int(df["Credit_Score"].min()), int(df["Credit_Score"].max())))

    view = df.copy()
    if status_f != "All":
        view = view[view["Loan_Approved"] == status_f]
    view = view[view["Employment_Status"].isin(employ_f)]
    view = view[view["Credit_Score"].between(*cs_range_d)]
    if search:
        mask = view.apply(lambda col: col.astype(str).str.contains(search, case=False)).any(axis=1)
        view = view[mask]

    st.caption(f"Showing **{len(view):,}** of **{len(df):,}** records")
    st.dataframe(view.reset_index(drop=True), use_container_width=True, height=420)

    # Download button
    csv = view.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download filtered CSV", csv, "filtered_applicants.csv",
                       "text/csv", use_container_width=True)
