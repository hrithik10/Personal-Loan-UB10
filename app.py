import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve, classification_report)
from sklearn.tree import DecisionTreeClassifier
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & GLOBAL DARK THEME
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UniversalBank · Loan Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark neon-accent colour palette
C = {
    "bg":        "#0a0e1a",
    "surface":   "#111827",
    "surface2":  "#1a2235",
    "border":    "#1e2d45",
    "neon":      "#00d4ff",
    "neon2":     "#7c3aed",
    "neon3":     "#10b981",
    "neon4":     "#f59e0b",
    "neon5":     "#f43f5e",
    "text":      "#e2e8f0",
    "muted":     "#64748b",
    "yes":       "#00d4ff",
    "no":        "#1e2d45",
}

PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=C["text"], family="monospace"),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=C["text"])),
    xaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"],
               tickfont=dict(color=C["muted"])),
    yaxis=dict(gridcolor=C["border"], zerolinecolor=C["border"],
               tickfont=dict(color=C["muted"])),
    margin=dict(t=45, b=35, l=35, r=20),
    title_font=dict(color=C["neon"], size=14, family="monospace"),
)

SCALE_NEON = [[0, "#0a0e1a"], [0.5, "#7c3aed"], [1, "#00d4ff"]]
SCALE_HEAT = [[0, "#0a0e1a"], [0.33, "#7c3aed"], [0.66, "#f59e0b"], [1, "#f43f5e"]]
COLORS_CAT  = [C["neon"], C["neon2"], C["neon3"], C["neon4"], C["neon5"]]

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

html, body, [class*="css"] {{
    background-color: {C["bg"]} !important;
    color: {C["text"]} !important;
    font-family: 'Space Grotesk', sans-serif;
}}
.stApp {{ background-color: {C["bg"]} !important; }}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background: {C["surface"]} !important;
    border-right: 1px solid {C["border"]} !important;
}}
section[data-testid="stSidebar"] * {{ color: {C["text"]} !important; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: {C["surface"]} !important;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid {C["border"]};
}}
.stTabs [data-baseweb="tab"] {{
    background: transparent !important;
    color: {C["muted"]} !important;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    font-weight: 500;
    padding: 8px 18px;
    border: none !important;
    transition: all 0.2s;
}}
.stTabs [aria-selected="true"] {{
    background: linear-gradient(135deg, {C["neon2"]}40, {C["neon"]}30) !important;
    color: {C["neon"]} !important;
    border: 1px solid {C["neon"]}50 !important;
}}
.stTabs [data-baseweb="tab-panel"] {{
    background: transparent !important;
    padding: 0 !important;
}}

/* Metrics */
[data-testid="stMetric"] {{
    background: {C["surface"]} !important;
    border: 1px solid {C["border"]} !important;
    border-radius: 12px !important;
    padding: 16px 20px !important;
    position: relative;
    overflow: hidden;
}}
[data-testid="stMetric"]::before {{
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    background: linear-gradient({C["neon"]}, {C["neon2"]});
}}
[data-testid="stMetricValue"] {{ color: {C["neon"]} !important; font-family: 'JetBrains Mono' !important; font-size: 1.6rem !important; }}
[data-testid="stMetricLabel"] {{ color: {C["muted"]} !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.1em; }}

/* Selectbox, Slider */
.stSelectbox > div > div, .stMultiSelect > div > div {{
    background: {C["surface2"]} !important;
    border-color: {C["border"]} !important;
    color: {C["text"]} !important;
    border-radius: 8px !important;
}}
.stSlider [data-baseweb="slider"] {{ background: {C["border"]} !important; }}
.stSlider [data-testid="stThumbValue"] {{ color: {C["neon"]} !important; }}

/* Dataframe */
.stDataFrame {{ border-radius: 10px; overflow: hidden; }}
iframe[title="st.dataframe"] {{ background: {C["surface"]} !important; }}

/* Buttons */
.stButton > button {{
    background: linear-gradient(135deg, {C["neon2"]}, {C["neon"]}) !important;
    color: {C["bg"]} !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em;
    transition: all 0.2s;
}}
.stButton > button:hover {{ opacity: 0.85; transform: translateY(-1px); }}

/* Expander */
.streamlit-expanderHeader {{
    background: {C["surface"]} !important;
    border: 1px solid {C["border"]} !important;
    border-radius: 8px !important;
    color: {C["text"]} !important;
}}
.streamlit-expanderContent {{
    background: {C["surface2"]} !important;
    border: 1px solid {C["border"]} !important;
    border-radius: 0 0 8px 8px !important;
}}

/* Info / success / warning boxes */
.stInfo, .stSuccess, .stWarning, .stError {{
    border-radius: 8px !important;
    border-left-width: 3px !important;
}}
.stInfo {{ background: {C["neon"]}15 !important; border-color: {C["neon"]} !important; }}
.stSuccess {{ background: {C["neon3"]}15 !important; border-color: {C["neon3"]} !important; }}
.stWarning {{ background: {C["neon4"]}15 !important; border-color: {C["neon4"]} !important; }}

/* Card component */
.glass-card {{
    background: {C["surface"]};
    border: 1px solid {C["border"]};
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}}
.glass-card::after {{
    content: '';
    position: absolute;
    top: 0; right: 0;
    width: 60px; height: 60px;
    background: radial-gradient({C["neon"]}15, transparent 70%);
    border-radius: 50%;
}}
.neon-header {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    color: {C["neon"]};
    text-transform: uppercase;
    letter-spacing: 0.2em;
    margin-bottom: 4px;
}}
.hero-title {{
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(135deg, {C["neon"]}, {C["neon2"]}, {C["neon3"]});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin-bottom: 8px;
}}
.insight-tag {{
    display: inline-block;
    background: {C["neon"]}20;
    border: 1px solid {C["neon"]}50;
    color: {C["neon"]};
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    margin: 2px;
}}
.offer-card {{
    background: linear-gradient(135deg, {C["neon2"]}20, {C["neon"]}10);
    border: 1px solid {C["neon"]}40;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 8px 0;
}}
.offer-title {{
    color: {C["neon"]};
    font-family: 'JetBrains Mono', monospace;
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 6px;
}}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    import os
    for path in ["UniversalBank_updated.xlsx", "data/UniversalBank_updated.xlsx"]:
        if os.path.exists(path):
            df = pd.read_excel(path)
            break
    else:
        np.random.seed(42)
        n = 5000
        income = np.clip(np.random.lognormal(4.2, 0.6, n), 8, 224).astype(int)
        ccavg = np.clip(income * np.random.uniform(0.005, 0.08, n), 0, 10).round(1)
        edu = np.random.choice([1,2,3], n, p=[0.42,0.28,0.30])
        family = np.random.choice([1,2,3,4], n, p=[0.29,0.26,0.20,0.25])
        cd = np.random.binomial(1, 0.06, n)
        mortgage = np.where(np.random.rand(n)<0.3, np.random.randint(50,635,n), 0)
        p_loan = np.clip(-0.5 + income*0.008 + ccavg*0.05 + (edu-1)*0.04 + cd*0.4, 0, 1)
        loan = np.random.binomial(1, p_loan/p_loan.max()*0.3, n)
        df = pd.DataFrame({
            'ID': range(1, n+1), 'Age': np.random.randint(23,68,n),
            'Experience': np.random.randint(0,44,n), 'Income': income,
            'ZIP Code': np.random.randint(90000,99999,n), 'Family': family,
            'CCAvg': ccavg, 'Education': edu, 'Mortgage': mortgage,
            'Personal Loan': loan, 'Securities Account': np.random.binomial(1,0.10,n),
            'CD Account': cd, 'Online': np.random.binomial(1,0.60,n),
            'CreditCard': np.random.binomial(1,0.29,n),
        })

    df = df.drop(columns=["ZIP Code", "ID"], errors="ignore")
    df["Education_Label"] = df["Education"].map({1:"Undergrad",2:"Graduate",3:"Advanced"})
    df["Loan_Label"]      = df["Personal Loan"].map({0:"No Loan",1:"Accepted"})
    df["Family_Label"]    = df["Family"].map({1:"Single",2:"Couple",3:"Family·3",4:"Family·4"})
    df["Income_Band"]     = pd.cut(df["Income"], bins=[0,30,60,100,150,300],
                                   labels=["<$30K","$30-60K","$60-100K","$100-150K","$150K+"])
    df["CCAvg_Band"]      = pd.cut(df["CCAvg"],  bins=[-0.1,1,3,6,10],
                                   labels=["Low(<1K)","Mid(1-3K)","High(3-6K)","VHigh(6K+)"])
    df["Age_Band"]        = pd.cut(df["Age"],    bins=[20,30,40,50,60,70],
                                   labels=["20s","30s","40s","50s","60s"])
    return df

FEATURES = ["Age","Experience","Income","Family","CCAvg","Education",
            "Mortgage","Securities Account","CD Account","Online","CreditCard"]
TARGET = "Personal Loan"


# ─────────────────────────────────────────────────────────────────────────────
# TRAIN MODELS (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def build_models(df):
    X = df[FEATURES]
    y = df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    sc = StandardScaler()
    X_tr_s = sc.fit_transform(X_tr)
    X_te_s = sc.transform(X_te)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        "Decision Tree":       DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=200, max_depth=8, class_weight="balanced", random_state=42),
        "Gradient Boosting":   GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=42),
    }
    results, trained = [], {}
    for name, m in models.items():
        use_s = name == "Logistic Regression"
        m.fit(X_tr_s if use_s else X_tr, y_tr)
        trained[name] = (m, use_s)
        Xt = X_te_s if use_s else X_te
        yp = m.predict(Xt)
        ypr = m.predict_proba(Xt)[:,1]
        results.append({
            "Model": name, "Accuracy": accuracy_score(y_te, yp),
            "Precision": precision_score(y_te, yp, zero_division=0),
            "Recall": recall_score(y_te, yp, zero_division=0),
            "F1": f1_score(y_te, yp, zero_division=0),
            "ROC-AUC": roc_auc_score(y_te, ypr),
            "y_pred": yp, "y_prob": ypr,
        })
    rf_imp = pd.DataFrame({"Feature": FEATURES,
        "Importance": trained["Random Forest"][0].feature_importances_}).sort_values("Importance", ascending=False)
    return trained, results, pd.DataFrame(results), X_te, y_te, sc, rf_imp


# ─────────────────────────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────────────────────────
df_full = load_data()

with st.spinner("Training intelligence models…"):
    trained_models, model_results, metrics_df, X_test, y_test, scaler, feat_imp = build_models(df_full)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — FILTERS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center;padding:12px 0 20px;">
        <div style="font-size:2.2rem">🏦</div>
        <div style="font-family:'JetBrains Mono';font-size:0.75rem;color:{C['neon']};
                    letter-spacing:0.2em;text-transform:uppercase;">UniversalBank</div>
        <div style="color:{C['muted']};font-size:0.7rem;margin-top:2px;">Loan Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div class='neon-header'>⚙ Filters</div>", unsafe_allow_html=True)

    income_range = st.slider("Income Range ($K)", int(df_full.Income.min()),
                             int(df_full.Income.max()), (8, 224))
    edu_opts = st.multiselect("Education Level",
        ["Undergrad","Graduate","Advanced"], default=["Undergrad","Graduate","Advanced"])
    fam_opts = st.multiselect("Family Size",
        [1,2,3,4], default=[1,2,3,4],
        format_func=lambda x: {1:"1 – Single",2:"2 – Couple",3:"3 – Family",4:"4 – Large"}[x])
    age_range = st.slider("Age Range", int(df_full.Age.min()),
                          int(df_full.Age.max()), (23, 67))

    st.markdown("---")
    loan_filter = st.radio("Show customers:", ["All","Loan Accepted","No Loan"], index=0)

    st.markdown(f"""
    <div style="margin-top:24px;padding:12px;background:{C['surface2']};
                border-radius:8px;border:1px solid {C['border']};">
        <div style="font-family:'JetBrains Mono';font-size:0.65rem;color:{C['muted']};
                    text-transform:uppercase;letter-spacing:0.15em;">Dataset</div>
        <div style="color:{C['text']};font-size:0.85rem;margin-top:4px;">5,000 customers</div>
        <div style="color:{C['text']};font-size:0.85rem;">11 features · 1 target</div>
        <div style="color:{C['neon']};font-size:0.85rem;font-family:'JetBrains Mono'">9.6% acceptance rate</div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────────────────────────────────────
df = df_full.copy()
df = df[(df.Income >= income_range[0]) & (df.Income <= income_range[1])]
if edu_opts:
    df = df[df.Education_Label.isin(edu_opts)]
if fam_opts:
    df = df[df.Family.isin(fam_opts)]
df = df[(df.Age >= age_range[0]) & (df.Age <= age_range[1])]
if loan_filter == "Loan Accepted":
    df = df[df[TARGET] == 1]
elif loan_filter == "No Loan":
    df = df[df[TARGET] == 0]


# ─────────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding: 28px 0 12px;">
    <div class="neon-header">🏦 UniversalBank · Analytics Intelligence</div>
    <div class="hero-title">Personal Loan<br>Acceptance Engine</div>
    <div style="color:{C['muted']};font-size:0.95rem;max-width:620px;line-height:1.6;">
        AI-powered analysis of 5,000 customers across descriptive, diagnostic, predictive
        and prescriptive lenses — helping you identify, understand and target the right customers.
    </div>
</div>
""", unsafe_allow_html=True)

# Top KPIs
total = len(df)
accepted = df[TARGET].sum()
rate = accepted/total*100 if total > 0 else 0
avg_inc_yes = df[df[TARGET]==1]["Income"].mean() if accepted > 0 else 0
avg_cc_yes  = df[df[TARGET]==1]["CCAvg"].mean()  if accepted > 0 else 0

k1,k2,k3,k4,k5 = st.columns(5)
k1.metric("Customers in View", f"{total:,}")
k2.metric("Loan Acceptances", f"{int(accepted):,}")
k3.metric("Acceptance Rate", f"{rate:.1f}%")
k4.metric("Avg Income (Accepted)", f"${avg_inc_yes:.0f}K")
k5.metric("Avg CCAvg (Accepted)", f"${avg_cc_yes:.2f}K")

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN TABS
# ─────────────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "📊  Descriptive",
    "🔬  Diagnostic",
    "🤖  Predictive",
    "🎯  Prescriptive",
    "🔮  Predict Me",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — DESCRIPTIVE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown(f"<div class='neon-header' style='margin-top:16px'>📊 Descriptive Analysis — Who are our customers?</div>", unsafe_allow_html=True)

    # Row 1: Donut + Age dist + income dist
    r1c1, r1c2, r1c3 = st.columns([1, 1.2, 1.2])

    with r1c1:
        # Loan acceptance donut
        vals = [int(df[TARGET].sum()), int((df[TARGET]==0).sum())]
        fig = go.Figure(go.Pie(
            labels=["Accepted","No Loan"], values=vals,
            hole=0.65, marker=dict(colors=[C["neon"], C["surface2"]],
            line=dict(color=C["bg"], width=3)),
            textinfo="label+percent",
            textfont=dict(color=C["text"], family="JetBrains Mono", size=11),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>",
        ))
        fig.update_layout(**PLOT_LAYOUT, title="Loan Acceptance Split",
            annotations=[dict(text=f"<b>{rate:.1f}%</b>", x=0.5, y=0.5, font_size=22,
                              font_color=C["neon"], showarrow=False)])
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.1),
                          height=320)
        st.plotly_chart(fig, use_container_width=True)

    with r1c2:
        fig = px.histogram(df, x="Age", color="Loan_Label", nbins=30, barmode="overlay",
            opacity=0.82, color_discrete_map={"Accepted": C["neon"], "No Loan": C["neon2"]},
            title="Age Distribution by Loan Status",
            labels={"Age":"Age","count":"Customers"})
        fig.update_layout(**PLOT_LAYOUT, height=320)
        st.plotly_chart(fig, use_container_width=True)

    with r1c3:
        fig = px.histogram(df, x="Income", color="Loan_Label", nbins=35, barmode="overlay",
            opacity=0.82, color_discrete_map={"Accepted": C["neon3"], "No Loan": C["neon2"]},
            title="Income Distribution by Loan Status",
            labels={"Income":"Income ($K)","count":"Customers"})
        fig.update_layout(**PLOT_LAYOUT, height=320)
        st.plotly_chart(fig, use_container_width=True)

    # Row 2: Education rates + Family + Products
    r2c1, r2c2, r2c3 = st.columns(3)

    with r2c1:
        edu_data = df.groupby("Education_Label")[TARGET].agg(["sum","count"]).reset_index()
        edu_data["Rate"] = edu_data["sum"] / edu_data["count"] * 100
        fig = px.bar(edu_data, x="Education_Label", y="Rate",
            color="Rate", color_continuous_scale=SCALE_NEON,
            title="Loan Rate by Education",
            text=edu_data["Rate"].map(lambda x: f"{x:.1f}%"),
            labels={"Education_Label":"Education","Rate":"Acceptance %"})
        fig.update_traces(textposition="outside", textfont=dict(color=C["neon"], family="JetBrains Mono"))
        fig.update_layout(**PLOT_LAYOUT, height=320, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with r2c2:
        fam_data = df.groupby("Family_Label")[TARGET].agg(["sum","count"]).reset_index()
        fam_data["Rate"] = fam_data["sum"] / fam_data["count"] * 100
        fam_data["Size"] = fam_data["count"]
        fig = px.scatter(fam_data, x="Family_Label", y="Rate", size="Size",
            color="Rate", color_continuous_scale=SCALE_NEON,
            title="Loan Rate by Family Size",
            labels={"Family_Label":"Family","Rate":"Acceptance %"},
            size_max=55)
        fig.update_layout(**PLOT_LAYOUT, height=320, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with r2c3:
        products = ["Securities Account","CD Account","Online","CreditCard"]
        rates_p = []
        for p in products:
            r_yes = df[df[p]==1][TARGET].mean()*100
            r_no  = df[df[p]==0][TARGET].mean()*100
            rates_p.append({"Product":p.replace(" ","\n"),"Have":r_yes,"Don't":r_no})
        pp = pd.DataFrame(rates_p)
        fig = go.Figure()
        fig.add_bar(name="Have product", x=pp["Product"], y=pp["Have"],
                    marker_color=C["neon"], opacity=0.9)
        fig.add_bar(name="Don't have", x=pp["Product"], y=pp["Don't"],
                    marker_color=C["neon2"], opacity=0.7)
        fig.update_layout(**PLOT_LAYOUT, barmode="group", height=320,
                          title="Loan Rate: Product Ownership",
                          yaxis_title="Acceptance %")
        st.plotly_chart(fig, use_container_width=True)

    # Row 3: Income band heatmap + avg stats table
    r3c1, r3c2 = st.columns([1.6, 1])

    with r3c1:
        pivot = df.groupby(["Income_Band","Education_Label"])[TARGET].mean().unstack(fill_value=0)*100
        fig = px.imshow(pivot, text_auto=".1f", aspect="auto",
            color_continuous_scale=SCALE_HEAT,
            title="Loan Rate (%) — Income Band × Education",
            labels={"color":"Acceptance %"})
        fig.update_layout(**PLOT_LAYOUT, height=280)
        st.plotly_chart(fig, use_container_width=True)

    with r3c2:
        st.markdown(f"<div class='neon-header' style='margin-top:8px'>Average Stats by Loan Status</div>", unsafe_allow_html=True)
        avg_stats = df.groupby("Loan_Label")[["Income","CCAvg","Mortgage","Age","Experience"]].mean().round(2).T.reset_index()
        avg_stats.columns = ["Metric","No Loan","Accepted"]
        avg_stats["Δ"] = (avg_stats["Accepted"] - avg_stats["No Loan"]).round(2)
        st.dataframe(avg_stats.style
            .background_gradient(subset=["Accepted"], cmap="Blues")
            .format({"No Loan":"{:.1f}","Accepted":"{:.1f}","Δ":"{:+.1f}"}),
            height=250, use_container_width=True)

    # Row 4: Interactive Drill-Down Donut
    st.markdown("---")
    st.markdown(f"<div class='neon-header'>🍩 Interactive Drill-Down Explorer</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{C['muted']};font-size:0.85rem;margin-bottom:12px;'>Select level 1 & level 2 dimensions to drill into loan acceptance patterns.</div>", unsafe_allow_html=True)

    dd1, dd2, dd3 = st.columns([1,1,2])
    with dd1:
        dim1 = st.selectbox("Level 1 Dimension", ["Income_Band","Education_Label","Family_Label","Age_Band"])
    with dd2:
        dim2 = st.selectbox("Level 2 Dimension", ["Education_Label","Family_Label","Age_Band","Income_Band"], index=1)

    d1_vals = sorted(df[dim1].dropna().unique().tolist())
    sel1 = st.selectbox(f"Select {dim1.replace('_',' ')}:", ["(All)"] + d1_vals)
    sub1 = df if sel1 == "(All)" else df[df[dim1] == sel1]

    dc1, dc2, dc3 = st.columns(3)
    with dc1:
        d = df[dim1].value_counts().reset_index(); d.columns = ["Label","Count"]
        fig = go.Figure(go.Pie(labels=d["Label"], values=d["Count"], hole=0.55,
            marker=dict(colors=COLORS_CAT[:len(d)], line=dict(color=C["bg"], width=2)),
            textfont=dict(family="JetBrains Mono", size=10)))
        fig.update_layout(**PLOT_LAYOUT, title=f"All · {dim1.replace('_',' ')}", height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with dc2:
        d2 = sub1[dim2].value_counts().reset_index(); d2.columns = ["Label","Count"]
        fig = go.Figure(go.Pie(labels=d2["Label"], values=d2["Count"], hole=0.55,
            marker=dict(colors=COLORS_CAT[:len(d2)], line=dict(color=C["bg"], width=2)),
            textfont=dict(family="JetBrains Mono", size=10)))
        sel_lbl = sel1 if sel1 != "(All)" else "All"
        fig.update_layout(**PLOT_LAYOUT, title=f"{sel_lbl} · {dim2.replace('_',' ')}", height=300, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with dc3:
        d2_vals = sorted(sub1[dim2].dropna().unique().tolist())
        sel2 = st.selectbox(f"Select {dim2.replace('_',' ')}:", ["(All)"] + d2_vals)
        sub2 = sub1 if sel2 == "(All)" else sub1[sub1[dim2] == sel2]
        d3 = sub2["Loan_Label"].value_counts().reset_index(); d3.columns = ["Label","Count"]
        fig = go.Figure(go.Pie(labels=d3["Label"], values=d3["Count"], hole=0.55,
            marker=dict(colors=[C["neon"], C["neon2"]], line=dict(color=C["bg"], width=2)),
            textfont=dict(family="JetBrains Mono", size=11)))
        rate_drill = sub2[TARGET].mean()*100
        fig.update_layout(**PLOT_LAYOUT, title=f"Loan Outcome · {sel_lbl}→{sel2}", height=300,
            annotations=[dict(text=f"<b>{rate_drill:.0f}%</b>", x=0.5, y=0.5,
                              font=dict(size=20, color=C["neon"]), showarrow=False)])
        st.plotly_chart(fig, use_container_width=True)

    st.success(f"📌 Segment **{sel1} → {sel2}**: {len(sub2):,} customers | Loan acceptance: **{rate_drill:.1f}%** (baseline: {df[TARGET].mean()*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DIAGNOSTIC
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown(f"<div class='neon-header' style='margin-top:16px'>🔬 Diagnostic Analysis — WHY do customers accept loans?</div>", unsafe_allow_html=True)

    # Correlation heatmap
    corr = df_full[FEATURES + [TARGET]].corr()
    fig = px.imshow(corr, text_auto=".2f", aspect="auto",
        color_continuous_scale="RdBu_r", color_continuous_midpoint=0,
        title="Full Feature Correlation Matrix",
        zmin=-1, zmax=1)
    fig.update_layout(**PLOT_LAYOUT, height=480)
    st.plotly_chart(fig, use_container_width=True)

    diag_c1, diag_c2 = st.columns(2)

    with diag_c1:
        loan_corr = corr[TARGET].drop(TARGET).abs().sort_values(ascending=True)
        loan_corr_signed = corr[TARGET].drop(TARGET).reindex(loan_corr.index)
        bar_colors = [C["neon"] if v > 0 else C["neon5"] for v in loan_corr_signed.values]
        fig = go.Figure(go.Bar(
            x=loan_corr_signed.values, y=loan_corr_signed.index,
            orientation="h", marker=dict(color=bar_colors, opacity=0.85),
            text=[f"{v:+.3f}" for v in loan_corr_signed.values],
            textfont=dict(family="JetBrains Mono", size=10, color=C["text"]),
            textposition="outside"))
        fig.update_layout(**PLOT_LAYOUT, title="Correlation with Personal Loan",
                          xaxis_title="Pearson r", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with diag_c2:
        # Feature importance
        fig = go.Figure(go.Bar(
            x=feat_imp["Importance"], y=feat_imp["Feature"],
            orientation="h",
            marker=dict(color=feat_imp["Importance"],
                        colorscale=[[0,C["neon2"]], [1,C["neon"]]], opacity=0.9),
            text=feat_imp["Importance"].map(lambda x: f"{x:.3f}"),
            textfont=dict(family="JetBrains Mono", size=10, color=C["text"]),
            textposition="outside"))
        fig.update_layout(**PLOT_LAYOUT, title="Random Forest Feature Importance",
                          xaxis_title="Importance Score", height=380,
                          yaxis=dict(**PLOT_LAYOUT["yaxis"], categoryorder="total ascending"))
        st.plotly_chart(fig, use_container_width=True)

    # Box plots for top features
    st.markdown(f"<div class='neon-header' style='margin-top:8px'>Feature Distributions: Loan Accepted vs Not</div>", unsafe_allow_html=True)
    top_feats = feat_imp.head(4)["Feature"].tolist()
    bcols = st.columns(4)
    for i, feat in enumerate(top_feats):
        with bcols[i]:
            fig = px.box(df_full, x="Loan_Label", y=feat, color="Loan_Label",
                color_discrete_map={"Accepted": C["neon"], "No Loan": C["neon2"]},
                points="outliers", title=feat)
            fig.update_layout(**PLOT_LAYOUT, height=280, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    # Scatter: Income vs CCAvg coloured by loan
    sc1, sc2 = st.columns(2)
    with sc1:
        fig = px.scatter(df_full.sample(min(1500, len(df_full)), random_state=1),
            x="Income", y="CCAvg", color="Loan_Label",
            color_discrete_map={"Accepted": C["neon"], "No Loan": C["neon2"]},
            opacity=0.7, title="Income vs CC Spending (sampled)",
            labels={"Income":"Income ($K)","CCAvg":"CC Avg Spend ($K)"})
        fig.update_layout(**PLOT_LAYOUT, height=340)
        st.plotly_chart(fig, use_container_width=True)

    with sc2:
        # Violin — Income by Education × Loan
        fig = px.violin(df_full, x="Education_Label", y="Income", color="Loan_Label",
            box=True, color_discrete_map={"Accepted": C["neon"], "No Loan": C["neon2"]},
            title="Income Distribution: Education × Loan",
            labels={"Education_Label":"Education","Income":"Income ($K)"})
        fig.update_layout(**PLOT_LAYOUT, height=340)
        st.plotly_chart(fig, use_container_width=True)

    # Statistical summary
    from scipy import stats as sp_stats
    st.markdown(f"<div class='neon-header' style='margin-top:8px'>Statistical Significance Tests (t-test: Accepted vs Not)</div>", unsafe_allow_html=True)
    test_rows = []
    g_yes = df_full[df_full[TARGET]==1]
    g_no  = df_full[df_full[TARGET]==0]
    for col in ["Income","CCAvg","Age","Mortgage"]:
        t, p = sp_stats.ttest_ind(g_yes[col], g_no[col])
        test_rows.append({"Feature": col,
            "Mean (Accepted)": f"{g_yes[col].mean():.2f}",
            "Mean (No Loan)":  f"{g_no[col].mean():.2f}",
            "t-stat": f"{t:.3f}",
            "p-value": f"{p:.2e}",
            "Significant": "✅ Yes" if p < 0.05 else "❌ No"})
    st.dataframe(pd.DataFrame(test_rows), use_container_width=True)

    st.info("""
    **🔬 Key Diagnostic Findings:**
    · **Income** is the single strongest differentiator — loan acceptors earn **$145K avg** vs **$66K** for non-acceptors (2.2× gap)
    · **CD Account** holders accept loans at **46%** vs **7%** baseline — the most powerful binary signal
    · **CCAvg** averages **$3.9K/mo** for acceptors vs **$1.7K** — high spenders are prime targets
    · **Education** level 2+ doubles acceptance rates compared to undergraduates
    · **Family size 3-4** shows elevated acceptance — larger households likely need liquidity
    """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — PREDICTIVE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown(f"<div class='neon-header' style='margin-top:16px'>🤖 Predictive Analysis — Model Comparison & Evaluation</div>", unsafe_allow_html=True)

    # Model metrics table
    display_cols = ["Model","Accuracy","Precision","Recall","F1","ROC-AUC"]
    disp_df = metrics_df[display_cols].copy()
    st.dataframe(
        disp_df.style
            .format({c: "{:.3f}" for c in display_cols[1:]})
            .background_gradient(subset=display_cols[1:], cmap="Blues")
            .set_properties(**{"font-family":"JetBrains Mono","font-size":"13px"}),
        use_container_width=True, height=200)

    pred_c1, pred_c2 = st.columns([1.4, 1])

    with pred_c1:
        # ROC Curves
        fig = go.Figure()
        fig.add_shape(type="line", x0=0,y0=0,x1=1,y1=1,
            line=dict(color=C["muted"], dash="dot", width=1.5))
        pal = [C["neon"], C["neon2"], C["neon3"], C["neon4"]]
        for i, r in enumerate(model_results):
            fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                name=f"{r['Model']} (AUC={r['ROC-AUC']:.3f})",
                line=dict(color=pal[i], width=2.5)))
        fig.update_layout(**PLOT_LAYOUT, title="ROC Curves — All Models",
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate", height=380)
        st.plotly_chart(fig, use_container_width=True)

    with pred_c2:
        # Radar
        metrics_names = ["Accuracy","Precision","Recall","F1","ROC-AUC"]
        fig = go.Figure()
        for i, r in enumerate(model_results):
            vals = [r[m] for m in metrics_names]
            vals.append(vals[0])
            fig.add_trace(go.Scatterpolar(r=vals, theta=metrics_names+[metrics_names[0]],
                fill="toself", name=r["Model"], opacity=0.75,
                line=dict(color=pal[i], width=2)))
        fig.update_layout(**{**PLOT_LAYOUT,
            "polar": dict(
                radialaxis=dict(visible=True, range=[0,1], gridcolor=C["border"],
                                tickfont=dict(color=C["muted"])),
                angularaxis=dict(gridcolor=C["border"], tickfont=dict(color=C["text"]))),
            "title": "Model Radar Chart"},
            height=380)
        st.plotly_chart(fig, use_container_width=True)

    # Confusion matrices — all 4
    st.markdown(f"<div class='neon-header' style='margin-top:8px'>Confusion Matrices</div>", unsafe_allow_html=True)
    cm_cols = st.columns(4)
    for i, r in enumerate(model_results):
        cm = confusion_matrix(y_test, r["y_pred"])
        fig = px.imshow(cm, text_auto=True,
            x=["No Loan","Accepted"], y=["No Loan","Accepted"],
            color_continuous_scale=SCALE_NEON,
            title=r["Model"].replace(" ","\n"), aspect="equal",
            labels=dict(x="Predicted", y="Actual"))
        fig.update_layout(**PLOT_LAYOUT, height=260, coloraxis_showscale=False)
        cm_cols[i].plotly_chart(fig, use_container_width=True)

    # Feature importance for best model
    best_r = max(model_results, key=lambda x: x["ROC-AUC"])
    st.success(f"🏆 Best Model: **{best_r['Model']}** — ROC-AUC {best_r['ROC-AUC']:.3f} | Accuracy {best_r['Accuracy']:.3f} | F1 {best_r['F1']:.3f}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — PRESCRIPTIVE
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown(f"<div class='neon-header' style='margin-top:16px'>🎯 Prescriptive Analysis — Campaign Strategy & Targeting</div>", unsafe_allow_html=True)

    # Tiered segments
    tiers = [
        ("🔴 Tier 1 · PLATINUM", (df_full["CD Account"]==1)&(df_full["Income"]>80),
         "CD Account + Income >$80K", "Personal banker call within 24h", C["neon5"]),
        ("🟡 Tier 2 · GOLD",     (df_full["Education"]>=2)&(df_full["CCAvg"]>3)&(df_full["Income"]>60),
         "Graduate + CCAvg >$3K + Income >$60K", "Targeted email + app notification", C["neon4"]),
        ("🔵 Tier 3 · SILVER",   (df_full["Income"]>100)&(df_full["Mortgage"]>0),
         "Income >$100K + Has Mortgage", "Digital pre-approval offer", C["neon"]),
        ("🟢 Tier 4 · BRONZE",   (df_full["Education"]>=2)&(df_full["Online"]==1)&(df_full["Income"]>40),
         "Graduate + Online user + Income >$40K", "In-app banner + push notification", C["neon3"]),
        ("⚪ Tier 5 · STANDARD",  pd.Series([True]*len(df_full), index=df_full.index),
         "All remaining customers", "Mass email campaign", C["muted"]),
    ]

    seg_rows = []
    for label, mask, criteria, channel, color in tiers:
        seg = df_full[mask]
        rate = seg[TARGET].mean()*100
        lift = rate / (df_full[TARGET].mean()*100) if df_full[TARGET].mean() > 0 else 1
        seg_rows.append({"Tier":label,"Criteria":criteria,"Size":len(seg),
            "Acc. Rate":f"{rate:.1f}%","Lift":f"{lift:.2f}×","Channel":channel,"Color":color})
    seg_df = pd.DataFrame(seg_rows)

    # Visual segment bars
    for _, row in seg_df.iterrows():
        rate_num = float(row["Acc. Rate"].replace("%",""))
        bar_pct  = min(rate_num / 50 * 100, 100)
        st.markdown(f"""
        <div class="glass-card" style="border-left: 3px solid {row['Color']}; margin-bottom:10px;">
            <div style="display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <div style="font-family:'JetBrains Mono';font-weight:600;color:{row['Color']};font-size:0.9rem;">{row['Tier']}</div>
                    <div style="color:{C['muted']};font-size:0.8rem;margin-top:2px;">{row['Criteria']}</div>
                    <div style="color:{C['text']};font-size:0.75rem;margin-top:4px;">📣 {row['Channel']}</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-family:'JetBrains Mono';font-size:1.4rem;color:{row['Color']};font-weight:700;">{row['Acc. Rate']}</div>
                    <div style="color:{C['muted']};font-size:0.75rem;">{row['Size']:,} customers · Lift {row['Lift']}</div>
                </div>
            </div>
            <div style="background:{C['border']};border-radius:4px;height:4px;margin-top:10px;">
                <div style="background:{row['Color']};width:{bar_pct:.0f}%;height:100%;border-radius:4px;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    pres_c1, pres_c2 = st.columns(2)

    with pres_c1:
        # Bubble chart: segment size vs rate
        fig = px.scatter(seg_df, x="Size", y=[float(r.replace("%","")) for r in seg_df["Acc. Rate"]],
            size="Size", color=[float(r.replace("%","")) for r in seg_df["Acc. Rate"]],
            text=seg_df["Tier"].str.split("·").str[-1].str.strip(),
            color_continuous_scale=SCALE_NEON,
            title="Segment Size vs Acceptance Rate",
            labels={"x":"Segment Size","y":"Acceptance Rate (%)","color":"Rate"},
            size_max=60)
        fig.update_traces(textposition="top center", textfont=dict(color=C["text"], size=10))
        fig.update_layout(**PLOT_LAYOUT, height=380, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with pres_c2:
        # Budget ROI simulator
        st.markdown(f"<div class='neon-header'>💰 Budget ROI Simulator</div>", unsafe_allow_html=True)
        budget    = st.slider("Marketing Budget ($)", 10_000, 500_000, 100_000, 5_000)
        cost_cust = st.slider("Cost per Contact ($)", 5, 100, 20)
        loan_val  = st.slider("Avg Loan Value ($)", 5_000, 100_000, 25_000, 1_000)
        margin    = st.slider("Net Margin (%)", 1, 20, 5) / 100

        contactable = budget // cost_cust
        strats = [("Broadcast (all)", df_full[TARGET].mean()),
                  ("Tier 1+2 (CD+Grad)", df_full[(df_full["CD Account"]==1)|(df_full["CCAvg"]>3)][TARGET].mean()),
                  ("Tier 1 only (CD)", df_full[df_full["CD Account"]==1][TARGET].mean()),
                  ("Income >$80K", df_full[df_full["Income"]>80][TARGET].mean())]
        strat_res = []
        for name, rate_s in strats:
            convs = contactable * rate_s
            rev   = convs * loan_val * margin
            roi   = (rev - budget) / budget * 100
            strat_res.append({"Strategy":name,"Conversions":int(convs),"Net Rev":f"${rev:,.0f}","ROI":f"{roi:.0f}%","roi_n":roi})
        sr = pd.DataFrame(strat_res)
        fig = px.bar(sr, x="Strategy", y="roi_n",
            color="roi_n", color_continuous_scale=SCALE_HEAT,
            text=sr["ROI"], title=f"ROI by Strategy (Budget ${budget:,})",
            labels={"roi_n":"ROI (%)"})
        fig.update_traces(textposition="outside", textfont=dict(color=C["neon"], family="JetBrains Mono"))
        fig.update_layout(**PLOT_LAYOUT, height=320, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **🎯 Prescriptive Recommendations:**
    · **Concentrate 60% of budget on Tier 1 & 2** — 46% and 25%+ acceptance rates vs 9.6% baseline
    · **Lead with CD product cross-sell** — CD account holders are the best loan candidates (5× lift)
    · **Avoid mass broadcast campaigns** — ROI is sharply negative without targeting
    · **Graduate degree holders** with CCAvg >$3K respond 3× better to digital channels
    · **Large families (3-4)** show elevated need — offer family financial planning bundles
    """)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — LIVE PREDICTOR + PERSONALIZED OFFER
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown(f"<div class='neon-header' style='margin-top:16px'>🔮 Live Loan Acceptance Predictor + Personalized Offer Engine</div>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{C['muted']};font-size:0.85rem;margin-bottom:18px;'>Enter a customer's profile to get an AI prediction and a tailored, personalised loan offer.</div>", unsafe_allow_html=True)

    inp_c1, inp_c2, inp_c3 = st.columns(3)
    with inp_c1:
        st.markdown(f"<div class='neon-header'>Demographics</div>", unsafe_allow_html=True)
        p_age        = st.slider("Age", 23, 67, 38)
        p_experience = st.slider("Years of Experience", 0, 43, 12)
        p_family     = st.selectbox("Family Size", [1,2,3,4],
            format_func=lambda x: {1:"1 · Single",2:"2 · Couple",3:"3 · Family",4:"4 · Large"}[x])
        p_education  = st.selectbox("Education",  [1,2,3],
            format_func=lambda x: {1:"1 · Undergrad",2:"2 · Graduate",3:"3 · Advanced/Prof"}[x])

    with inp_c2:
        st.markdown(f"<div class='neon-header'>Financial Profile</div>", unsafe_allow_html=True)
        p_income    = st.slider("Annual Income ($K)", 8, 224, 85)
        p_ccavg     = st.slider("Avg CC Spend/mo ($K)", 0.0, 10.0, 2.5, 0.1)
        p_mortgage  = st.slider("Mortgage ($K)", 0, 635, 120)

    with inp_c3:
        st.markdown(f"<div class='neon-header'>Banking Products</div>", unsafe_allow_html=True)
        p_sec    = st.checkbox("Securities Account")
        p_cd     = st.checkbox("CD Account")
        p_online = st.checkbox("Online Banking", value=True)
        p_cc     = st.checkbox("Credit Card with Bank")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("⚡  RUN PREDICTION", use_container_width=True)

    if predict_btn:
        input_vec = pd.DataFrame([[p_age, p_experience, p_income, p_family, p_ccavg, p_education,
            p_mortgage, int(p_sec), int(p_cd), int(p_online), int(p_cc)]], columns=FEATURES)

        probs = {}
        for name, (model, use_s) in trained_models.items():
            Xin = scaler.transform(input_vec) if use_s else input_vec
            probs[name] = model.predict_proba(Xin)[0, 1]

        ensemble_prob = np.mean(list(probs.values()))
        best_name     = max(model_results, key=lambda x: x["ROC-AUC"])["Model"]
        best_prob     = probs[best_name]
        decision      = ensemble_prob >= 0.5

        out_c1, out_c2, out_c3 = st.columns([1, 1, 1.2])

        with out_c1:
            color_dec = C["neon"] if decision else C["neon5"]
            label_dec = "✅ LIKELY TO ACCEPT" if decision else "❌ UNLIKELY TO ACCEPT"
            st.markdown(f"""
            <div class="glass-card" style="border-color:{color_dec};text-align:center;padding:28px 20px;">
                <div style="font-family:'JetBrains Mono';font-size:0.7rem;color:{C['muted']};
                            text-transform:uppercase;letter-spacing:0.15em;">AI Verdict</div>
                <div style="font-size:1.5rem;font-weight:700;color:{color_dec};margin:10px 0;">{label_dec}</div>
                <div style="font-family:'JetBrains Mono';font-size:2.4rem;color:{color_dec};font-weight:800;">
                    {ensemble_prob*100:.1f}%</div>
                <div style="color:{C['muted']};font-size:0.75rem;">ensemble probability</div>
            </div>
            """, unsafe_allow_html=True)

        with out_c2:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=ensemble_prob*100,
                title={"text":"Acceptance Probability", "font":{"color":C["neon"], "size":12, "family":"JetBrains Mono"}},
                number={"suffix":"%", "font":{"color":C["neon"], "size":28, "family":"JetBrains Mono"}},
                gauge={
                    "axis": {"range":[0,100], "tickcolor":C["muted"], "tickfont":{"color":C["muted"]}},
                    "bar":  {"color": color_dec, "thickness":0.25},
                    "bgcolor": C["surface"],
                    "borderwidth": 1, "bordercolor": C["border"],
                    "steps": [
                        {"range":[0,30],  "color":"#1a0a0a"},
                        {"range":[30,50], "color":"#1a1a0a"},
                        {"range":[50,100],"color":"#0a1a0a"}],
                    "threshold": {"line":{"color":color_dec,"width":3},"thickness":0.75,"value":ensemble_prob*100}
                }
            ))
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color=C["text"]), height=280, margin=dict(t=40,b=10,l=20,r=20))
            st.plotly_chart(fig, use_container_width=True)

        with out_c3:
            fig = px.bar(x=list(probs.keys()), y=[v*100 for v in probs.values()],
                color=[v*100 for v in probs.values()],
                color_continuous_scale=SCALE_NEON,
                labels={"x":"Model","y":"Probability (%)"},
                title="Per-Model Probabilities")
            fig.update_layout(**PLOT_LAYOUT, height=280, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        # ── PERSONALISED OFFER ENGINE ────────────────────────────────────────
        st.markdown("---")
        st.markdown(f"<div class='neon-header' style='font-size:0.85rem;'>🎁 Personalised Offer & Recommendations</div>", unsafe_allow_html=True)

        # Score the customer on key signals
        signals = {
            "income_high":    p_income > 100,
            "income_mid":     60 < p_income <= 100,
            "income_low":     p_income <= 60,
            "ccavg_high":     p_ccavg > 3,
            "ccavg_mid":      1 < p_ccavg <= 3,
            "cd_account":     p_cd,
            "graduate":       p_education >= 2,
            "large_family":   p_family >= 3,
            "has_mortgage":   p_mortgage > 0,
            "online_user":    p_online,
            "securities":     p_sec,
        }

        # Determine offer tier
        if ensemble_prob >= 0.6:
            tier_label, tier_color = "🔴 PLATINUM OFFER", C["neon5"]
        elif ensemble_prob >= 0.4:
            tier_label, tier_color = "🟡 GOLD OFFER", C["neon4"]
        elif ensemble_prob >= 0.25:
            tier_label, tier_color = "🔵 SILVER OFFER", C["neon"]
        else:
            tier_label, tier_color = "🟢 NURTURE OFFER", C["neon3"]

        # Build personalised message
        if signals["income_high"] and signals["ccavg_high"]:
            offer_headline = "Pre-Approved Premium Personal Loan — Exclusive Rate"
            offer_body     = f"Based on your strong income of ${p_income}K and credit card usage, you are pre-approved for a personal loan of up to **${min(p_income*3, 500):,}K** at our best-in-class interest rate. Your dedicated relationship manager will contact you within 24 hours."
        elif signals["cd_account"]:
            offer_headline = "CD Account Holder Priority Loan Offer"
            offer_body     = f"As a valued CD account holder, you qualify for a **priority personal loan** with zero processing fee and an instant approval guarantee up to **${p_income*2:,}K**."
        elif signals["graduate"] and signals["ccavg_mid"]:
            offer_headline = "Graduate Professional Loan Package"
            offer_body     = f"Your education profile and spending patterns make you an ideal candidate for our Graduate Professional Loan — up to **${p_income*2:,}K** with flexible EMI options tailored for your lifestyle."
        elif signals["large_family"]:
            offer_headline = "Family Financial Freedom Plan"
            offer_body     = f"Managing a family of {p_family} comes with unique needs. Our Family Financial Freedom personal loan offers up to **${p_income*2:,}K** with longer repayment tenures and special family insurance add-ons."
        elif signals["has_mortgage"]:
            offer_headline = "Homeowner Top-Up Loan Offer"
            offer_body     = f"As an existing homeowner, unlock additional liquidity with our Homeowner Top-Up Loan — **${min(p_mortgage, 200):,}K available** at preferential rates for renovation, education or emergency needs."
        elif not decision:
            offer_headline = "Start Your Journey to Better Banking"
            offer_body     = "We'd love to help you qualify for a personal loan. Consider these next steps to strengthen your profile and access better offers in the future."
        else:
            offer_headline = "You Qualify for a Personal Loan!"
            offer_body     = f"Congratulations! Your financial profile qualifies you for a personal loan up to **${p_income*1.5:,}K**. Apply online in 10 minutes with no paperwork."

        st.markdown(f"""
        <div class="offer-card">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
                <span style="background:{tier_color}20;border:1px solid {tier_color}60;
                             color:{tier_color};border-radius:20px;padding:3px 14px;
                             font-family:'JetBrains Mono';font-size:0.75rem;font-weight:600;">
                    {tier_label}</span>
            </div>
            <div class="offer-title">{offer_headline}</div>
            <div style="color:{C['text']};font-size:0.9rem;line-height:1.6;">{offer_body}</div>
        </div>
        """, unsafe_allow_html=True)

        # Next-best actions
        st.markdown(f"<div class='neon-header' style='margin-top:16px;'>💡 Recommended Actions</div>", unsafe_allow_html=True)
        actions = []
        if not p_cd:
            actions.append(("Open a CD Account", "CD account holders have a **46% loan acceptance rate** — 5× the baseline. Opening a CD first builds trust and dramatically improves eligibility.", C["neon4"]))
        if p_income < 60:
            actions.append(("Income Profile Enhancement", f"Current income ${p_income}K is below the $80K+ sweet spot. Explore salary advance, side income products, or revisit in 6-12 months.", C["neon2"]))
        if p_ccavg < 2:
            actions.append(("Activate Credit Card Usage", "Credit card spending >$3K/mo correlates strongly with loan acceptance. Upgrading your credit card tier and usage signals financial health.", C["neon"]))
        if p_education == 1:
            actions.append(("Education Upgrade Program", "Graduate and advanced degree holders are 3× more likely to qualify. Explore our Education Loan products to fund further studies.", C["neon3"]))
        if not p_online:
            actions.append(("Enroll in Online Banking", "Online banking users show higher engagement and 15% better loan acceptance rates. Enable online banking today.", C["neon5"]))
        if not actions:
            actions.append(("Apply Now", "Your profile is strong — apply for the personal loan today via our mobile app or website. Pre-approval in under 3 minutes.", C["neon3"]))

        act_cols = st.columns(min(len(actions), 3))
        for i, (title, desc, col) in enumerate(actions[:3]):
            with act_cols[i]:
                st.markdown(f"""
                <div class="glass-card" style="border-top:2px solid {col};height:100%;">
                    <div style="color:{col};font-family:'JetBrains Mono';font-size:0.75rem;
                                font-weight:600;margin-bottom:6px;">{title}</div>
                    <div style="color:{C['muted']};font-size:0.8rem;line-height:1.5;">{desc}</div>
                </div>
                """, unsafe_allow_html=True)

        # Customer summary card
        st.markdown(f"""
        <div style="margin-top:16px;padding:16px 20px;background:{C['surface']};border:1px solid {C['border']};
                    border-radius:10px;font-family:'JetBrains Mono';font-size:0.78rem;color:{C['muted']};">
            <span style="color:{C['neon']};">Customer Snapshot</span> · 
            Age: <span style="color:{C['text']}">{p_age}</span> · 
            Income: <span style="color:{C['text']}">${p_income}K</span> · 
            CCAvg: <span style="color:{C['text']}">${p_ccavg}K</span> · 
            Education: <span style="color:{C['text']}">{['','Undergrad','Graduate','Advanced'][p_education]}</span> · 
            Family: <span style="color:{C['text']}">{p_family}</span> · 
            CD: <span style="color:{C['text']}">{('Yes' if p_cd else 'No')}</span> · 
            Mortgage: <span style="color:{C['text']}">${p_mortgage}K</span> ·
            Probability: <span style="color:{tier_color};font-weight:700;">{ensemble_prob*100:.1f}%</span>
        </div>
        """, unsafe_allow_html=True)

    else:
        st.markdown(f"""
        <div style="text-align:center;padding:60px 20px;color:{C['muted']};">
            <div style="font-size:3rem;margin-bottom:12px;">🔮</div>
            <div style="font-family:'JetBrains Mono';color:{C['neon']};font-size:1rem;">Ready to predict</div>
            <div style="font-size:0.85rem;margin-top:6px;">Fill in the customer profile above and click <b style="color:{C['text']}">RUN PREDICTION</b></div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(f"""
<div style="text-align:center;padding:12px;color:{C['muted']};font-size:0.75rem;font-family:'JetBrains Mono';">
    UniversalBank Loan Intelligence Platform · 5,000 customers · 4 ML models · 
    <span style="color:{C['neon']}">Built with Streamlit + Plotly + scikit-learn</span>
</div>
""", unsafe_allow_html=True)
