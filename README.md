# 🏦 UniversalBank · Loan Intelligence Platform

> An AI-powered dark-theme Streamlit dashboard for analysing customer loan acceptance patterns — featuring descriptive, diagnostic, predictive and prescriptive analytics with a live personalised offer engine.

---

## 🚀 Live Demo

Deploy instantly → **[Streamlit Cloud](https://streamlit.io/cloud)**

---

## 📊 Dashboard Modules

| Tab | What it covers |
|-----|---------------|
| 📊 **Descriptive** | Demographics, income & age distributions, drill-down donut charts (2-level interactive), product ownership analysis |
| 🔬 **Diagnostic** | Correlation heatmap, feature importance, box plots, violin charts, statistical t-tests, effect sizes |
| 🤖 **Predictive** | 4 ML models compared (LR · DT · RF · GBT) — ROC curves, radar chart, confusion matrices |
| 🎯 **Prescriptive** | 5-tier targeting segments, budget ROI simulator, campaign messaging by tier |
| 🔮 **Predict Me** | Live customer predictor with ensemble probability gauge + **personalised loan offer engine** |

---

## 🛠️ Local Setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/universalbank-loan-dashboard.git
cd universalbank-loan-dashboard

# 2. Install
pip install -r requirements.txt

# 3. Place data file in root directory
# UniversalBank_updated.xlsx  ← drop here

# 4. Run
streamlit run app.py
```

---

## 🌐 Deploy on Streamlit Cloud (free)

1. Push this repo to GitHub
2. Go to **[share.streamlit.io](https://share.streamlit.io)**
3. Connect GitHub → select this repo
4. Main file path: `app.py`
5. Click **Deploy** ✅

> **Note**: Include `UniversalBank_updated.xlsx` in your repo root, or the app falls back to synthetic data automatically.

---

## 🎨 Design

- **Dark neon theme** — deep navy `#0a0e1a` with cyan `#00d4ff`, violet `#7c3aed`, emerald `#10b981` accents
- **JetBrains Mono** for data labels, **Space Grotesk** for body text
- Glassmorphism cards, animated metric bars, gradient headers

---

## 📦 Stack

```
streamlit · plotly · scikit-learn · pandas · numpy · scipy
```

---

## 🔑 Key Findings

- Only **9.6%** of customers accept personal loans
- **CD Account holders** accept at **46%** — 5× the baseline
- Loan acceptors earn **$145K avg income** vs **$66K** for non-acceptors
- **CCAvg** is 2.3× higher for acceptors ($3.9K vs $1.7K/mo)
- Top 20% of customers (by AI score) capture ~55% of potential loans

---

## 📁 File Structure

```
universalbank-loan-dashboard/
├── app.py                        ← complete single-file dashboard
├── requirements.txt
├── .streamlit/
│   └── config.toml               ← dark theme config
├── UniversalBank_updated.xlsx    ← dataset (add to repo)
└── README.md
```
