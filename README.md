# AURA India — Analytics Dashboard

A data-driven analytics dashboard for the **AURA Art Experience Pod** business concept,
built with Streamlit. Covers the full analytical pipeline from descriptive to prescriptive analysis.

## 🚀 Live App

Deploy on [Streamlit Community Cloud](https://streamlit.io/cloud):

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repo
4. Set `app.py` as the main file
5. Click Deploy

---

## 📊 Dashboard Tabs

| Tab | Analysis Type | Algorithms |
|-----|--------------|------------|
| Overview | Descriptive | Charts, KPIs, distributions |
| Diagnostic | Diagnostic | Cross-tabs, correlation, funnel |
| Clustering | Predictive | K-Means, PCA, Silhouette |
| Classification | Predictive | Random Forest, XGBoost, Logistic Regression |
| Association Rules | Predictive | Apriori (mlxtend + fallback) |
| Regression | Predictive | Random Forest, Gradient Boosting, Linear |
| Predict New | Prescriptive | Upload CSV → instant scoring |

---

## 📁 File Structure

```
app.py                    # Main Streamlit entry point
aura_theme.py             # Brand colors, Plotly template, CSS
aura_data.py              # Data loading, preprocessing, model training
tab_overview.py           # Tab 1 — Descriptive
tab_diagnostic.py         # Tab 2 — Diagnostic
tab_clustering.py         # Tab 3 — Clustering
tab_classification.py     # Tab 4 — Classification
tab_arm.py                # Tab 5 — Association Rules
tab_regression.py         # Tab 6 — Regression
tab_predict.py            # Tab 7 — New Customer Scoring
requirements.txt          # Pinned dependencies
.streamlit/config.toml    # AURA dark theme config
aura_survey1_n2000.csv    # Main survey dataset (2,000 rows)
aura_survey2_n1314.csv    # Deep profile dataset (1,314 rows)
aura_arm_transactions.csv # ARM binary basket matrix
aura_combined_wide.csv    # Merged wide dataset
```

---

## ⚙️ Local Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/aura-india-dashboard.git
cd aura-india-dashboard

# Create virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## 🔮 New Customer Prediction

The **Predict New** tab accepts a CSV upload with new survey respondents and outputs:

- `predicted_interest` — Interested / Maybe / Not Interested
- `confidence_score` — Model confidence (0–1)
- `predicted_wtp_inr` — Estimated willingness-to-pay in ₹
- `assigned_cluster` — Customer persona name
- `recommended_action` — Plain-English marketing action

Download the template CSV from within the app to get the correct column format.

---

## 📦 Key Dependencies

- `streamlit==1.35.0` — Dashboard framework
- `scikit-learn==1.5.0` — ML models (RF, KMeans, LR)
- `xgboost==2.0.3` — Gradient boosted classification
- `plotly==5.22.0` — Interactive charts
- `mlxtend==0.23.1` — Apriori association rules
- `imbalanced-learn==0.12.3` — SMOTE class balancing
- `pandas==2.2.2` + `numpy==1.26.4` — Data handling

---

## 🎨 AURA Brand

This dashboard uses the AURA brand palette:
- **Gold** `#e8c547` — Primary accent
- **Teal** `#5ec4a1` — Positive / success
- **Orange** `#e07c3a` — Warning / secondary
- **Rose** `#e06b8b` — Negative / alert
- **Dark Background** `#0e0c0a`

---

*AURA India · 2026 · Confidential Business Intelligence*
