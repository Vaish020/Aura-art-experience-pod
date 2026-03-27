"""
app.py — AURA India Analytics Dashboard
========================================
Main entry point for the Streamlit application.
Run: streamlit run app.py

Tab Structure:
  1. Overview        — Descriptive Analysis
  2. Diagnostic      — Why customers are / aren't interested
  3. Clustering      — K-Means persona segmentation
  4. Classification  — Interest prediction models
  5. Association     — ARM product & experience bundling
  6. Regression      — WTP / spend prediction
  7. Predict New     — Upload & score new customers (prescriptive)
"""

import streamlit as st

# ── PAGE CONFIG (must be first Streamlit call) ────────────────
st.set_page_config(
    page_title="AURA India — Analytics Dashboard",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "AURA India — Data-Driven Art Experience Analytics"
    }
)

# ── IMPORTS ───────────────────────────────────────────────────
import pandas as pd
import numpy as np

from aura_theme import GLOBAL_CSS, GOLD, TEAL, ORANGE, MUTED, SURFACE, SURFACE2, INK, BG
from aura_data import (
    load_data,
    train_classification_models,
    train_regression_models,
    train_clustering,
)

# Apply global CSS
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────
with st.sidebar:
    # Logo / Brand
    st.markdown(f"""
    <div style="padding:24px 0 8px;border-bottom:1px solid rgba(255,255,255,0.07);margin-bottom:20px;">
        <div style="font-size:28px;font-weight:900;color:{GOLD};letter-spacing:0.06em;line-height:1;">AURA</div>
        <div style="font-size:10px;letter-spacing:0.2em;text-transform:uppercase;color:{MUTED};margin-top:4px;">
            India Analytics Dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="color:{MUTED};font-size:11px;line-height:1.7;margin-bottom:20px;">
        Data-driven decision making for the AURA Art Experience Pod business — 
        2,000 respondents · 4 ML algorithms · India market
    </div>
    """, unsafe_allow_html=True)

    # Navigation
    st.markdown(f"<div style='color:{GOLD};font-size:10px;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:10px;font-weight:700;'>Navigation</div>", unsafe_allow_html=True)

    tabs_config = [
        ("📊", "Overview", "Descriptive Analysis"),
        ("🔍", "Diagnostic", "Why Customers Convert"),
        ("🧩", "Clustering", "Persona Segmentation"),
        ("🤖", "Classification", "Interest Prediction"),
        ("🔗", "Association Rules", "Product Bundling"),
        ("📈", "Regression", "WTP Prediction"),
        ("🚀", "Predict New", "Score New Customers"),
    ]

    selected_tab = st.radio(
        "Select analysis:",
        [f"{icon} {name}" for icon, name, _ in tabs_config],
        label_visibility="collapsed"
    )

    # Status indicators
    st.markdown(f"""
    <div style="margin-top:24px;padding-top:16px;border-top:1px solid rgba(255,255,255,0.06);">
        <div style="color:{MUTED};font-size:10px;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:10px;">Dataset Status</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="display:flex;flex-direction:column;gap:6px;">
        <div style="font-size:11px;color:{TEAL};">● Survey 1: 2,000 respondents</div>
        <div style="font-size:11px;color:{TEAL};">● Survey 2: 1,314 deep profiles</div>
        <div style="font-size:11px;color:{GOLD};">● ARM Transactions: 2,000 rows</div>
        <div style="font-size:11px;color:{ORANGE};">● Features: 81 columns</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div style="margin-top:24px;padding-top:16px;border-top:1px solid rgba(255,255,255,0.06);">
        <div style="color:{MUTED};font-size:10px;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:10px;">Analysis Pipeline</div>
        <div style="font-size:11px;color:{MUTED};line-height:1.8;">
            ✓ Descriptive<br>✓ Diagnostic<br>✓ Clustering (K-Means)<br>
            ✓ Classification (RF, XGB, LR)<br>✓ ARM (Apriori)<br>
            ✓ Regression (RF, GB, LR)<br>✓ Prescriptive (Upload)
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"<div style='margin-top:32px;color:{MUTED};font-size:10px;'>AURA India · 2026 · Confidential</div>", unsafe_allow_html=True)

# ── LOAD DATA ─────────────────────────────────────────────────
with st.spinner("Loading AURA dataset..."):
    df1, df2, arm, wide = load_data()

# ── TRAIN MODELS (cached) ─────────────────────────────────────
with st.spinner("Training models (cached after first run)..."):
    clf_models, clf_results, clf_feat_imp, X_test_clf, y_test_clf, X_train_clf, y_train_clf = \
        train_classification_models(df1)

    reg_models, reg_results, reg_feat_imp, X_test_reg, y_test_reg, reg_scaler = \
        train_regression_models(df1)

    km_model, df_clustered, km_scaler, best_k, k_range, inertias, silhouettes, pca = \
        train_clustering(df1)

# ── RENDER SELECTED TAB ───────────────────────────────────────
tab_name = selected_tab.split(" ", 1)[1]  # strip emoji

if tab_name == "Overview":
    import tab_overview
    tab_overview.render(df1, df2, arm, wide)

elif tab_name == "Diagnostic":
    import tab_diagnostic
    tab_diagnostic.render(df1, df2, arm, wide)

elif tab_name == "Clustering":
    import tab_clustering
    tab_clustering.render(
        df1, df2, arm, wide,
        km_model, df_clustered, km_scaler,
        best_k, k_range, inertias, silhouettes, pca
    )

elif tab_name == "Classification":
    import tab_classification
    tab_classification.render(
        df1, df2, arm, wide,
        clf_models, clf_results, clf_feat_imp,
        X_test_clf, y_test_clf, X_train_clf, y_train_clf
    )

elif tab_name == "Association Rules":
    import tab_arm
    tab_arm.render(df1, df2, arm, wide)

elif tab_name == "Regression":
    import tab_regression
    tab_regression.render(
        df1, df2, arm, wide,
        reg_models, reg_results, reg_feat_imp,
        X_test_reg, y_test_reg, reg_scaler
    )

elif tab_name == "Predict New":
    import tab_predict
    tab_predict.render(
        df1, df2, arm, wide,
        clf_models, reg_models, km_model, km_scaler
    )
