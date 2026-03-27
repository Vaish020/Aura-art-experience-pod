"""tab_classification.py — Classification: Interest Prediction Models"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from aura_theme import (GOLD, TEAL, ORANGE, INDIGO, ROSE, MUTED, SURFACE, SURFACE2,
                         INK, PALETTE, page_header, section_header, info_card, kpi_row)


LABEL_NAMES = {0: "Not Interested", 1: "Maybe", 2: "Interested"}
CLASS_COLORS = {0: ROSE, 1: GOLD, 2: TEAL}


def render(df1, df2, arm, wide, models, results, feat_imp, X_test, y_test, X_train, y_train):
    page_header(
        "Classification Models",
        "Predicting whether a new customer will be Interested, Maybe, or Not Interested in AURA — before they even visit.",
        "04 — Predictive Analysis · Classification"
    )

    # ── MODEL SELECTOR ────────────────────────────────────────
    model_names = list(results.keys())
    selected_model = st.radio("Select model to inspect:", model_names, horizontal=True)
    res = results[selected_model]

    # ── KPI ROW ───────────────────────────────────────────────
    kpi_row([
        ("Accuracy",  f"{res['accuracy']*100:.1f}%", None),
        ("Precision", f"{res['precision']*100:.1f}%", None),
        ("Recall",    f"{res['recall']*100:.1f}%",    None),
        ("F1-Score",  f"{res['f1']*100:.1f}%",        None),
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    # ── ALL MODELS COMPARISON ─────────────────────────────────
    section_header("Model Performance Comparison", GOLD)

    comp_df = pd.DataFrame({
        "Model": list(results.keys()),
        "Accuracy":  [results[m]["accuracy"]  for m in results],
        "Precision": [results[m]["precision"] for m in results],
        "Recall":    [results[m]["recall"]    for m in results],
        "F1-Score":  [results[m]["f1"]        for m in results],
    })

    metrics_melt = comp_df.melt(id_vars="Model", var_name="Metric", value_name="Score")
    fig_comp = px.bar(
        metrics_melt, x="Model", y="Score", color="Metric",
        barmode="group",
        color_discrete_sequence=[GOLD, TEAL, ORANGE, INDIGO],
        text=metrics_melt["Score"].apply(lambda x: f"{x:.3f}"),
        title="Classification Model Comparison"
    )
    fig_comp.update_traces(textposition="outside")
    fig_comp.update_layout(height=380, yaxis=dict(range=[0, 1.1]), legend_title="Metric")
    st.plotly_chart(fig_comp, use_container_width=True)

    # ── CONFUSION MATRIX + ROC CURVE ─────────────────────────
    section_header(f"Model Details — {selected_model}")
    c1, c2 = st.columns(2)

    with c1:
        cm = res["cm"]
        class_names = ["Not Interested", "Maybe", "Interested"]
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        fig_cm = go.Figure(go.Heatmap(
            z=cm_pct,
            x=class_names,
            y=class_names,
            colorscale=[[0, SURFACE2], [0.5, ORANGE], [1, GOLD]],
            text=[[f"{cm[i][j]}<br>({cm_pct[i][j]:.1f}%)" for j in range(len(class_names))]
                  for i in range(len(class_names))],
            texttemplate="%{text}",
            textfont=dict(size=11),
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z:.0f}%<extra></extra>",
            showscale=False
        ))
        fig_cm.update_layout(
            title="Confusion Matrix (%)",
            xaxis_title="Predicted", yaxis_title="Actual",
            height=340
        )
        st.plotly_chart(fig_cm, use_container_width=True)

    with c2:
        # ROC Curves (one vs rest)
        y_prob = res.get("y_prob")
        fig_roc = go.Figure()

        if y_prob is not None and y_prob.shape[1] >= 3:
            y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
            for i, (class_name, color) in enumerate(zip(class_names, [ROSE, GOLD, TEAL])):
                try:
                    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
                    roc_auc = auc(fpr, tpr)
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr, mode="lines",
                        name=f"{class_name} (AUC={roc_auc:.3f})",
                        line=dict(color=color, width=2)
                    ))
                except Exception:
                    pass
        else:
            st.info("ROC curve requires probability outputs from the model.")

        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            line=dict(dash="dash", color=MUTED, width=1),
            name="Random Baseline", showlegend=True
        ))
        fig_roc.update_layout(
            title="ROC Curves (One-vs-Rest)",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=340
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # ── PER-CLASS METRICS ─────────────────────────────────────
    section_header("Per-Class Performance Report")

    report = res.get("report", {})
    rows = []
    for cls_key, cls_name in [("0", "Not Interested"), ("1", "Maybe"), ("2", "Interested")]:
        if cls_key in report:
            r = report[cls_key]
            rows.append({
                "Class": cls_name,
                "Precision": round(r.get("precision", 0), 3),
                "Recall": round(r.get("recall", 0), 3),
                "F1-Score": round(r.get("f1-score", 0), 3),
                "Support": int(r.get("support", 0))
            })
    if rows:
        report_df = pd.DataFrame(rows)
        col_r1, col_r2 = st.columns([1, 1.5])
        with col_r1:
            st.dataframe(report_df.set_index("Class"), use_container_width=True)
        with col_r2:
            fig_cls = go.Figure()
            for metric, color in [("Precision", GOLD), ("Recall", TEAL), ("F1-Score", ORANGE)]:
                fig_cls.add_trace(go.Bar(
                    name=metric,
                    x=report_df["Class"],
                    y=report_df[metric],
                    marker_color=color, opacity=0.85,
                    text=report_df[metric].apply(lambda x: f"{x:.3f}"),
                    textposition="outside"
                ))
            fig_cls.update_layout(
                barmode="group", title="Per-Class Metrics",
                height=300, yaxis=dict(range=[0, 1.15])
            )
            st.plotly_chart(fig_cls, use_container_width=True)

    # ── FEATURE IMPORTANCE ────────────────────────────────────
    section_header("Feature Importance — Random Forest")
    top_n = st.slider("Show top N features:", 5, len(feat_imp), 15)
    fi_top = feat_imp.head(top_n)

    colors_fi = [GOLD if i == 0 else TEAL if i < 3 else ORANGE if i < 7 else MUTED
                 for i in range(len(fi_top))]

    fig_fi = go.Figure(go.Bar(
        x=fi_top["importance"],
        y=[f.replace("_", " ").title() for f in fi_top["feature"]],
        orientation="h",
        marker_color=colors_fi,
        text=fi_top["importance"].apply(lambda x: f"{x:.4f}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>"
    ))
    fig_fi.update_layout(
        title=f"Top {top_n} Features by Importance (Random Forest)",
        height=max(320, top_n * 26),
        yaxis=dict(autorange="reversed")
    )
    st.plotly_chart(fig_fi, use_container_width=True)

    # ── PREDICTION DISTRIBUTION ───────────────────────────────
    section_header("Prediction Distribution on Test Set")

    pred_counts = pd.Series(res["y_pred"]).map(LABEL_NAMES).value_counts()
    actual_counts = pd.Series(np.array(y_test)).map(LABEL_NAMES).value_counts()
    combined = pd.DataFrame({"Actual": actual_counts, "Predicted": pred_counts}).fillna(0)

    fig_dist = go.Figure()
    for col, color in [("Actual", TEAL), ("Predicted", GOLD)]:
        fig_dist.add_trace(go.Bar(
            name=col, x=combined.index, y=combined[col],
            marker_color=color, opacity=0.85,
            text=combined[col].astype(int), textposition="outside"
        ))
    fig_dist.update_layout(
        barmode="group",
        title="Actual vs Predicted Label Distribution (Test Set)",
        height=320
    )
    st.plotly_chart(fig_dist, use_container_width=True)

    info_card(
        "Classification Insight",
        "Random Forest achieves the highest balanced F1-Score across all three classes. "
        "The top 3 predictive features are <b>creative_self_identity</b>, <b>participation_barrier</b>, and <b>visit_frequency_intent</b> — "
        "confirming that <b>2–3 quick psychographic questions</b> can replace a 35-question survey for rapid field screening. "
        "The 'Maybe' class is the hardest to classify — these are price-sensitive or information-seeking customers best converted via a free trial offer.",
        GOLD
    )
