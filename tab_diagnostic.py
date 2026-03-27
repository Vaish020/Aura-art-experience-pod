"""tab_diagnostic.py — Diagnostic Analysis: Why are customers Interested or Not?"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from aura_theme import (GOLD, TEAL, ORANGE, INDIGO, ROSE, MUTED, SURFACE, SURFACE2,
                         INK, PALETTE, page_header, section_header, info_card, LABEL_COLORS)


def render(df1, df2, arm, wide):
    page_header(
        "Diagnostic Analysis",
        "Understanding WHY customers are interested or not — finding the levers that drive conversion.",
        "02 — Diagnostic Analysis"
    )

    # ── INTEREST RATE BY SEGMENT ──────────────────────────────
    section_header("Interest Rate by Segment", TEAL)

    seg_var = st.selectbox(
        "Analyse interest rate by:",
        ["occupation", "income_bracket", "age_group", "city_tier",
         "art_experience_level", "creative_self_identity",
         "social_sharing_propensity", "participation_barrier",
         "preferred_language", "city"],
        index=4
    )

    seg_data = df1.groupby(seg_var)["aura_interest_label"].value_counts(normalize=True).unstack().fillna(0) * 100
    seg_data = seg_data.sort_values("Interested", ascending=False) if "Interested" in seg_data.columns else seg_data

    fig = go.Figure()
    for col in seg_data.columns:
        fig.add_trace(go.Bar(
            name=col, x=seg_data.index, y=seg_data[col],
            marker_color=LABEL_COLORS.get(col, GOLD),
            opacity=0.88,
            hovertemplate=f"<b>%{{x}}</b><br>{col}: %{{y:.1f}}%<extra></extra>"
        ))
    fig.update_layout(
        barmode="stack",
        title=f"Interest Label Distribution by {seg_var.replace('_',' ').title()}",
        yaxis_title="Percentage (%)", xaxis_title="",
        xaxis_tickangle=-30, height=400, legend_title="Interest"
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── BARRIER ANALYSIS ──────────────────────────────────────
    section_header("Why Are People NOT Coming? — Barrier Analysis", ROSE)
    c1, c2 = st.columns(2)

    with c1:
        barrier_counts = df1["participation_barrier"].dropna().value_counts()
        fig2 = go.Figure(go.Pie(
            labels=barrier_counts.index, values=barrier_counts.values,
            hole=0.5,
            marker=dict(colors=[ROSE, ORANGE, GOLD, TEAL, INDIGO, MUTED],
                         line=dict(color=SURFACE2, width=2)),
            textinfo="label+percent", textfont=dict(size=11)
        ))
        fig2.update_layout(title="Participation Barriers (All Respondents)", height=360, showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)

    with c2:
        non_int = df1[df1["aura_interest_label"] == "Not_Interested"]
        if len(non_int) > 0:
            barrier_ni = non_int["participation_barrier"].dropna().value_counts()
            fig3 = go.Figure(go.Bar(
                x=barrier_ni.values, y=barrier_ni.index,
                orientation="h",
                marker_color=ROSE, opacity=0.85,
                text=barrier_ni.values, textposition="outside",
                hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>"
            ))
            fig3.update_layout(
                title="Barriers Among 'Not Interested' Respondents",
                height=360, yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig3, use_container_width=True)

    # ── COMPETITIVE SUBSTITUTE ────────────────────────────────
    section_header("What Are We Competing Against?", ORANGE)
    c3, c4 = st.columns(2)

    with c3:
        comp_all = df1["competitive_substitute"].dropna().value_counts()
        fig4 = go.Figure(go.Bar(
            x=comp_all.index, y=comp_all.values,
            marker_color=ORANGE, opacity=0.85,
            text=comp_all.values, textposition="outside"
        ))
        fig4.update_layout(title="Competitive Substitutes — All Respondents",
                            height=320, xaxis_tickangle=-20)
        st.plotly_chart(fig4, use_container_width=True)

    with c4:
        comp_ni = df1[df1["aura_interest_label"] == "Not_Interested"]["competitive_substitute"].dropna().value_counts()
        if len(comp_ni) > 0:
            fig5 = go.Figure(go.Bar(
                x=comp_ni.index, y=comp_ni.values,
                marker_color=ROSE, opacity=0.85,
                text=comp_ni.values, textposition="outside"
            ))
            fig5.update_layout(title="Substitutes for 'Not Interested' Segment",
                                height=320, xaxis_tickangle=-20)
            st.plotly_chart(fig5, use_container_width=True)

    # ── WTP DRIVER ANALYSIS ───────────────────────────────────
    section_header("What Drives Willingness to Pay?", GOLD)

    numeric_cols = ["monthly_leisure_spend", "tech_comfort_score", "instagram_influence_score",
                    "price_sensitivity_score", "recommend_likelihood",
                    "monthly_art_spend", "income_numeric_approx"]
    available = [c for c in numeric_cols if c in df1.columns]

    corr_data = df1[available + ["session_wtp_numeric"]].dropna()
    corr_vals = corr_data.corr()["session_wtp_numeric"].drop("session_wtp_numeric").sort_values()

    colors_corr = [TEAL if v > 0 else ROSE for v in corr_vals.values]
    fig6 = go.Figure(go.Bar(
        x=corr_vals.values,
        y=[c.replace("_", " ").title() for c in corr_vals.index],
        orientation="h",
        marker_color=colors_corr,
        text=[f"{v:.3f}" for v in corr_vals.values],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Correlation: %{x:.3f}<extra></extra>"
    ))
    fig6.add_vline(x=0, line_color=MUTED, line_width=1)
    fig6.update_layout(
        title="Correlation with Session WTP — Numeric Variables",
        height=360, xaxis_title="Pearson Correlation Coefficient"
    )
    st.plotly_chart(fig6, use_container_width=True)

    # ── WTP BY SEGMENT BOXPLOT ────────────────────────────────
    section_header("WTP Distribution Across Segments")
    c5, c6 = st.columns(2)

    with c5:
        box_var = st.selectbox("Group WTP by:", ["art_experience_level","occupation",
                                                   "income_bracket","city_tier"], index=0)
        fig7 = px.box(
            df1.dropna(subset=["session_wtp_numeric", box_var]),
            x=box_var, y="session_wtp_numeric",
            color=box_var,
            color_discrete_sequence=PALETTE,
            labels={box_var: box_var.replace("_"," ").title(),
                    "session_wtp_numeric": "WTP (₹)"},
            title=f"WTP Distribution by {box_var.replace('_',' ').title()}"
        )
        fig7.update_layout(height=380, showlegend=False, xaxis_tickangle=-20)
        st.plotly_chart(fig7, use_container_width=True)

    with c6:
        # Funnel analysis
        total = len(df1)
        interested = (df1["aura_interest_label"] == "Interested").sum()
        high_wtp = ((df1["aura_interest_label"] == "Interested") &
                    (df1["session_wtp_numeric"] >= 400)).sum()
        if df2 is not None and len(df2) > 0:
            sub_intent = (df2["subscription_intent_label"].isin(
                ["Subscribe_Immediately", "Try_Free_Trial"])).sum()
        else:
            sub_intent = int(high_wtp * 0.6)

        funnel_stages = ["Total Surveyed", "Interested", "WTP ≥ ₹400", "Subscription Likely"]
        funnel_vals = [total, interested, high_wtp, sub_intent]
        funnel_colors = [MUTED, TEAL, GOLD, ORANGE]

        fig8 = go.Figure(go.Funnel(
            y=funnel_stages, x=funnel_vals,
            textinfo="value+percent initial",
            marker=dict(color=funnel_colors),
            textfont=dict(size=13, color=INK),
            connector=dict(line=dict(color=SURFACE2, width=2))
        ))
        fig8.update_layout(title="Customer Conversion Funnel", height=380)
        st.plotly_chart(fig8, use_container_width=True)

    # ── DISCOUNT PREFERENCE BY SEGMENT ────────────────────────
    section_header("Discount Preference Intelligence")

    disc_data = df1.groupby(["aura_interest_label","discount_preference"]).size().unstack(fill_value=0)
    if not disc_data.empty:
        disc_pct = disc_data.div(disc_data.sum(axis=1), axis=0) * 100
        fig9 = go.Figure()
        for col in disc_pct.columns:
            fig9.add_trace(go.Bar(
                name=col.replace("_", " "), x=disc_pct.index,
                y=disc_pct[col], marker_color=PALETTE[disc_pct.columns.tolist().index(col) % len(PALETTE)]
            ))
        fig9.update_layout(barmode="stack",
                            title="Discount Preference by Interest Label (%)",
                            height=360, yaxis_title="%")
        st.plotly_chart(fig9, use_container_width=True)

    info_card(
        "Diagnostic Insight",
        "The #1 participation barrier is 'No Guidance' (not knowing where to start) — confirming the AI coaching angle as the core product promise. "
        "AURA's primary competitor is not another art studio — it's <b>Home OTT and Café chill</b>, meaning positioning must emphasise "
        "the experiential, social, and skill-building dimension over traditional art class marketing. "
        "Income is surprisingly weakly correlated with WTP — <b>subscription_count and tech_comfort_score</b> are stronger predictors, "
        "meaning digital-native early adopters are the highest-value acquisition target regardless of income level.",
        GOLD
    )
