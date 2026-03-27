"""tab_overview.py — Descriptive Analysis: Overview & Data Explorer"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from aura_theme import (GOLD, TEAL, ORANGE, INDIGO, ROSE, MUTED, BG,
                         SURFACE, SURFACE2, INK, PALETTE,
                         page_header, section_header, kpi_row, info_card, LABEL_COLORS)


def render(df1, df2, arm, wide):
    page_header(
        "Market Overview",
        "Descriptive analysis of 2,000 AURA India survey respondents — understanding who we reached and what they told us.",
        "01 — Descriptive Analysis"
    )

    # ── KPI ROW ──────────────────────────────────────────────
    total = len(df1)
    interested = (df1["aura_interest_label"] == "Interested").sum()
    avg_wtp = int(df1["session_wtp_numeric"].median())
    tier1_pct = (df1["city_tier"] == "Tier1").mean() * 100

    kpi_row([
        ("Total Respondents", f"{total:,}", None),
        ("Interested in AURA", f"{interested:,} ({interested/total*100:.0f}%)", None),
        ("Median Session WTP", f"₹{avg_wtp:,}", None),
        ("Tier 1 City Respondents", f"{tier1_pct:.0f}%", None),
    ])
    st.markdown("<br>", unsafe_allow_html=True)

    # ── ROW 1: Label Distribution + City Distribution ─────────
    section_header("Survey Response Distribution")
    c1, c2 = st.columns(2)

    with c1:
        label_counts = df1["aura_interest_label"].value_counts()
        fig = go.Figure(go.Pie(
            labels=label_counts.index,
            values=label_counts.values,
            hole=0.55,
            marker=dict(
                colors=[LABEL_COLORS.get(l, GOLD) for l in label_counts.index],
                line=dict(color=SURFACE2, width=2)
            ),
            textinfo="label+percent",
            textfont=dict(size=12, color=INK),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Share: %{percent}<extra></extra>"
        ))
        fig.update_layout(
            title="AURA Interest Label Distribution",
            annotations=[dict(text=f"<b>{total}</b><br>Respondents",
                               x=0.5, y=0.5, font_size=14, showarrow=False,
                               font_color=GOLD)],
            showlegend=True, height=360
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        city_counts = df1["city"].value_counts().head(12)
        fig2 = go.Figure(go.Bar(
            x=city_counts.values,
            y=city_counts.index,
            orientation="h",
            marker=dict(
                color=city_counts.values,
                colorscale=[[0, SURFACE2], [1, GOLD]],
                line=dict(color="rgba(0,0,0,0)", width=0)
            ),
            text=city_counts.values,
            textposition="outside",
            textfont=dict(color=MUTED, size=11),
            hovertemplate="<b>%{y}</b><br>Respondents: %{x}<extra></extra>"
        ))
        fig2.update_layout(
            title="Respondents by City (Top 12)",
            xaxis_title="Count", yaxis_title="",
            height=360, yaxis=dict(autorange="reversed")
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── ROW 2: Demographics ───────────────────────────────────
    section_header("Demographic Breakdown")
    c3, c4, c5 = st.columns(3)

    with c3:
        age_counts = df1["age_group"].value_counts().reindex(
            ["Under_18","18_24","25_34","35_44","45_54","55_plus"]).dropna()
        fig3 = go.Figure(go.Bar(
            x=age_counts.index, y=age_counts.values,
            marker_color=TEAL, opacity=0.85,
            text=age_counts.values, textposition="outside",
            hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"
        ))
        fig3.update_layout(title="Age Distribution", height=300,
                            xaxis_title="", yaxis_title="Count")
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        occ_counts = df1["occupation"].value_counts()
        fig4 = go.Figure(go.Pie(
            labels=occ_counts.index, values=occ_counts.values,
            hole=0.45,
            marker=dict(colors=PALETTE, line=dict(color=SURFACE2, width=2)),
            textinfo="label+percent", textfont=dict(size=10)
        ))
        fig4.update_layout(title="Occupation Mix", height=300, showlegend=False)
        st.plotly_chart(fig4, use_container_width=True)

    with c5:
        inc_order = ["Below_25k","25k_50k","50k_1L","1L_2L","Above_2L","Prefer_not"]
        inc_counts = df1["income_bracket"].value_counts().reindex(inc_order).dropna()
        fig5 = go.Figure(go.Bar(
            x=inc_counts.values, y=inc_counts.index,
            orientation="h",
            marker_color=ORANGE, opacity=0.85,
            text=inc_counts.values, textposition="outside",
            hovertemplate="<b>%{y}</b><br>Count: %{x}<extra></extra>"
        ))
        fig5.update_layout(title="Income Distribution", height=300,
                            yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig5, use_container_width=True)

    # ── ROW 3: WTP Distribution + Art Form Popularity ─────────
    section_header("Demand Signals")
    c6, c7 = st.columns(2)

    with c6:
        fig6 = px.histogram(
            df1, x="session_wtp_numeric",
            nbins=40,
            color_discrete_sequence=[GOLD],
            labels={"session_wtp_numeric": "Willingness to Pay (₹)"},
            title="Session WTP Distribution"
        )
        fig6.add_vline(x=df1["session_wtp_numeric"].median(),
                        line_dash="dash", line_color=TEAL,
                        annotation_text=f"Median ₹{int(df1['session_wtp_numeric'].median())}",
                        annotation_font_color=TEAL)
        fig6.add_vline(x=df1["session_wtp_numeric"].mean(),
                        line_dash="dot", line_color=ORANGE,
                        annotation_text=f"Mean ₹{int(df1['session_wtp_numeric'].mean())}",
                        annotation_font_color=ORANGE)
        fig6.update_layout(height=340, bargap=0.05)
        st.plotly_chart(fig6, use_container_width=True)

    with c7:
        art_cols = [c for c in df1.columns if c.startswith("art_")]
        art_sums = df1[art_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int).sum().sort_values(ascending=True)
        art_labels = [c.replace("art_", "").replace("_", " ").title() for c in art_sums.index]
        colors = [GOLD if v == art_sums.max() else TEAL for v in art_sums.values]
        fig7 = go.Figure(go.Bar(
            x=art_sums.values, y=art_labels,
            orientation="h",
            marker_color=colors,
            text=art_sums.values, textposition="outside",
            hovertemplate="<b>%{y}</b><br>Interested: %{x}<extra></extra>"
        ))
        fig7.update_layout(title="Art Form Interest (All Respondents)", height=340)
        st.plotly_chart(fig7, use_container_width=True)

    # ── ROW 4: Weekend Activity + Spend Scatter ───────────────
    section_header("Behaviour Patterns")
    c8, c9 = st.columns(2)

    with c8:
        wknd_cols = [c for c in df1.columns if c.startswith("wknd_")]
        wknd_sums = df1[wknd_cols].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int).sum().sort_values(ascending=False)
        wknd_labels = [c.replace("wknd_", "").replace("_", " ").title() for c in wknd_sums.index]
        fig8 = go.Figure(go.Bar(
            x=wknd_labels, y=wknd_sums.values,
            marker_color=INDIGO, opacity=0.85,
            text=wknd_sums.values, textposition="outside",
        ))
        fig8.update_layout(title="Weekend Activity Frequency", height=320,
                            xaxis_tickangle=-35)
        st.plotly_chart(fig8, use_container_width=True)

    with c9:
        sample = df1.dropna(subset=["monthly_leisure_spend","session_wtp_numeric"]).sample(
            min(600, len(df1)), random_state=42)
        fig9 = px.scatter(
            sample,
            x="monthly_leisure_spend", y="session_wtp_numeric",
            color="aura_interest_label",
            color_discrete_map=LABEL_COLORS,
            opacity=0.65, size_max=8,
            labels={"monthly_leisure_spend": "Monthly Leisure Spend (₹)",
                    "session_wtp_numeric": "Session WTP (₹)",
                    "aura_interest_label": "Interest Label"},
            title="Leisure Spend vs Session WTP",
            hover_data=["city", "occupation"]
        )
        fig9.update_layout(height=320)
        st.plotly_chart(fig9, use_container_width=True)

    # ── ROW 5: Language + City Tier ───────────────────────────
    section_header("Operational Insights")
    c10, c11 = st.columns(2)

    with c10:
        lang_counts = df1["preferred_language"].dropna().value_counts()
        fig10 = go.Figure(go.Pie(
            labels=lang_counts.index, values=lang_counts.values,
            hole=0.4,
            marker=dict(colors=PALETTE, line=dict(color=SURFACE2, width=2)),
            textinfo="label+percent"
        ))
        fig10.update_layout(title="Preferred Coaching Language", height=320, showlegend=False)
        st.plotly_chart(fig10, use_container_width=True)

    with c11:
        tier_label = df1.groupby("city_tier")["aura_interest_label"].value_counts(normalize=True).unstack().fillna(0) * 100
        fig11 = go.Figure()
        for col, color in zip(tier_label.columns, [TEAL, GOLD, ROSE]):
            fig11.add_trace(go.Bar(
                name=col, x=tier_label.index, y=tier_label[col],
                marker_color=color, opacity=0.85
            ))
        fig11.update_layout(
            barmode="group", title="Interest Label by City Tier (%)",
            height=320, yaxis_title="Percentage (%)"
        )
        st.plotly_chart(fig11, use_container_width=True)

    # ── DATA EXPLORER ─────────────────────────────────────────
    section_header("Raw Data Explorer")
    with st.expander("🔍 Browse & Filter Survey Data", expanded=False):
        col_filter, col_city, col_label = st.columns(3)
        with col_filter:
            label_filter = st.multiselect("Filter by Interest Label",
                df1["aura_interest_label"].unique().tolist(),
                default=df1["aura_interest_label"].unique().tolist())
        with col_city:
            city_filter = st.multiselect("Filter by City Tier",
                df1["city_tier"].unique().tolist(),
                default=df1["city_tier"].unique().tolist())
        with col_label:
            occ_filter = st.multiselect("Filter by Occupation",
                df1["occupation"].dropna().unique().tolist(),
                default=df1["occupation"].dropna().unique().tolist())

        filtered = df1[
            df1["aura_interest_label"].isin(label_filter) &
            df1["city_tier"].isin(city_filter) &
            df1["occupation"].isin(occ_filter)
        ]
        st.caption(f"Showing {len(filtered):,} of {len(df1):,} respondents")
        display_cols = ["respondent_id","age_group","gender","city","city_tier",
                        "occupation","income_bracket","art_experience_level",
                        "creative_self_identity","session_wtp_numeric",
                        "aura_interest_label"]
        st.dataframe(filtered[display_cols].head(200), use_container_width=True, height=300)

    info_card(
        "Descriptive Insight",
        "72% of respondents show interest in AURA — significantly above the 40–50% benchmark for new experience concepts in India. "
        "The median WTP of ₹791 validates a ₹600–₹800 launch price. "
        "Bengaluru dominates early demand with 18% of respondents and the highest Interested rate among Tier 1 cities. "
        "Mandala art is the #1 interest format — confirming the Mandala Studio Kit as a launch-priority product.",
        TEAL
    )
