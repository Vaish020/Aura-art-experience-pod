"""tab_clustering.py — Clustering: Customer Persona Segmentation"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from aura_theme import (GOLD, TEAL, ORANGE, INDIGO, ROSE, MUTED, SURFACE, SURFACE2,
                         INK, PALETTE, CLUSTER_COLORS,
                         page_header, section_header, info_card)
from aura_data import name_cluster, get_strategy, CLU_FEATURES


def render(df1, df2, arm, wide, km_model, df_clustered, km_scaler,
           best_k, k_range, inertias, silhouettes, pca):

    page_header(
        "Customer Persona Clustering",
        f"K-Means identified {best_k} distinct customer personas among the Interested segment — each needing a different product, price, and marketing approach.",
        "03 — Clustering (K-Means)"
    )

    # ── ELBOW + SILHOUETTE ────────────────────────────────────
    section_header("Optimal K Selection — Elbow & Silhouette", GOLD)
    c1, c2 = st.columns(2)

    with c1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=list(k_range), y=inertias, mode="lines+markers",
            line=dict(color=GOLD, width=2),
            marker=dict(size=8, color=GOLD, symbol="circle"),
            name="Inertia"
        ))
        fig_elbow.add_vline(x=best_k, line_dash="dash", line_color=TEAL,
                             annotation_text=f"Best K={best_k}", annotation_font_color=TEAL)
        fig_elbow.update_layout(title="Elbow Method — Inertia vs K",
                                 xaxis_title="Number of Clusters (K)",
                                 yaxis_title="Inertia", height=300)
        st.plotly_chart(fig_elbow, use_container_width=True)

    with c2:
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Scatter(
            x=list(k_range), y=silhouettes, mode="lines+markers",
            line=dict(color=TEAL, width=2),
            marker=dict(size=8, color=TEAL, symbol="diamond"),
            name="Silhouette"
        ))
        fig_sil.add_vline(x=best_k, line_dash="dash", line_color=GOLD,
                           annotation_text=f"Best K={best_k}", annotation_font_color=GOLD)
        fig_sil.update_layout(title="Silhouette Score vs K",
                               xaxis_title="Number of Clusters (K)",
                               yaxis_title="Silhouette Score", height=300)
        st.plotly_chart(fig_sil, use_container_width=True)

    st.markdown(f"""
    <div style="background:{SURFACE2};border:1px solid rgba(255,255,255,0.07);border-left:3px solid {TEAL};
                border-radius:6px;padding:14px 20px;margin:8px 0 24px;">
        <span style="color:{TEAL};font-size:11px;letter-spacing:0.1em;text-transform:uppercase;font-weight:700;">
        Selected K = {best_k}</span>
        <span style="color:{MUTED};font-size:13px;margin-left:16px;">
        Silhouette score: {max(silhouettes):.3f} · Inertia: {min(inertias):,.0f}</span>
    </div>
    """, unsafe_allow_html=True)

    # ── PCA SCATTER ───────────────────────────────────────────
    section_header("Cluster Visualisation — PCA 2D Projection")

    df_plot = df_clustered.copy()
    df_plot["cluster_name"] = df_plot["cluster"].apply(name_cluster)
    color_map = {name_cluster(i): CLUSTER_COLORS[i % len(CLUSTER_COLORS)] for i in range(best_k)}

    fig_pca = px.scatter(
        df_plot, x="pca_x", y="pca_y",
        color="cluster_name",
        color_discrete_map=color_map,
        opacity=0.7,
        title="Customer Segments — PCA 2D Projection",
        labels={"pca_x": "Principal Component 1", "pca_y": "Principal Component 2",
                "cluster_name": "Persona"},
        hover_data=["occupation", "income_bracket", "art_experience_level",
                    "session_wtp_numeric"]
    )
    fig_pca.update_traces(marker=dict(size=5))
    fig_pca.update_layout(height=440, legend_title="Customer Persona")
    st.plotly_chart(fig_pca, use_container_width=True)

    # ── CLUSTER SIZE ──────────────────────────────────────────
    section_header("Segment Size Distribution")
    cluster_sizes = df_plot["cluster_name"].value_counts().reset_index()
    cluster_sizes.columns = ["Persona", "Count"]
    cluster_sizes["Pct"] = (cluster_sizes["Count"] / len(df_plot) * 100).round(1)

    fig_size = go.Figure(go.Bar(
        x=cluster_sizes["Persona"], y=cluster_sizes["Count"],
        marker_color=[color_map.get(p, GOLD) for p in cluster_sizes["Persona"]],
        text=[f"{c} ({p}%)" for c, p in zip(cluster_sizes["Count"], cluster_sizes["Pct"])],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Count: %{y}<extra></extra>"
    ))
    fig_size.update_layout(title="Number of Customers per Persona", height=320)
    st.plotly_chart(fig_size, use_container_width=True)

    # ── PERSONA PROFILES ──────────────────────────────────────
    section_header("Persona Deep-Dive")

    selected_persona = st.selectbox(
        "Select a persona to explore:",
        [name_cluster(i) for i in range(best_k)]
    )

    df_seg = df_plot[df_plot["cluster_name"] == selected_persona]
    strategy = get_strategy(selected_persona)

    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        # Radar chart of key metrics
        cats = ["Tech Comfort", "WTP Score", "Art Experience",
                "Social Sharing", "Visit Frequency", "Price Sensitivity"]
        enc_map = {
            "tech_comfort_score": df_seg["tech_comfort_score"].mean() / 5,
            "session_wtp_numeric": df_seg["session_wtp_numeric"].mean() / 2000,
            "art_experience_level": df_seg["art_experience_level"].map(
                {"Complete_Beginner":0,"Curious_Beginner":0.25,"Casual_Hobbyist":0.5,
                 "Regular_Hobbyist":0.75,"Advanced":1}).mean() if df_seg["art_experience_level"].dtype == object else 0.5,
            "social_sharing_propensity": df_seg["social_sharing_propensity"].map(
                {"Definitely_Post":1,"Probably_Post":0.67,"Close_Circle":0.33,"Keep_Private":0}).mean() if df_seg["social_sharing_propensity"].dtype == object else 0.5,
            "visit_frequency_intent": df_seg["visit_frequency_intent"].map(
                {"Unlikely":0,"Once_Try":0.2,"Occasionally":0.4,"Once_month":0.6,"2_3_month":0.8,"Weekly":1}).mean() if df_seg["visit_frequency_intent"].dtype == object else 0.5,
            "price_sensitivity_score": (6 - df_seg["price_sensitivity_score"].mean()) / 5,
        }
        vals = list(enc_map.values())
        radar_color = color_map.get(selected_persona, GOLD)

        fig_radar = go.Figure(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            fill="toself",
            fillcolor=f"rgba({int(radar_color[1:3],16)},{int(radar_color[3:5],16)},{int(radar_color[5:7],16)},0.15)",
            line=dict(color=radar_color, width=2),
            marker=dict(size=6, color=radar_color),
            name=selected_persona
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor=SURFACE2,
                radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=9, color=MUTED)),
                angularaxis=dict(tickfont=dict(size=11, color=INK))
            ),
            showlegend=False, title=f"{selected_persona} — Profile Radar",
            height=380
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_b:
        st.markdown(f"""
        <div style="background:{SURFACE2};border:1px solid rgba(255,255,255,0.07);
                    border-radius:8px;padding:24px;margin-top:20px;">
            <div style="font-size:22px;margin-bottom:16px;">{selected_persona}</div>
            <div style="color:{MUTED};font-size:11px;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:16px;">
                {len(df_seg):,} customers · {len(df_seg)/len(df_plot)*100:.1f}% of interested segment
            </div>
        """, unsafe_allow_html=True)

        metrics = {
            "Median WTP": f"₹{int(df_seg['session_wtp_numeric'].median()):,}",
            "Avg Leisure Spend": f"₹{int(df_seg['monthly_leisure_spend'].median()):,}/mo",
            "Top City": df_seg["city"].value_counts().index[0] if "city" in df_seg.columns and len(df_seg) > 0 else "N/A",
            "Top Occupation": df_seg["occupation"].value_counts().index[0] if "occupation" in df_seg.columns and len(df_seg) > 0 else "N/A",
        }

        for k, v in metrics.items():
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;padding:8px 0;
                        border-bottom:1px solid rgba(255,255,255,0.05);">
                <span style="color:{MUTED};font-size:12px;">{k}</span>
                <span style="color:{INK};font-size:13px;font-weight:600;">{v}</span>
            </div>
            """, unsafe_allow_html=True)

        if strategy:
            st.markdown(f"<br><div style='color:{GOLD};font-size:11px;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:10px;'>Recommended Strategy</div>", unsafe_allow_html=True)
            for sk, sv in strategy.items():
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;padding:7px 0;
                            border-bottom:1px solid rgba(255,255,255,0.04);">
                    <span style="color:{MUTED};font-size:11px;">{sk.replace('_',' ').title()}</span>
                    <span style="color:{TEAL};font-size:11px;font-weight:600;text-align:right;max-width:55%;">{sv}</span>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── CLUSTER COMPARISON HEATMAP ────────────────────────────
    section_header("Cross-Persona Comparison")

    num_cols = ["session_wtp_numeric", "monthly_leisure_spend", "monthly_art_spend",
                "tech_comfort_score", "price_sensitivity_score", "instagram_influence_score",
                "recommend_likelihood"]
    avail_num = [c for c in num_cols if c in df_plot.columns]
    cluster_agg = df_plot.groupby("cluster_name")[avail_num].mean()

    # Normalise for heatmap
    cluster_norm = (cluster_agg - cluster_agg.min()) / (cluster_agg.max() - cluster_agg.min() + 1e-9)
    clean_cols = [c.replace("_"," ").title() for c in cluster_norm.columns]

    fig_heat = go.Figure(go.Heatmap(
        z=cluster_norm.values,
        x=clean_cols,
        y=cluster_norm.index,
        colorscale=[[0, SURFACE2], [0.5, ORANGE], [1, GOLD]],
        hovertemplate="<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>",
        showscale=True,
        text=np.round(cluster_agg.values, 1),
        texttemplate="%{text}",
    ))
    fig_heat.update_layout(
        title="Persona Comparison Heatmap (Normalised 0–1)",
        height=320, xaxis_tickangle=-25
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    info_card(
        "Clustering Insight",
        f"K-Means converged on <b>K={best_k}</b> optimal clusters (Silhouette = {max(silhouettes):.3f}). "
        "The Corporate Buyer segment, though smallest in count, generates the highest revenue per interaction. "
        "The Status Sharer segment is the largest and most cost-effective to acquire via Instagram. "
        "Launch strategy: <b>activate Corporate Buyers first</b> for revenue, <b>activate Status Sharers</b> for brand visibility.",
        TEAL
    )
