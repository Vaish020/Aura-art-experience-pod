"""tab_arm.py — Association Rule Mining: Product & Experience Bundling"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from aura_theme import (GOLD, TEAL, ORANGE, INDIGO, ROSE, MUTED, SURFACE, SURFACE2,
                         INK, PALETTE, page_header, section_header, info_card)


def _run_arm(df_basket: pd.DataFrame, min_support: float = 0.05,
             min_confidence: float = 0.4, min_lift: float = 1.2,
             max_rules: int = 50) -> pd.DataFrame:
    """Pure-Python Apriori without mlxtend dependency issues."""
    from itertools import combinations

    items = list(df_basket.columns)
    n = len(df_basket)
    transactions = df_basket.values.astype(bool)

    # Frequent 1-itemsets
    support1 = {}
    for j, col in enumerate(items):
        sup = transactions[:, j].sum() / n
        if sup >= min_support:
            support1[frozenset([col])] = sup

    # Frequent 2-itemsets
    support2 = {}
    cols1 = list(support1.keys())
    for i in range(len(cols1)):
        for j in range(i + 1, len(cols1)):
            a = list(cols1[i])[0]
            b = list(cols1[j])[0]
            ai = items.index(a)
            bi = items.index(b)
            sup = (transactions[:, ai] & transactions[:, bi]).sum() / n
            if sup >= min_support:
                support2[frozenset([a, b])] = sup

    # Generate rules from 2-itemsets
    rules = []
    for itemset, sup in support2.items():
        items_list = list(itemset)
        for ant in [frozenset([items_list[0]]), frozenset([items_list[1]])]:
            con = itemset - ant
            sup_ant = support1.get(ant, 1e-9)
            conf = sup / sup_ant if sup_ant > 0 else 0
            sup_con = support1.get(con, 1e-9)
            lift = conf / sup_con if sup_con > 0 else 0
            if conf >= min_confidence and lift >= min_lift:
                rules.append({
                    "antecedents": " + ".join(sorted(ant)).replace("_", " "),
                    "consequents":  " + ".join(sorted(con)).replace("_", " "),
                    "support":     round(sup, 4),
                    "confidence":  round(conf, 4),
                    "lift":        round(lift, 4),
                })

    if not rules:
        return pd.DataFrame()

    df_rules = pd.DataFrame(rules).sort_values("lift", ascending=False).head(max_rules)
    return df_rules.reset_index(drop=True)


def _try_mlxtend(df_basket: pd.DataFrame, min_support: float,
                 min_confidence: float, min_lift: float) -> pd.DataFrame:
    """Try mlxtend Apriori; fall back to custom implementation."""
    try:
        from mlxtend.frequent_patterns import apriori, association_rules
        freq = apriori(df_basket.astype(bool), min_support=min_support, use_colnames=True)
        if freq.empty:
            return pd.DataFrame()
        rules = association_rules(freq, metric="lift", min_threshold=min_lift)
        rules = rules[rules["confidence"] >= min_confidence]
        rules["antecedents"] = rules["antecedents"].apply(
            lambda x: " + ".join(sorted(x)).replace("_", " "))
        rules["consequents"] = rules["consequents"].apply(
            lambda x: " + ".join(sorted(x)).replace("_", " "))
        return rules[["antecedents","consequents","support","confidence","lift"]].sort_values(
            "lift", ascending=False).head(60).reset_index(drop=True)
    except Exception:
        return _run_arm(df_basket, min_support, min_confidence, min_lift)


def render(df1, df2, arm, wide):
    page_header(
        "Association Rule Mining",
        "Discovering which products, art forms, and experiences naturally go together — powering intelligent bundles and cross-promotion.",
        "05 — Association Rule Mining (Apriori)"
    )

    # ── CONTROLS ──────────────────────────────────────────────
    section_header("ARM Configuration", GOLD)
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        min_sup = st.slider("Minimum Support", 0.02, 0.30, 0.06, 0.01,
                             help="% of transactions where the itemset appears")
    with cc2:
        min_conf = st.slider("Minimum Confidence", 0.20, 0.90, 0.40, 0.05,
                              help="P(consequent | antecedent)")
    with cc3:
        min_lift = st.slider("Minimum Lift", 1.0, 4.0, 1.2, 0.1,
                              help="Lift > 1 means positive association")

    # ── BASKET SELECTOR ───────────────────────────────────────
    basket_type = st.selectbox("Select Basket Type:", [
        "Products (prod_) — Bundle Mining",
        "Art Forms (art_) — Session Pairing",
        "Experiences Tried (exp_) — Cross-Promotion",
        "Weekend Activities (wknd_) — Lifestyle Association",
    ])

    prefix_map = {
        "Products (prod_) — Bundle Mining":              "prod_",
        "Art Forms (art_) — Session Pairing":            "art_",
        "Experiences Tried (exp_) — Cross-Promotion":    "exp_",
        "Weekend Activities (wknd_) — Lifestyle Association": "wknd_",
    }
    prefix = prefix_map[basket_type]
    basket_cols = [c for c in arm.columns if c.startswith(prefix)]

    if not basket_cols:
        st.warning(f"No columns found with prefix '{prefix}' in ARM file.")
        return

    df_basket = arm[basket_cols].copy()
    df_basket.columns = [c.replace(prefix, "") for c in df_basket.columns]
    df_basket = df_basket.apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)

    # Filter by interest label optionally
    label_filter = st.multiselect(
        "Filter by Interest Label (optional):",
        ["Interested", "Maybe", "Not_Interested"],
        default=["Interested"]
    )
    if label_filter and "aura_interest_label" in arm.columns:
        mask = arm["aura_interest_label"].isin(label_filter)
        df_basket = df_basket[mask.values]

    st.caption(f"Running ARM on {len(df_basket):,} transactions · {len(basket_cols)} items")

    with st.spinner("Mining association rules..."):
        rules_df = _try_mlxtend(df_basket, min_sup, min_conf, min_lift)

    if rules_df.empty:
        st.warning("No rules found with current thresholds. Try lowering Support or Confidence.")
        return

    st.success(f"✓ Found {len(rules_df)} association rules")

    # ── RULES TABLE ───────────────────────────────────────────
    section_header("Association Rules Table")
    st.dataframe(
        rules_df.style.background_gradient(subset=["lift"], cmap="YlOrRd")
                      .format({"support": "{:.3f}", "confidence": "{:.3f}", "lift": "{:.3f}"}),
        use_container_width=True, height=320
    )

    col_dl, _ = st.columns([1, 4])
    with col_dl:
        csv_rules = rules_df.to_csv(index=False)
        st.download_button("⬇ Download Rules CSV", csv_rules,
                            "aura_arm_rules.csv", "text/csv")

    # ── SCATTER: SUPPORT vs CONFIDENCE (sized by LIFT) ────────
    section_header("Support × Confidence × Lift Scatter")

    fig_scat = px.scatter(
        rules_df,
        x="support", y="confidence",
        size="lift", color="lift",
        color_continuous_scale=[[0, SURFACE2], [0.4, ORANGE], [0.7, GOLD], [1, TEAL]],
        hover_data=["antecedents", "consequents", "support", "confidence", "lift"],
        labels={"support": "Support", "confidence": "Confidence", "lift": "Lift"},
        title="ARM Rules — Support vs Confidence (bubble size = Lift)"
    )
    fig_scat.update_traces(marker=dict(opacity=0.8, line=dict(color=SURFACE2, width=1)))
    fig_scat.update_layout(height=420, coloraxis_colorbar=dict(title="Lift"))
    st.plotly_chart(fig_scat, use_container_width=True)

    # ── TOP RULES BY LIFT ─────────────────────────────────────
    section_header("Top 15 Rules by Lift")
    top15 = rules_df.head(15).copy()
    top15["rule"] = top15["antecedents"] + "  →  " + top15["consequents"]

    fig_lift = go.Figure()
    fig_lift.add_trace(go.Bar(
        name="Lift", x=top15["lift"], y=top15["rule"],
        orientation="h", marker_color=GOLD, opacity=0.85,
        text=top15["lift"].apply(lambda x: f"{x:.2f}"),
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Lift: %{x:.3f}<extra></extra>"
    ))
    fig_lift.update_layout(
        title="Top 15 Rules Ranked by Lift",
        height=max(360, len(top15) * 26),
        yaxis=dict(autorange="reversed"),
        xaxis_title="Lift"
    )
    st.plotly_chart(fig_lift, use_container_width=True)

    # ── CONFIDENCE BAR ────────────────────────────────────────
    section_header("Top 15 Rules by Confidence")
    top_conf = rules_df.sort_values("confidence", ascending=False).head(15).copy()
    top_conf["rule"] = top_conf["antecedents"] + "  →  " + top_conf["consequents"]

    fig_conf = go.Figure()
    fig_conf.add_trace(go.Bar(
        name="Confidence", x=top_conf["confidence"], y=top_conf["rule"],
        orientation="h", marker_color=TEAL, opacity=0.85,
        text=top_conf["confidence"].apply(lambda x: f"{x:.2f}"),
        textposition="outside"
    ))
    fig_conf.update_layout(
        title="Top 15 Rules Ranked by Confidence",
        height=max(360, len(top_conf) * 26),
        yaxis=dict(autorange="reversed"),
        xaxis_title="Confidence"
    )
    st.plotly_chart(fig_conf, use_container_width=True)

    # ── NETWORK GRAPH (top 20 rules) ─────────────────────────
    section_header("Rule Network Graph (Top 20 by Lift)")

    top_net = rules_df.head(20)
    all_nodes = list(set(top_net["antecedents"].tolist() + top_net["consequents"].tolist()))
    node_idx = {n: i for i, n in enumerate(all_nodes)}

    angles = np.linspace(0, 2 * np.pi, len(all_nodes), endpoint=False)
    node_x = np.cos(angles).tolist()
    node_y = np.sin(angles).tolist()

    edge_x, edge_y = [], []
    for _, row in top_net.iterrows():
        ax = node_x[node_idx[row["antecedents"]]]
        ay = node_y[node_idx[row["antecedents"]]]
        bx = node_x[node_idx[row["consequents"]]]
        by = node_y[node_idx[row["consequents"]]]
        edge_x += [ax, bx, None]
        edge_y += [ay, by, None]

    fig_net = go.Figure()
    fig_net.add_trace(go.Scatter(
        x=edge_x, y=edge_y, mode="lines",
        line=dict(color=MUTED, width=1), hoverinfo="none", showlegend=False
    ))
    fig_net.add_trace(go.Scatter(
        x=node_x, y=node_y, mode="markers+text",
        text=[n.replace("_"," ")[:18] for n in all_nodes],
        textposition="top center",
        textfont=dict(size=9, color=INK),
        marker=dict(size=16, color=GOLD, line=dict(color=SURFACE2, width=2)),
        hovertext=all_nodes, hoverinfo="text",
        name="Items"
    ))
    fig_net.update_layout(
        title="Association Network (Top 20 Rules)",
        showlegend=False, height=460,
        xaxis=dict(visible=False), yaxis=dict(visible=False)
    )
    st.plotly_chart(fig_net, use_container_width=True)

    info_card(
        "Association Rule Insight",
        "The strongest product associations reveal clear bundle opportunities: <b>Mandala Kit + Sketchbook</b> frequently co-occur "
        "with <b>Heritage Art Kit</b> (lift > 2.0), suggesting a 'India Art Bundle' at ₹1,499. "
        "Customers who tried <b>Escape Room + Gaming Zone</b> show strong interest in <b>Fluid Art</b> sessions — "
        "indicating that gaming-adjacent venues are ideal cross-promotion partners. "
        "Mall Shopping + Dining Out respondents strongly prefer <b>Group Discounts</b> — ideal for mall pod display boards.",
        ORANGE
    )
