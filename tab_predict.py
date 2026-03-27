"""tab_predict.py — Prescriptive: Predict & Score New Customer Data"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from aura_theme import (GOLD, TEAL, ORANGE, INDIGO, ROSE, MUTED, SURFACE, SURFACE2,
                         INK, PALETTE, page_header, section_header, info_card)
from aura_data import (predict_new_customers, CLF_FEATURES, REG_FEATURES,
                        CLU_FEATURES, ORDINAL_MAPS)

LABEL_COLORS = {"Interested": TEAL, "Maybe": GOLD, "Not Interested": ROSE}


def _make_template() -> pd.DataFrame:
    """Create a downloadable template CSV with sample rows."""
    sample_rows = [
        {"age_group":"25_34","gender":"Female","city":"Bengaluru","city_tier":"Tier1",
         "occupation":"Salaried_IT","income_bracket":"50k_1L","preferred_language":"English",
         "monthly_leisure_spend":4500,"subscription_count":"3_5","online_exp_purchase_freq":"2_3_month",
         "social_platforms":"Instagram|YouTube","art_experience_level":"Curious_Beginner",
         "creative_self_identity":"Want_To_Be","creativity_mindset":"Growth",
         "social_orientation":"Ambivert","reward_preference":"Both",
         "social_sharing_propensity":"Probably_Post","participation_barrier":"No_Guidance",
         "creative_motivation":"Skill_Building","tech_comfort_score":4,
         "instagram_influence_score":4,"price_sensitivity_score":3,
         "session_wtp":"400_700","monthly_art_spend":500,
         "discount_preference":"Bundle_Pack","recommend_likelihood":8,
         "visit_frequency_intent":"2_3_month","decision_autonomy":"Fully_Independent"},
        {"age_group":"18_24","gender":"Male","city":"Mumbai","city_tier":"Tier1",
         "occupation":"Student","income_bracket":"Below_25k","preferred_language":"Hindi",
         "monthly_leisure_spend":1200,"subscription_count":"1_2","online_exp_purchase_freq":"Once_month",
         "social_platforms":"Instagram|Reels_Shorts","art_experience_level":"Complete_Beginner",
         "creative_self_identity":"Somewhat_Creative","creativity_mindset":"Growth",
         "social_orientation":"Extrovert","reward_preference":"Instant",
         "social_sharing_propensity":"Definitely_Post","participation_barrier":"Too_Expensive",
         "creative_motivation":"Social_Fun","tech_comfort_score":4,
         "instagram_influence_score":5,"price_sensitivity_score":5,
         "session_wtp":"Below_200","monthly_art_spend":0,
         "discount_preference":"First_Free","recommend_likelihood":7,
         "visit_frequency_intent":"Occasionally","decision_autonomy":"Peer_Influenced"},
        {"age_group":"35_44","gender":"Male","city":"Hyderabad","city_tier":"Tier1",
         "occupation":"Business_Owner","income_bracket":"Above_2L","preferred_language":"Telugu",
         "monthly_leisure_spend":12000,"subscription_count":"6_plus","online_exp_purchase_freq":"Weekly",
         "social_platforms":"LinkedIn|Instagram","art_experience_level":"Casual_Hobbyist",
         "creative_self_identity":"Am_Creative","creativity_mindset":"Growth",
         "social_orientation":"Ambivert","reward_preference":"Delayed",
         "social_sharing_propensity":"Probably_Post","participation_barrier":"No_Time",
         "creative_motivation":"Stress_Relief","tech_comfort_score":5,
         "instagram_influence_score":3,"price_sensitivity_score":1,
         "session_wtp":"Above_1200","monthly_art_spend":2500,
         "discount_preference":"Monthly_Pass","recommend_likelihood":9,
         "visit_frequency_intent":"2_3_month","decision_autonomy":"Fully_Independent"},
    ]
    return pd.DataFrame(sample_rows)


def render(df1, df2, arm, wide, clf_models, reg_models, km_model, km_scaler):
    page_header(
        "Predict New Customers",
        "Upload a CSV of new respondents — instantly get interest predictions, WTP estimates, persona assignments, and personalised marketing actions.",
        "07 — Prescriptive Analysis · New Customer Scoring"
    )

    # ── HOW IT WORKS ──────────────────────────────────────────
    st.markdown(f"""
    <div style="background:{SURFACE2};border:1px solid rgba(255,255,255,0.07);border-radius:8px;
                padding:24px 32px;margin-bottom:28px;">
        <div style="color:{GOLD};font-size:12px;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:14px;font-weight:700;">
            How This Works</div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;">
            <div style="text-align:center;padding:16px 8px;">
                <div style="font-size:28px;margin-bottom:8px;">📂</div>
                <div style="color:{INK};font-size:13px;font-weight:600;margin-bottom:4px;">1. Upload CSV</div>
                <div style="color:{MUTED};font-size:12px;">Download the template, fill in your new respondent data, upload here.</div>
            </div>
            <div style="text-align:center;padding:16px 8px;">
                <div style="font-size:28px;margin-bottom:8px;">🤖</div>
                <div style="color:{INK};font-size:13px;font-weight:600;margin-bottom:4px;">2. Auto Prediction</div>
                <div style="color:{MUTED};font-size:12px;">Classification, regression, and clustering run automatically on your data.</div>
            </div>
            <div style="text-align:center;padding:16px 8px;">
                <div style="font-size:28px;margin-bottom:8px;">🎯</div>
                <div style="color:{INK};font-size:13px;font-weight:600;margin-bottom:4px;">3. Scores & Actions</div>
                <div style="color:{MUTED};font-size:12px;">Each customer gets a label, WTP estimate, persona, and marketing action.</div>
            </div>
            <div style="text-align:center;padding:16px 8px;">
                <div style="font-size:28px;margin-bottom:8px;">⬇️</div>
                <div style="color:{INK};font-size:13px;font-weight:600;margin-bottom:4px;">4. Download Results</div>
                <div style="color:{MUTED};font-size:12px;">Export scored CSV for your marketing team — no data science needed.</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── TEMPLATE DOWNLOAD ─────────────────────────────────────
    section_header("Step 1 — Download Template")
    template_df = _make_template()
    csv_template = template_df.to_csv(index=False)
    col_dl, col_note = st.columns([1, 3])
    with col_dl:
        st.download_button(
            "⬇ Download Template CSV",
            csv_template, "aura_new_customers_template.csv", "text/csv"
        )
    with col_note:
        st.markdown(f"<p style='color:{MUTED};font-size:13px;margin-top:8px;'>Template contains 3 sample rows showing the required column format. Delete sample rows and add your own data before uploading.</p>", unsafe_allow_html=True)

    with st.expander("📋 Preview Template", expanded=False):
        st.dataframe(template_df, use_container_width=True)
        st.caption(f"Required columns: {', '.join(CLF_FEATURES[:8])} + others (see template)")

    # ── FILE UPLOAD ───────────────────────────────────────────
    section_header("Step 2 — Upload New Customer Data")
    uploaded = st.file_uploader(
        "Upload your CSV file (must match template column format)",
        type=["csv"],
        help="Max 10,000 rows. Columns must match the template."
    )

    # ── DEMO MODE ─────────────────────────────────────────────
    use_demo = st.checkbox("Use demo data (100 synthetic new customers)", value=True)

    if use_demo and uploaded is None:
        # Generate 100 demo rows from a subset of df1
        np.random.seed(123)
        demo_df = df1.sample(min(100, len(df1)), random_state=123).copy()
        if "aura_interest_label" in demo_df.columns:
            demo_df = demo_df.drop(columns=["aura_interest_label"])
        st.info("Using 100 demo respondents from the existing dataset. Upload your own CSV to score real data.")
        df_new = demo_df
    elif uploaded is not None:
        try:
            df_new = pd.read_csv(uploaded)
            st.success(f"✓ Loaded {len(df_new):,} respondents from uploaded file.")
        except Exception as e:
            st.error(f"Error reading file: {e}")
            return
    else:
        st.info("Upload a CSV file or enable demo mode above.")
        return

    # ── VALIDATE COLUMNS ─────────────────────────────────────
    missing_cols = [c for c in CLF_FEATURES if c not in df_new.columns]
    if missing_cols:
        st.warning(f"⚠ Missing {len(missing_cols)} columns: {missing_cols[:5]}{'...' if len(missing_cols)>5 else ''}")
        st.info("Proceeding with available columns — missing features will be imputed with median values.")

    # ── RUN PREDICTIONS ───────────────────────────────────────
    section_header("Step 3 — Prediction Results")

    # Select best models
    best_clf_name = max(clf_models, key=lambda m: 0)  # use RF as default
    best_clf = clf_models.get("Random Forest", list(clf_models.values())[0])
    best_reg = reg_models.get("Random Forest", list(reg_models.values())[0])

    with st.spinner("Running predictions..."):
        try:
            result_df = predict_new_customers(
                df_new, best_clf, best_reg, km_model, km_scaler
            )
            st.success(f"✓ Scored {len(result_df):,} customers successfully!")
        except Exception as e:
            st.error(f"Prediction error: {e}")
            return

    # ── SUMMARY KPIs ─────────────────────────────────────────
    pred_counts = result_df["predicted_interest"].value_counts()
    total_pred = len(result_df)
    interested_pred = pred_counts.get("Interested", 0)
    avg_wtp_pred = int(result_df["predicted_wtp_inr"].mean())
    high_priority = ((result_df["predicted_interest"] == "Interested") &
                     (result_df["predicted_wtp_inr"] >= 500)).sum()

    c_k1, c_k2, c_k3, c_k4 = st.columns(4)
    c_k1.metric("Total Scored", f"{total_pred:,}")
    c_k2.metric("Predicted Interested", f"{interested_pred} ({interested_pred/total_pred*100:.0f}%)")
    c_k3.metric("Avg Predicted WTP", f"₹{avg_wtp_pred:,}")
    c_k4.metric("High Priority Leads", str(int(high_priority)))

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CHARTS ────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        pred_cnt = result_df["predicted_interest"].value_counts()
        fig1 = go.Figure(go.Pie(
            labels=pred_cnt.index, values=pred_cnt.values,
            hole=0.5,
            marker=dict(colors=[LABEL_COLORS.get(l, GOLD) for l in pred_cnt.index],
                         line=dict(color=SURFACE2, width=2)),
            textinfo="label+percent"
        ))
        fig1.update_layout(title="Predicted Interest Distribution", height=300, showlegend=False)
        st.plotly_chart(fig1, use_container_width=True)

    with c2:
        fig2 = px.histogram(
            result_df, x="predicted_wtp_inr", nbins=30,
            color_discrete_sequence=[GOLD],
            labels={"predicted_wtp_inr": "Predicted WTP (₹)"},
            title="Predicted WTP Distribution"
        )
        fig2.add_vline(x=result_df["predicted_wtp_inr"].median(),
                        line_dash="dash", line_color=TEAL,
                        annotation_text=f"Median ₹{int(result_df['predicted_wtp_inr'].median())}",
                        annotation_font_color=TEAL)
        fig2.update_layout(height=300)
        st.plotly_chart(fig2, use_container_width=True)

    with c3:
        cluster_cnt = result_df["assigned_cluster"].value_counts()
        fig3 = go.Figure(go.Bar(
            x=cluster_cnt.values, y=cluster_cnt.index,
            orientation="h",
            marker_color=TEAL, opacity=0.85,
            text=cluster_cnt.values, textposition="outside"
        ))
        fig3.update_layout(title="Cluster Assignment", height=300,
                            yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig3, use_container_width=True)

    # ── WTP BY CLUSTER ────────────────────────────────────────
    section_header("Predicted WTP by Persona Cluster")
    fig_box = px.box(
        result_df, x="assigned_cluster", y="predicted_wtp_inr",
        color="assigned_cluster", color_discrete_sequence=PALETTE,
        labels={"assigned_cluster": "Persona", "predicted_wtp_inr": "Predicted WTP (₹)"},
        title="WTP Distribution per Customer Persona"
    )
    fig_box.update_layout(height=360, showlegend=False, xaxis_tickangle=-15)
    st.plotly_chart(fig_box, use_container_width=True)

    # ── MARKETING ACTION SUMMARY ──────────────────────────────
    section_header("Marketing Action Plan")
    action_counts = result_df["recommended_action"].value_counts()
    action_df = action_counts.reset_index()
    action_df.columns = ["Action", "Count"]
    action_df["% of Customers"] = (action_df["Count"] / len(result_df) * 100).round(1)

    fig_act = go.Figure(go.Bar(
        x=action_df["Count"], y=action_df["Action"],
        orientation="h",
        marker_color=[TEAL, GOLD, ORANGE, ROSE, MUTED][:len(action_df)],
        text=[f"{c} ({p}%)" for c, p in zip(action_df["Count"], action_df["% of Customers"])],
        textposition="outside"
    ))
    fig_act.update_layout(title="Recommended Marketing Actions", height=320,
                           yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig_act, use_container_width=True)

    # ── RESULT TABLE ──────────────────────────────────────────
    section_header("Scored Customer Records")

    interest_filter = st.multiselect(
        "Filter by predicted interest:",
        result_df["predicted_interest"].unique().tolist(),
        default=result_df["predicted_interest"].unique().tolist()
    )
    filtered_result = result_df[result_df["predicted_interest"].isin(interest_filter)]

    display_cols_avail = ["respondent_id"] if "respondent_id" in result_df.columns else []
    for c in ["age_group", "city", "occupation", "income_bracket",
              "predicted_interest", "confidence_score", "predicted_wtp_inr",
              "assigned_cluster", "recommended_action"]:
        if c in result_df.columns:
            display_cols_avail.append(c)

    def color_label(val):
        colors = {"Interested": f"color: {TEAL}", "Maybe": f"color: {GOLD}",
                  "Not Interested": f"color: {ROSE}"}
        return colors.get(val, "")

    st.dataframe(
        filtered_result[display_cols_avail].head(200).style.map(
            color_label, subset=["predicted_interest"]
        ),
        use_container_width=True, height=360
    )
    st.caption(f"Showing {min(200, len(filtered_result))} of {len(filtered_result):,} filtered records")

    # ── DOWNLOAD RESULTS ──────────────────────────────────────
    section_header("Step 4 — Download Scored Results")
    csv_out = result_df.to_csv(index=False)
    col_dl2, col_note2 = st.columns([1, 3])
    with col_dl2:
        st.download_button(
            "⬇ Download Scored CSV",
            csv_out, "aura_scored_customers.csv", "text/csv"
        )
    with col_note2:
        st.markdown(f"<p style='color:{MUTED};font-size:13px;margin-top:8px;'>Full scored dataset with predictions, WTP estimates, cluster assignments, and marketing actions ready for your team.</p>", unsafe_allow_html=True)

    info_card(
        "Prescriptive Insight",
        "This scoring engine turns raw survey responses into a <b>prioritised marketing action list</b> — instantly. "
        "High-confidence 'Interested' customers with WTP ≥ ₹700 should receive premium package outreach within 24 hours of data collection. "
        "For future surveys, <b>only 8 key columns are strictly required</b> for accurate prediction (creative_self_identity, participation_barrier, "
        "visit_frequency_intent, tech_comfort_score, income_bracket, subscription_count, art_experience_level, instagram_influence_score). "
        "This enables a <b>rapid 2-minute screening form</b> for mall activations and corporate events.",
        TEAL
    )
