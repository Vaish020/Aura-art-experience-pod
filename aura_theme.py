"""
aura_theme.py
AURA brand colors, Plotly template, and shared styling utilities.
"""

# ── BRAND PALETTE ──────────────────────────────────────────────
GOLD      = "#e8c547"
GOLD_SOFT = "#f5e199"
TEAL      = "#5ec4a1"
ORANGE    = "#e07c3a"
INDIGO    = "#7b9fe8"
ROSE      = "#e06b8b"
MUTED     = "#7a7268"
BG        = "#0e0c0a"
SURFACE   = "#161310"
SURFACE2  = "#1c1814"
INK       = "#f2ede6"
BORDER    = "rgba(255,255,255,0.07)"

PALETTE = [GOLD, TEAL, ORANGE, INDIGO, ROSE, "#a3e635", "#fb923c", "#c084fc"]

LABEL_COLORS = {
    "Interested":     TEAL,
    "Maybe":          GOLD,
    "Not_Interested": ROSE,
    "Not Interested": ROSE,
}

CLUSTER_COLORS = [GOLD, TEAL, ORANGE, INDIGO, ROSE]

# ── PLOTLY TEMPLATE ────────────────────────────────────────────
import plotly.graph_objects as go
import plotly.io as pio

AURA_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE2,
        font=dict(family="'Plus Jakarta Sans', sans-serif", color=INK, size=12),
        title=dict(font=dict(color=GOLD, size=16, family="sans-serif"), x=0.03),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.05)", gridwidth=1,
            linecolor="rgba(255,255,255,0.1)", tickcolor=MUTED,
            zerolinecolor="rgba(255,255,255,0.05)",
        ),
        yaxis=dict(
            gridcolor="rgba(255,255,255,0.05)", gridwidth=1,
            linecolor="rgba(255,255,255,0.1)", tickcolor=MUTED,
            zerolinecolor="rgba(255,255,255,0.05)",
        ),
        legend=dict(
            bgcolor="rgba(22,19,16,0.8)",
            bordercolor="rgba(255,255,255,0.08)",
            borderwidth=1,
            font=dict(color=INK, size=11),
        ),
        colorway=PALETTE,
        margin=dict(l=40, r=20, t=50, b=40),
        hoverlabel=dict(
            bgcolor=SURFACE2,
            bordercolor=GOLD,
            font=dict(color=INK, size=12),
        ),
    )
)

pio.templates["aura"] = AURA_TEMPLATE
pio.templates.default = "aura"


# ── STREAMLIT CSS ──────────────────────────────────────────────
GLOBAL_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {{
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    background-color: {BG};
    color: {INK};
}}

/* Sidebar */
section[data-testid="stSidebar"] {{
    background-color: {SURFACE} !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}}

/* Metric cards */
div[data-testid="metric-container"] {{
    background: {SURFACE2};
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 8px;
    padding: 16px 20px;
    border-top: 2px solid {GOLD};
}}
div[data-testid="metric-container"] label {{
    color: {MUTED} !important;
    font-size: 11px !important;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {{
    color: {GOLD} !important;
    font-size: 28px !important;
    font-weight: 700;
}}

/* Tabs */
button[data-baseweb="tab"] {{
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 12px !important;
    letter-spacing: 0.08em;
    color: {MUTED} !important;
    font-weight: 500;
}}
button[data-baseweb="tab"][aria-selected="true"] {{
    color: {GOLD} !important;
    border-bottom: 2px solid {GOLD} !important;
}}

/* DataFrames */
div[data-testid="stDataFrame"] {{
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 6px;
}}

/* Buttons */
button[kind="primary"], .stButton > button {{
    background: {GOLD} !important;
    color: {BG} !important;
    border: none !important;
    border-radius: 4px !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
}}
button[kind="primary"]:hover, .stButton > button:hover {{
    background: {TEAL} !important;
    color: {BG} !important;
}}

/* File uploader */
section[data-testid="stFileUploaderDropzone"] {{
    background: {SURFACE2} !important;
    border: 1px dashed rgba(232,197,71,0.3) !important;
    border-radius: 8px;
}}

/* Selectbox, multiselect */
div[data-baseweb="select"] {{
    background: {SURFACE2} !important;
}}

/* Expanders */
details summary {{
    color: {GOLD} !important;
    font-weight: 600;
}}

/* Scrollbar */
::-webkit-scrollbar {{ width: 4px; height: 4px; }}
::-webkit-scrollbar-track {{ background: {SURFACE}; }}
::-webkit-scrollbar-thumb {{ background: {MUTED}; border-radius: 2px; }}

/* Hide Streamlit branding */
#MainMenu, footer {{ visibility: hidden; }}
header[data-testid="stHeader"] {{ background: {BG}; border-bottom: 1px solid rgba(255,255,255,0.05); }}
</style>
"""


def page_header(title: str, subtitle: str = "", tag: str = ""):
    """Render a branded page header."""
    import streamlit as st
    tag_html = f'<span style="font-family:monospace;font-size:10px;letter-spacing:0.18em;text-transform:uppercase;color:{MUTED};display:block;margin-bottom:8px;">{tag}</span>' if tag else ""
    sub_html = f'<p style="color:{MUTED};font-size:15px;margin:0;line-height:1.6;">{subtitle}</p>' if subtitle else ""
    st.markdown(f"""
    <div style="padding:32px 0 24px;border-bottom:1px solid rgba(255,255,255,0.06);margin-bottom:32px;">
        {tag_html}
        <h1 style="font-size:clamp(24px,4vw,36px);font-weight:700;color:{INK};margin:0 0 8px;line-height:1.1;">{title}</h1>
        {sub_html}
    </div>
    """, unsafe_allow_html=True)


def section_header(title: str, color: str = GOLD):
    import streamlit as st
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:12px;margin:28px 0 16px;">
        <div style="width:3px;height:20px;background:{color};border-radius:2px;"></div>
        <span style="font-size:14px;font-weight:700;color:{INK};letter-spacing:0.04em;text-transform:uppercase;">{title}</span>
    </div>
    """, unsafe_allow_html=True)


def kpi_row(metrics: list):
    """metrics = [(label, value, delta), ...]"""
    import streamlit as st
    cols = st.columns(len(metrics))
    for col, (label, value, delta) in zip(cols, metrics):
        col.metric(label, value, delta)


def info_card(title: str, body: str, color: str = GOLD):
    import streamlit as st
    st.markdown(f"""
    <div style="background:{SURFACE2};border:1px solid rgba(255,255,255,0.07);border-left:3px solid {color};
                border-radius:6px;padding:20px 24px;margin:8px 0;">
        <div style="font-size:12px;font-weight:700;color:{color};letter-spacing:0.1em;
                    text-transform:uppercase;margin-bottom:8px;">{title}</div>
        <div style="font-size:14px;color:{INK};opacity:0.75;line-height:1.7;">{body}</div>
    </div>
    """, unsafe_allow_html=True)
