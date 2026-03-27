"""
aura_data.py
Data loading, cleaning, feature engineering, and model training utilities.
All heavy computation is cached with st.cache_data / st.cache_resource.
"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc, classification_report, confusion_matrix,
    mean_squared_error, mean_absolute_error, r2_score
)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except ImportError:
    HAS_SMOTE = False

# ── ORDINAL ENCODINGS ──────────────────────────────────────────
ORDINAL_MAPS = {
    "age_group": {"Under_18": 0, "18_24": 1, "25_34": 2, "35_44": 3, "45_54": 4, "55_plus": 5},
    "income_bracket": {"Below_25k": 0, "25k_50k": 1, "50k_1L": 2, "1L_2L": 3, "Above_2L": 4, "Prefer_not": 2},
    "art_experience_level": {
        "Complete_Beginner": 0, "Curious_Beginner": 1, "Casual_Hobbyist": 2,
        "Regular_Hobbyist": 3, "Advanced": 4
    },
    "subscription_count": {"0": 0, "1_2": 1, "3_5": 2, "6_plus": 3},
    "online_exp_purchase_freq": {"Rarely": 0, "Occasionally": 1, "Once_month": 2, "2_3_month": 3, "Weekly": 4},
    "visit_frequency_intent": {"Unlikely": 0, "Once_Try": 1, "Occasionally": 2, "Once_month": 3, "2_3_month": 4, "Weekly": 5},
    "price_sensitivity_score": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    "tech_comfort_score": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    "instagram_influence_score": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
    "city_tier": {"Tier1": 2, "Tier2": 1, "Tier3": 0},
    "decision_autonomy": {
        "Fully_Independent": 3, "Mention_After": 2, "Consult_Partner": 1, "Peer_Influenced": 0
    },
    "creative_self_identity": {
        "Am_Creative": 3, "Somewhat_Creative": 2, "Want_To_Be": 1, "Not_Creative": 0
    },
    "social_sharing_propensity": {
        "Definitely_Post": 3, "Probably_Post": 2, "Close_Circle": 1, "Keep_Private": 0
    },
    "participation_barrier": {
        "No_Barrier": 4, "Social_Anxiety": 3, "No_Guidance": 2,
        "No_Time": 1, "Too_Expensive": 0, "Not_Interested": -1
    },
    "session_wtp": {
        "Below_200": 0, "200_400": 1, "400_700": 2, "700_1200": 3, "Above_1200": 4
    },
    "recommend_likelihood": {i: i for i in range(11)},
    "social_orientation": {"Introvert": 0, "Ambivert": 1, "Extrovert": 2},
    "reward_preference": {"Instant": 0, "Both": 1, "Delayed": 2},
    "travel_tolerance": {"Walking_5min": 0, "15min": 1, "30min": 2, "60min": 3, "No_Limit": 4},
    "gifting_frequency": {"Never": 0, "Rarely": 1, "Occasionally": 2, "Frequently": 3},
    "creativity_mindset": {"Fixed": 0, "Unsure": 1, "Growth": 2},
}

# ── CLASSIFICATION FEATURES ────────────────────────────────────
CLF_FEATURES = [
    "age_group", "income_bracket", "city_tier", "art_experience_level",
    "creative_self_identity", "social_sharing_propensity", "participation_barrier",
    "visit_frequency_intent", "tech_comfort_score", "instagram_influence_score",
    "subscription_count", "online_exp_purchase_freq", "price_sensitivity_score",
    "recommend_likelihood", "creativity_mindset", "social_orientation",
    "reward_preference", "decision_autonomy", "monthly_leisure_spend",
    "session_wtp",
]

# ── CLUSTERING FEATURES ────────────────────────────────────────
CLU_FEATURES = [
    "age_group", "income_bracket", "art_experience_level", "creative_self_identity",
    "social_sharing_propensity", "subscription_count", "session_wtp",
    "visit_frequency_intent", "tech_comfort_score", "instagram_influence_score",
    "price_sensitivity_score", "social_orientation", "reward_preference",
    "city_tier", "decision_autonomy", "monthly_leisure_spend",
]

# ── REGRESSION FEATURES ────────────────────────────────────────
REG_FEATURES = [
    "income_bracket", "art_experience_level", "subscription_count",
    "tech_comfort_score", "city_tier", "social_sharing_propensity",
    "instagram_influence_score", "monthly_leisure_spend",
    "price_sensitivity_score", "age_group", "online_exp_purchase_freq",
    "creative_self_identity", "recommend_likelihood", "visit_frequency_intent",
]


# ═══════════════════════════════════════════════════════════════
# LOAD DATA
# ═══════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_data():
    """Load all datasets with error handling."""
    try:
        df1 = pd.read_csv("aura_survey1_n2000.csv")
        df2 = pd.read_csv("aura_survey2_n1314.csv")
        arm = pd.read_csv("aura_arm_transactions.csv")
        wide = pd.read_csv("aura_combined_wide.csv")
        return df1, df2, arm, wide
    except FileNotFoundError as e:
        st.error(f"Data file not found: {e}. Ensure CSV files are in the same directory as app.py")
        st.stop()


# ═══════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════

def encode_features(df: pd.DataFrame, feature_list: list) -> pd.DataFrame:
    """Encode features using ordinal maps + label encoding for unknowns."""
    df_enc = df.copy()
    for col in feature_list:
        if col not in df_enc.columns:
            df_enc[col] = 0
            continue
        if col in ORDINAL_MAPS:
            df_enc[col] = df_enc[col].map(ORDINAL_MAPS[col])
        elif df_enc[col].dtype == object:
            le = LabelEncoder()
            df_enc[col] = le.fit_transform(df_enc[col].astype(str).fillna("Unknown"))
        df_enc[col] = pd.to_numeric(df_enc[col], errors="coerce")
        df_enc[col] = df_enc[col].fillna(df_enc[col].median() if df_enc[col].notna().any() else 0)
    return df_enc[feature_list]


def prepare_clf_data(df: pd.DataFrame):
    """Prepare X, y for classification."""
    df_clean = df.dropna(subset=["aura_interest_label"]).copy()
    X = encode_features(df_clean, CLF_FEATURES)
    y = df_clean["aura_interest_label"].map({
        "Interested": 2, "Maybe": 1, "Not_Interested": 0
    }).fillna(0).astype(int)
    return X, y, df_clean


def prepare_reg_data(df: pd.DataFrame):
    """Prepare X, y for regression."""
    df_clean = df.dropna(subset=["session_wtp_numeric"]).copy()
    df_clean = df_clean[df_clean["session_wtp_numeric"] > 0].copy()
    X = encode_features(df_clean, REG_FEATURES)
    y = df_clean["session_wtp_numeric"].astype(float)
    return X, y


def prepare_cluster_data(df: pd.DataFrame):
    """Prepare scaled X for clustering (interested segment only)."""
    df_int = df[df["aura_interest_label"] == "Interested"].copy()
    if len(df_int) < 50:
        df_int = df.copy()
    X = encode_features(df_int, CLU_FEATURES)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, df_int, scaler


# ═══════════════════════════════════════════════════════════════
# MODEL TRAINING
# ═══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def train_classification_models(_df):
    """Train RF, XGBoost (if available), and Logistic Regression classifiers."""
    X, y, df_clean = prepare_clf_data(_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Handle imbalance
    if HAS_SMOTE:
        try:
            sm = SMOTE(random_state=42, k_neighbors=min(5, min(np.bincount(y_train)) - 1))
            X_train_res, y_train_res = sm.fit_resample(X_train, y_train)
        except Exception:
            X_train_res, y_train_res = X_train, y_train
    else:
        X_train_res, y_train_res = X_train, y_train

    models = {}
    results = {}

    # Random Forest
    rf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42,
                                 class_weight="balanced", n_jobs=-1)
    rf.fit(X_train_res, y_train_res)
    models["Random Forest"] = rf
    results["Random Forest"] = _eval_clf(rf, X_test, y_test)

    # XGBoost
    if HAS_XGB:
        try:
            xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                 random_state=42, eval_metric="mlogloss", verbosity=0)
            xgb.fit(X_train_res, y_train_res)
            models["XGBoost"] = xgb
            results["XGBoost"] = _eval_clf(xgb, X_test, y_test)
        except Exception:
            pass

    # Logistic Regression
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train_res)
    X_te_sc = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced",
                             solver="lbfgs")
    lr.fit(X_tr_sc, y_train_res)
    models["Logistic Regression"] = (lr, scaler)
    results["Logistic Regression"] = _eval_clf(lr, X_te_sc, y_test)

    # Feature importance from RF
    feat_imp = pd.DataFrame({
        "feature": CLF_FEATURES,
        "importance": rf.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return models, results, feat_imp, X_test, y_test, X_train, y_train


def _eval_clf(model, X_test, y_test):
    """Evaluate classifier and return metrics dict."""
    if isinstance(model, tuple):
        m = model[0]
    else:
        m = model
    y_pred = m.predict(X_test)
    y_prob = None
    try:
        y_prob = m.predict_proba(X_test)
    except Exception:
        pass

    res = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
        "y_pred":    y_pred,
        "y_prob":    y_prob,
        "cm":        confusion_matrix(y_test, y_pred),
        "report":    classification_report(y_test, y_pred,
                                           target_names=["Not Interested","Maybe","Interested"],
                                           output_dict=True, zero_division=0),
    }
    return res


@st.cache_resource(show_spinner=False)
def train_regression_models(_df):
    """Train regression models for WTP prediction."""
    X, y = prepare_reg_data(_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train)
    X_te_sc = scaler.transform(X_test)

    models = {}
    results = {}

    # Random Forest Regressor
    rfr = RandomForestRegressor(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    rfr.fit(X_train, y_train)
    models["Random Forest"] = rfr
    results["Random Forest"] = _eval_reg(rfr, X_test, y_test)

    # Gradient Boosting
    gbr = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    gbr.fit(X_train, y_train)
    models["Gradient Boosting"] = gbr
    results["Gradient Boosting"] = _eval_reg(gbr, X_test, y_test)

    # Linear Regression (baseline)
    lr = LinearRegression()
    lr.fit(X_tr_sc, y_train)
    models["Linear Regression"] = (lr, scaler)
    results["Linear Regression"] = _eval_reg(lr, X_te_sc, y_test)

    feat_imp = pd.DataFrame({
        "feature": REG_FEATURES,
        "importance": rfr.feature_importances_
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    return models, results, feat_imp, X_test, y_test, scaler


def _eval_reg(model, X_test, y_test):
    if isinstance(model, tuple):
        m = model[0]
    else:
        m = model
    y_pred = m.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    return {
        "rmse": round(rmse, 2),
        "mae":  round(float(mean_absolute_error(y_test, y_pred)), 2),
        "r2":   round(float(r2_score(y_test, y_pred)), 4),
        "y_pred": y_pred,
        "y_test": np.array(y_test),
    }


@st.cache_resource(show_spinner=False)
def train_clustering(_df):
    """Train KMeans, select K via elbow + silhouette."""
    X_scaled, df_int, scaler = prepare_cluster_data(_df)

    inertias, silhouettes = [], []
    K_range = range(2, 9)
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    best_k = K_range[int(np.argmax(silhouettes))]
    best_k = max(4, min(best_k, 6))

    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=15)
    cluster_labels = km_final.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    df_int = df_int.copy()
    df_int["cluster"] = cluster_labels
    df_int["pca_x"] = X_pca[:, 0]
    df_int["pca_y"] = X_pca[:, 1]

    return km_final, df_int, scaler, best_k, list(K_range), inertias, silhouettes, pca


# ═══════════════════════════════════════════════════════════════
# ARM DATA PREP
# ═══════════════════════════════════════════════════════════════

def get_arm_basket(df_arm: pd.DataFrame, prefix: str = "prod_") -> pd.DataFrame:
    """Extract binary basket columns for a given prefix."""
    cols = [c for c in df_arm.columns if c.startswith(prefix)]
    return df_arm[cols].copy()


# ═══════════════════════════════════════════════════════════════
# CLUSTER NAMING
# ═══════════════════════════════════════════════════════════════

CLUSTER_NAMES = {
    0: "🎨 Weekend Creative",
    1: "📸 Status Sharer",
    2: "🖌️ Serious Hobbyist",
    3: "💼 Corporate Buyer",
    4: "👨‍👩‍👧 Curious Parent",
    5: "🎓 Student Explorer",
}

CLUSTER_STRATEGY = {
    "🎨 Weekend Creative": {
        "price": "₹400–₹700/session",
        "discount": "Buy 3 Get 1 Free Bundle",
        "channel": "Instagram + Mall Display",
        "product": "Mandala Kit + Fluid Art Kit",
        "pod_time": "Evening & Weekends",
    },
    "📸 Status Sharer": {
        "price": "₹199 intro session",
        "discount": "First Session Free / Heavy Intro Discount",
        "channel": "Instagram Reels + College Campus",
        "product": "AURA Sketchbook + Mandala Kit",
        "pod_time": "Afternoons & Evenings",
    },
    "🖌️ Serious Hobbyist": {
        "price": "₹700–₹1,200/session",
        "discount": "Monthly Unlimited Pass",
        "channel": "YouTube + LinkedIn",
        "product": "Heritage Art Kit + Watercolour Set",
        "pod_time": "Morning & Weekends",
    },
    "💼 Corporate Buyer": {
        "price": "₹900–₹2,500/pod/event",
        "discount": "Corporate Package + Diwali Bundle",
        "channel": "LinkedIn + HR Network + Email",
        "product": "Corporate Gift Bundle + Session Credits",
        "pod_time": "Weekday Daytime",
    },
    "👨‍👩‍👧 Curious Parent": {
        "price": "₹350–₹600 family session",
        "discount": "Family Pack (2 adults + 1 child)",
        "channel": "Instagram + WhatsApp Communities",
        "product": "Session Kit + Heritage Kit",
        "pod_time": "Weekends & School Holidays",
    },
    "🎓 Student Explorer": {
        "price": "₹150–₹300/session",
        "discount": "Student ID Discount + Group Discount",
        "channel": "Instagram + College Partnerships",
        "product": "Sketchbook + Mandala Kit",
        "pod_time": "Afternoons & Evenings",
    },
}


def name_cluster(cid: int) -> str:
    return CLUSTER_NAMES.get(cid, f"Segment {cid+1}")


def get_strategy(cluster_name: str) -> dict:
    return CLUSTER_STRATEGY.get(cluster_name, {})


# ═══════════════════════════════════════════════════════════════
# PREDICTION FOR NEW DATA UPLOAD
# ═══════════════════════════════════════════════════════════════

def predict_new_customers(df_new: pd.DataFrame, clf_model, reg_model,
                           km_model, km_scaler, reg_scaler=None):
    """Run full prediction pipeline on new uploaded data."""
    LABEL_MAP = {2: "Interested", 1: "Maybe", 0: "Not Interested"}

    # Classification
    X_clf = encode_features(df_new, CLF_FEATURES)
    X_clf = X_clf.fillna(X_clf.median())

    if isinstance(clf_model, tuple):
        m, sc = clf_model
        X_clf_sc = sc.transform(X_clf)
        y_pred_clf = m.predict(X_clf_sc)
        try:
            y_prob_clf = m.predict_proba(X_clf_sc).max(axis=1)
        except Exception:
            y_prob_clf = np.ones(len(df_new)) * 0.5
    else:
        y_pred_clf = clf_model.predict(X_clf)
        try:
            y_prob_clf = clf_model.predict_proba(X_clf).max(axis=1)
        except Exception:
            y_prob_clf = np.ones(len(df_new)) * 0.5

    # Regression
    X_reg = encode_features(df_new, REG_FEATURES)
    X_reg = X_reg.fillna(X_reg.median())
    if isinstance(reg_model, tuple):
        m_reg, sc_reg = reg_model
        X_reg_sc = sc_reg.transform(X_reg)
        y_pred_wtp = m_reg.predict(X_reg_sc)
    else:
        y_pred_wtp = reg_model.predict(X_reg)

    # Clustering
    X_clu = encode_features(df_new, CLU_FEATURES)
    X_clu = X_clu.fillna(X_clu.median())
    X_clu_sc = km_scaler.transform(X_clu)
    cluster_ids = km_model.predict(X_clu_sc)

    result = df_new.copy()
    result["predicted_interest"] = [LABEL_MAP.get(int(p), "Maybe") for p in y_pred_clf]
    result["confidence_score"] = np.round(y_prob_clf, 3)
    result["predicted_wtp_inr"] = np.round(y_pred_wtp).astype(int)
    result["assigned_cluster"] = [name_cluster(int(c)) for c in cluster_ids]
    result["recommended_action"] = result.apply(_get_action, axis=1)
    return result


def _get_action(row):
    label = row.get("predicted_interest", "Maybe")
    cluster = row.get("assigned_cluster", "")
    wtp = row.get("predicted_wtp_inr", 400)
    if label == "Interested" and wtp >= 700:
        return "🟢 High priority — Send premium package offer immediately"
    elif label == "Interested" and wtp < 400:
        return "🟡 Interested but price-sensitive — Send intro discount"
    elif label == "Interested":
        return "🟢 Send standard onboarding offer and pod location details"
    elif label == "Maybe":
        if "Corporate" in cluster:
            return "🟠 Warm lead — Send corporate team-building brochure"
        return "🟠 Nurture — Send AURA story content + first session offer"
    else:
        return "🔴 Low priority — Add to awareness-only email list"
