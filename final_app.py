from pathlib import Path
import base64

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBRegressor

# ==========================================
# CALORIEPRO CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="CaloriePro | Fitness Intelligence",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).resolve().parent

IMAGE_PATHS = [
    BASE_DIR / "caloriepro_hero.png",
    BASE_DIR / "caloriepro_hero.jpg"
]

CSV_PATHS = [
    BASE_DIR / "exercise_preprocessing_practice.csv",
    BASE_DIR / "exercise.csv"
]

MODEL_PATHS = [
    BASE_DIR / "xgboost_model (1).json",
    BASE_DIR / "xgboost_model.json"
]

FEATURES = [
    "Gender", "Age", "Height", "Weight",
    "Duration", "Heart_Rate", "Body_Temp"
]


def find_file(paths):
    return next((p for p in paths if p.exists()), None)


# ==========================================
# DARK NAVY + TEAL THEME
# ==========================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background: radial-gradient(
        ellipse at 70% 0%, #0b2632 0%,
        #04121d 45%, #020910 100%
    );
    color: #edf5fc;
    font-family: Inter, sans-serif;
}

[data-testid="stHeader"] {
    background: rgba(2, 9, 16, 0.9);
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020a11, #061521);
    border-right: 1px solid #12424f;
}

.block-container {
    max-width: 1800px;
    padding-top: 1.3rem;
    padding-bottom: 1rem;
}

h1, h2, h3, h4 {
    color: #f5f9ff !important;
}

p, label {
    color: #c2d1df;
}

.brand {
    text-align: center;
    padding: 12px 0 20px;
}

.brand-icon {
    font-size: 34px;
    color: #12e1cc;
}

.brand-name {
    color: #f5f9ff;
    font-size: 24px;
    font-weight: 800;
    letter-spacing: 1px;
}

.brand-name span {
    color: #12e1cc;
}

.brand-subtitle {
    color: #9ab4c9;
    font-size: 9px;
    letter-spacing: 2px;
    margin-top: 5px;
}

.nav-heading {
    color: #79b9df;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.5px;
}

.hero {
    min-height: 280px;
    border: 1px solid #10505c;
    border-radius: 17px;
    padding: 30px;
    margin-bottom: 22px;
    display: flex;
    align-items: center;
    background-color: #03121c;
    background-size: cover;
    background-position: center 43%;
    box-shadow: 0 12px 35px rgba(0,0,0,.25);
}

.hero-content {
    max-width: 650px;
}

.eyebrow {
    color: #12e6d2;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 3px;
}

.hero-title {
    color: #f8fbff;
    font-size: clamp(30px, 3.2vw, 46px);
    font-weight: 800;
    line-height: 1.08;
    margin: 14px 0 10px;
}

.hero-title span {
    color: #10e1cd;
}

.hero-copy {
    color: #c4d2df;
    font-size: 15px;
    line-height: 1.55;
}

.teal-line {
    height: 3px;
    width: 38px;
    background: #12e1cc;
    margin-top: 18px;
}

.metric-card {
    display: flex;
    align-items: center;
    gap: 13px;
    min-height: 108px;
    padding: 14px;
    border: 1px solid #10505c;
    border-radius: 13px;
    background: linear-gradient(145deg, #0b202d, #03121d);
    box-sizing: border-box;
}

.metric-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 48px;
    height: 48px;
    flex-shrink: 0;
    border-radius: 50%;
    background: #092a35;
    color: #12e1cc;
    font-size: 23px;
}

.metric-label {
    color: #a8bfd4;
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 1px;
}

.metric-value {
    color: #f7fbff;
    font-size: clamp(18px, 1.6vw, 25px);
    font-weight: 800;
    margin-top: 6px;
    white-space: nowrap;
}

.metric-note {
    color: #12e1cc;
    font-size: 11px;
    margin-top: 3px;
}

.panel {
    background: linear-gradient(145deg, #071b29, #03121e);
    border: 1px solid #103f4d;
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 16px;
}

.panel-title {
    color: #f5f9ff;
    font-size: 18px;
    font-weight: 800;
}

.panel-subtitle {
    color: #a9bfd2;
    font-size: 12px;
    margin: 4px 0 14px;
}

.stat-box {
    background: #071b29;
    border: 1px solid #173b4b;
    border-radius: 9px;
    padding: 11px;
    min-height: 72px;
}

.stat-label {
    color: #a9bfd2;
    font-size: 11px;
}

.stat-value {
    color: #f5f9ff;
    font-size: 20px;
    font-weight: 800;
    margin-top: 5px;
}

.quote {
    color: #e6f1fa;
    font-size: 12px;
    font-weight: 800;
    line-height: 1.9;
    letter-spacing: 3px;
    border-left: 2px solid #12e1cc;
    padding-left: 13px;
    margin-top: 12px;
}

.stButton > button,
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #087f80, #12e1cc);
    color: #021018;
    font-weight: 800;
    border: none;
    border-radius: 9px;
    min-height: 42px;
}

div[data-testid="stMetric"] {
    background: #071b29;
    border: 1px solid #164451;
    border-radius: 10px;
    padding: 12px;
}

div[data-testid="stMetricValue"] {
    color: #12e1cc;
}

hr {
    border-color: #183746;
}

footer {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)


# ==========================================
# HELPERS
# ==========================================
def metric_card(label, value, note, icon):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div>
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-note">{note}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def chart_style(fig, height=280):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#c5d6e5", family="Arial", size=11),
        height=height,
        margin=dict(l=15, r=15, t=20, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    fig.update_xaxes(gridcolor="#183543")
    fig.update_yaxes(gridcolor="#183543")
    return fig


@st.cache_data
def read_csv(path):
    return pd.read_csv(path)


@st.cache_resource
def load_model(path):
    model = XGBRegressor()
    model.load_model(path)
    return model


def clean_for_analysis(data):
    data = data.copy()

    if "Gender" in data.columns:
        data["Gender"] = (
            data["Gender"].astype("string")
            .str.strip().str.lower()
            .replace({
                "m": "male",
                "f": "female",
                "": pd.NA,
                "nan": pd.NA
            })
        )

    numeric_columns = [
        "Age", "Height", "Weight", "Duration",
        "Heart_Rate", "Body_Temp", "Calories"
    ]

    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(
                data[col], errors="coerce"
            )

    return data


# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.markdown("""
    <div class="brand">
        <div class="brand-icon">▥</div>
        <div class="brand-name">CALORIE<span>PRO</span></div>
        <div class="brand-subtitle">
            SMART ACTIVITY INTELLIGENCE
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown(
        '<div class="nav-heading">NAVIGATION</div>',
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Calorie Prediction",
            
        ],
        label_visibility="collapsed"
    )

    st.divider()

    st.markdown(
        '<div class="nav-heading">ABOUT</div>',
        unsafe_allow_html=True
    )

    st.write(
        "An ML-powered calorie prediction project "
        "using XGBoost regression."
    )
    st.caption("Output: estimated calories burned")


    st.markdown("""
    <div class="quote">
        DISCIPLINE<br>
        BUILDS<br>
        FREEDOM
    </div>
    """, unsafe_allow_html=True)


# ==========================================
# LOAD DATASET AND MODEL
# ==========================================

# LOAD DATASET AND MODEL

exercise_path = BASE_DIR / "exercise_new.csv"
calories_path = BASE_DIR / "calories.csv"

df = None

try:
    exercise_data = pd.read_csv(exercise_path)
    calories_data = pd.read_csv(calories_path)

    exercise_data.columns = exercise_data.columns.str.strip()
    calories_data.columns = calories_data.columns.str.strip()

    df = pd.merge(
        exercise_data,
        calories_data[["User_ID", "Calories"]],
        on="User_ID",
        how="inner"
    )

    df = clean_for_analysis(df)



except Exception as e:
    st.error(f"Dataset error: {e}")

# LOAD TRAINED XGBOOST MODEL
model_path = BASE_DIR / "xgboost_model (1).json"
model = None

if model_path.exists():
    try:
        model = load_model(str(model_path))
    except Exception as e:
        st.error(f"Model loading error: {e}")
else:
    st.error("XGBoost model file not found.")


# ==========================================
# HERO BANNER WITH GYM IMAGE
# ==========================================
hero_path = find_file(IMAGE_PATHS)

if hero_path:
    image_bytes = base64.b64encode(
        hero_path.read_bytes()
    ).decode("utf-8")

    mime = (
        "image/png"
        if hero_path.suffix.lower() == ".png"
        else "image/jpeg"
    )

    background = (
        "linear-gradient(90deg,"
        "rgba(2,10,17,.98) 0%,"
        "rgba(2,14,21,.87) 42%,"
        "rgba(2,12,19,.10) 100%),"
        f"url('data:{mime};base64,{image_bytes}')"
    )
else:
    background = (
        "linear-gradient(110deg,#061723,#0d3a46)"
    )

st.markdown(f"""
<div class="hero" style="background-image:{background};">
    <div class="hero-content">
        <div class="eyebrow">
            PERFORMANCE • DATA • PROGRESS
        </div>
        <div class="hero-title">
            MAKE EVERY<br>
            <span>MOVEMENT COUNT.</span>
        </div>
        <div class="hero-copy">
            Estimate calories burned, explore activity patterns,<br>
            and transform exercise data into meaningful.
        </div>
        <div class="teal-line"></div>
    </div>
</div>
""", unsafe_allow_html=True)

if not hero_path:
    st.warning(
        "Gym photo not found. Put caloriepro_hero.png "
        "beside app.py."
    )


# ==========================================
# OVERVIEW PAGE
# ==========================================
if page == "Overview":

    if df is None:
        st.warning(
            "Dataset not found. Put "
            "exercise_preprocessing_practice.csv "
            "beside app.py."
        )

    else:
        records = len(df)

        users = (
            df["User_ID"].nunique()
            if "User_ID" in df.columns
            else records
        )

        avg_hr = (
            df["Heart_Rate"].mean()
            if "Heart_Rate" in df.columns
            else np.nan
        )

        avg_cal = (
            df["Calories"].mean()
            if "Calories" in df.columns
            else np.nan
        )

        k1, k2, k3, k4 = st.columns(4)

        with k1:
            metric_card(
                "TOTAL RECORDS",
                f"{records:,}",
                "Complete dataset",
                "♨"
            )

        with k2:
            metric_card(
                "UNIQUE USERS",
                f"{users:,}",
                "Distinct users",
                "♟"
            )

        with k3:
            metric_card(
                "AVG HEART RATE",
                f"{avg_hr:.1f} bpm"
                if pd.notna(avg_hr) else "N/A",
                "Dataset average",
                "♥"
            )

        with k4:
            metric_card(
                "AVG CALORIES",
                f"{avg_cal:.1f}"
                if pd.notna(avg_cal) else "N/A",
                "Per session",
                "↗"
            )

        # QUICK OVERVIEW AND GENDER
        left, right = st.columns([1, 1.05])

        with left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)

            st.markdown("""
            <div class="panel-title">
                ▥ &nbsp; Quick Overview
            </div>
            <div class="panel-subtitle">
                Key statistics about your exercise dataset.
            </div>
            """, unsafe_allow_html=True)

            stats = [
                ("Mean Age", "Age", ""),
                ("Mean Height", "Height", ""),
                ("Mean Weight", "Weight", ""),
                ("Mean Duration", "Duration", " min")
            ]

            cols = st.columns(4)

            for col_ui, (label, col, suffix) in zip(cols, stats):
                value = (
                    df[col].mean()
                    if col in df.columns
                    else np.nan
                )

                shown = (
                    f"{value:.1f}{suffix}"
                    if pd.notna(value)
                    else "N/A"
                )

                with col_ui:
                    st.markdown(f"""
                    <div class="stat-box">
                        <div class="stat-label">{label}</div>
                        <div class="stat-value">{shown}</div>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("<div style='height:35px'></div>",
                        unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)

            st.markdown("""
            <div class="panel-title">
                ◉ &nbsp; Gender Distribution
            </div>
            <div class="panel-subtitle">
                Distribution of users in the dataset.
            </div>
            """, unsafe_allow_html=True)

            if "Gender" in df.columns:
                gender_counts = (
                    df["Gender"].dropna().str.title()
                    .value_counts()
                    .rename_axis("Gender")
                    .reset_index(name="Count")
                )

                if not gender_counts.empty:
                    fig = px.pie(
                        gender_counts,
                        names="Gender",
                        values="Count",
                        hole=0.58,
                        color_discrete_sequence=[
                            "#22a8ed", "#ed4d87", "#10dcca"
                        ]
                    )

                    fig.update_traces(
                        textinfo="percent",
                        textfont_color="#07131b"
                    )

                    st.plotly_chart(
                        chart_style(fig, 245),
                        use_container_width=True,
                        config={"displayModeBar": False}
                    )
                else:
                    st.info("No valid gender values found.")
            else:
                st.info("Gender column is missing.")

            st.markdown("</div>", unsafe_allow_html=True)

        # AGE AND CALORIE CHARTS
        left, right = st.columns([1, 1.05])

        with left:
            st.markdown('<div class="panel">', unsafe_allow_html=True)

            st.markdown("""
            <div class="panel-title">
                ♟ &nbsp; Age Distribution
            </div>
            <div class="panel-subtitle">
                Distribution of user ages.
            </div>
            """, unsafe_allow_html=True)

            if "Age" in df.columns:
                fig = px.histogram(
                    df.dropna(subset=["Age"]),
                    x="Age",
                    nbins=15,
                    color_discrete_sequence=["#10cdbb"]
                )

                fig.update_layout(
                    xaxis_title="Age",
                    yaxis_title="Count",
                    showlegend=False,
                    bargap=0.12
                )

                st.plotly_chart(
                    chart_style(fig, 280),
                    use_container_width=True,
                    config={"displayModeBar": False}
                )
            else:
                st.info("Age column is missing.")

            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="panel">', unsafe_allow_html=True)

            st.markdown("""
            <div class="panel-title">
                ⚡ &nbsp; Calories vs Duration
            </div>
            <div class="panel-subtitle">
                Relationship between exercise duration and calories burned.
            </div>
            """, unsafe_allow_html=True)

            if {"Duration", "Calories"}.issubset(df.columns):
                plot_df = df.dropna(
                    subset=["Duration", "Calories"]
                )

                fig = px.scatter(
                    plot_df,
                    x="Duration",
                    y="Calories",
                    opacity=0.60,
                    color_discrete_sequence=["#10cdbb"]
                )

                if (
                    len(plot_df) >= 3
                    and plot_df["Duration"].nunique() > 1
                ):
                    x = plot_df["Duration"].to_numpy(float)
                    y = plot_df["Calories"].to_numpy(float)

                    slope, intercept = np.polyfit(x, y, 1)
                    xs = np.linspace(x.min(), x.max(), 80)

                    fig.add_trace(go.Scatter(
                        x=xs,
                        y=slope * xs + intercept,
                        mode="lines",
                        line=dict(color="#12e1cc", width=3),
                        showlegend=False
                    ))

                fig.update_layout(
                    xaxis_title="Duration (minutes)",
                    yaxis_title="Calories",
                    showlegend=False
                )

                st.plotly_chart(
                    chart_style(fig, 280),
                    use_container_width=True,
                    config={"displayModeBar": False}
                )
            else:
                st.info(
                    "A Calories column is required for this chart."
                )

            st.markdown("</div>", unsafe_allow_html=True)



# ==========================================
# CALORIE PREDICTION
# ==========================================
elif page == "Calorie Prediction":

    st.subheader("⚡ Calorie Prediction")
    st.write("Enter your activity details to estimate calories burned.")

    if model is None:
        st.error(
            "XGBoost model not found. Put your model JSON file "
            "beside app.py."
        )

    else:
        with st.form("prediction_form"):

            c1, c2, c3 = st.columns(3)

            with c1:
                gender_label = st.selectbox(
                    "Gender", ["Male", "Female"]
                )
                age = st.slider("Age", 10, 100, 25)
                height = st.number_input(
                    "Height (cm)", 100.0, 230.0, 170.0
                )

            with c2:
                weight = st.number_input(
                    "Weight (kg)", 25.0, 250.0, 70.0
                )
                duration = st.slider(
                    "Duration (minutes)", 1, 180, 30
                )
                heart_rate = st.slider(
                    "Heart Rate (bpm)", 40, 220, 100
                )

            with c3:
                body_temp = st.number_input(
                    "Body Temperature (°C)",
                    35.0, 42.0, 37.0, step=0.1
                )

            submitted = st.form_submit_button(
                "⚡ Predict Calories",
                use_container_width=True
            )

        if submitted:
            gender = 0 if gender_label == "Male" else 1

            input_df = pd.DataFrame(
                [[
                    gender, age, height, weight,
                    duration, heart_rate, body_temp
                ]],
                columns=FEATURES
            )

            try:
                prediction = max(
                    0.0,
                    float(model.predict(input_df)[0])
                )

                st.markdown("### Prediction Result")

                metric_card(
                    "ESTIMATED CALORIES BURNED",
                    f"{prediction:.1f} kcal",
                    "XGBoost model estimate",
                    "⚡"
                )

                st.caption(
                    "This is a machine-learning estimate, "
                    "not a direct measurement."
                )

            except Exception as e:
                st.error(f"Prediction failed: {e}")




# ==========================================
# FOOTER
# ==========================================
st.divider()

st.markdown("""
<div style="
    text-align:center;
    color:#6f9cb4;
    font-size:10px;
    letter-spacing:2px;
    padding:8px;">
    CALORIEPRO · FITNESS INTELLIGENCE · XGBOOST REGRESSION
</div>
""", unsafe_allow_html=True)