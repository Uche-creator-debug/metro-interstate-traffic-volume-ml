# ============================================
# TRAFFIC VOLUME PREDICTION APP
# ============================================

# =========================
# IMPORTS
# =========================

import sys
import os

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

sys.path.append(BASE_DIR)

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

from src.feature_engineering import engineer_features


# ============================================
# PAGE CONFIG
# ============================================

st.set_page_config(
    page_title="Traffic Volume Prediction",
    page_icon="🚦",
    layout="wide"
)


# ============================================
# LOAD MODEL
# ============================================

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "xgb_model.joblib"
)

FEATURE_PATH = os.path.join(
    BASE_DIR,
    "model",
    "xgb_model_columns.joblib"
)

model = joblib.load(MODEL_PATH)

model_features = joblib.load(FEATURE_PATH)


# ============================================
# CUSTOM STYLING
# ============================================

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.block-container {
    padding-top: 2rem;
}

h1, h2, h3 {
    color: white;
}

.metric-container {
    background-color: #1E1E1E;
    padding: 1rem;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================
# TITLE
# ============================================

st.title("🚦 Interstate Traffic Volume Prediction")

st.divider()

with st.expander("About This Project"):

    st.write("""
    This project predicts interstate traffic volume
    using weather conditions and temporal traffic patterns.

    The model was trained using XGBoost
    and optimized with Optuna.
    """)

st.divider()

st.sidebar.header("Model Performance")

st.sidebar.write("R² Score: 0.94")

st.sidebar.write("RMSE: 478.28")

st.divider()
# ============================================
# SIDEBAR INPUTS
# ============================================

st.sidebar.header("Input Features")


# -------------------------
# Time Inputs
# -------------------------

hour = st.sidebar.slider(
    "Hour of Day",
    0,
    23,
    12
)

month = st.sidebar.slider(
    "Month",
    1,
    12,
    6
)

weekday = st.sidebar.selectbox(
    "Weekday",
    [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday"
    ]
)


# -------------------------
# Weather Inputs
# -------------------------

temp_c = st.sidebar.slider(
    "Temperature (°C)",
    -30.0,
    45.0,
    20.0
)

rain_1h = st.sidebar.slider(
    "Rainfall in Last Hour (mm)",
    0.0,
    50.0,
    0.0
)

clouds_all = st.sidebar.slider(
    "Cloud Coverage (%)",
    0,
    100,
    40
)

weather_main = st.sidebar.selectbox(
    "Weather Condition",
    [
        "Clear",
        "Clouds",
        "Rain",
        "Snow",
        "Mist",
        "Fog",
        "Haze"
    ]
)


# -------------------------
# Holiday Input
# -------------------------

is_holiday = st.sidebar.selectbox(
    "Holiday",
    ["No", "Yes"]
)


# ============================================
# CREATE INPUT DATAFRAME
# ============================================

input_df = pd.DataFrame({

    "holiday": [
        "holiday" if is_holiday == "Yes"
        else "noholiday"
    ],

    "temp_c": [temp_c],

    "rain_1h": [rain_1h],

    "snow_1h": [0.0],

    "clouds_all": [clouds_all],

    "weather_main": [
        weather_main.lower()
    ],

    "weather_description": [
        weather_main.lower()
    ],

    "date_time": [
        pd.Timestamp(
            year=2024,
            month=month,
            day=15,
            hour=hour
        )
    ]

})

st.sidebar.markdown(
    "[GitHub Repository](https://github.com/Uche-creator-debug/metro-interstate-traffic-volume-ml.git)"
)


# ============================================
# FEATURE ENGINEERING
# ============================================

input_df = engineer_features(input_df)


# ============================================
# DROP REDUNDANT COLUMNS
# ============================================

columns_to_drop = [

    "date_time",

    "hour",

    "month"

]

existing_cols = [
    col for col in columns_to_drop
    if col in input_df.columns
]

input_df = input_df.drop(
    columns=existing_cols
)


# ============================================
# ENCODING
# ============================================

input_df = pd.get_dummies(
    input_df,
    drop_first=True
)


# ============================================
# ALIGN FEATURES
# ============================================

input_df = input_df.reindex(
    columns=model_features,
    fill_value=0
)


# ============================================
# PREDICTION
# ============================================

prediction = model.predict(input_df)[0]


# ============================================
# TRAFFIC CATEGORY
# ============================================

def traffic_category(prediction):

    if prediction < 2000:
        return "🟢 Low Traffic"

    elif prediction < 5000:
        return "🟡 Moderate Traffic"

    else:
        return "🔴 Heavy Traffic"


traffic_status = traffic_category(prediction)


# ============================================
# MAIN DASHBOARD
# ============================================

col1, col2 = st.columns([1, 1])


# -------------------------
# Prediction Display
# -------------------------

with col1:

    st.subheader("Predicted Traffic Volume")

    st.metric(
        label="Estimated Vehicles",
        value=f"{prediction:,.0f}"
    )

    st.markdown(f"### {traffic_status}")


# -------------------------
# Gauge Chart
# -------------------------

with col2:

    gauge = go.Figure(go.Indicator(

        mode="gauge+number",

        value=prediction,

        title={
            "text": "Traffic Intensity"
        },

        gauge={

                "axis": {

                    "range": [0, 7000],

                    "tickmode": "array",

                    "tickvals": [0, 2000, 4000, 6000],

                    "ticktext": ["0", "2000", "4000", "6000"]

                },

                "bar": {
                    "color": "darkblue"
                },

                "steps": [

                    {
                        "range": [0, 2000],
                        "color": "lightgreen"
                    },

                    {
                        "range": [2000, 5000],
                        "color": "gold"
                    },

                    {
                        "range": [5000, 7000],
                        "color": "salmon"
                    }

                ]

            }
    ))

    st.plotly_chart(
        gauge,
        use_container_width=True
    )


st.divider()


# ============================================
# FEATURE SUMMARY
# ============================================

st.subheader("Input Summary")


summary_df = pd.DataFrame({

    "Feature": [
        "Hour",
        "Month",
        "Weekday",
        "Temperature",
        "Rainfall",
        "Cloud Coverage",
        "Weather",
        "Holiday"
    ],

    "Value": [
        hour,
        month,
        weekday,
        f"{temp_c} °C",
        f"{rain_1h} mm",
        f"{clouds_all} %",
        weather_main,
        is_holiday
    ]

})

st.dataframe(
    summary_df,
    use_container_width=True
)


# ============================================
# INTERPRETATION SECTION
# ============================================

st.subheader("Prediction Interpretation")


if prediction < 2000:

    st.info("""
    Traffic conditions are expected to remain relatively light.
    
    Roads are likely to experience smooth traffic flow with minimal congestion.
    """)

elif prediction < 5000:

    st.warning("""
    Moderate traffic conditions are expected.
    
    Some congestion may occur depending on road activity and commuting patterns.
    """)

else:

    st.error("""
    Heavy traffic conditions are expected.
    
    Significant congestion and slower movement may occur during this period.
    """)


# ============================================
# FOOTER
# ============================================

st.divider()

st.caption("""
Developed using:
- Streamlit
- XGBoost
- SHAP Explainability
- Feature Engineering & Time-Series Modeling
""")