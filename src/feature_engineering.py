# ============================================
# FEATURE ENGINEERING SCRIPT
# ============================================

import pandas as pd
import numpy as np


# ============================================
# RUSH HOUR FUNCTION
# ============================================

def is_rush_hour(hour):

    morning_rush = 6 <= hour <= 9

    evening_rush = 15 <= hour <= 18

    return int(
        morning_rush or evening_rush
    )


# ============================================
# SEASON FUNCTION
# ============================================

def get_season(month):

    if month in [12, 1, 2]:
        return "winter"

    elif month in [3, 4, 5]:
        return "spring"

    elif month in [6, 7, 8]:
        return "summer"

    else:
        return "fall"


# ============================================
# MAIN FEATURE ENGINEERING
# ============================================

def engineer_features(df):

    df = df.copy()

    # ========================================
    # DATETIME FEATURES
    # ========================================

    if "date_time" in df.columns:

        df["date_time"] = pd.to_datetime(
            df["date_time"]
        )

        # Hour
        df["hour"] = (
            df["date_time"].dt.hour
        )

        # Weekday
        df["weekday"] = (
            df["date_time"]
            .dt.day_name()
        )

        # Month
        df["month"] = (
            df["date_time"].dt.month
        )


    # ========================================
    # WEEKEND FEATURE
    # ========================================

    if "weekday" in df.columns:

        df["is_weekend"] = (
            df["weekday"]
            .isin([
                "Saturday",
                "Sunday"
            ])
            .astype(int)
        )


    # ========================================
    # RUSH HOUR FEATURE
    # ========================================

    if "hour" in df.columns:

        df["rush_hour"] = (
            df["hour"]
            .apply(is_rush_hour)
        )


    # ========================================
    # SEASON FEATURE
    # ========================================

    if "month" in df.columns:

        df["season"] = (
            df["month"]
            .apply(get_season)
        )


    # ========================================
    # TEMPERATURE CONVERSION
    # ========================================

    # Convert Kelvin to Celsius
    # only if temp exists

    if "temp" in df.columns:

        df["temp_c"] = (
            df["temp"] - 273.15
        )


    # ========================================
    # CYCLICAL ENCODING
    # ========================================

    # Hour Encoding
    if "hour" in df.columns:

        df["hour_sin"] = np.sin(
            2 * np.pi * df["hour"] / 24
        )

        df["hour_cos"] = np.cos(
            2 * np.pi * df["hour"] / 24
        )


    # Month Encoding
    if "month" in df.columns:

        df["month_sin"] = np.sin(
            2 * np.pi * df["month"] / 12
        )

        df["month_cos"] = np.cos(
            2 * np.pi * df["month"] / 12
        )


    # ========================================
    # HOLIDAY FEATURE
    # ========================================

    if "holiday" in df.columns:

        df["is_holiday"] = (
            df["holiday"]
            .str.lower()
            .ne("noholiday")
            .astype(int)
        )


    # ========================================
    # TEXT CLEANING
    # ========================================

    categorical_cols = [

        "holiday",

        "weather_main",

        "weather_description",

        "season",

        "weekday"

    ]

    for col in categorical_cols:

        if col in df.columns:

            df[col] = (

                df[col]

                .astype(str)

                .str.lower()

                .str.strip()

            )


    return df