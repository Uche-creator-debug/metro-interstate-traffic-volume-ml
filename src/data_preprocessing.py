import numpy as np


def clean_data(df):

    # Remove duplicates
    df = df.drop_duplicates()

    # Fix holiday missing values
    df["holiday"] = df["holiday"].fillna("NoHoliday")

    # Standardize text columns
    categorical_cols = [
        "holiday",
        "weather_main",
        "weather_description"
    ]

    for col in categorical_cols:
        df[col] = (
            df[col]
            .str.lower()
            .str.strip()
        )

    # Remove impossible temperature values
    df = df[df["temp"] != 0]

    # Remove extreme rainfall outlier
    df = df[df["rain_1h"] < 1000]

    return df