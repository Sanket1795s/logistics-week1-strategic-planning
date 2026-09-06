"""
Week 2 - Data Collection, Cleaning, and Preprocessing for Logistics Analysis
Logistics Data Analyst Internship

This script implements the cleaning pipeline described in the Week 2 report:
profiling, missing-value handling, deduplication, outlier capping (IQR),
format standardisation, and Min-Max normalisation.

Input: orders_deliveries_raw.csv (placeholder; replace with your dataset,
structured similarly to the DataCo Smart Supply Chain schema referenced
in the report).
Output: orders_deliveries_clean.csv
"""

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def profile_data(df):
    """Print a basic data-quality profile before cleaning."""
    print("--- Shape ---")
    print(df.shape)
    print("\n--- Dtypes ---")
    print(df.dtypes)
    print("\n--- Missing values per column ---")
    print(df.isnull().sum())
    print("\n--- Numeric summary ---")
    print(df.describe())


def handle_missing_values(df):
    """Impute or drop missing values depending on field criticality."""
    df["delivery_time_hrs"] = df["delivery_time_hrs"].fillna(
        df["delivery_time_hrs"].median()
    )

    if "route_id" in df.columns and "scan_sequence" in df.columns:
        df = df.sort_values(["route_id", "scan_sequence"])
        if "gps_timestamp" in df.columns:
            df["gps_timestamp"] = df.groupby("route_id")["gps_timestamp"].ffill()

    df = df.dropna(subset=["order_id", "warehouse_id"])
    return df


def remove_duplicates(df):
    """Keep only the most recently updated record per order_id."""
    if "last_updated" in df.columns:
        df = df.sort_values("last_updated", ascending=False)
    return df.drop_duplicates(subset="order_id", keep="first")


def cap_outliers_iqr(df, column="delivery_time_hrs"):
    """Cap outliers in a numeric column using the IQR method."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
    return df


def standardise_formats(df):
    """Unify date formats and categorical warehouse codes."""
    if "order_date" in df.columns:
        df["order_date"] = pd.to_datetime(df["order_date"], errors="coerce", dayfirst=True)

    if "warehouse_id" in df.columns:
        df["warehouse_id"] = (
            df["warehouse_id"]
            .astype(str)
            .str.upper()
            .str.replace(" ", "", regex=False)
            .str.replace("WH-", "WH", regex=False)
        )
    return df


def normalise_features(df, columns=("distance_km", "order_volume", "shipping_cost")):
    """Scale numeric features to a common [0, 1] range."""
    cols = [c for c in columns if c in df.columns]
    if cols:
        scaler = MinMaxScaler()
        df[cols] = scaler.fit_transform(df[cols])
    return df


def validate(df):
    """Re-check the dataset after cleaning."""
    assert df["order_id"].is_unique, "Duplicate order_id values remain."
    print("Remaining nulls:\n", df.isnull().sum())
    if "delivery_time_hrs" in df.columns:
        print(
            "Delivery time range after capping:",
            df["delivery_time_hrs"].min(),
            "-",
            df["delivery_time_hrs"].max(),
        )


def run_pipeline(input_path="orders_deliveries_raw.csv", output_path="orders_deliveries_clean.csv"):
    df = pd.read_csv(input_path)
    profile_data(df)

    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = cap_outliers_iqr(df)
    df = standardise_formats(df)
    df = normalise_features(df)

    validate(df)
    df.to_csv(output_path, index=False)
    print(f"\nClean dataset written to {output_path}")
    return df


if __name__ == "__main__":
    run_pipeline()
