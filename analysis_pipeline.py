"""
Week 1 - Strategic Planning and Data Exploration in Logistics
Logistics Data Analyst Internship

This script illustrates the proposed end-to-end analytical approach described
in the Week 1 strategic planning report: data loading & cleaning, exploratory
analysis, a delivery-time prediction model, and route clustering.

Note: This is an illustrative / pseudocode-style pipeline intended to
demonstrate approach and structure, using placeholder CSV inputs. Replace the
file paths and column names with your actual dataset in later weeks.
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.cluster import KMeans


def load_and_clean_data(orders_path="orders.csv", deliveries_path="deliveries.csv"):
    """Load raw order and delivery data, merge, and handle missing/duplicate records."""
    orders = pd.read_csv(orders_path)
    deliveries = pd.read_csv(deliveries_path)

    df = orders.merge(deliveries, on="order_id", how="left")
    df = df.drop_duplicates(subset="order_id")

    df["delivery_time_hrs"] = df["delivery_time_hrs"].fillna(df["delivery_time_hrs"].median())
    df = df.dropna(subset=["warehouse_id", "destination_zip"])
    return df


def explore_data(df):
    """Produce a basic exploratory chart: late-delivery rate by day of week."""
    df["is_late"] = df["delivery_time_hrs"] > df["promised_time_hrs"]
    late_by_day = df.groupby("day_of_week")["is_late"].mean()

    sns.barplot(x=late_by_day.index, y=late_by_day.values)
    plt.title("Late Delivery Rate by Day of Week")
    plt.ylabel("Late Delivery Rate")
    plt.xlabel("Day of Week")
    plt.tight_layout()
    plt.savefig("late_delivery_rate_by_day.png")
    plt.close()
    return late_by_day


def train_delivery_time_model(df, features):
    """Train a regression model to predict delivery time in hours."""
    X_train, X_test, y_train, y_test = train_test_split(
        df[features], df["delivery_time_hrs"], test_size=0.2, random_state=42
    )

    model = GradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Delivery Time Prediction MAE: {mae:.2f} hours")
    return model, mae


def cluster_routes(df, route_features, n_clusters=4):
    """Segment delivery routes into clusters based on cost/time characteristics."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["route_cluster"] = kmeans.fit_predict(df[route_features])
    return df, kmeans


if __name__ == "__main__":
    # Example usage (requires orders.csv and deliveries.csv with the expected columns)
    df = load_and_clean_data()
    explore_data(df)

    features = ["distance_km", "order_volume", "day_of_week_num", "warehouse_load"]
    model, mae = train_delivery_time_model(df, features)

    route_features = ["avg_distance_km", "avg_delivery_time_hrs", "avg_cost_per_order"]
    df, kmeans_model = cluster_routes(df, route_features)

    print("Pipeline complete. See late_delivery_rate_by_day.png for the EDA chart.")
