"""
Week 3 - Advanced Data Analysis and Visualization in Logistics
Simulates the cleaned dataset from Week 2 (same mid-size e-commerce, 3-warehouse
scenario) and performs EDA + visualization, saving chart images and printing
summary statistics used directly in the Week 3 report.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
ORANGE = "#D2691E"
PALETTE = ["#D2691E", "#F4A460", "#8B4513", "#DEB887"]

rng = np.random.default_rng(42)
N = 6000

warehouses = rng.choice(["WH1", "WH2", "WH3"], size=N, p=[0.4, 0.35, 0.25])
days = rng.choice(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    size=N, p=[0.16, 0.15, 0.15, 0.15, 0.18, 0.12, 0.09]
)
distance_km = np.clip(rng.gamma(shape=4.0, scale=6.0, size=N), 1, 120)

# Warehouse load factor: WH2 is systematically busier -> more delay
load_factor = np.where(warehouses == "WH2", 1.35, np.where(warehouses == "WH3", 1.05, 1.0))
# Weekday factor: Fri/Sat afternoons are the peak-delay window described in Week 1
day_factor = pd.Series(days).map({
    "Monday": 1.0, "Tuesday": 1.0, "Wednesday": 1.05, "Thursday": 1.05,
    "Friday": 1.35, "Saturday": 1.25, "Sunday": 0.9
}).values

base_time = 4 + 0.35 * distance_km
noise = rng.normal(0, 2.5, size=N)
delivery_time_hrs = np.clip(base_time * load_factor * day_factor + noise, 1, None)
# Inject a small number of genuine extreme outliers (logging errors) - already capped in Week 2,
# here we show the pre-capped distribution for the "before" chart context.
promised_time_hrs = 8 + 0.3 * distance_km + rng.normal(0, 1.5, size=N)

order_volume = np.clip(rng.normal(12, 5, size=N), 1, None)
shipping_cost = 40 + 6.5 * distance_km + 1.8 * order_volume + rng.normal(0, 25, size=N)

df = pd.DataFrame({
    "warehouse_id": warehouses,
    "day_of_week": days,
    "distance_km": distance_km,
    "order_volume": order_volume,
    "delivery_time_hrs": delivery_time_hrs,
    "promised_time_hrs": promised_time_hrs,
    "shipping_cost": shipping_cost,
})
df["is_late"] = df["delivery_time_hrs"] > df["promised_time_hrs"]

day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ---------------- Summary statistics (printed for report) ----------------
print("=== Central tendency: delivery_time_hrs ===")
print(df["delivery_time_hrs"].agg(["mean", "median", "std", "min", "max"]))

print("\n=== Late rate by warehouse ===")
print(df.groupby("warehouse_id")["is_late"].mean().round(3))

print("\n=== Late rate by day of week ===")
print(df.groupby("day_of_week")["is_late"].mean().reindex(day_order).round(3))

print("\n=== Correlation matrix ===")
corr = df[["distance_km", "order_volume", "delivery_time_hrs", "shipping_cost"]].corr()
print(corr.round(2))

print("\n=== Avg cost & time by warehouse ===")
print(df.groupby("warehouse_id")[["shipping_cost", "delivery_time_hrs"]].mean().round(2))

# ---------------- Chart 1: Distribution of delivery time ----------------
plt.figure(figsize=(7, 4.2))
sns.histplot(df["delivery_time_hrs"], bins=40, color=ORANGE, kde=True)
plt.axvline(df["delivery_time_hrs"].mean(), color="#333333", linestyle="--", label=f"Mean = {df['delivery_time_hrs'].mean():.1f}h")
plt.axvline(df["delivery_time_hrs"].median(), color="#8B4513", linestyle=":", label=f"Median = {df['delivery_time_hrs'].median():.1f}h")
plt.title("Distribution of Delivery Time (hours)")
plt.xlabel("Delivery Time (hours)")
plt.ylabel("Number of Orders")
plt.legend()
plt.tight_layout()
plt.savefig("chart1_delivery_time_distribution.png", dpi=160)
plt.close()

# ---------------- Chart 2: Late delivery rate by day of week ----------------
late_by_day = df.groupby("day_of_week")["is_late"].mean().reindex(day_order)
plt.figure(figsize=(7, 4.2))
sns.barplot(x=late_by_day.index, y=late_by_day.values, color=ORANGE)
plt.title("Late Delivery Rate by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Late Delivery Rate")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("chart2_late_rate_by_day.png", dpi=160)
plt.close()

# ---------------- Chart 3: Correlation heatmap ----------------
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, cmap="Oranges", fmt=".2f", square=True, cbar_kws={"shrink": 0.8})
plt.title("Correlation Between Key Numeric Variables")
plt.tight_layout()
plt.savefig("chart3_correlation_heatmap.png", dpi=160)
plt.close()

# ---------------- Chart 4: Distance vs shipping cost ----------------
plt.figure(figsize=(7, 4.5))
sns.scatterplot(data=df.sample(1200, random_state=1), x="distance_km", y="shipping_cost",
                 hue="warehouse_id", palette=PALETTE[:3], alpha=0.6, s=25)
plt.title("Shipping Cost vs Distance by Warehouse")
plt.xlabel("Distance (km)")
plt.ylabel("Shipping Cost")
plt.tight_layout()
plt.savefig("chart4_cost_vs_distance.png", dpi=160)
plt.close()

# ---------------- Chart 5: Avg delivery time by warehouse ----------------
plt.figure(figsize=(6.5, 4.2))
wh_time = df.groupby("warehouse_id")["delivery_time_hrs"].mean().sort_values()
sns.barplot(x=wh_time.index, y=wh_time.values, palette=PALETTE[:3])
plt.title("Average Delivery Time by Warehouse")
plt.xlabel("Warehouse")
plt.ylabel("Average Delivery Time (hours)")
plt.tight_layout()
plt.savefig("chart5_avg_time_by_warehouse.png", dpi=160)
plt.close()

df.to_csv("orders_deliveries_week3_sample.csv", index=False)
print("\nCharts and sample dataset saved.")
