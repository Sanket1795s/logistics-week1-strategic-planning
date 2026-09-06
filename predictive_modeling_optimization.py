"""
Week 4 - Predictive Modeling and Optimization in Logistics Systems
Uses the same mid-size e-commerce, 3-warehouse scenario as Weeks 1-3.
Trains and compares predictive models for delivery time, evaluates them,
tunes hyperparameters, and solves a small route-optimization example with PuLP.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

sns.set_theme(style="whitegrid")
ORANGE = "#D2691E"
PALETTE = ["#D2691E", "#F4A460", "#8B4513", "#DEB887"]

# ---------------- Simulate dataset (same generative structure as Week 3) ----------------
rng = np.random.default_rng(42)
N = 6000

warehouses = rng.choice(["WH1", "WH2", "WH3"], size=N, p=[0.4, 0.35, 0.25])
days = rng.choice(
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
    size=N, p=[0.16, 0.15, 0.15, 0.15, 0.18, 0.12, 0.09]
)
distance_km = np.clip(rng.gamma(shape=4.0, scale=6.0, size=N), 1, 120)
load_factor = np.where(warehouses == "WH2", 1.35, np.where(warehouses == "WH3", 1.05, 1.0))
day_factor = pd.Series(days).map({
    "Monday": 1.0, "Tuesday": 1.0, "Wednesday": 1.05, "Thursday": 1.05,
    "Friday": 1.35, "Saturday": 1.25, "Sunday": 0.9
}).values
base_time = 4 + 0.35 * distance_km
noise = rng.normal(0, 2.5, size=N)
delivery_time_hrs = np.clip(base_time * load_factor * day_factor + noise, 1, None)
order_volume = np.clip(rng.normal(12, 5, size=N), 1, None)
shipping_cost = 40 + 6.5 * distance_km + 1.8 * order_volume + rng.normal(0, 25, size=N)

df = pd.DataFrame({
    "warehouse_id": warehouses,
    "day_of_week": days,
    "distance_km": distance_km,
    "order_volume": order_volume,
    "shipping_cost": shipping_cost,
    "delivery_time_hrs": delivery_time_hrs,
})

target = "delivery_time_hrs"
numeric_features = ["distance_km", "order_volume", "shipping_cost"]
categorical_features = ["warehouse_id", "day_of_week"]

X = df[numeric_features + categorical_features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

preprocess = ColumnTransformer([
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
], remainder="passthrough")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
}

results = []
fitted_pipelines = {}
for name, model in models.items():
    pipe = Pipeline([("prep", preprocess), ("model", model)])
    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    cv_scores = cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_absolute_error")
    cv_mae = -cv_scores.mean()

    results.append({"model": name, "MAE": mae, "RMSE": rmse, "R2": r2, "CV_MAE": cv_mae})
    fitted_pipelines[name] = pipe
    print(f"{name}: MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}  CV_MAE={cv_mae:.3f}")

results_df = pd.DataFrame(results)
print("\n=== Model comparison ===")
print(results_df)

best_name = results_df.sort_values("MAE").iloc[0]["model"]
print(f"\nBest model by test MAE: {best_name}")

# ---------------- Hyperparameter tuning on Gradient Boosting ----------------
gb_pipe = Pipeline([("prep", preprocess), ("model", GradientBoostingRegressor(random_state=42))])
param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [2, 3, 4],
    "model__learning_rate": [0.05, 0.1],
}
grid = GridSearchCV(gb_pipe, param_grid, cv=3, scoring="neg_mean_absolute_error", n_jobs=-1)
grid.fit(X_train, y_train)
print("\n=== GridSearchCV best params (Gradient Boosting) ===")
print(grid.best_params_)
best_gb_preds = grid.best_estimator_.predict(X_test)
tuned_mae = mean_absolute_error(y_test, best_gb_preds)
tuned_rmse = np.sqrt(mean_squared_error(y_test, best_gb_preds))
tuned_r2 = r2_score(y_test, best_gb_preds)
print(f"Tuned Gradient Boosting: MAE={tuned_mae:.3f}  RMSE={tuned_rmse:.3f}  R2={tuned_r2:.3f}")

# ---------------- Feature importance (Random Forest) ----------------
rf_pipe = fitted_pipelines["Random Forest"]
ohe_cols = rf_pipe.named_steps["prep"].named_transformers_["cat"].get_feature_names_out(categorical_features)
all_cols = list(ohe_cols) + numeric_features
importances = rf_pipe.named_steps["model"].feature_importances_
fi = pd.Series(importances, index=all_cols).sort_values(ascending=False).head(8)
print("\n=== Top feature importances (Random Forest) ===")
print(fi.round(3))

# ---------------- Chart 1: Predicted vs Actual (best model) ----------------
best_pipe = fitted_pipelines[best_name]
best_preds = best_pipe.predict(X_test)

plt.figure(figsize=(6.2, 5.2))
plt.scatter(y_test, best_preds, alpha=0.35, color=ORANGE, s=18)
lims = [min(y_test.min(), best_preds.min()), max(y_test.max(), best_preds.max())]
plt.plot(lims, lims, "--", color="#333333", linewidth=1.5)
plt.xlabel("Actual Delivery Time (hours)")
plt.ylabel("Predicted Delivery Time (hours)")
plt.title(f"Predicted vs Actual Delivery Time — {best_name}")
plt.tight_layout()
plt.savefig("w4_chart1_pred_vs_actual.png", dpi=160)
plt.close()

# ---------------- Chart 2: Residuals ----------------
residuals = y_test.values - best_preds
plt.figure(figsize=(6.5, 4.2))
sns.histplot(residuals, bins=40, color=ORANGE, kde=True)
plt.axvline(0, color="#333333", linestyle="--")
plt.title(f"Residual Distribution — {best_name}")
plt.xlabel("Residual (Actual - Predicted, hours)")
plt.tight_layout()
plt.savefig("w4_chart2_residuals.png", dpi=160)
plt.close()

# ---------------- Chart 3: Feature importance ----------------
plt.figure(figsize=(7, 4.5))
sns.barplot(x=fi.values, y=fi.index, color=ORANGE)
plt.title("Top Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.tight_layout()
plt.savefig("w4_chart3_feature_importance.png", dpi=160)
plt.close()

# ---------------- Chart 4: Model comparison bar chart ----------------
plt.figure(figsize=(6.5, 4.2))
sns.barplot(x="model", y="MAE", data=results_df, color=ORANGE)
plt.title("Model Comparison — Test MAE (hours)")
plt.ylabel("MAE (hours, lower is better)")
plt.xlabel("")
plt.tight_layout()
plt.savefig("w4_chart4_model_comparison.png", dpi=160)
plt.close()

results_df.to_csv("w4_model_results.csv", index=False)

# ================= OPTIMIZATION: small route sequencing example with PuLP =================
import pulp

# 6 delivery stops from a single warehouse, with a symmetric distance matrix (km)
stops = ["Depot", "A", "B", "C", "D", "E"]
np.random.seed(7)
coords = {
    "Depot": (0, 0), "A": (4, 2), "B": (6, 6), "C": (2, 7), "D": (-3, 4), "E": (-2, -3)
}
def dist(p1, p2):
    (x1, y1), (x2, y2) = coords[p1], coords[p2]
    return round(((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5, 2)

pairs = [(i, j) for i in stops for j in stops if i != j]
d = {(i, j): dist(i, j) for i, j in pairs}

prob = pulp.LpProblem("Route_Optimization", pulp.LpMinimize)
x = pulp.LpVariable.dicts("x", pairs, cat="Binary")
u = pulp.LpVariable.dicts("u", stops, lowBound=0, upBound=len(stops), cat="Continuous")

prob += pulp.lpSum(d[i, j] * x[i, j] for i, j in pairs)

for k in stops:
    prob += pulp.lpSum(x[i, k] for i in stops if i != k) == 1
    prob += pulp.lpSum(x[k, j] for j in stops if j != k) == 1

n = len(stops)
for i in stops:
    if i == "Depot":
        continue
    for j in stops:
        if j == "Depot" or i == j:
            continue
        prob += u[i] - u[j] + n * x[i, j] <= n - 1

prob.solve(pulp.PULP_CBC_CMD(msg=0))

route_edges = [(i, j) for i, j in pairs if pulp.value(x[i, j]) == 1]
# reconstruct route order
route = ["Depot"]
current = "Depot"
for _ in range(len(stops) - 1):
    nxt = [j for (i, j) in route_edges if i == current][0]
    route.append(nxt)
    current = nxt
route.append("Depot")
total_distance = sum(d[route[k], route[k+1]] for k in range(len(route)-1))

print("\n=== Optimized route ===")
print(" -> ".join(route))
print(f"Total distance: {total_distance:.2f} km")

# naive (unoptimized) order for comparison — reflects a dispatcher sequencing
# stops in the order orders were received, ignoring geography
naive_route = ["C", "E", "A", "D", "B"]
naive_full = ["Depot"] + naive_route + ["Depot"]
naive_distance = sum(dist(naive_full[k], naive_full[k+1]) for k in range(len(naive_full)-1))
print(f"Naive sequential route distance: {naive_distance:.2f} km")
print(f"Distance saved: {naive_distance - total_distance:.2f} km ({(1 - total_distance/naive_distance)*100:.1f}% reduction)")

with open("w4_optimization_result.txt", "w") as f:
    f.write(f"Optimized route: {' -> '.join(route)}\n")
    f.write(f"Optimized distance: {total_distance:.2f} km\n")
    f.write(f"Naive route: {' -> '.join(naive_full)}\n")
    f.write(f"Naive distance: {naive_distance:.2f} km\n")
    f.write(f"Reduction: {(1 - total_distance/naive_distance)*100:.1f}%\n")

print("\nAll charts and results saved.")
