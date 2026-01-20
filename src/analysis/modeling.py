######################## Libraries ##########################
# import libraries
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    mean_squared_error,
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
)

######################## Global Reproducibility ##########################
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

##################### Loading and Reformatting Datasets ###################
# paths to datasets
DATA_PATH = "/Users/kimmyempringham/Downloads/311/311_data_merged.csv"
DISTRICTS_PATH = "/Users/kimmyempringham/Downloads/Council_District.geojson"

######################## Helper Functions ##########################
def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

# safely calculates a percentage while avoiding division by zero or bad data by returning NaN
def safe_percent(numer, denom):
    denom = pd.to_numeric(denom, errors="coerce")
    numer = pd.to_numeric(numer, errors="coerce")
    return np.where(denom > 0, (numer / denom) * 100, np.nan)

# converts categorical columns into numeric dummy variables and splits X/y
def encode_features(df_in: pd.DataFrame, categorical_cols: list, target_col: str):
    df_work = df_in.copy()
    df_encoded = pd.get_dummies(df_work, columns=categorical_cols, drop_first=True)
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]
    return X, y

######################## Load & Clean ##########################
# read in 311 dataset
df = pd.read_csv(DATA_PATH)

# parse dates and drop rows with missing request date
df["reformat_request_date"] = pd.to_datetime(df["reformat_request_date"], errors="coerce")
df = df.dropna(subset=["reformat_request_date"])

# make sure resolution hours is numeric and filter impossible/extreme values
df["resolution_hours"] = pd.to_numeric(df["resolution_hours"], errors="coerce")
df = df[df["resolution_hours"].between(0, 24 * 30)]  # <= 30 days

# feature engineering (time + demographics)
df["day_of_week"] = df["reformat_request_date"].dt.day_name()
df["month_of_year"] = df["reformat_request_date"].dt.month
df["year"] = df["reformat_request_date"].dt.year

# convert demographic features to percentage of population
df["white_perc"] = safe_percent(df["white"], df["race_total"])
df["black_perc"] = safe_percent(df["black"], df["race_total"])
df["asian_perc"] = safe_percent(df["asian"], df["race_total"])
df["aian_perc"] = safe_percent(df["aian"], df["race_total"])
df["nhpi_perc"] = safe_percent(df["nhpi"], df["race_total"])
df["perc_pop_25plus"] = safe_percent(df["pop_25plus"], df["pop_total"])

# target variables
df["log_reso_time"] = np.log1p(df["resolution_hours"])
df["under_72"] = pd.to_numeric(df["under_72"], errors="coerce")
df = df[df["under_72"].isin([0, 1])]  # keep binary only
df["under_72"] = df["under_72"].astype(int)

######################## Spatial Subset (District Assignment) ##########################
# keep rows with coordinates
df_geo = df.dropna(subset=["Latitude", "Longitude"]).copy()

# load district polygons
districts = gpd.read_file(DISTRICTS_PATH).to_crs("EPSG:4326")
districts["geometry"] = districts.geometry.make_valid()

# create point geometry from lat/long
gdf_points = gpd.GeoDataFrame(
    df_geo,
    geometry=gpd.points_from_xy(df_geo["Longitude"], df_geo["Latitude"]),
    crs="EPSG:4326"
)

# spatial join to assign district
joined = gpd.sjoin(
    gdf_points,
    districts[["DISTRICTINT", "geometry"]],
    how="left",
    predicate="within"
)

df_geo = pd.DataFrame(joined.drop(columns="geometry"))

# keep only rows that successfully got a district
df_geo = df_geo.dropna(subset=["DISTRICTINT"])

# convert IDs to categorical strings
df_geo["DISTRICTINT"] = df_geo["DISTRICTINT"].astype(str)
df_geo["zcta"] = df_geo["zcta"].astype(str)

######################## Feature Sets ##########################
# reduced feature set (more stable for linear/logistic)
features_reduced = [
    "poverty_below",
    "Service Type",
    "day_of_week",
    "month_of_year",
    "pop_total",
    "white_perc",
    "black_perc",
    "asian_perc",
    "owner_occupied",
    "TMAX",
    "PRCP",
    "perc_pop_25plus",
    "ba_degree",
    "DISTRICTINT",
]

# full feature set (ok for ridge/RF)
features_full = features_reduced + [
    "per_capita_income",
    "median_rent",
    "median_home_value",
    "AWND",
    "TMIN",
    "zcta",
]

cat_reduced = ["Service Type", "day_of_week", "month_of_year", "DISTRICTINT"]
cat_full = cat_reduced + ["zcta"]

######################## Time-Based Split ##########################
train_geo = df_geo[df_geo["year"].between(2019, 2023)].copy()
test_geo = df_geo[df_geo["year"] == 2024].copy()

print("Spatial subset rows (has district):")
print("  Train (2019–2023):", len(train_geo))
print("  Test (2024):", len(test_geo))

######################## Model A: Linear Regression (Spatial) ##########################
lin_train = train_geo[features_reduced + ["log_reso_time"]].dropna()
lin_test = test_geo[features_reduced + ["log_reso_time"]].dropna()

X_train_lin, y_train_lin = encode_features(lin_train, cat_reduced, "log_reso_time")
X_test_lin, y_test_lin = encode_features(lin_test, cat_reduced, "log_reso_time")
X_test_lin = X_test_lin.reindex(columns=X_train_lin.columns, fill_value=0)

lin_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])

lin_model.fit(X_train_lin, y_train_lin)
pred_lin = lin_model.predict(X_test_lin)

print("\nModel A (Time split): Linear Regression on log(resolution_hours) [spatial subset]")
print("  R^2:", lin_model.score(X_test_lin, y_test_lin))
print("  RMSE:", rmse(y_test_lin, pred_lin))

######################## Model B: Ridge Regression (Spatial) ##########################
ridge_train = train_geo[features_full + ["log_reso_time"]].dropna()
ridge_test = test_geo[features_full + ["log_reso_time"]].dropna()

X_train_ridge, y_train_ridge = encode_features(ridge_train, cat_full, "log_reso_time")
X_test_ridge, y_test_ridge = encode_features(ridge_test, cat_full, "log_reso_time")
X_test_ridge = X_test_ridge.reindex(columns=X_train_ridge.columns, fill_value=0)

ridge_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0))
])

ridge_model.fit(X_train_ridge, y_train_ridge)
pred_ridge = ridge_model.predict(X_test_ridge)

print("\nModel B (Time split): Ridge Regression on log(resolution_hours) [spatial subset]")
print("  R^2:", ridge_model.score(X_test_ridge, y_test_ridge))
print("  RMSE:", rmse(y_test_ridge, pred_ridge))

######################## Model C: Logistic Regression (Spatial) ##########################
log_train = train_geo[features_reduced + ["under_72"]].dropna()
log_test = test_geo[features_reduced + ["under_72"]].dropna()

X_train_c, y_train_c = encode_features(log_train, cat_reduced, "under_72")
X_test_c, y_test_c = encode_features(log_test, cat_reduced, "under_72")
X_test_c = X_test_c.reindex(columns=X_train_c.columns, fill_value=0)

log_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(solver="liblinear", max_iter=2000))
])

log_model.fit(X_train_c, y_train_c)
pred_c = log_model.predict(X_test_c)
proba_c = log_model.predict_proba(X_test_c)[:, 1]

print("\nModel C (Time split): Logistic Regression for under_72 [spatial subset]")
print("  Base rate (under_72):", f"{y_test_c.mean() * 100:.2f}%  |  N={len(y_test_c):,}")
print("  Accuracy:", accuracy_score(y_test_c, pred_c))
print("  ROC-AUC:", roc_auc_score(y_test_c, proba_c))
print("  PR-AUC:", average_precision_score(y_test_c, proba_c))
print("  F1:", f1_score(y_test_c, pred_c))

######################## Model D: Random Forest (Spatial) ##########################
rf_train = train_geo[features_full + ["under_72"]].dropna()
rf_test = test_geo[features_full + ["under_72"]].dropna()

X_train_d, y_train_d = encode_features(rf_train, cat_full, "under_72")
X_test_d, y_test_d = encode_features(rf_test, cat_full, "under_72")
X_test_d = X_test_d.reindex(columns=X_train_d.columns, fill_value=0)

rf_model = RandomForestClassifier(
    n_estimators=300,
    random_state=SEED,
    n_jobs=-1
)

rf_model.fit(X_train_d, y_train_d)
pred_d = rf_model.predict(X_test_d)
proba_d = rf_model.predict_proba(X_test_d)[:, 1]

print("\nModel D (Time split): Random Forest for under_72 [spatial subset]")
print("  Base rate (under_72):", f"{y_test_d.mean() * 100:.2f}%  |  N={len(y_test_d):,}")
print("  Accuracy:", accuracy_score(y_test_d, pred_d))
print("  ROC-AUC:", roc_auc_score(y_test_d, proba_d))
print("  PR-AUC:", average_precision_score(y_test_d, proba_d))
print("  F1:", f1_score(y_test_d, pred_d))

######################## Model 5: Baseline Logistic (No Location) ##########################
# select model to visualize (deployable to all requests, no geo required)
features_nospatial = [
    "Service Type",
    "day_of_week",
    "month_of_year",
    "TMAX",
    "TMIN",
    "PRCP",
    "AWND",
]
cat_nospatial = ["Service Type", "day_of_week", "month_of_year"]

train_all = df[df["year"].between(2019, 2023)].copy()
test_all = df[df["year"] == 2024].copy()

base_train = train_all[features_nospatial + ["under_72"]].dropna()
base_test = test_all[features_nospatial + ["under_72"]].dropna()

X_train_b, y_train_b = encode_features(base_train, cat_nospatial, "under_72")
X_test_b, y_test_b = encode_features(base_test, cat_nospatial, "under_72")
X_test_b = X_test_b.reindex(columns=X_train_b.columns, fill_value=0)

base_model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(solver="liblinear", max_iter=2000))
])

base_model.fit(X_train_b, y_train_b)
proba_b = base_model.predict_proba(X_test_b)[:, 1]

# threshold-based decision rule (chosen operating point)
final_threshold = 0.60
pred_b = (proba_b >= final_threshold).astype(int)

print("\nModel 5 (Time split): Baseline Logistic Regression for under_72 [no location]")
print("  Base rate (under_72):", f"{y_test_b.mean() * 100:.2f}%  |  N={len(y_test_b):,}")
print("  ROC-AUC:", roc_auc_score(y_test_b, proba_b))
print("  PR-AUC:", average_precision_score(y_test_b, proba_b))
print("  F1 (threshold=0.60):", f1_score(y_test_b, pred_b))
print("  Accuracy (threshold=0.60):", accuracy_score(y_test_b, pred_b))

######################## Confusion Matrix (Model 5 @ threshold) ##########################
cm_final = confusion_matrix(y_test_b, pred_b)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_final,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Miss SLA (>72h)", "Meet SLA (≤72h)"],
    yticklabels=["Miss SLA (>72h)", "Meet SLA (≤72h)"]
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"Confusion Matrix (Model 5, Threshold = {final_threshold})")
plt.tight_layout()
plt.show()

######################## ROC Curve (Model 5) ##########################
fpr, tpr, _ = roc_curve(y_test_b, proba_b)
auc = roc_auc_score(y_test_b, proba_b)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve – Model 5 Baseline Logistic (No Location)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

