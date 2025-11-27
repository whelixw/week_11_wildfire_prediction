#load datasets
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from datetime import date

df_cities = pd.read_csv('data/cities.csv')
df_fires = pd.read_csv('data/wildfires_fires_table.csv')
df_us_weather = pd.read_parquet('data/us_daily_weather_1992_2015.parquet')





# Keep only US stations
df_cities_us = df_cities[df_cities["country"] == "United States of America"].copy()

lat_col = "latitude"    # adjust if different
lon_col = "longitude"   # adjust if different

coords = df_cities_us[[lat_col, lon_col]].to_numpy()

n_regions = 20  # try 10, 20, etc.
kmeans = KMeans(
    n_clusters=n_regions,
    random_state=42,
    n_init="auto",
)
df_cities_us["region"] = kmeans.fit_predict(coords)

# Keep only needed columns to merge later
station_regions = df_cities_us[["station_id", "region"]].copy()


df_us_weather = df_us_weather.merge(
    station_regions,
    on="station_id",
    how="inner",  # drop stations not in df_cities_us
)

#set weather regions

# Ensure consistent datetime
df_us_weather["date"] = pd.to_datetime(df_us_weather["date"])

# Basic region–day aggregates (keep it small at first)
agg_dict = {
    "avg_temp_c": "mean",
    "min_temp_c": "mean",
    "max_temp_c": "mean",
    "precipitation_mm": "sum",      # sum over stations in region
    "avg_wind_speed_kmh": "mean",
    "avg_sea_level_pres_hpa": "mean",
}

weather_region_day = (
    df_us_weather.groupby(["region", "date"])
    .agg(agg_dict)
    .reset_index()
)

# Optional: change dtypes to save memory
float_cols = [c for c in weather_region_day.columns if weather_region_day[c].dtype == "float64"]
weather_region_day[float_cols] = weather_region_day[float_cols].astype("float32")


# Add rolling features
weather_region_day = weather_region_day.sort_values(["region", "date"])
grouped = weather_region_day.groupby("region", group_keys=False)

# 7-day rolling precipitation sum
weather_region_day["prcp_7d_sum"] = (
    grouped["precipitation_mm"]
    .rolling(7, min_periods=1)
    .sum()
    .reset_index(level=0, drop=True)
)

# 14-day rolling mean max_temp
weather_region_day["tmax_14d_mean"] = (
    grouped["max_temp_c"]
    .rolling(14, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)

# Day-of-year (for seasonality)
weather_region_day["doy"] = weather_region_day["date"].dt.dayofyear


weather_region_day["sin_doy"] = np.sin(2 * np.pi * weather_region_day["doy"] / 365.0)
weather_region_day["cos_doy"] = np.cos(2 * np.pi * weather_region_day["doy"] / 365.0)


#assign regions to fires

# Clean up fire dates & range
df_fires = df_fires.dropna(subset=["DISCOVERY_DATE", "LATITUDE", "LONGITUDE"]).copy()

df_fires["DISCOVERY_DATE"] = pd.to_datetime(df_fires["DISCOVERY_DATE"])

df_fires = df_fires[
    (df_fires["DISCOVERY_DATE"] >= "1992-01-01")
    & (df_fires["DISCOVERY_DATE"] <= "2015-12-31")
]

# Compute region for each fire using previously fitted kmeans
fire_coords = df_fires[["LATITUDE", "LONGITUDE"]].to_numpy()
df_fires["region"] = kmeans.predict(fire_coords)

# Region–day fire count and binary label
fires_region_day = (
    df_fires.groupby(["region", "DISCOVERY_DATE"])
    .size()
    .reset_index(name="fire_count")
)
fires_region_day["fire_occurred"] = (fires_region_day["fire_count"] > 0).astype(int)

#merge weather and fire data

data = weather_region_day.merge(
    fires_region_day[["region", "DISCOVERY_DATE", "fire_occurred"]],
    left_on=["region", "date"],
    right_on=["region", "DISCOVERY_DATE"],
    how="left",
)

data["fire_occurred"] = data["fire_occurred"].fillna(0).astype(int)

# Drop redundant column
data = data.drop(columns=["DISCOVERY_DATE"])

#train split

train = data[data["date"] < "2011-01-01"].copy()
val = data[(data["date"] >= "2011-01-01") & (data["date"] < "2014-01-01")].copy()
test = data[data["date"] >= "2014-01-01"].copy()

feature_cols = [
    "avg_temp_c",
    "min_temp_c",
    "max_temp_c",
    "precipitation_mm",
    "avg_wind_speed_kmh",
    "avg_sea_level_pres_hpa",
    "prcp_7d_sum",
    "tmax_14d_mean",
    "sin_doy",
    "cos_doy",
]

X_train, y_train = train[feature_cols], train["fire_occurred"]
X_val, y_val = val[feature_cols], val["fire_occurred"]
X_test, y_test = test[feature_cols], test["fire_occurred"]


from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score

clf = HistGradientBoostingClassifier(
    max_depth=6,
    learning_rate=0.1,
    max_iter=200,
    class_weight="balanced",
    random_state=42,
)

clf.fit(X_train, y_train)

probs_val = clf.predict_proba(X_val)[:, 1]
probs_test = clf.predict_proba(X_test)[:, 1]

print("Val ROC-AUC:", roc_auc_score(y_val, probs_val))
print("Val PR-AUC:", average_precision_score(y_val, probs_val))
print("Test ROC-AUC:", roc_auc_score(y_test, probs_test))
print("Test PR-AUC:", average_precision_score(y_test, probs_test))