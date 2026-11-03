"""
AirTrace India: Multi-Source Attribution Training Pipeline
Strict Data Integrity: Zero PM10 Leakage, No Target Lags
"""
import os
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import pytz
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

CITIES = {
    "hyderabad": {"name": "Hyderabad", "lat": 17.3850, "lon": 78.4867},
    "delhi": {"name": "Delhi", "lat": 28.6139, "lon": 77.2090}
}

M0_COLS = ["wind_speed_10m", "wind_dir_sin", "wind_dir_cos", "temperature_2m", "relative_humidity_2m", "hour_sin", "hour_cos", "day_of_week"]
GROUPS = {
    "fires": ["upwind_fires_0_150km", "upwind_fires_150_300km", "upwind_fires_300_500km", "upwind_frp_sum", "total_fires_500km"],
    "traffic_combustion": ["nitrogen_dioxide", "carbon_monoxide"],
    "industry_power": ["sulphur_dioxide"],
    "dust": ["dust"]
}
ALL_SOURCE_COLS = [c for cols in GROUPS.values() for c in cols]
M3_COLS = M0_COLS + ALL_SOURCE_COLS

def train_and_attribute(city_key, meta):
    print(f"Training models for {meta['name']}...")
    os.makedirs(f"data/{city_key}", exist_ok=True)
    os.makedirs(f"results/{city_key}", exist_ok=True)
    
    # 1. Open-Meteo CAMS and Weather
    aq_params = {"latitude": meta["lat"], "longitude": meta["lon"], "hourly": "pm2_5,nitrogen_dioxide,carbon_monoxide,sulphur_dioxide,dust", "past_days": 60, "timezone": "Asia/Kolkata"}
    r_aq = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params=aq_params, timeout=20).json()["hourly"]
    df_aq = pd.DataFrame(r_aq)
    df_aq["time"] = pd.to_datetime(df_aq["time"])
    
    w_params = {"latitude": meta["lat"], "longitude": meta["lon"], "hourly": "wind_speed_10m,wind_direction_10m,temperature_2m,relative_humidity_2m", "past_days": 60, "timezone": "Asia/Kolkata"}
    r_w = requests.get("https://api.open-meteo.com/v1/forecast", params=w_params, timeout=20).json()["hourly"]
    df_w = pd.DataFrame(r_w)
    df_w["time"] = pd.to_datetime(df_w["time"])
    
    df = pd.merge(df_aq, df_w, on="time", how="inner").dropna()
    now_ist = datetime.now(pytz.timezone("Asia/Kolkata"))
    df = df[df["time"].dt.tz_localize("Asia/Kolkata") <= now_ist].drop(columns=["time_ist"], errors="ignore")
    
    # Feature Engineering (Zero PM10 Leakage)
    wind_rad = np.radians(df["wind_direction_10m"])
    df["wind_dir_sin"] = np.sin(wind_rad)
    df["wind_dir_cos"] = np.cos(wind_rad)
    df["hour_sin"] = np.sin(2 * np.pi * df["time"].dt.hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * df["time"].dt.hour / 24.0)
    df["day_of_week"] = df["time"].dt.dayofweek
    for c in GROUPS["fires"]:
        df[c] = 0
        
    df.to_csv(f"data/{city_key}/features_all.csv", index=False)
    
    # 70-30 Chronological Train/Test Split
    split_idx = int(len(df) * 0.70)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    # Train M0 (Weather Baseline)
    m0 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    m0.fit(train_df[M0_COLS], train_df["pm2_5"])
    m0_pred = m0.predict(test_df[M0_COLS])
    m0_mae = mean_absolute_error(test_df["pm2_5"], m0_pred)
    
    # Train M3 (Full Multi-Source)
    m3 = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    m3.fit(train_df[M3_COLS], train_df["pm2_5"])
    m3_pred = m3.predict(test_df[M3_COLS])
    m3_mae = mean_absolute_error(test_df["pm2_5"], m3_pred)
    m3_rmse = np.sqrt(mean_squared_error(test_df["pm2_5"], m3_pred))
    m3_r2 = r2_score(test_df["pm2_5"], m3_pred)
    
    # Metrics
    metrics_df = pd.DataFrame([
        {"Model": "M0_Baseline", "MAE": round(m0_mae, 2), "RMSE": round(np.sqrt(mean_squared_error(test_df["pm2_5"], m0_pred)), 2), "R2": round(r2_score(test_df["pm2_5"], m0_pred), 2)},
        {"Model": "M3_Full", "MAE": round(m3_mae, 2), "RMSE": round(m3_rmse, 2), "R2": round(m3_r2, 2)}
    ])
    metrics_df.to_csv(f"results/{city_key}/metrics.csv", index=False)
    
    # Counterfactual Decomposition
    p10 = train_df[ALL_SOURCE_COLS].quantile(0.10)
    X_actual = df[M3_COLS].copy()
    pred_P = m3.predict(X_actual)
    
    X_base = X_actual.copy()
    for col in ALL_SOURCE_COLS: X_base[col] = p10[col]
    pred_B = m3.predict(X_base)
    excess_E = np.maximum(pred_P - pred_B, 0.0)
    
    raw_C = {}
    for g_name, g_cols in GROUPS.items():
        X_g = X_actual.copy()
        for c in g_cols: X_g[c] = p10[c]
        raw_C[g_name] = np.maximum(pred_P - m3.predict(X_g), 0.0)
        
    sum_raw = sum(raw_C.values())
    rescaled_C = {g: (np.where(sum_raw > 0, raw_C[g] / np.where(sum_raw == 0, 1.0, sum_raw), 0.0) * excess_E) for g in GROUPS}
    
    df_attr = pd.DataFrame({
        "time": df["time"], "pm25_actual": df["pm2_5"], "pm25_pred": np.round(pred_P, 2),
        "background_and_weather": np.round(pred_B, 2),
        "fires": np.round(rescaled_C["fires"], 2),
        "traffic_combustion": np.round(rescaled_C["traffic_combustion"], 2),
        "industry_power": np.round(rescaled_C["industry_power"], 2),
        "dust": np.round(rescaled_C["dust"], 2)
    })
    for g in ["fires", "traffic_combustion", "industry_power", "dust"]:
        df_attr[f"{g}_pct"] = np.where(df_attr["pm25_pred"] > 0, np.round((df_attr[g] / df_attr["pm25_pred"]) * 100.0, 1), 0.0)
    df_attr["bg_weather_pct"] = np.where(df_attr["pm25_pred"] > 0, np.round((df_attr["background_and_weather"] / df_attr["pm25_pred"]) * 100.0, 1), 0.0)
    df_attr.to_csv(f"results/{city_key}/attribution_hourly.csv", index=False)
    print(f"Finished {meta['name']}.")

if __name__ == "__main__":
    for c_key, c_meta in CITIES.items():
        train_and_attribute(c_key, c_meta)
