import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="AirTrace India", page_icon="🌬️", layout="wide")

# Map degrees to 16-point compass
def deg_to_compass(d):
    val = int((d / 22.5) + 0.5)
    arr = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return arr[(val % 16)]

# CPCB Approximate PM2.5 bands
def get_cpcb_category(val):
    if val <= 30:
        return "Good", "#28a745"
    elif val <= 60:
        return "Satisfactory", "#70b043"
    elif val <= 90:
        return "Moderate", "#e5a700"
    elif val <= 120:
        return "Poor", "#e67e22"
    elif val <= 250:
        return "Very Poor", "#d9534f"
    else:
        return "Severe", "#8e44ad"

# 1. Header
st.title("🌬️ AirTrace India: Urban Source Attribution")
st.markdown("**Hourly PM2.5 Source Group Estimation for Indian Metros** | *Neural Storm 2026 (JNTUH) GP-01*")

# City Selector (Hyderabad default)
CITY_CONFIG = {
    "Hyderabad (Hero City)": "hyderabad",
    "Delhi": "delhi"
}
selected_label = st.selectbox("Select Metro Area:", list(CITY_CONFIG.keys()), index=0)
city = CITY_CONFIG[selected_label]

# Load cached data
@st.cache_data
def load_data(city_name):
    attr_path = f"results/{city_name}/attribution_hourly.csv"
    feat_path = f"data/{city_name}/features_all.csv"
    met_path = f"results/{city_name}/metrics.csv"
    
    df_attr = pd.read_csv(attr_path) if os.path.exists(attr_path) else None
    df_feat = pd.read_csv(feat_path) if os.path.exists(feat_path) else None
    df_met = pd.read_csv(met_path) if os.path.exists(met_path) else None
    
    if df_attr is not None:
        df_attr["time"] = pd.to_datetime(df_attr["time"])
    if df_feat is not None:
        df_feat["time"] = pd.to_datetime(df_feat["time"])
    return df_attr, df_feat, df_met

df_attr, df_feat, df_met = load_data(city)

if df_attr is None or df_feat is None:
    st.error(f"Data files missing for {city}. Please ensure Stages S2-S6 are complete.")
    st.stop()

# 2. Right Now Cards
latest_attr = df_attr.iloc[-1]
latest_feat = df_feat.iloc[-1]

cat_label, cat_color = get_cpcb_category(latest_attr["pm25_actual"])
wind_dir = deg_to_compass(latest_feat["wind_direction_10m"])

# Determine highest non-background source
source_shares = {
    "Traffic & Combustion": latest_attr["traffic_combustion_pct"],
    "Industry & Power": latest_attr["industry_power_pct"],
    "Dust": latest_attr["dust_pct"],
    "Fires": latest_attr["fires_pct"],
}
top_source = max(source_shares, key=source_shares.get)
top_share = source_shares[top_source]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Latest PM2.5", f"{latest_attr['pm25_actual']:.1f} µg/m³")
col2.markdown(f"**Category (Approx CPCB)**<br><span style='color:{cat_color}; font-size:1.4rem; font-weight:bold;'>{cat_label}</span>", unsafe_allow_html=True)
col3.metric("Wind Vector", f"{wind_dir} ({int(latest_feat['wind_direction_10m'])}°)")
col4.metric("Active Fires (24h <500km)", int(latest_feat["total_fires_500km"]))
col5.metric("Top Source Group", f"{top_source} ({top_share:.1f}%)")

# 3. Dynamic Narrative Sentence
st.info(
    f"**Indicative Assessment:** In **{city.capitalize()}** at the latest observation hour, an estimated "
    f"**{latest_attr['traffic_combustion_pct']:.1f}%** of PM2.5 is linked to traffic/combustion, "
    f"**{latest_attr['industry_power_pct']:.1f}%** to industrial/power signals, "
    f"**{latest_attr['dust_pct']:.1f}%** to dust, and "
    f"**{latest_attr['fires_pct']:.1f}%** to upwind fires, superimposed on a weather baseline of "
    f"**{latest_attr['bg_weather_pct']:.1f}%** (*statistical proxy estimate, not causal proof*)."
)

# 4. Interactive Charts
st.subheader(f"📈 Source Decomposition Timeline: {city.capitalize()}")
tab1, tab2, tab3 = st.tabs(["Stacked Source Area", "Actual vs Model Prediction", "Ablation Study (Error Drop)"])

with tab1:
    fig, ax = plt.subplots(figsize=(12, 4))
    # Rolling 14 days for visual clarity
    df_plot = df_attr.tail(336)
    ax.stackplot(
        df_plot["time"],
        df_plot["background_and_weather"],
        df_plot["traffic_combustion"],
        df_plot["industry_power"],
        df_plot["dust"],
        df_plot["fires"],
        labels=["Background & Weather", "Traffic / Combustion", "Industry / Power", "Dust", "Upwind Fires"],
        colors=["#bdc3c7", "#e74c3c", "#9b59b6", "#f39c12", "#c0392b"],
        alpha=0.85
    )
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

with tab2:
    fig, ax = plt.subplots(figsize=(12, 4))
    df_plot = df_attr.tail(336)
    ax.plot(df_plot["time"], df_plot["pm25_actual"], label="CAMS Atmospheric PM2.5 (Actual)", color="#2c3e50", lw=1.5)
    ax.plot(df_plot["time"], df_plot["pm25_pred"], label="M3 Random Forest Estimate", color="#27ae60", lw=1.5, linestyle="--")
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

with tab3:
    if df_met is not None:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        bars = ax.bar(df_met["Model"], df_met["MAE"], color=["#7f8c8d", "#e67e22", "#3498db", "#2ecc71"])
        ax.set_ylabel("Mean Absolute Error (µg/m³)")
        ax.set_title(f"Ablation Impact: Incremental Error Reduction ({city.capitalize()})")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f}", ha="center", va="bottom")
        st.pyplot(fig)
        st.dataframe(df_met[["Model", "MAE", "RMSE", "R2", "MAE_Improvement_pct"]])

# 5. Pan-India Comparison View
st.subheader("🌐 Pan-India Cross-Metro Source Comparison")
shares_path = "results/source_shares_by_city.csv"
if os.path.exists(shares_path):
    df_shares = pd.read_csv(shares_path)
    df_shares_plot = df_shares.set_index("City")
    
    fig, ax = plt.subplots(figsize=(10, 4))
    df_shares_plot.plot(
        kind="bar",
        stacked=True,
        colormap="Spectral",
        ax=ax
    )
    ax.set_ylabel("Mean Attributed Share (%)")
    ax.set_title("Cross-City Source Composition Contrast")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("Notice: Hyderabad reflects dominant urban combustion and baseline weather dispersion, contrasting with northern cities where agricultural fires feature more strongly.")

# 6. Sanity Check Visuals
st.subheader("🔍 Scientific Sanity Checks")
c_col1, c_col2 = st.columns(2)
wind_img = f"results/{city}/sanity_wind_sector.png"
diurnal_img = f"results/{city}/sanity_diurnal_traffic.png"

if os.path.exists(wind_img):
    c_col1.image(wind_img, caption="Sanity Check 1: Attributed Pollution vs Wind Sector")
if os.path.exists(diurnal_img):
    c_col2.image(diurnal_img, caption="Sanity Check 2: Diurnal (Hourly) Combustion Distribution")

# 7. Methodological Limitations & Transparency
with st.expander("ℹ️ How it Works & Methodological Caveats (Non-Negotiable)"):
    st.markdown("""
    - **Modelled Atmospheric Composition:** All target and tracer variables originate from the European CAMS model via Open-Meteo (~40 km regional resolution), not local ground-level monitoring stations.
    - **Proxy Tracers:** NO2/CO, SO2, and dust serve as atmospheric indicators, not direct tailpipe or smokestack emissions.
    - **Aerodynamic Filtering:** Satellite fires (NASA FIRMS VIIRS) are filtered within a 24h window and a ±45° arc aligned with incoming wind direction.
    - **Counterfactual Attribution:** Group contributions represent statistical differences from a 10th-percentile clean baseline rescaled to excess PM2.5.
    - **Correlation vs Causation:** These metrics indicate statistical association across a recent 60-day window, not regulatory proof of origin.
    """)

st.markdown("---")
st.markdown("<small>Data Sources: Open-Meteo (CC BY 4.0), Copernicus Atmosphere Monitoring Service (CAMS), NASA FIRMS / LANCE.</small>", unsafe_allow_html=True)
