import os
import io
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="AirTrace India | Urban Attribution & Policy Sandbox", page_icon="🌐", layout="wide")

st.markdown("""
<style>
    .metric-box {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

def deg_to_compass(d):
    val = int((d / 22.5) + 0.5)
    arr = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
    return arr[(val % 16)]

def get_cpcb_category(val):
    if val <= 30: return "Good", "#10b981"
    elif val <= 60: return "Satisfactory", "#84cc16"
    elif val <= 90: return "Moderate", "#eab308"
    elif val <= 120: return "Poor", "#f97316"
    elif val <= 250: return "Very Poor", "#ef4444"
    else: return "Severe", "#7c3aed"

st.title("🌐 AirTrace India: Multi-Source Attribution & Policy Sandbox")
st.caption("Statistical Urban Decomposition Platform | Neural Storm 2026 (JNTUH) GP-01")

CITY_CONFIG = {
    "Hyderabad (Hero City)": {"key": "hyderabad", "lat": 17.3850, "lon": 78.4867},
    "Delhi": {"key": "delhi", "lat": 28.6139, "lon": 77.2090}
}

selected_label = st.selectbox("Select Target Metropolitan Area:", list(CITY_CONFIG.keys()), index=0)
city_meta = CITY_CONFIG[selected_label]
city = city_meta["key"]

@st.cache_data
def load_data(city_name):
    attr_path = f"results/{city_name}/attribution_hourly.csv"
    feat_path = f"data/{city_name}/features_all.csv"
    met_path = f"results/{city_name}/metrics.csv"
    
    if os.path.exists(attr_path) and os.path.exists(feat_path):
        df_attr = pd.read_csv(attr_path)
        df_feat = pd.read_csv(feat_path)
        df_met = pd.read_csv(met_path) if os.path.exists(met_path) else None
        df_attr["time"] = pd.to_datetime(df_attr["time"])
        df_feat["time"] = pd.to_datetime(df_feat["time"])
        return df_attr, df_feat, df_met
    else:
        return None, None, None

df_attr, df_feat, df_met = load_data(city)

if df_attr is None or df_feat is None:
    st.error(f"Result files for {city} are missing in the repo (results/{city}/ and data/{city}/).")
    st.stop()

latest_attr = df_attr.iloc[-1]
latest_feat = df_feat.iloc[-1]
cat_label, cat_color = get_cpcb_category(latest_attr["pm25_actual"])
wind_dir = deg_to_compass(latest_feat["wind_direction_10m"])

# KPI Header
col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Modelled PM2.5 (CAMS)", f"{latest_attr['pm25_actual']:.1f} µg/m³")
col2.markdown(f"**CPCB Air Quality Band**<br><span style='color:{cat_color}; font-size:1.3rem; font-weight:bold;'>{cat_label}</span>", unsafe_allow_html=True)
col3.metric("Prevailing Wind", f"{wind_dir} ({int(latest_feat['wind_direction_10m'])}°)")
col4.metric("Active 24h Fires (<500km)", int(latest_feat.get("total_fires_500km", 0)))

r2 = df_met.iloc[-1]["R2"] if (df_met is not None and "R2" in df_met.columns) else None
col5.metric("Model R² (test)", f"{r2:.2f}" if r2 is not None else "n/a")

dominant_source = max(
    {"Traffic & Combustion": latest_attr["traffic_combustion_pct"],
     "Industry & Power": latest_attr["industry_power_pct"],
     "Dust": latest_attr["dust_pct"],
     "Biomass Fires": latest_attr["fires_pct"]}.items(),
    key=lambda x: x[1]
)

st.markdown("### 🔍 Real-Time Indicator & Municipal Source Signals")
if dominant_source[0] == "Traffic & Combustion":
    advisory = "Elevated vehicular proxy signals detected. Statistical indicator suggests prioritizing transit corridors and local traffic congestion controls."
elif dominant_source[0] == "Biomass Fires":
    advisory = "Upwind satellite fire detections detected within the aerodynamic envelope. Statistical indicator suggests regional agricultural burn tracking."
elif dominant_source[0] == "Dust":
    advisory = "Elevated coarse mineral dust signals detected. Statistical indicator suggests road dust suppression and construction zone monitoring."
else:
    advisory = "Elevated sulfur dioxide industrial proxies detected. Statistical indicator suggests peri-urban industrial corridor emission surveillance."

st.info(f"Indicative signal only (proxy-based): Dominant source signal is {dominant_source[0]} ({dominant_source[1]:.1f}% estimated share). {advisory}")

tabs = st.tabs([
    "📊 Source Decomposition",
    "📉 Model fit (not a forecast)",
    "🎛️ Policy Sandbox (Illustrative)",
    "🗺️ 500 km study area",
    "📉 Model Ablation & Validation",
    "🌐 Multi-Metro Comparison"
])

# TAB 1: Source Decomposition
with tabs[0]:
    st.subheader(f"Hourly Source Decomposition: {selected_label}")
    df_plot = df_attr.tail(336)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.stackplot(
        df_plot["time"],
        df_plot["background_and_weather"],
        df_plot["traffic_combustion"],
        df_plot["industry_power"],
        df_plot["dust"],
        df_plot["fires"],
        labels=["Background & Weather", "Traffic / Combustion", "Industry / Power", "Dust", "Upwind Fires"],
        colors=["#64748b", "#ef4444", "#a855f7", "#f59e0b", "#dc2626"],
        alpha=0.85
    )
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# TAB 2: Model fit (not a forecast)
with tabs[1]:
    st.subheader("Model prediction vs modelled PM2.5")
    recent = df_attr.tail(120).copy()
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(recent["time"], recent["pm25_actual"], label="Modelled PM2.5 (CAMS)", color="#38bdf8", lw=2)
    ax.plot(recent["time"], recent["pm25_pred"], label="M3 model prediction", color="#f43f5e", linestyle="--", lw=1.8)
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("Tracks the in-sample reconstructive model fit over the last 5 days without autoregressive target lags.")

# TAB 3: Policy Sandbox
with tabs[2]:
    st.subheader("Illustrative Policy Intervention Sandbox")
    st.caption("Illustrative arithmetic on the estimated shares for the latest hour. Not a prediction of real policy effects.")
    sc_col1, sc_col2 = st.columns(2)
    traffic_reduction = sc_col1.slider("Illustrative Traffic Curb (% reduction in vehicular proxies)", 0, 80, 20)
    dust_reduction = sc_col2.slider("Illustrative Dust Suppression (% reduction in coarse dust)", 0, 80, 25)
    
    sim_traffic = latest_attr["traffic_combustion"] * (1 - traffic_reduction / 100.0)
    sim_dust = latest_attr["dust"] * (1 - dust_reduction / 100.0)
    sim_pm25 = latest_attr["background_and_weather"] + sim_traffic + latest_attr["industry_power"] + sim_dust + latest_attr["fires"]
    
    sim_cat, sim_color = get_cpcb_category(sim_pm25)
    saved_pm25 = max(latest_attr["pm25_pred"] - sim_pm25, 0.0)
    
    sim_c1, sim_c2, sim_c3 = st.columns(3)
    sim_c1.metric("Counterfactual PM2.5 Level", f"{sim_pm25:.1f} µg/m³", delta=f"-{saved_pm25:.1f} µg/m³", delta_color="inverse")
    sim_c2.markdown(f"**Projected CPCB Band**<br><span style='color:{sim_color}; font-size:1.3rem; font-weight:bold;'>{sim_cat}</span>", unsafe_allow_html=True)
    sim_c3.metric("Estimated Improvement", f"{(saved_pm25 / max(latest_attr['pm25_pred'], 1.0)) * 100:.1f}%")

# TAB 4: 500 km study area
with tabs[3]:
    st.subheader("Regional Study Area (500 km Geometric Perimeter)")
    map_data = pd.DataFrame({"lat": [city_meta["lat"]], "lon": [city_meta["lon"]]})
    angles = np.radians(np.linspace(0, 360, 32))
    radii = 500 / 111.0
    ring_lats = city_meta["lat"] + radii * np.sin(angles)
    ring_lons = city_meta["lon"] + (radii / np.cos(np.radians(city_meta["lat"]))) * np.cos(angles)
    map_envelope = pd.concat([map_data, pd.DataFrame({"lat": ring_lats, "lon": ring_lons})], ignore_index=True)
    st.map(map_envelope, zoom=5)
    st.caption("Visual boundary represents the 500 km spatial search radius used during NASA FIRMS satellite fire filtering.")

# TAB 5: Model Ablation
with tabs[4]:
    st.subheader("Model Ablation & Validation Benchmarking")
    if df_met is not None and "MAE" in df_met.columns:
        fig, ax = plt.subplots(figsize=(8, 3.5))
        bars = ax.bar(df_met["Model"], df_met["MAE"], color=["#64748b", "#10b981"][:len(df_met)])
        ax.set_ylabel("Mean Absolute Error (µg/m³)")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.3, f"{yval:.2f}", ha="center", va="bottom")
        st.pyplot(fig)
        st.dataframe(df_met)

# TAB 6: Multi-Metro Comparison
with tabs[5]:
    st.subheader("Cross-Metro Attribution Comparison")
    shares_path = "results/source_shares_by_city.csv"
    if os.path.exists(shares_path):
        df_shares = pd.read_csv(shares_path).set_index("City")
        fig, ax = plt.subplots(figsize=(8, 4))
        df_shares.plot(kind="bar", stacked=True, colormap="tab10", ax=ax)
        ax.set_ylabel("Mean Attributed Share (%)")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Cross-metro summary available when source shares are aggregated.")

st.markdown("---")
csv_data = df_attr.tail(168).to_csv(index=False).encode("utf-8")
st.download_button(
    label=f"📥 Download 7-Day Attribution Dataset ({city.capitalize()})",
    data=csv_data,
    file_name=f"AirTrace_{city}_attribution.csv",
    mime="text/csv"
)
