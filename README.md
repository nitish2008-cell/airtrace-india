# 🌐 AirTrace India: Multi-Source Attribution & Urban Policy Sandbox
> **Track:** Green Planet (GP-01) | **Event:** Neural Storm 2026 (JNTUH)  
> **Target Focus:** Hyderabad (Hero City) & Delhi Metropolitan Benchmark

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Problem & Concept
Standard urban AQI monitors report gross particulate mass without attributing the contribution of distinct urban activities. **AirTrace India** implements an ML-driven statistical decomposition framework to estimate the hourly contribution of:
1. **Background & Meteorology:** Atmospheric boundary-layer ventilation and stagnation.
2. **Traffic & Combustion:** Proxied via NO₂ and CO atmospheric signatures.
3. **Heavy Industry & Power:** Proxied via SO₂ signatures.
4. **Coarse Fugitive Dust:** Mineral and construction dust signatures.
5. **Upwind Biomass Fires:** Satellite-derived active fire radiative power filtered aerodynamically within a 45° upwind dispersion cone.

---

## 🔬 Scientific Methodology & Leakage Prevention
- **Strict Leakage Prevention:** PM10 is completely excluded from feature space to avoid target leakage (since PM2.5 is mechanically a subset of PM10). Autoregressive target lags are also excluded to force genuine emission-proxy reliance.
- **Counterfactual Attribution:** Isolates source groups by evaluating model predictions against a 10th-percentile clean-air baseline, normalized proportionally to conserve particulate mass.

---

## 📊 Empirical Ablation Metrics (Test Set)

| Urban Airshed | Model Stage | MAE (µg/m³) | RMSE (µg/m³) | R² Score | MAE Reduction vs M0 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Hyderabad (Hero City)** | M0 (Weather Baseline) | 16.42 | 21.05 | 0.12 | Baseline |
| **Hyderabad** | **M3 (Full Multi-Source)** | **8.92** | **12.18** | **0.71** | **+45.7%** |
| **Delhi (NCR)** | M0 (Weather Baseline) | 38.10 | 48.50 | 0.20 | Baseline |
| **Delhi (NCR)** | **M3 (Full Multi-Source)** | **18.70** | **25.30** | **0.75** | **+50.9%** |

*Key Empirical Takeaway:* Hyderabad is dominated by background ventilation and localized vehicular proxies, whereas Delhi exhibits distinct seasonal sensitivity to regional transport and transboundary fires.

---

## ⚠️ Scientific Limitations & Honesty Statement
1. **Modelled Input Nature:** Atmospheric inputs reflect ECMWF Copernicus Atmosphere Monitoring Service (CAMS) regional gridded model fields (~40 km grid) retrieved via Open-Meteo, rather than direct hyperlocal regulatory monitoring stations.
2. **Correlation vs. Causation:** Identified attribution shares reflect statistical correlation against chemical tracer proxies rather than physical chemical mass balance assays.
3. **Temporal Scope:** Model training reflects a single multi-month seasonal window; generalized cross-season transferability remains unverified.
4. **CAMS Underlying Physics:** CAMS inputs already integrate emission inventories within their meteorological modeling framework.

---

## 📜 Acknowledgements & Data Attribution
- **Open-Meteo Air Quality & Weather API:** CC BY 4.0
- **Copernicus Atmosphere Monitoring Service (CAMS):** ECMWF
- **NASA FIRMS / LANCE:** VIIRS 375m active fire telemetry
- Built for **Neural Storm 2026** (GP-01, Green Planet Track).
