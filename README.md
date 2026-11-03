# AirTrace India: Urban Air Quality & Source Attribution
Track: Green Planet (GP-01) | Neural Storm 2026 (JNTUH)

## Overview
AirTrace India estimates hourly PM2.5 attributed to 4 source groups (Fires, Traffic/Combustion, Industry/Power, Dust) and weather baseline.

## Key Results
- Hyderabad: Dominated by urban combustion proxies and baseline weather dispersion.
- Delhi: Demonstrates clear transboundary upwind biomass fire contributions.

## Quickstart
```bash
pip install -r requirements.txt
streamlit run app.py
```

## ⚠️ Limitations & Honesty Statement
1. **Atmospheric Model Inputs:** Data is retrieved from the Copernicus Atmosphere Monitoring Service (CAMS) regional model (~40 km resolution) via Open-Meteo, representing municipal airsheds rather than hyperlocal street-level microclimates.
2. **Proxy Tracers:** Chemical species serve as atmospheric signatures rather than direct tailpipe emission measurements.
3. **Statistical Attribution:** Quantified allocations are indicative counterfactual estimates across the study window, not regulatory chemical mass balance laboratory assays.

## 📜 Acknowledgements & Data Attribution
- **Open-Meteo Air Quality & Weather API:** CC BY 4.0
- **Copernicus Atmosphere Monitoring Service (CAMS):** ECMWF
- **NASA FIRMS / LANCE:** VIIRS 375m active fire telemetry
- Built for **Neural Storm 2026** (GP-01, Green Planet Track).
