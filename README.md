# AI-ML26-C03-weather-forecasting

A machine learning project for multi-horizon weather forecasting across five Bulgarian cities, using geographic feature encoding and transfer learning techniques. Part of the SoftUni Upskill Program AI and Machine Learning February 2026.

## Project Overview
Current weather applications often provide unreliable forecasts that change significantly within short timeframes and fail to communicate prediction uncertainty. This project addresses these limitations by building interpretable ML models that:

- Forecast temperature 1, 3, and 7 days ahead (with expected degrading at 7-day forecasting)
- Predict rain probability (binary classification)
- Discover natural weather patterns via clustering
- Generalize across cities using geographic feature encoding
- Quantify prediction uncertainty

## Study Area
10 Bulgarian cities were selected to capture the full climatic diversity of the country as a starting point in this project but further cities might be added along the project as well:

| City | Region | Elevation | Climate |
|------|--------|-----------|---------|
| Sofia | Western | ~555m | Continental |
| Varna | Eastern (Black Sea coast) | ~68m | Temperate/Coastal |
| Veliko Tarnovo | Central | ~226m | Continental |
| Ruse | Northern (Danube) | ~36m | Continental |
| Kardzhali | Southern (Rhodopes) | ~248m | Transitional |
| Pleven | Central North | ~96m | Continental |
| Burgas | Eastern (Black Sea coast) | ~32m | Temperate/Coastal |
| Haskovo | Southern | ~195m | Sub-Mediterranean |
| Vidin | Northwestern (Danube) | ~43m | Continental |
| Yundola | Western Mountains | ~1213m | Alpine/Subalpine |


## Data
**Source:** [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api) - ERA5 reanalysis blended with ECMWF historical forecast model  
**Period:** January 2016 - May 2026 (10 years)  
**Features:** 33 variables per city including temperature (mean/min/max), precipitation, rain, snowfall, wind (speed/gusts/direction), cloud cover, humidity, dew point, pressure (MSL and surface), vapour pressure deficit, soil moisture, and soil temperature
**Data quality note:** Model-derived reanalysis data - spatially complete with zero missing values across all cities and variables.


## Methodology
The project follows a four-stage pipeline, with one notebook per stage. Each stage produces artifacts (cleaned data, trained models, figures) consumed by the next.

### 1. Data Collection & EDA (`01_data_collection_and_eda.ipynb`)
- City-by-city exploration of the 10 selected Open-Meteo datasets: distributions, missing values, seasonal cycles
- Cross-city comparisons to characterize climatic diversity (coastal vs. continental vs. alpine)
- Correlation analysis against `tavg` (mean temperature) to identify candidate predictors
- Output: a shortlist of ~15 variables to carry forward, with documented rationale

### 2. Preprocessing & Feature Engineering (`02_preprocessing_and_features.ipynb`)
- Merge all 10 city datasets into a single unified frame with a `city` identifier column
- Apply column renaming for readability (raw Open-Meteo names - short codes)
- Variable selection: reduce to the ~15 predictors identified in EDA
- Redundancy analysis: correlation matrix and Variance Inflation Factor (VIF) to confirm exclusion of collinear variables
- Feature engineering:
  - Lag features (t-1, t-3, t-7) for autoregressive signal
  - Rolling means to capture short-term trends
  - Pressure tendency (`pres_max - pres_min`) as a synoptic-change indicator
  - Circular encoding (sin/cos) of wind direction
- Time-aware train / validation / test splits (2016-2023 / 2024 / 2025-2026 (primary & recency)), with all scalers, encoders, and imputers fit on training data only
- Output: cleaned, model-ready datasets saved to disk

### 3. Modelling (`03_modelling.ipynb`)
Two prediction tasks, each evaluated across 1-, 3-, and 7-day horizons:

**Temperature regression (predicting `tavg`)**
- Baseline: persistence model (naive forecast)
- Linear Regression
- Random Forest / Gradient Boosting (XGBoost or LightGBM)
- Optional: a simple MLP neural network
- Evaluated with RMSE and MAE, per horizon and per city

**Rain classification (predicting next-day rain)**
- Baseline: majority-class predictor
- Logistic Regression
- Random Forest / Gradient Boosting
- Evaluated with F1, AUC-ROC, and precision-recall (accounting for class imbalance)

All experiments are tracked with MLFlow (model type, hyperparameters, metrics, feature counts). The notebook concludes with a model selection table comparing approaches and identifying the final candidates.

### 4. Results & Deployment (`04_results_and_deployment.ipynb`)
- Final candidate models retrained on combined train + validation data, evaluated once on the held-out test set (2025, plus a separate 2026 recency check)
- SHAP-based feature importance, linking back to the variable selection rationale from stage 1-2
- Per-city performance breakdown to assess generalization across climate zones
- Discussion (markdown): adapting the pipeline to live Open-Meteo API calls, retraining cadence, a Streamlit/Gradio forecast UI concept, and limitations / next steps

## Repository Structure
TBD

## How to work with this project
TBD

 