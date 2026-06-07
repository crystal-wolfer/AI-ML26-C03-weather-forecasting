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
**Period:** January 2016 – May 2026 (10 years)  
**Features:** 33 variables per city including temperature (mean/min/max), precipitation, rain, snowfall, wind (speed/gusts/direction), cloud cover, humidity, dew point, pressure (MSL and surface), vapour pressure deficit, soil moisture, and soil temperature
**Data quality note:** Model-derived reanalysis data - spatially complete with zero missing values across all cities and variables.


## Methodology
TBD

## Repository Structure
TBD

## How to work with this project
TBD

 