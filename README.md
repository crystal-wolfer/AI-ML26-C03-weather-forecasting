# AI-ML26-C03-weather-forecasting

A machine learning project for multi-horizon weather forecasting across ten Bulgarian cities, using geographic feature encoding techniques. Part of the SoftUni Upskill Program AI and Machine Learning February 2026.

**Live demo:** [Bulgaria Weather Forecast app](https://ai-ml26-c03-weather-forecasting-tvfwdjuxrh955sbtbeukur.streamlit.app/) - pick a city and date to see the model's +1/+3/+7-day temperature and rain outlook. No setup required.

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

### Downloading the dataset
The raw CSVs are **not committed to the repository** (they are excluded via `.gitignore`). To run the project you must obtain them separately:

1. Download the dataset archive (10 CSV files): **[RAW Datasets](https://limewire.com/d/3fxN9#rrdJ2UD9A9)**
2. Unzip it and place all 10 CSV files directly inside `data/raw/` (no extra sub-folder).
3. Keep the **original filenames unchanged** - notebook `01` maps each city to an exact filename, so renaming will break loading. The expected files are:

   ```
   data/raw/
       open-meteo-sofia-2016-2026-42.71N23.40E555m.csv
       open-meteo-varna-2016-2026-43.20N27.94E68m.csv
       open-meteo-tarnovo-2016-2026-43.13N25.62E226m.csv
       open-meteo-ruse-2016-2026-43.83N26.01E36m.csv
       open-meteo-kurdzhali-2016-2026-41.65N25.36E248m.csv
       open-meteo-pleven-2016-2026-43.41N24.56E96m.csv
       open-meteo-burgas-2016-2026-42.57N27.44E32m.csv
       open-meteo-haskovo-2016-2026-41.93N25.51E195m.csv
       open-meteo-vidin-2016-2026-43.97N22.94E43m.csv
       open-meteo-yundola-2016-2026-42.00N23.84E1213m.csv
   ```

Each raw CSV has two metadata rows plus a blank line before the header, so they are read with `pd.read_csv(path, skiprows=3, encoding='utf-8')`. The exact Open-Meteo API query used to generate them is recorded in `data/raw/data_sources.md`.

> Note: Haskovo's column order differs slightly (`wind_direction_10m_dominant` is last instead of 3rd-to-last). All notebooks access columns by name, so this is handled transparently.


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
Two prediction tasks (temperature regression & rain classification) across 1-, 3-, and 7-day horizons. A single geographically-encoded model is trained per family, then evaluated per city to test generalisation.
**Temperature regression (tavg)**
- Baselines (persistence, climatology)
- Ridge
- Random Forest / HistGradientBoosting
- optional MLP.
- Metrics: RMSE, MAE, skill scores per horizon and city.

**Rain classification**
- Baselines (majority-class, rain-persistence)
- Logistic Regression
- Random Forest / HistGradientBoosting. Class imbalance handled via balanced weighting
- Metrics: F1, AUC-ROC, PR-AUC, Brier score; threshold tuned on validation.

Experiments logged to CSV via a lightweight in-notebook tracker (no MLFlow dependency). Reproducibility ensured by fixed seeds, recorded library versions, and leakage assertions. Notebook concludes with a model selection table feeding into notebook 04.

### 4. Results & Deployment (`04_results.ipynb`)
- Final candidate models retrained on combined train + validation data, evaluated once on the held-out test set (2025, plus a separate 2026 recency check)
- SHAP-based feature importance, linking back to the variable selection rationale from stage 1-2 (falls back to permutation importance automatically if `shap` is not installed)
- Per-city performance breakdown to assess generalization across climate zones
- Final models saved to `models/` as joblib files (one per task and horizon)
- Discussion (markdown): adapting the pipeline to live Open-Meteo API calls, retraining cadence, a Streamlit forecast UI, and limitations / next steps

## Repository Structure
AI-ML26-C03-weather-forecasting/
├── README.md
├── .gitignore
├── data/
│   ├── raw/                 # Open-Meteo CSVs (gitignored - see raw/data_sources.md)
│   └── processed/           # regenerated by notebook 02
├── notebooks/               # 01_eda, 02_preprocessing, 03_modelling, 04_results
├── ui_app/                  # self-contained Streamlit demo (loads the final models)
├── models/                  # empty - notebook 04 regenerates the final models here
├── reports/                 # empty - notebooks 03/04 regenerate metrics + figures here
│   └── figures/
└── artefacts/               # FROZEN reference outputs
    ├── models/              # the 6 final models (also what the UI app loads)
    └── reports/{figures,*.csv}

`models/` and `reports/` start empty on purpose: run the notebooks and your
outputs regenerate there, then diff them against the committed `artifacts/` copy.

## How to work with this project

### Prerequisites
- **Python 3.11-3.13** (the models were pickled under scikit-learn 1.7.2; see the version pin below).
- A virtual environment is recommended.

### Setup
1. Clone the repository and download the raw data as described in [Downloading the dataset](#downloading-the-dataset) so that `data/raw/` contains the 10 CSV files.

2. Create and activate a virtual environment, then install the dependencies:

   ```bash
   python -m venv .venv
   # Windows
   .venv\Scripts\activate
   # macOS / Linux
   source .venv/bin/activate

   pip install -r requirements.txt
   ```

   If a `requirements.txt` is not present, install the packages directly:

   ```bash
   pip install \
     pandas numpy scipy \
     matplotlib seaborn \
     scikit-learn==1.7.2 statsmodels \
     joblib pyarrow shap \
     jupyter ipykernel
   ```

   **Why pin scikit-learn to 1.7.2:** the trained models in `models/` were serialized with scikit-learn 1.7.2. Loading them with a different version may raise version-mismatch warnings or errors. `statsmodels` is needed for the VIF analysis in notebook 02, `pyarrow` for reading/writing the parquet files, and `shap` for the feature-attribution plots in notebook 04 (optional - the notebook falls back to permutation importance if it is missing).

### Running the notebooks
Run the notebooks **in numerical order** from the `notebooks/` folder - each stage produces artifacts the next one consumes:

1. **`01_data_collection_and_eda.ipynb`** - reads `data/raw/`, produces EDA figures.
2. **`02_preprocessing_and_features.ipynb`** - builds features and writes the processed splits to `data/processed/`. **Run this before 03 and 04**, otherwise they have no processed data to load.
3. **`03_modelling.ipynb`** - trains and compares models, writes the experiment log and selection tables to `reports/`.
4. **`04_results.ipynb`** - retrains the selected models on train+validation, evaluates once on the held-out 2025/2026 test sets, runs SHAP attribution, and saves the final models to `models/`.

### Troubleshooting
- **`FileNotFoundError` on a raw CSV in notebook 01** - a file is missing from `data/raw/` or was renamed. Confirm all 10 filenames match the list above exactly.
- **Notebook 03 or 04 cannot find `data/processed/...`** - run notebook `02` first; the processed files are not committed to git.
- **Model loading warnings/errors in notebook 04 or the app** - your scikit-learn version differs from 1.7.2; reinstall with the pinned version.
 