"""
Minimal Streamlit forecast demo for the Bulgaria multi-city weather project.
Run (local)
-----------
    pip install -r requirements.txt
    streamlit run app.py

Layout expected at the repo root: app.py, requirements.txt, models/ (the 6
joblib models), forecast_data/ (the 2 test parquets, scaler.joblib,
feature_columns.json).

Requires: streamlit, pandas, numpy, scikit-learn, joblib, pyarrow, matplotlib
"""

from __future__ import annotations

import datetime as dt
import glob
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Paths and constants
BASE_DIR = Path(__file__).resolve().parent

def _resolve_dir(name: str) -> Path:
    """Find a folder named `name` next to app.py or one level up (repo root)."""
    for base in (BASE_DIR, BASE_DIR.parent):
        candidate = base / name
        if candidate.exists():
            return candidate
    return BASE_DIR / name

MODELS_DIR = _resolve_dir("models")
PROCESSED_DIR = _resolve_dir("forecast_data")
FEATURE_COLS_PATH = PROCESSED_DIR / "feature_columns.json"
SCALER_PATH = PROCESSED_DIR / "scaler.joblib"
TEST_PARQUETS = [PROCESSED_DIR / "test_2025.parquet", PROCESSED_DIR / "test_2026.parquet"]

HORIZONS = [1, 3, 7]
TREND_DAYS_BEFORE = 10
TREND_DAYS_AFTER = 8

START_MIN = dt.date(2026, 1, 1)
START_MAX = dt.date(2026, 4, 24)
DEFAULT_START = dt.date(2026, 4, 24)

CITY_LABELS = {
    "sofia": "Sofia",
    "varna": "Varna",
    "veliko_tarnovo": "Veliko Tarnovo",
    "ruse": "Ruse",
    "kardzhali": "Kardzhali",
    "pleven": "Pleven",
    "burgas": "Burgas",
    "haskovo": "Haskovo",
    "vidin": "Vidin",
    "yundola": "Yundola",
}


# Cached loaders
def _find_model_file(task: str, horizon: int) -> str | None:
    """Locate one model file, searching the resolved dir then recursively.

    Checks MODELS_DIR directly, then does a recursive search under the repo root, so the file is found wherever it
    actually landed in the checkout.
    """
    pattern = f"final_{task}_*_{horizon}d.joblib"
    direct = glob.glob(str(MODELS_DIR / pattern))
    if direct:
        return direct[0]
    for root in (BASE_DIR, BASE_DIR.parent):
        deep = glob.glob(str(root / "**" / pattern), recursive=True)
        if deep:
            return deep[0]
    return None


@st.cache_resource
def load_models() -> dict:
    """Load all final models, keyed by (task, horizon)."""
    models = {}
    for task in ("temperature", "rain"):
        for horizon in HORIZONS:
            path = _find_model_file(task, horizon)
            if path:
                models[(task, horizon)] = joblib.load(path)
    return models


def _diagnostics() -> str:
    """List candidate directories and their contents for troubleshooting."""
    lines = []
    seen = set()
    for label, directory in [
        ("app dir (BASE_DIR)", BASE_DIR),
        ("repo root", BASE_DIR.parent),
        ("resolved MODELS_DIR", MODELS_DIR),
    ]:
        key = str(directory)
        if key in seen:
            continue
        seen.add(key)
        try:
            entries = sorted(p.name for p in directory.iterdir())
            shown = ", ".join(entries[:60]) if entries else "(empty)"
        except Exception as error:
            shown = f"<cannot list: {error}>"
        lines.append(f"{label}: {directory}\n    {shown}")
    return "\n".join(lines)


@st.cache_resource
def load_tavg_stats() -> tuple:
    """Return (mean, scale) for tavg to invert the scaler for display."""
    scaler = joblib.load(SCALER_PATH)
    idx = list(scaler.feature_names_in_).index("tavg")
    return float(scaler.mean_[idx]), float(scaler.scale_[idx])


@st.cache_data
def load_feature_cols() -> list:
    """Load the ordered list of 76 model input columns."""
    with open(FEATURE_COLS_PATH, "r", encoding="utf-8") as handle:
        return json.load(handle)["feature_cols"]


@st.cache_data
def load_data() -> pd.DataFrame:
    """Load and concatenate the 2025 + 2026 scaled feature rows."""
    frames = [pd.read_parquet(path) for path in TEST_PARQUETS]
    frame = pd.concat(frames, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"])
    return frame


# Forecast logic
def forecast_from_date(
    city_df: pd.DataFrame,
    start: pd.Timestamp,
    feature_cols: list,
    models: dict,
) -> dict:
    """Forecast the next 1/3/7 days from a single issue date.

    Args:
        city_df: Rows for one city, indexed by date.
        start: The issue date ("today"); must be present in city_df.
        feature_cols: Ordered list of model input columns.
        models: Mapping of (task, horizon) -> fitted estimator.

    Returns:
        Mapping horizon -> dict with forecast_date, temp, rain_prob, obs_temp,
        obs_rain. Empty if the issue date has no feature row.
    """
    if start not in city_df.index:
        return {}

    source = city_df.loc[start]
    design = source[feature_cols].to_frame().T.astype(float)

    results = {}
    for horizon in HORIZONS:
        temp = float(models[("temperature", horizon)].predict(design)[0])
        rain_prob = float(models[("rain", horizon)].predict_proba(design)[0, 1])

        obs_temp = source.get(f"tavg_target_{horizon}d")
        obs_rain = source.get(f"rain_target_{horizon}d")
        results[horizon] = {
            "forecast_date": (start + pd.Timedelta(days=horizon)).date(),
            "temp": temp,
            "rain_prob": rain_prob,
            "obs_temp": None if pd.isna(obs_temp) else float(obs_temp),
            "obs_rain": None if pd.isna(obs_rain) else int(obs_rain),
        }
    return results


def build_trend(
    city_df: pd.DataFrame,
    start: pd.Timestamp,
    feature_cols: list,
    temp_model,
    tavg_mean: float,
    tavg_scale: float,
) -> pd.DataFrame:
    """Build observed vs 1-day-ahead forecast temperature over a window. Days outside the dataset stay NaN.

    Args:
        city_df: Rows for one city, indexed by date.
        start: The issue date the window is centred on.
        feature_cols: Ordered list of model input columns.
        temp_model: The 1-day temperature model.
        tavg_mean: Scaler mean for tavg.
        tavg_scale: Scaler scale for tavg.

    Returns:
        DataFrame indexed by date with 'Observed' and 'Forecast' columns (degC).
    """
    window = pd.date_range(
        start - pd.Timedelta(days=TREND_DAYS_BEFORE),
        start + pd.Timedelta(days=TREND_DAYS_AFTER),
    )
    records = {}
    for day in window:
        observed = None
        if day in city_df.index:
            observed = float(city_df.loc[day, "tavg"]) * tavg_scale + tavg_mean
        forecast = None
        source = day - pd.Timedelta(days=1)
        if source in city_df.index:
            design = city_df.loc[source, feature_cols].to_frame().T.astype(float)
            forecast = float(temp_model.predict(design)[0])
        records[day] = {"Observed": observed, "Forecast": forecast}
    return pd.DataFrame(records).T


def render_trend_chart(trend: pd.DataFrame, start: pd.Timestamp, results: dict):
    """Draw the observed vs forecast trend with the forward forecast points."""
    figure, axis = plt.subplots(figsize=(8, 3.5))
    axis.plot(
        trend.index, trend["Observed"], marker="o", markersize=4,
        color="#1f77b4", label="Observed",
    )
    axis.plot(
        trend.index, trend["Forecast"], marker="o", markersize=3,
        linestyle="--", color="#ff7f0e", label="1-day-ahead forecast",
    )
    axis.axvline(start, color="grey", linestyle=":", linewidth=1, label="Issue date")

    # The +1/+3/+7-day forecasts, plotted at the dates they apply to.
    lead_colors = {1: "#2ca02c", 3: "#9467bd", 7: "#8c564b"}
    for horizon in sorted(results):
        axis.scatter(
            pd.Timestamp(results[horizon]["forecast_date"]),
            results[horizon]["temp"], s=70, zorder=5,
            color=lead_colors.get(horizon, "#333333"),
            label=f"+{horizon}-day forecast",
        )

    axis.set_ylabel("Temperature (degC)")
    axis.set_title("Temperature trend and outlook")
    axis.legend(fontsize=8, loc="best")
    axis.grid(True, alpha=0.3)
    figure.autofmt_xdate()
    figure.tight_layout()
    return figure


# App
def main() -> None:
    st.set_page_config(page_title="Bulgaria Weather Forecast Demo", page_icon="*")
    st.title("Bulgaria Weather Forecast")
    st.caption("Select a city and a date - get the temperature and rain forecast.")

    models = load_models()
    feature_cols = load_feature_cols()
    data = load_data()
    tavg_mean, tavg_scale = load_tavg_stats()

    if not models:
        st.error(f"No model files found in {MODELS_DIR}.")
        st.caption(
            "If the models folder is empty or missing the large .joblib files, the big "
            "RandomForest models were not pulled into the deployment."
        )
        st.code(_diagnostics())
        st.stop()

    # Inputs
    left, right = st.columns(2)
    with left:
        cities = sorted(data["city"].unique())
        city = st.selectbox(
            "City", cities, format_func=lambda c: CITY_LABELS.get(c, c)
        )
    with right:
        start_date = st.date_input(
            "Forecast from (issue date)",
            value=DEFAULT_START,
            min_value=START_MIN,
            max_value=START_MAX,
            format="DD/MM/YYYY",
        )

    city_df = data[data["city"] == city].set_index("date").sort_index()
    start = pd.Timestamp(start_date)
    results = forecast_from_date(city_df, start, feature_cols, models)

    st.divider()
    st.subheader(
        f"{CITY_LABELS.get(city, city)} - outlook from {start_date.strftime('%d %b %Y')}"
    )

    if not results:
        st.warning("No weather data available for this issue date.")
        st.stop()

    # Headline = nearest forecast (+1 day).
    head = results[min(results)]
    rain_verdict = "Rain likely" if head["rain_prob"] >= 0.5 else "Likely dry"

    head_left, head_right = st.columns(2)
    head_left.metric(
        f"Tomorrow ({head['forecast_date'].strftime('%d %b')})",
        f"{head['temp']:.1f} degC",
    )
    head_right.metric("Rain", f"{head['rain_prob'] * 100:.0f}%", rain_verdict)

    # Forward outlook table, with observed values where known.
    st.markdown("**Outlook for the days ahead**")
    rows = []
    for horizon in sorted(results):
        item = results[horizon]
        obs_temp = (
            f"{item['obs_temp']:.1f} degC" if item["obs_temp"] is not None else "-"
        )
        obs_rain = (
            ("yes" if item["obs_rain"] == 1 else "no")
            if item["obs_rain"] is not None
            else "-"
        )
        rows.append(
            {
                "Lead time": f"+{horizon} day(s)",
                "Forecast date": item["forecast_date"].strftime("%d %b %Y"),
                "Forecast temp": f"{item['temp']:.1f} degC",
                "Observed temp": obs_temp,
                "Rain prob": f"{item['rain_prob'] * 100:.0f}%",
                "Rained": obs_rain,
            }
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    if any(results[h]["obs_temp"] is not None for h in results):
        st.caption(
            "Observed columns are the actual recorded values - shown for dates that "
            "fall inside the dataset so you can compare the forecast against reality."
        )

    # Trend chart: observed vs forecast around the issue date.
    st.markdown("**Temperature trend**")
    trend = build_trend(
        city_df, start, feature_cols, models[("temperature", 1)],
        tavg_mean, tavg_scale,
    )
    st.pyplot(render_trend_chart(trend, start, results))
    st.caption(
        "Blue = recorded temperature  \n"
        "Orange dashed = the model's 1-day-ahead forecast for each day  \n"
        "Dotted grey = the issue date  \n"
        "Coloured dots = the +1/+3/+7-day forecasts plotted on the dates they apply to  \n"
        "The observed line stops where the dataset ends."
    )

    st.divider()
    st.caption(
        "Offline-evaluation demo. Forecasts project forward from the issue date using "
        "its real ERA5 features; issue dates run to 2026-04-24 (the last available data), "
        "so the outlook reaches 2026-05-01. This application was built for the SoftUni "
        "AI/ML course, using Claude."
    )


if __name__ == "__main__":
    main()
