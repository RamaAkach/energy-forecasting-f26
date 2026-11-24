# forecast-24h: YOUR model forecasts Connecticut electricity load for the next 24 hours, every hour.
#
# Every hour (:20, after the weather scrapers at :05 and the ISO-NE scraper at :15) this function:
#   1. loads your live model:          gs://<bucket>/energy/models/model.joblib (+ model_card.json)
#   2. picks the next 24 FULL hours:   if it runs at 2:20 p.m., it forecasts 3 p.m. today ... 2 p.m. tomorrow
#   3. builds the features with make_features(), the SAME code as the Lab 5 notebook:
#        weather  -> the newest NWS forecast for each hour (bdl_weather_forecast_master.csv)
#        load     -> the load at the same hour yesterday (ct_load_5min_hourly_master.csv, live five-minute load)
#   4. adds a low and high value (an 80% range: 10th and 90th percentiles) if your model card says how (Lab 6):
#        "range_method": "quantile_models"  -> two more models, energy/models/model_low.joblib and model_high.joblib
#        "range_method": "offsets_by_hour"  -> "range_offsets_mw": {"low": [24 MW], "high": [24 MW]}, by Connecticut hour
#        "range_method": "range_model"      -> Final project: ONE model, energy/models/model_qrf.joblib, fed the NWS
#                                              forecast only. model.predict(X) returns 3 columns: low, expected, high
#                                              (e.g. a pipeline ending in a quantile forest with default_quantiles=
#                                              [0.1, 0.5, 0.9]). A plain quantile forest also works: it is asked for its
#                                              10th, 50th and 90th percentiles. "quantile_forest" = same thing (older name)
#        "range_method": "hrrr_qrf_pooled"  -> Lab 6C: energy/models/model_qrf.joblib (a quantile forest) on EVERY recent HRRR
#                                              run that covers each hour (weather uncertainty), pooled into one low,
#                                              expected and high. HRRR runs come from Dr. Dave's class file (HRRR_RUNS_URL);
#                                              if it is missing or stale, the NWS forecast is used instead
#        no "range_method" (Lab 5)          -> expected value only (the low and high columns stay empty)
#   5. writes 24 rows (target_hour_utc, predicted_low_mw, predicted_load_mw, predicted_high_mw, ...):
#        gs://<bucket>/energy/forecasts/latest.csv                       <- replaced every hour (Lab 6 reads this one)
#        gs://<bucket>/energy/forecasts/history/<issued time>.csv        <- one file per hour, never overwritten
#        gs://<bucket>/energy/datasets/forecasts_master.csv              <- every forecast you ever made, in one table
# No model yet? It politely does nothing ("status": "no_model"). No ISO-NE data yet? "status": "no_load_data".
# HTTP entrypoint: forecast_24h_http

import io
import json
import logging
import os
import traceback
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from google.cloud import storage
from pandas.tseries.holiday import USFederalHolidayCalendar

PROJECT_ID = os.getenv("PROJECT_ID", "")
GCS_BUCKET = os.getenv("GCS_BUCKET", "")
ENERGY_PREFIX = os.getenv("ENERGY_PREFIX", "energy")
HOURS_AHEAD = int(os.getenv("HOURS_AHEAD", "24"))

MODEL_KEY = f"{ENERGY_PREFIX}/models/model.joblib"
CARD_KEY = f"{ENERGY_PREFIX}/models/model_card.json"
LOW_MODEL_KEY = f"{ENERGY_PREFIX}/models/model_low.joblib"
HIGH_MODEL_KEY = f"{ENERGY_PREFIX}/models/model_high.joblib"
QRF_MODEL_KEY = f"{ENERGY_PREFIX}/models/model_qrf.joblib"
# Lab 6C: the class HRRR file, rewritten every hour at :05 by Dr. Dave's hrrr-bdl job (8 newest runs, 0-48 hours ahead)
HRRR_RUNS_URL = os.getenv("HRRR_RUNS_URL", "https://storage.googleapis.com/opim5512-leaderboard-f26/weather/bdl_hrrr_runs.csv")
HRRR_MAX_AGE_HOURS = 14        # newest run older than this = something is wrong upstream: use the NWS forecast instead
# 20 percentiles, each the middle of a 5% slice of the chances (2.5%, 7.5%, ..., 97.5%), same as the Lab 6C notebook
POOL_QUANTILES = [0.025, 0.075, 0.125, 0.175, 0.225, 0.275, 0.325, 0.375, 0.425, 0.475, 0.525, 0.575, 0.625, 0.675, 0.725,
                  0.775, 0.825, 0.875, 0.925, 0.975]
WEATHER_KEY = f"{ENERGY_PREFIX}/datasets/bdl_weather_forecast_master.csv"
LOAD_KEY = f"{ENERGY_PREFIX}/datasets/ct_load_5min_hourly_master.csv"
# Final project (optional): NWS forecasts at 9 airports (scraper-weather-forecast). These 7 are averaged with population
# weights into one Connecticut temperature; JFK and Boston are inputs of their own.
CT_WEATHER_KEY = f"{ENERGY_PREFIX}/datasets/ct_weather_forecast_master.csv"
STATION_WEIGHTS = {"BDL": 0.45, "HFD": 0.45, "BDR": 0.40, "DXR": 0.20, "HPN": 0.36, "HVN": 0.86, "GON": 0.27}
LATEST_KEY = f"{ENERGY_PREFIX}/forecasts/latest.csv"
HISTORY_PREFIX = f"{ENERGY_PREFIX}/forecasts/history"
MASTER_KEY = f"{ENERGY_PREFIX}/datasets/forecasts_master.csv"

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(message)s")

# ----------------------------------------------------------------------------------------------------------------
# make_features: copied word for word from the Lab 5 notebook (Part 3). Change it there, change it here.
# ----------------------------------------------------------------------------------------------------------------
WEATHER = ["temp_f", "dewpoint_f", "rel_humidity_pct", "wind_speed_mph", "cloud_cover_pct"]
CALENDAR = ["hour", "day_of_week", "month", "is_weekend", "is_holiday"]
FEATURES = CALENDAR + WEATHER + ["load_yesterday_mw"]
# Final project (optional): 5 more inputs. A model uses them only if its model card's "features" list names them.
EXTRA_FEATURES = ["ct_temp_f", "ct_dewpoint_f", "solar_proxy", "jfk_temp_f", "bos_temp_f"]
# Each extra weather input and the Bradley column that stands in when it's missing
EXTRA_WEATHER = {"ct_temp_f": "temp_f", "ct_dewpoint_f": "dewpoint_f", "jfk_temp_f": "temp_f", "bos_temp_f": "temp_f"}
# The temperature inputs that "nudge_temp_sd" (model card) shifts together, like a forecast that runs warm or cold
TEMPERATURE_INPUTS = ["temp_f", "dewpoint_f", "ct_temp_f", "ct_dewpoint_f", "jfk_temp_f", "bos_temp_f"]


def solar_proxy(hours_utc, cloud_cover_pct):
    """Sunlight reaching the ground at Bradley (W/m2): clear-sky sunlight from the sun's height, cut by cloud cover.
    Rooftop solar lowers the load the grid sees on sunny middays."""
    hours_utc = pd.DatetimeIndex(hours_utc).tz_convert("UTC")
    day_of_year = np.asarray(hours_utc.dayofyear)
    middle_of_hour = np.asarray(hours_utc.hour) + 0.5
    latitude, longitude = np.radians(41.94), -72.68
    declination = np.radians(23.44) * np.sin(2 * np.pi * (284 + day_of_year) / 365)
    hour_angle = np.radians(15 * ((middle_of_hour + longitude / 15) - 12))
    sun_height = np.arcsin(np.sin(latitude) * np.sin(declination) + np.cos(latitude) * np.cos(declination) * np.cos(hour_angle))
    clear_sky = 1000 * np.clip(np.sin(sun_height), 0, None) ** 1.15
    return clear_sky * (1 - 0.75 * (np.asarray(cloud_cover_pct, dtype=float) / 100.0) ** 3.4)


def make_features(hours_utc, weather, load, extra=False):
    """One row of model inputs per hour.
    hours_utc: the hours to describe (UTC).  weather: table indexed by UTC hour with the WEATHER columns.
    load: Connecticut load in MW, indexed by UTC hour (only hours that are already over).
    extra=True also builds EXTRA_FEATURES: the Connecticut temperature and dew point and the JFK and Boston
    temperatures (columns of the same name in weather; Bradley's values where they're missing) and the solar proxy."""
    hours_utc = pd.DatetimeIndex(hours_utc)
    local = hours_utc.tz_convert("America/New_York")           # people use electricity on local time
    out = pd.DataFrame(index=hours_utc)
    out["hour"] = local.hour
    out["day_of_week"] = local.dayofweek                       # 0 = Monday ... 6 = Sunday
    out["month"] = local.month
    out["is_weekend"] = (local.dayofweek >= 5).astype(int)
    holidays = USFederalHolidayCalendar().holidays(start="2020-01-01", end="2030-12-31")
    out["is_holiday"] = local.tz_localize(None).normalize().isin(holidays).astype(int)

    for column in WEATHER:
        out[column] = weather[column].reindex(hours_utc).values

    # Same hour yesterday. If that hour isn't over yet (only the last hour of a live forecast),
    # use the hour just before it.
    yesterday = load.reindex(hours_utc - pd.Timedelta(hours=24)).values
    hour_before = load.reindex(hours_utc - pd.Timedelta(hours=25)).values
    out["load_yesterday_mw"] = np.where(np.isnan(yesterday), hour_before, yesterday)
    if not extra:
        return out[FEATURES]
    for column, bradley in EXTRA_WEATHER.items():
        values = weather[column].reindex(hours_utc).values if column in weather.columns else np.full(len(out), np.nan)
        out[column] = np.where(np.isnan(np.asarray(values, dtype=float)), out[bradley].values, values)
    out["solar_proxy"] = solar_proxy(hours_utc, out["cloud_cover_pct"].values)
    return out[FEATURES + EXTRA_FEATURES]
# ----------------------------------------------------------------------------------------------------------------


def read_csv(bucket, key):
    blob = bucket.blob(key)
    if not blob.exists():
        return None
    return pd.read_csv(io.BytesIO(blob.download_as_bytes()))


def latest_weather(forecasts: pd.DataFrame) -> pd.DataFrame:
    """For every target hour, the NEWEST forecast that covers it."""
    forecasts["scraped_hour_utc"] = pd.to_datetime(forecasts["scraped_hour_utc"], utc=True)
    forecasts["target_hour_utc"] = pd.to_datetime(forecasts["target_hour_utc"], utc=True)
    forecasts = forecasts.sort_values(["target_hour_utc", "scraped_hour_utc"])
    newest = forecasts.drop_duplicates("target_hour_utc", keep="last").set_index("target_hour_utc")
    return newest


LATEST_LOAD_FEATURES = ["hours_since_load", "latest_load_mw", "today_vs_yesterday_mw"]


def latest_load_inputs(target_hours, load, newest):
    """Final project (optional): what the load already told us when the forecast was made. newest = the newest hour of
    load that is over (one time for a live forecast, or one per row when building training data). Returns how many hours
    after it each target is, that load, and how far it ran above (+) or below (-) the same hour the day before."""
    target_hours = pd.DatetimeIndex(target_hours)
    if isinstance(newest, pd.Timestamp):
        newest = pd.DatetimeIndex([newest] * len(target_hours))
    newest = pd.DatetimeIndex(newest)
    latest = load.reindex(newest).values
    day_before = load.reindex(newest - pd.Timedelta(hours=24)).values
    out = pd.DataFrame(index=target_hours)
    out["hours_since_load"] = np.asarray((target_hours - newest) / pd.Timedelta(hours=1), dtype=float)
    out["latest_load_mw"] = latest
    out["today_vs_yesterday_mw"] = np.where(np.isnan(day_before), 0.0, latest - day_before)
    return out[LATEST_LOAD_FEATURES]


# Final project (optional): ISO New England's own Connecticut load forecast (saved every hour by scraper-isone) as an
# input. Only versions ISO created by the top of this hour count, exactly like the training data.
ISONE_FORECAST_KEY = f"{ENERGY_PREFIX}/datasets/isone_ct_forecast_master.csv"
ISONE_FEATURES = ["isone_forecast_mw", "isone_age_hours"]
ISONE_MISSING_AGE_HOURS = 48.0


def isone_inputs(bucket, target_hours, made_by) -> pd.DataFrame:
    """For every target hour, ISO-NE's newest forecast created at or before made_by, and how many hours before the
    target that version was created. Hours with no version stay empty (run_once fills them)."""
    target_hours = pd.DatetimeIndex(target_hours)
    out = pd.DataFrame(index=target_hours, columns=ISONE_FEATURES, dtype=float)
    table = read_csv(bucket, ISONE_FORECAST_KEY)
    if table is None or len(table) == 0:
        return out
    table["issued_at_utc"] = pd.to_datetime(table["issued_at_utc"], utc=True)
    table["target_hour_utc"] = pd.to_datetime(table["target_hour_utc"], utc=True)
    table = table[(table["issued_at_utc"] <= made_by) & table["isone_forecast_mw"].notna()]
    newest = table.sort_values("issued_at_utc").drop_duplicates("target_hour_utc", keep="last").set_index("target_hour_utc")
    newest = newest.reindex(target_hours)
    out["isone_forecast_mw"] = newest["isone_forecast_mw"].astype(float).values
    out["isone_age_hours"] = np.asarray((target_hours - pd.DatetimeIndex(newest["issued_at_utc"])) / pd.Timedelta(hours=1), dtype=float)
    return out


def ct_weather(bucket) -> pd.DataFrame:
    """Final project: for every target hour, from the NEWEST forecast of each airport, the population-weighted
    Connecticut temperature and dew point (weights re-scaled over the airports that have a value, so one missing
    airport just means six) and the JFK and Boston temperatures. Empty table if the file isn't there yet."""
    names = ["ct_temp_f", "ct_dewpoint_f", "jfk_temp_f", "bos_temp_f"]
    table = read_csv(bucket, CT_WEATHER_KEY)
    if table is None or len(table) == 0:
        return pd.DataFrame(columns=names)
    table["scraped_hour_utc"] = pd.to_datetime(table["scraped_hour_utc"], utc=True)
    table["target_hour_utc"] = pd.to_datetime(table["target_hour_utc"], utc=True)
    newest = table.sort_values("scraped_hour_utc").drop_duplicates(["station", "target_hour_utc"], keep="last")
    out = pd.DataFrame(index=pd.DatetimeIndex(sorted(newest["target_hour_utc"].unique()), name="target_hour_utc"))
    ct = newest[newest["station"].isin(STATION_WEIGHTS)]
    for column, name in [("temp_f", "ct_temp_f"), ("dewpoint_f", "ct_dewpoint_f")]:
        wide = ct.pivot(index="target_hour_utc", columns="station", values=column)
        weights = pd.Series(STATION_WEIGHTS)[wide.columns]
        total = (wide * weights).sum(axis=1, min_count=1)
        out[name] = total / wide.notna().mul(weights).sum(axis=1).replace(0, np.nan)
    for station, name in [("JFK", "jfk_temp_f"), ("BOS", "bos_temp_f")]:
        out[name] = newest[newest["station"] == station].set_index("target_hour_utc")["temp_f"]
    return out[names]


def forecast_range(bucket, card, X, preds, targets):
    """The low and high ends of your 80% range for each hour, plus how they were made.
    Returns (low, high, method), or (None, None, "none") when your model card asks for no range."""
    method = str(card.get("range_method", "none"))
    if method == "quantile_models":
        if not bucket.blob(LOW_MODEL_KEY).exists() or not bucket.blob(HIGH_MODEL_KEY).exists():
            raise ValueError("Your model card says quantile_models, but model_low.joblib or model_high.joblib is missing.")
        low_model = joblib.load(io.BytesIO(bucket.blob(LOW_MODEL_KEY).download_as_bytes()))
        high_model = joblib.load(io.BytesIO(bucket.blob(HIGH_MODEL_KEY).download_as_bytes()))
        low = low_model.predict(X[FEATURES])
        high = high_model.predict(X[FEATURES])
    elif method == "offsets_by_hour":
        offsets = card["range_offsets_mw"]
        if len(offsets["low"]) != 24 or len(offsets["high"]) != 24:
            raise ValueError("range_offsets_mw needs 24 low and 24 high offsets, one per Connecticut hour (0 = midnight).")
        local_hours = pd.DatetimeIndex(targets).tz_convert("America/New_York").hour
        low = preds + np.array(offsets["low"], dtype=float)[local_hours]
        high = preds + np.array(offsets["high"], dtype=float)[local_hours]
    else:
        return None, None, "none"

    # "range_widen": 1.3 stretches the range 30% around the expected value, no retraining (Lab 6: if the leaderboard
    # says only 65% of actual loads landed inside your 80% range, your range is too narrow)
    widen = float(card.get("range_widen", 1.0))
    low = preds - widen * (preds - low)
    high = preds + widen * (high - preds)
    # The expected value always sits inside its own range
    low = np.minimum(low, preds)
    high = np.maximum(high, preds)
    return low, high, method


def fill_inputs(X, load):
    """Honest gap filling (rare): a missing weather value takes its neighbor; a missing yesterday load the newest load."""
    X[WEATHER] = X[WEATHER].interpolate(limit_direction="both")
    X["load_yesterday_mw"] = X["load_yesterday_mw"].fillna(float(load.iloc[-1]))
    return X


def hrrr_run_tables(card, now):
    """The recent HRRR runs from the class file, newest last: a list of weather tables indexed by UTC hour.
    Returns an empty list when the file is missing or stale."""
    try:
        runs = pd.read_csv(HRRR_RUNS_URL)
    except Exception as err:
        logging.warning("HRRR runs not available: %s", err)
        return []
    runs["init_utc"] = pd.to_datetime(runs["init_utc"], utc=True)
    runs["hour_utc"] = pd.to_datetime(runs["hour_utc"], utc=True)
    inits = sorted(runs["init_utc"].unique())
    if len(inits) == 0 or pd.Timestamp(now) - pd.Timestamp(inits[-1]) > pd.Timedelta(hours=HRRR_MAX_AGE_HOURS):
        return []
    n_runs = int(card.get("hrrr_runs", 8))
    tables = []
    for init in inits[-n_runs:]:
        tables.append(runs[runs["init_utc"] == init].set_index("hour_utc")[WEATHER])
    return tables


def forest_range(bucket, card, X):
    """Final project: ONE model on the NWS forecast only. If model.predict(X) gives 3 columns, they are low, expected
    and high; otherwise the model is a quantile forest and is asked for its 10th, 50th and 90th percentiles. Each row is
    sorted so low <= expected <= high, then range_widen stretches low and high around the expected value.
    Returns (low, expected, high)."""
    if not bucket.blob(QRF_MODEL_KEY).exists():
        raise ValueError("Your model card says " + str(card.get("range_method")) + ", but energy/models/model_qrf.joblib is missing.")
    model = joblib.load(io.BytesIO(bucket.blob(QRF_MODEL_KEY).download_as_bytes()))
    # The inputs this model was trained on: the model card's "features" list (the 11 Lab 5 inputs if it has none)
    model_features = list(card.get("features", FEATURES))
    missing = [name for name in model_features if name not in X.columns]
    if missing:
        raise ValueError("forecast-24h doesn't build these inputs from your model card: " + str(missing))
    def three_columns(inputs):
        """low, expected, high for each row, sorted"""
        q = np.asarray(model.predict(inputs), dtype=float)
        if q.ndim != 2 or q.shape[1] != 3:
            q = np.asarray(model.predict(inputs, quantiles=[0.1, 0.5, 0.9]), dtype=float)
        return np.sort(q, axis=1)

    q = three_columns(X[model_features])
    low = q[:, 0]
    expected = q[:, 1]
    high = q[:, 2]
    widen = float(card.get("range_widen", 1.0))
    low_side = widen * (expected - low)
    high_side = widen * (high - expected)
    # Optional (model card "nudge_temp_sd", e.g. 2.7 °F): the weather forecast is off by a few degrees, so nudge every
    # temperature input up or down together, many times, and widen each hour's range by how far the expected value moves
    nudge_sd = float(card.get("nudge_temp_sd", 0))
    if nudge_sd > 0:
        rng = np.random.default_rng(int(card.get("seed", 42)))
        moved = []
        for draw in range(int(card.get("nudge_draws", 30))):
            nudged = X[model_features].copy()
            shift = rng.normal(0, nudge_sd, len(nudged))
            for column in TEMPERATURE_INPUTS:
                if column in nudged.columns:
                    nudged[column] = nudged[column] + shift
            moved.append(three_columns(nudged)[:, 1])
        weather_spread = 1.2816 * np.std(np.array(moved), axis=0)        # 1.2816 = how far the 10th/90th sit from the middle
        low_side = np.sqrt(low_side ** 2 + weather_spread ** 2)
        high_side = np.sqrt(high_side ** 2 + weather_spread ** 2)
    return expected - low_side, expected, expected + high_side


def pooled_range(bucket, card, targets, load, nws_weather, now):
    """Lab 6C: the quantile forest's 20 percentiles (2.5%, 7.5%, ..., 97.5%) for every HRRR run that covers an hour, pooled
    into one pile per hour; low, expected and high are the 10th, 50th and 90th percentile of the pile ("hazen": each
    value is the middle of its 5% slice, so a single run gives back exactly its own 10th, 50th and 90th).
    The model's uncertainty is inside each run's percentiles; the weather's is in how much the runs disagree.
    Returns (low, expected, high, info)."""
    if not bucket.blob(QRF_MODEL_KEY).exists():
        raise ValueError("Your model card says hrrr_qrf_pooled, but energy/models/model_qrf.joblib is missing.")
    qrf = joblib.load(io.BytesIO(bucket.blob(QRF_MODEL_KEY).download_as_bytes()))
    tables = hrrr_run_tables(card, now)
    weather_source = "hrrr"
    if len(tables) == 0:
        tables = [nws_weather]
        weather_source = "nws_fallback"
    piles = []
    for i in range(len(targets)):
        piles.append([])
    for table in tables:
        covered = np.asarray(targets.isin(table.index))
        if not covered.any():
            continue
        X = fill_inputs(make_features(targets[covered], table, load), load)
        if X[WEATHER].isna().any().any():
            continue
        q = qrf.predict(X[FEATURES], quantiles=POOL_QUANTILES)
        positions = np.where(covered)[0]
        for j in range(len(positions)):
            piles[positions[j]].extend(q[j])
    # an hour no HRRR run covers (should not happen) gets the NWS forecast
    empty = []
    for i in range(len(targets)):
        if len(piles[i]) == 0:
            empty.append(i)
    if len(empty) > 0:
        X = fill_inputs(make_features(targets[empty], nws_weather, load), load)
        q = qrf.predict(X[FEATURES], quantiles=POOL_QUANTILES)
        for j in range(len(empty)):
            piles[empty[j]].extend(q[j])
    low = np.zeros(len(targets))
    expected = np.zeros(len(targets))
    high = np.zeros(len(targets))
    runs_per_hour = []
    for i in range(len(targets)):
        low[i], expected[i], high[i] = np.quantile(piles[i], [0.1, 0.5, 0.9], method="hazen")
        runs_per_hour.append(len(piles[i]) // len(POOL_QUANTILES))
    widen = float(card.get("range_widen", 1.0))
    low = expected - widen * (expected - low)
    high = expected + widen * (high - expected)
    info = {"weather_source": weather_source, "hrrr_runs_read": len(tables) if weather_source == "hrrr" else 0,
            "runs_per_hour_min": int(min(runs_per_hour)), "runs_per_hour_max": int(max(runs_per_hour))}
    return low, expected, high, info


def run_once(dry_run: bool = False) -> dict:
    bucket = storage.Client(project=PROJECT_ID).bucket(GCS_BUCKET)
    if not bucket.blob(MODEL_KEY).exists():
        return {"status": "no_model", "message": f"Upload your model to gs://{GCS_BUCKET}/{MODEL_KEY} (Lab 5, Part 7)."}
    model = joblib.load(io.BytesIO(bucket.blob(MODEL_KEY).download_as_bytes()))
    card = {}
    if bucket.blob(CARD_KEY).exists():
        card = json.loads(bucket.blob(CARD_KEY).download_as_text())
    model_version = str(card.get("trained_at", "unknown"))

    forecasts = read_csv(bucket, WEATHER_KEY)
    if forecasts is None or len(forecasts) == 0:
        return {"status": "no_weather_data", "message": "scraper-weather-forecast has not written its master yet."}
    load_table = read_csv(bucket, LOAD_KEY)
    if load_table is None or len(load_table) == 0:
        return {"status": "no_load_data", "message": "No ISO-NE load yet: add the ISO_NE_USERNAME and ISO_NE_PASSWORD "
                "repository secrets, re-run 'Deploy scraper-isone', and wait for its next run (:15)."}
    load_table["hour_utc"] = pd.to_datetime(load_table["hour_utc"], utc=True)
    load = load_table.set_index("hour_utc")["load_mw_5min_avg"].sort_index()

    now = datetime.now(timezone.utc)
    this_hour = pd.Timestamp(now).floor("h")
    targets = pd.date_range(this_hour + pd.Timedelta(hours=1), periods=HOURS_AHEAD, freq="h")
    weather = latest_weather(forecasts)
    ct_hours_found = 0
    try:                         # final project extras: a problem here only means Bradley's values stand in
        ct = ct_weather(bucket)
        weather = weather.join(ct, how="left")
        ct_hours_found = int(ct.reindex(targets)["ct_temp_f"].notna().sum())
    except Exception as err:
        logging.warning("Connecticut weather not available: %s", err)
    X = make_features(targets, weather, load, extra=True)

    # Honest gap filling (rare): a missing weather value takes its neighbor; a missing yesterday load takes the newest load
    filled_weather = int(X[WEATHER].isna().sum().sum())
    X[WEATHER] = X[WEATHER].interpolate(limit_direction="both")
    filled_load = int(X["load_yesterday_mw"].isna().sum())
    X["load_yesterday_mw"] = X["load_yesterday_mw"].fillna(float(load.iloc[-1]))
    if X[WEATHER].isna().any().any():
        return {"status": "no_weather_data", "message": "The weather forecast does not cover the next 24 hours yet."}
    for column, bradley in EXTRA_WEATHER.items():                         # Bradley stands in for a missing extra value
        X[column] = X[column].fillna(X[bradley])
    X["solar_proxy"] = solar_proxy(targets, X["cloud_cover_pct"].values)  # again, after cloud cover gaps are filled
    # The newest load that is over right now (a model uses these only if its model card lists them)
    X = X.join(latest_load_inputs(targets, load, load.dropna().index.max()))
    isone_hours = 0
    if "isone_forecast_mw" in card.get("features", []):          # only models trained with ISO-NE's forecast read it
        try:
            X = X.join(isone_inputs(bucket, targets, this_hour))
            isone_hours = int(X["isone_forecast_mw"].notna().sum())
        except Exception as err:
            logging.warning("ISO-NE forecast not available: %s", err)
            X = X.drop(columns=ISONE_FEATURES, errors="ignore").join(pd.DataFrame(index=targets, columns=ISONE_FEATURES, dtype=float))
        # Honest gap filling (rare): no ISO version for an hour -> yesterday's load stands in, as in training
        X["isone_forecast_mw"] = X["isone_forecast_mw"].fillna(X["load_yesterday_mw"])
        X["isone_age_hours"] = X["isone_age_hours"].fillna(ISONE_MISSING_AGE_HOURS)

    preds = model.predict(X[FEATURES])
    range_error = ""
    pool_info = {}
    try:                         # a broken range never costs you your expected-value forecast
        if str(card.get("range_method", "none")) in ["range_model", "quantile_forest"]:
            low, model_expected, high = forest_range(bucket, card, X)
            preds = model_expected             # the expected value is the model's middle column (its median)
            range_method = str(card.get("range_method"))
        elif str(card.get("range_method", "none")) == "hrrr_qrf_pooled":
            low, pooled_expected, high, pool_info = pooled_range(bucket, card, targets, load, weather, now)
            preds = pooled_expected            # the expected value is the middle of the pooled pile
            range_method = "hrrr_qrf_pooled"
            if pool_info["weather_source"] != "hrrr":
                range_method = "hrrr_qrf_pooled_nws"
        else:
            low, high, range_method = forecast_range(bucket, card, X, preds, targets)
    except Exception as err:
        low, high, range_method = None, None, "error"
        range_error = str(err)[:300]
    if low is None:
        low = np.full(len(preds), np.nan)
        high = np.full(len(preds), np.nan)
    out = pd.DataFrame({"issued_at_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "target_hour_utc": targets.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        "lead_hours": np.arange(1, HOURS_AHEAD + 1),
                        "predicted_low_mw": np.round(low, 1),
                        "predicted_load_mw": np.round(preds, 1),
                        "predicted_high_mw": np.round(high, 1),
                        "model_version": model_version,
                        "range_method": range_method})
    summary = {"status": "ok", "model_version": model_version, "first_target_utc": out["target_hour_utc"].iloc[0],
               "last_target_utc": out["target_hour_utc"].iloc[-1], "newest_load_hour_utc": load.index.max().isoformat(),
               "min_mw": float(out["predicted_load_mw"].min()), "max_mw": float(out["predicted_load_mw"].max()),
               "range_method": range_method,
               "avg_range_width_mw": None if low is None or np.isnan(low).all() else round(float(np.mean(high - low)), 1),
               "range_error": range_error,
               "filled_weather_values": filled_weather, "filled_load_values": filled_load,
               "model_features": len(card.get("features", FEATURES)), "ct_weather_hours": ct_hours_found,
               "isone_forecast_hours": isone_hours}
    summary.update(pool_info)
    if dry_run:
        summary["dry_run"] = True
        return summary

    csv_text = out.to_csv(index=False)
    bucket.blob(LATEST_KEY).upload_from_string(csv_text, content_type="text/csv")
    bucket.blob(f"{HISTORY_PREFIX}/{now.strftime('%Y%m%dT%H%M%SZ')}.csv").upload_from_string(csv_text, content_type="text/csv")
    old = read_csv(bucket, MASTER_KEY)
    master = out if old is None else pd.concat([old, out], ignore_index=True)
    bucket.blob(MASTER_KEY).upload_from_string(master.to_csv(index=False), content_type="text/csv")
    summary["latest_key"] = LATEST_KEY
    summary["master_rows"] = int(len(master))
    return summary


def forecast_24h_http(request):
    try:
        body = request.get_json(silent=True) or {}
        result = run_once(dry_run=bool(body.get("dry_run", False)))
        logging.info(json.dumps(result)[:500])
        return (json.dumps(result), 200, {"Content-Type": "application/json"})
    except Exception as e:
        logging.error("Error: %s\n%s", e, traceback.format_exc())
        return (json.dumps({"status": "error", "error": str(e)}), 500, {"Content-Type": "application/json"})
