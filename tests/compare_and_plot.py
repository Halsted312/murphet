#!/usr/bin/env python
"""
compare_and_plot_optuna.py
Optuna‑tuned Murphet (10 trials) vs Prophet, 12‑month forecast plot
"""

import os, time, warnings, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import optuna
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ─── Murphet wrapper (MAP, 8 threads) ──────────────────────────────
os.environ["STAN_NUM_THREADS"] = "8"
from murphet.churn_model_parallel import fit_churn_model

# ─── helper metrics ────────────────────────────────────────────────
def rmse(a, b): return np.sqrt(mean_squared_error(a, b))

# ------------------------------------------------------------------
#  USER CONFIG
# ------------------------------------------------------------------
CSV_PATH          = "/home/halsted/Documents/python/murphet/data/churn_example.csv"
INIT_TRAIN_YRS    = 3
HORIZON_MONTHS    = 3
STEP_MONTHS       = 3
FORECAST_HORIZON  = 12
SEED              = 42
N_TRIALS          = 66

# ------------------------------------------------------------------
#  LOAD DATA
# ------------------------------------------------------------------
df = (pd.read_csv(CSV_PATH, parse_dates=["ds"])
        .sort_values("ds")
        .reset_index(drop=True))
df["t"] = np.arange(len(df))
y = df["y"].values

first_test_idx = 12 * INIT_TRAIN_YRS
fold_starts = list(range(first_test_idx,
                         len(df) - HORIZON_MONTHS + 1,
                         STEP_MONTHS))
print(f"Total folds for CV: {len(fold_starts)}")

# ------------------------------------------------------------------
#  OPTUNA OBJECTIVE
# ------------------------------------------------------------------
# ─── OPTUNA OBJECTIVE ─────────────────────────────────────────────
def objective(trial):
    # 1.  Sample hyper‑parameters
    cfg = dict(
        num_harmonics  = trial.suggest_int("num_harmonics", 1, 7),
        n_changepoints = trial.suggest_int("n_changepoints", 1, 7),
        delta_scale    = trial.suggest_float("delta_scale", 0.002, 0.25, log=True),
        adapt_delta    = trial.suggest_float("adapt_delta", 0.6, 0.99),
        period         = trial.suggest_categorical("period", [12.0, 6.0, 4.0])  # annual, half‑year, quarter
    )

    fold_errors = []
    for start in fold_starts:
        train_end = start
        test_end  = start + HORIZON_MONTHS

        y_train = y[:train_end]
        y_test  = y[start:test_end]
        t_train = np.arange(train_end)
        t_test  = np.arange(start, test_end)

        try:
            mur = fit_churn_model(
                t=t_train, y=y_train,
                chains=2, iter=3000, warmup=1000,
                inference="map",          # MAP keeps trials fast
                threads_per_chain=8,
                **cfg,
                seed=SEED)

            preds = mur.predict(t_test)
            fold_errors.append(rmse(y_test, preds))

        except RuntimeError as e:
            # Stan failed to optimise → penalise trial heavily
            trial.set_user_attr("fail_msg", str(e)[:120])
            return 1e3   # large RMSE

    return float(np.mean(fold_errors))


# ------------------------------------------------------------------
#  RUN OPTUNA STUDY  (unchanged)
# ------------------------------------------------------------------
sampler = optuna.samplers.TPESampler(seed=SEED)
study   = optuna.create_study(direction="minimize", sampler=sampler)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

print("\n=== OPTUNA BEST PARAMETERS ===")
print(json.dumps(study.best_params, indent=2))
print(f"Best CV RMSE: {study.best_value:.4f}")

# ------------------------------------------------------------------
#  REFIT ON FULL DATA WITH **DEDUPLICATED** PARAMS
# ------------------------------------------------------------------
best_cfg   = study.best_params.copy()
period_val = best_cfg.pop("period")      # avoid duplicate keyword

mur_full = fit_churn_model(
    t=df["t"], y=y,
    inference="map",
    threads_per_chain=8,
    period=period_val,
    iter=5000, warmup=50,
    **best_cfg,
    seed=SEED)

# Prophet stays unchanged
prophet_full = Prophet(
    seasonality_prior_scale=10.0,
    changepoint_prior_scale=0.05,
    n_changepoints=min(25, len(df)//2),
    interval_width=0.8)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    prophet_full.fit(df[["ds", "y"]])


# ------------------------------------------------------------------
#  12‑MONTH FORECAST
# ------------------------------------------------------------------
future_dates = pd.date_range(df["ds"].iloc[-1] + pd.DateOffset(months=1),
                             periods=FORECAST_HORIZON, freq="MS")
t_future     = np.arange(len(df), len(df)+FORECAST_HORIZON)

mur_hist   = mur_full.predict(df["t"])
mur_future = mur_full.predict(t_future)

prop_fcst  = prophet_full.predict(
                prophet_full.make_future_dataframe(FORECAST_HORIZON, freq="MS"))
prop_hist   = prop_fcst["yhat"].values[:len(df)]
prop_future = prop_fcst["yhat"].values[len(df):]
prop_low    = prop_fcst["yhat_lower"].values[len(df):]
prop_upp    = prop_fcst["yhat_upper"].values[len(df):]

# ------------------------------------------------------------------
#  PLOT
# ------------------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.style.use("seaborn-v0_8")

plt.plot(df["ds"], y, "k.-", label="Actual", lw=2)
plt.plot(df["ds"], mur_hist, color="royalblue", lw=2, label="Murphet fit")
plt.plot(df["ds"], prop_hist, color="darkorange", lw=2, label="Prophet fit")

plt.plot(future_dates, mur_future, "o--", color="royalblue",
         label="Murphet forecast")
plt.plot(future_dates, prop_future, "o--", color="darkorange",
         label="Prophet forecast")
plt.fill_between(future_dates, prop_low, prop_upp,
                 color="darkorange", alpha=0.25, label="Prophet 80% CI")

plt.axvline(df["ds"].iloc[-1], color="gray", ls="dotted")
plt.title("Churn Rate – Actuals, Fits, and 12‑Month Forecast (Optuna‑tuned Murphet)")
plt.ylabel("Churn rate" + (" (%)" if y.max() > 1 else ""))
plt.xlabel("Date")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_forecast.png", dpi=300)
plt.show()

# ------------------------------------------------------------------
#  IN‑SAMPLE ERROR
# ------------------------------------------------------------------
print("\n=== In‑sample error ===")
print(f"Murphet  RMSE={rmse(y, mur_hist):.4f}  "
      f"MAE={mean_absolute_error(y, mur_hist):.4f}")
print(f"Prophet  RMSE={rmse(y, prop_hist):.4f}  "
      f"MAE={mean_absolute_error(y, prop_hist):.4f}")
