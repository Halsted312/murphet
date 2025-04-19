"""
compare_murphet_prophet_cv.py
Rolling‑origin cross‑validation for churn‑rate forecasting.
"""

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ────────────────────────────────────────────────────────────────────
# 1) Prep Murphet helpers (MAP + 8‑threads)
# ────────────────────────────────────────────────────────────────────
os.environ["STAN_NUM_THREADS"] = "8"   # one‑time global env var BEFORE import
from murphet.churn_model_parallel import fit_churn_model   # your threaded wrapper

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) /
                         (np.abs(y_true) + np.abs(y_pred)))

def mape(y_true, y_pred):
    return 100 * np.mean(np.abs((y_true - y_pred) / y_true))

# ────────────────────────────────────────────────────────────────────
# 2) User config
# ────────────────────────────────────────────────────────────────────
CSV_PATH       = "/home/halsted/Documents/python/murphet/data/churn_example.csv"
INIT_TRAIN_YRS = 3                    # first training window (years)
HORIZON_MONTHS = 6                    # forecast horizon per fold
STEP_MONTHS    = 3                    # slide step
NUM_HARMONICS  = 2
SEED           = 42

# ────────────────────────────────────────────────────────────────────
# 3) Load / prepare data
# ────────────────────────────────────────────────────────────────────
df  = pd.read_csv(CSV_PATH, parse_dates=["ds"]).sort_values("ds").reset_index(drop=True)
df["t"] = np.arange(len(df))          # Murphet numeric index
y      = df["y"].values

# ────────────────────────────────────────────────────────────────────
# 4) Generate fold boundaries
# ────────────────────────────────────────────────────────────────────
first_test_idx  = 12 * INIT_TRAIN_YRS
horizon         = HORIZON_MONTHS
step            = STEP_MONTHS
fold_starts     = list(range(first_test_idx, len(df) - horizon + 1, step))

print(f"Total folds: {len(fold_starts)}  (initial train={INIT_TRAIN_YRS} yrs, "
      f"horizon={horizon} mo, step={step} mo)")

# Storage
records = []

# ────────────────────────────────────────────────────────────────────
# 5) Cross‑validation loop
# ────────────────────────────────────────────────────────────────────
for fold, test_start in enumerate(fold_starts, 1):
    train_end   = test_start
    test_end    = test_start + horizon

    df_train    = df.iloc[:train_end].copy()
    df_test     = df.iloc[test_start:test_end].copy()

    # ── Fit Murphet (MAP for speed) ────────────────────────────────
    t_train = df_train["t"].values
    y_train = df_train["y"].values

    t0 = time.time()
    murphet = fit_churn_model(t=t_train,
                              y=y_train,
                              num_harmonics=NUM_HARMONICS,
                              period=12.0,
                              n_changepoints=3,
                              delta_scale=0.03,
                              chains=2,
                              iter=2000,
                              warmup=800,
                              inference="map",
                              threads_per_chain=8,
                              adapt_delta=0.9,
                              seed=SEED)
    mur_fit_sec = time.time() - t0

    t_test      = df_test["t"].values
    mur_pred    = murphet.predict(t_test, method="mean_params")

    # ── Fit Prophet ────────────────────────────────────────────────
    prop_df = df_train[["ds", "y"]].rename(columns={"y": "y"})
    prophet = Prophet(
        seasonality_prior_scale=10.0,
        changepoint_prior_scale=0.05,
        n_changepoints=min(25, len(prop_df)//2),
        interval_width=0.8
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # suppress fbprophet deprecation msgs
        t0 = time.time()
        prophet.fit(prop_df)
    prop_fit_sec = time.time() - t0

    future_df    = prophet.make_future_dataframe(periods=horizon, freq="MS").iloc[-horizon:]
    prophet_pred = prophet.predict(future_df)["yhat"].values

    # ── Metrics ────────────────────────────────────────────────────
    true = df_test["y"].values

    fold_metrics = {
        "fold": fold,
        "mur_rmse": np.sqrt(mean_squared_error(true, mur_pred)),
        "prop_rmse": np.sqrt(mean_squared_error(true, prophet_pred)),
        "mur_mae": mean_absolute_error(true, mur_pred),
        "prop_mae": mean_absolute_error(true, prophet_pred),
        "mur_smape": smape(true, mur_pred),
        "prop_smape": smape(true, prophet_pred),
        "mur_mape": mape(true, mur_pred),
        "prop_mape": mape(true, prophet_pred),
        "mur_fit_s": mur_fit_sec,
        "prop_fit_s": prop_fit_sec,
        "test_start": df_test["ds"].iloc[0]
    }
    records.append(fold_metrics)
    print(f"Fold {fold:2d}/{len(fold_starts)} "
          f"(train {train_end} pts → test {horizon}) "
          f"Murphet RMSE={fold_metrics['mur_rmse']:.4f}  "
          f"Prophet RMSE={fold_metrics['prop_rmse']:.4f}")

# ────────────────────────────────────────────────────────────────────
# 6) Aggregate results
# ────────────────────────────────────────────────────────────────────
cv = pd.DataFrame(records)

summary = (cv
           .filter(regex="^(mur|prop)_(rmse|mae|smape|mape)$")
           .agg(["mean", "std"])
           .T.rename(columns={"mean": "Mean", "std": "SD"}))

print("\n===========  CROSS‑VALIDATION SUMMARY  ===========")
print(summary.round(4))

# ────────────────────────────────────────────────────────────────────
# 7) Plot average error bars
# ────────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
metric_names = ["rmse", "mae", "smape", "mape"]
x            = np.arange(len(metric_names))
mur_means    = [cv[f"mur_{m}"].mean()  for m in metric_names]
prop_means   = [cv[f"prop_{m}"].mean() for m in metric_names]

plt.bar(x - 0.15, mur_means, width=0.3, label="Murphet", alpha=0.8)
plt.bar(x + 0.15, prop_means, width=0.3, label="Prophet", alpha=0.8)

plt.xticks(x, [m.upper() for m in metric_names])
plt.ylabel("Error")
plt.title("Rolling‑Origin CV – Mean Error (lower is better)")
plt.legend()
plt.tight_layout()
plt.savefig("cv_metric_comparison.png", dpi=300)
plt.show()
