#!/usr/bin/env python
"""
compare_murphet_prophet_bikes.py
Rolling‑origin CV on bike‑rental “casual share” (casual / cnt)
"""

import os, time, warnings, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ───────────────────────────────────────────────────────────────────
# 1)  Murphet import & threading
# ───────────────────────────────────────────────────────────────────
os.environ["STAN_NUM_THREADS"] = "8"
from murphet import fit_churn_model

def smape(a, f):      # symmetric MAPE
    return 100 * np.mean(2 * np.abs(f - a) / (np.abs(a) + np.abs(f)))

def mape(a, f):
    return 100 * np.mean(np.abs((a - f) / a))

# ───────────────────────────────────────────────────────────────────
# 2)  Configuration
# ───────────────────────────────────────────────────────────────────
# absolute path:  <repo root>/data/synthetic_churn_rates.csv
HERE      = pathlib.Path(__file__).resolve().parent      # tests/
ROOT      = HERE.parent                                  # repo root
CSV_PATH  = ROOT / "data" / "day_bikes.csv"
INIT_TRAIN_D   = 365 * 1           # first training window = 1 year
HORIZON_DAYS   = 180               # 6‑month forecast per fold
STEP_DAYS      = 90                # slide 3 months
SEED           = 42

MUR_KWARGS = dict(
    periods=[7.0, 365.25],         # weekly & yearly seasonality
    num_harmonics=[3, 6],
    n_changepoints=5,
    delta_scale=0.05,
    inference="map",
    chains=2,
    iter=2000,
    warmup=800,
    threads_per_chain=8,
    adapt_delta=0.9,
    seed=SEED,
)

# ───────────────────────────────────────────────────────────────────
# 3)  Load & prepare data
# ───────────────────────────────────────────────────────────────────
df = (pd.read_csv(CSV_PATH, parse_dates=["dteday"])
        .rename(columns={"dteday": "ds"}))
df["y"] = df["casual"] / df["cnt"]
df = df.sort_values("ds").reset_index(drop=True)
df["t"] = np.arange(len(df))       # Murphet numeric index

# ───────────────────────────────────────────────────────────────────
# 4)  Generate fold boundaries
# ───────────────────────────────────────────────────────────────────
first_test_idx = INIT_TRAIN_D
fold_starts    = list(range(first_test_idx,
                            len(df) - HORIZON_DAYS + 1,
                            STEP_DAYS))

print(f"Total folds: {len(fold_starts)}  "
      f"(initial train={INIT_TRAIN_D} d, horizon={HORIZON_DAYS} d, step={STEP_DAYS} d)")

records = []

# ───────────────────────────────────────────────────────────────────
# 5)  CV loop
# ───────────────────────────────────────────────────────────────────
for fold, idx in enumerate(fold_starts, 1):
    train_end = idx
    test_end  = idx + HORIZON_DAYS

    df_train  = df.iloc[:train_end]
    df_test   = df.iloc[idx:test_end]

    # —— Murphet ----------------------------------------------------
    t0 = time.time()
    mur = fit_churn_model(
        t=df_train["t"],
        y=df_train["y"],
        **MUR_KWARGS
    )
    mur_fit_s = time.time() - t0
    mur_pred  = mur.predict(df_test["t"])

    # —— Prophet ----------------------------------------------------
    prop_df = df_train[["ds", "y"]]
    prophet = Prophet(
        seasonality_prior_scale=10,
        changepoint_prior_scale=0.05,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.8,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.time()
        prophet.fit(prop_df)
    prop_fit_s = time.time() - t0

    future = prophet.make_future_dataframe(periods=HORIZON_DAYS, freq="D").iloc[-HORIZON_DAYS:]
    prop_pred = prophet.predict(future)["yhat"].values

    # —— Metrics ----------------------------------------------------
    true = df_test["y"].values
    rec  = dict(
        fold=fold,
        mur_rmse  = np.sqrt(mean_squared_error(true, mur_pred)),
        prop_rmse = np.sqrt(mean_squared_error(true, prop_pred)),
        mur_mae   = mean_absolute_error(true, mur_pred),
        prop_mae  = mean_absolute_error(true, prop_pred),
        mur_smape = smape(true, mur_pred),
        prop_smape= smape(true, prop_pred),
        mur_mape  = mape(true, mur_pred),
        prop_mape = mape(true, prop_pred),
        mur_fit_s = mur_fit_s,
        prop_fit_s= prop_fit_s,
        test_start= df_test["ds"].iloc[0],
    )
    records.append(rec)

    print(f"Fold {fold:2d}/{len(fold_starts)}  "
          f"Murphet RMSE={rec['mur_rmse']:.4f}  "
          f"Prophet RMSE={rec['prop_rmse']:.4f}")

# ───────────────────────────────────────────────────────────────────
# 6)  Aggregate & report
# ───────────────────────────────────────────────────────────────────
cv = pd.DataFrame(records)
summary = (cv.filter(regex="^(mur|prop)_(rmse|mae|smape|mape)$")
             .agg(["mean", "std"])
             .T.rename(columns={"mean": "Mean", "std": "SD"}))

print("\n===========  BIKE SHARE – CROSS‑VALIDATION ===========")
print(summary.round(4))

# ───────────────────────────────────────────────────────────────────
# 7)  Plot comparison
# ───────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
metrics = ["rmse", "mae", "smape", "mape"]
x = np.arange(len(metrics))
plt.bar(x-0.15, [cv[f"mur_{m}"].mean()  for m in metrics],
        width=0.3, label="Murphet", alpha=0.85)
plt.bar(x+0.15, [cv[f"prop_{m}"].mean() for m in metrics],
        width=0.3, label="Prophet", alpha=0.85)
plt.xticks(x, [m.upper() for m in metrics])
plt.ylabel("Error")
plt.title("Bike rentals – Rolling‑Origin CV (lower is better)")
plt.legend()
plt.tight_layout()

out = pathlib.Path("tests/bikes_cv_comparison.png")
out.parent.mkdir(exist_ok=True)
plt.savefig(out, dpi=300)
plt.show()
print(f"\nSaved plot to {out.relative_to(pathlib.Path.cwd())}")
