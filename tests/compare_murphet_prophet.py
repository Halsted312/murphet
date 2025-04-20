#!/usr/bin/env python
"""
compare_murphet_prophet_cv.py
Rolling‑origin cross‑validation: Murphet vs Prophet on % churn data
"""

import os, time, warnings, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

# ───────────────────────────────────────────────────────────────────
# 1)  Murphet import  (MAP + multithread by default)
# ───────────────────────────────────────────────────────────────────
os.environ["STAN_NUM_THREADS"] = "8"     # one‑time env var
from murphet import fit_churn_model      # unified wrapper

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) /
                         (np.abs(y_true) + np.abs(y_pred)))

def mape(y_true, y_pred):
    return 100 * np.mean(np.abs((y_true - y_pred) / y_true))

# ───────────────────────────────────────────────────────────────────
# 2)  User config
# ───────────────────────────────────────────────────────────────────
# absolute path:  <repo root>/data/synthetic_churn_rates.csv
HERE      = pathlib.Path(__file__).resolve().parent      # tests/
ROOT      = HERE.parent                                  # repo root
CSV_PATH  = ROOT / "data" / "synthetic_churn_rates.csv"

INIT_TRAIN_YRS = 3                    # first training window (yrs)
HORIZON_MONTHS = 6                    # forecast horizon per fold
STEP_MONTHS    = 2                    # slide step
SEED           = 42

MUR_KWARGS = dict(                    # Murphet hyper‑params
    periods=[12.0],                  # yearly seasonality on monthly data
    num_harmonics=[3],               # 3 Fourier pairs
    n_changepoints=3,
    delta_scale=0.03,
    inference="map",                 # fast optimisation
    chains=2,
    iter=2000,
    warmup=800,
    threads_per_chain=8,
    adapt_delta=0.9,
    seed=SEED,
)

# ───────────────────────────────────────────────────────────────────
# 3)  Load data
# ───────────────────────────────────────────────────────────────────
df = (pd.read_csv(CSV_PATH, parse_dates=["ds"])
        .sort_values("ds")
        .reset_index(drop=True))
df["t"] = np.arange(len(df))          # Murphet numeric index
y       = df["y"].values

# ───────────────────────────────────────────────────────────────────
# 4)  Generate fold boundaries
# ───────────────────────────────────────────────────────────────────
first_test_idx = 12 * INIT_TRAIN_YRS
fold_starts    = list(range(first_test_idx,
                            len(df) - HORIZON_MONTHS + 1,
                            STEP_MONTHS))

print(f"Total folds: {len(fold_starts)}  "
      f"(initial train={INIT_TRAIN_YRS} yrs, "
      f"horizon={HORIZON_MONTHS} mo, step={STEP_MONTHS} mo)")

records = []
# ───────────────────────────────────────────────────────────────────
# 5)  Cross‑validation loop
# ───────────────────────────────────────────────────────────────────
for fold, idx in enumerate(fold_starts, 1):
    train_end = idx
    test_end  = idx + HORIZON_MONTHS

    df_train  = df.iloc[:train_end]
    df_test   = df.iloc[idx:test_end]

    # —— Murphet (MAP) ————————————————————————————
    t0 = time.time()
    mur = fit_churn_model(t=df_train["t"],
                          y=df_train["y"],
                          **MUR_KWARGS)
    mur_fit_s = time.time() - t0

    mur_pred  = mur.predict(df_test["t"])

    # —— Prophet ————————————————————————————————
    prop_df = df_train[["ds", "y"]]
    prophet = Prophet(
        seasonality_prior_scale=10.0,
        changepoint_prior_scale=0.05,
        n_changepoints=min(25, len(prop_df)//2),
        interval_width=0.8,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.time()
        prophet.fit(prop_df)
    prop_fit_s = time.time() - t0

    future = prophet.make_future_dataframe(
        periods=HORIZON_MONTHS, freq="MS").iloc[-HORIZON_MONTHS:]
    prop_pred = prophet.predict(future)["yhat"].values

    # —— Metrics ————————————————————————————————
    true = df_test["y"].values
    rec  = dict(
        fold=fold,
        mur_rmse = np.sqrt(mean_squared_error(true, mur_pred)),
        prop_rmse= np.sqrt(mean_squared_error(true, prop_pred)),
        mur_mae  = mean_absolute_error(true, mur_pred),
        prop_mae = mean_absolute_error(true, prop_pred),
        mur_smape= smape(true, mur_pred),
        prop_smape=smape(true, prop_pred),
        mur_mape = mape(true, mur_pred),
        prop_mape= mape(true, prop_pred),
        mur_fit_s = mur_fit_s,
        prop_fit_s= prop_fit_s,
        test_start= df_test["ds"].iloc[0],
    )
    records.append(rec)

    print(f"Fold {fold:2d}/{len(fold_starts)}  "
          f"Murphet RMSE={rec['mur_rmse']:.4f}  "
          f"Prophet RMSE={rec['prop_rmse']:.4f}")

# ───────────────────────────────────────────────────────────────────
# 6)  Aggregate results
# ───────────────────────────────────────────────────────────────────
cv = pd.DataFrame(records)
summary = (cv.filter(regex="^(mur|prop)_(rmse|mae|smape|mape)$")
             .agg(["mean", "std"])
             .T.rename(columns={"mean": "Mean", "std": "SD"}))

print("\n===========  CROSS‑VALIDATION SUMMARY  ===========")
print(summary.round(4))

# ───────────────────────────────────────────────────────────────────
# 7)  Plot average error bars
# ───────────────────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
metrics = ["rmse", "mae", "smape", "mape"]
x       = np.arange(len(metrics))
plt.bar(x-0.15, [cv[f"mur_{m}"].mean()  for m in metrics],
        width=0.3, label="Murphet", alpha=0.85)
plt.bar(x+0.15, [cv[f"prop_{m}"].mean() for m in metrics],
        width=0.3, label="Prophet", alpha=0.85)

plt.xticks(x, [m.upper() for m in metrics])
plt.ylabel("Error")
plt.title("Rolling‑Origin CV – Mean Error (lower is better)")
plt.legend()
plt.tight_layout()

out = pathlib.Path("tests/cv_metric_comparison.png")
out.parent.mkdir(exist_ok=True, parents=True)
plt.savefig(out, dpi=300)
plt.show()
print(f"\nSaved plot to {out.relative_to(pathlib.Path.cwd())}")
