#!/usr/bin/env python
"""
optuna_retailIR_murphet_vs_prophet.py
Compare Murphet (MAP, Optuna‑tuned) vs Prophet on
the inverted Retail Inventories‑to‑Sales ratio (monthly).

Outputs
-------
• rolling‑origin CV          – best RMSE per model
• 24‑month hold‑out metrics  – RMSE, MAE, SMAPE, MAPE
• Ljung–Box p‑values, ACF & cumulative‑Q plots
• forecast plot (train fit + 24‑mo forecast)
"""

# ────────────────────────── imports ──────────────────────────
import os, warnings, time, pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
import optuna
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

os.environ["STAN_NUM_THREADS"] = "16"            # 32‑core box → 16 threads total
from murphet import fit_churn_model              # <-- use the updated package

# ─────────────────────── helpers / metrics ──────────────────────
rmse  = lambda a, f: np.sqrt(mean_squared_error(a, f))
smape = lambda a, f: 100 * np.mean(2*np.abs(f - a)/(np.abs(a)+np.abs(f)))
mape  = lambda a, f: 100 * np.mean(np.abs((a - f) / a))

# ─────────────────────────── config ────────────────────────────
ROOT            = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH        = ROOT / "data" / "RETAILIRNSA.csv"     # monthly FRED series
INIT_YEARS      = 8
CV_HORIZON      = 6           # months predicted in each fold
CV_STEP         = 3           # months between folds
HOLD_OUT_MO     = 12          # final test window
LB_LAGS         = 12
SEED            = 42
TRIALS_MUR      = 15
TRIALS_PROP     = 15

# ──────────────────────── load data ────────────────────────────
df = (pd.read_csv(CSV_PATH, parse_dates=["ds"])
        .sort_values("ds")
        .reset_index(drop=True))

df["y"] = (1/df["y"]).clip(1e-6, 1-1e-6)          # invert, keep 0<y<1
df["t"] = np.arange(len(df))

t_all, y_all = df["t"].values, df["y"].values
first_test   = INIT_YEARS * 12
fold_starts  = list(range(first_test,
                          len(df) - HOLD_OUT_MO - CV_HORIZON + 1,
                          CV_STEP))
if not fold_starts:
    raise SystemExit("Dataset too short for this CV setup.")

# ──────────────── Optuna search spaces ─────────────────────────
def mur_cfg(trial):
    # seasonal structure ----------------------------------------------------
    periods, harms = [12.0], [trial.suggest_int("harm_year", 1, 4)]
    if trial.suggest_categorical("add_half", [0, 1]):
        periods.append(6.0)
        harms.append(trial.suggest_int("harm_half", 1, 3))
    if trial.suggest_categorical("add_qtr", [0, 1]):
        periods.append(3.0)
        harms.append(trial.suggest_int("harm_qtr", 1, 3))

    return dict(
        periods            = periods,
        num_harmonics      = harms,
        n_changepoints     = trial.suggest_int("n_cp", 0, 6),
        delta_scale        = trial.suggest_float("delta", 0.01, 0.22, log=True),
        # NEW global prior on seasonal size
        season_scale       = trial.suggest_float("season_scale", 0.1, 1.5),
        inference          = "map",          # MAP only
        chains             = 2,
        iter               = 3000,
        warmup             = 0,
        threads_per_chain  = 16,
        seed               = SEED,
    )

def prop_cfg(trial):
    return dict(
        changepoint_prior_scale = trial.suggest_float("cp_scale", 0.005, 0.4, log=True),
        seasonality_prior_scale = trial.suggest_float("sea_scale", 0.1,   15, log=True),
        yearly_seasonality=True, weekly_seasonality=False,
        daily_seasonality=False, interval_width=0.8,
    )

def cv_objective(make_cfg, is_murphet):
    def _obj(trial):
        cfg = make_cfg(trial)
        errs = []
        for idx in fold_starts:
            tr_end, te_end = idx, idx + CV_HORIZON
            try:
                if is_murphet:
                    mod  = fit_churn_model(t=t_all[:tr_end], y=y_all[:tr_end], **cfg)
                    pred = mod.predict(t_all[tr_end:te_end])
                else:
                    mod = Prophet(**cfg)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        mod.fit(df.iloc[:tr_end][["ds", "y"]])
                    fut  = mod.make_future_dataframe(CV_HORIZON, freq="MS").iloc[-CV_HORIZON:]
                    pred = mod.predict(fut)["yhat"].values
                errs.append(rmse(y_all[tr_end:te_end], pred))
            except RuntimeError:
                return 1e6
        return float(np.mean(errs))
    return _obj

# ───────────── 1) tune Murphet (MAP) ─────────────
mur_study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
t0 = time.time()
mur_study.optimize(cv_objective(mur_cfg, True),
                   n_trials=TRIALS_MUR, show_progress_bar=True)
mur_time = time.time() - t0
mur_cfg_best = mur_cfg(mur_study.best_trial)

# ───────────── 2) tune Prophet ───────────────────
prop_study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=SEED))
t0 = time.time()
prop_study.optimize(cv_objective(prop_cfg, False),
                    n_trials=TRIALS_PROP, show_progress_bar=True)
prop_time = time.time() - t0
prop_cfg_best = prop_cfg(prop_study.best_trial)

# ───────────── 3)  final MAP refit & hold‑out ────
train_ho, test_ho = df.iloc[:-HOLD_OUT_MO], df.iloc[-HOLD_OUT_MO:]

mur_fit  = fit_churn_model(t=train_ho["t"], y=train_ho["y"], **mur_cfg_best)
mur_pred = mur_fit.predict(test_ho["t"])
mur_res  = train_ho["y"].values - mur_fit.predict(train_ho["t"])

prop_fit = Prophet(**prop_cfg_best)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    prop_fit.fit(train_ho[["ds", "y"]])
fut        = prop_fit.make_future_dataframe(HOLD_OUT_MO, freq="MS").iloc[-HOLD_OUT_MO:]
prop_df    = prop_fit.predict(fut)
prop_pred  = prop_df["yhat"].values
prop_low   = prop_df["yhat_lower"].values
prop_upp   = prop_df["yhat_upper"].values
prop_res   = train_ho["y"].values - prop_fit.predict(train_ho[["ds"]])["yhat"].values

# ───────────── 4) console report ────────────────
print("\n=== Best CV RMSE ===")
print(f"Murphet : {mur_study.best_value:.4f}")
print(f"Prophet : {prop_study.best_value:.4f}")

print("\n=== Optimisation time ===")
print(f"Murphet : {mur_time:5.1f}s ({mur_time/TRIALS_MUR:.2f}s / trial)")
print(f"Prophet : {prop_time:5.1f}s ({prop_time/TRIALS_PROP:.2f}s / trial)")

lb = lambda r: acorr_ljungbox(r, lags=LB_LAGS, return_df=True)["lb_pvalue"]
print(f"\n=== Ljung‑Box p‑values (first 12 lags) ===")
print(pd.concat([lb(mur_res), lb(prop_res)], axis=1,
                keys=["Murphet", "Prophet"]).head(12)
      .applymap(lambda p: f"{p:.2e}"))

metrics = pd.DataFrame({
    "Model": ["Murphet(MAP)", "Prophet"],
    "RMSE" : [rmse(test_ho["y"], mur_pred), rmse(test_ho["y"], prop_pred)],
    "MAE"  : [mean_absolute_error(test_ho["y"], mur_pred),
              mean_absolute_error(test_ho["y"], prop_pred)],
    "SMAPE": [smape(test_ho["y"], mur_pred), smape(test_ho["y"], prop_pred)],
    "MAPE" : [mape(test_ho["y"], mur_pred),  mape(test_ho["y"], prop_pred)],
}).set_index("Model")
print(f"\n=== {HOLD_OUT_MO}-month hold‑out metrics ===")
print(metrics.round(4))

# ───────────── 5) visuals ───────────────────────
plt.style.use("seaborn-v0_8")

# forecast plot
fig1, ax1 = plt.subplots(figsize=(13, 6))
ax1.plot(df["ds"], df["y"], "k.-", label="Actual", lw=1)
ax1.plot(train_ho["ds"], mur_fit.predict(train_ho["t"]),
         color="royalblue", lw=1.6, label="Murphet fit")
ax1.plot(train_ho["ds"], prop_fit.predict(train_ho[["ds"]])["yhat"],
         color="darkorange", lw=1.6, label="Prophet fit")
ax1.plot(test_ho["ds"], mur_pred, "o--", color="royalblue",
         label="Murphet forecast")
ax1.plot(test_ho["ds"], prop_pred, "o--", color="darkorange",
         label="Prophet forecast")
ax1.fill_between(test_ho["ds"], prop_low, prop_upp,
                 color="darkorange", alpha=0.25, label="Prophet 80% CI")
ax1.axvline(test_ho["ds"].iloc[0], color="gray", ls="dotted")
ax1.set_title(f"Retail I/S ratio (inverted) – {HOLD_OUT_MO}‑month hold‑out")
ax1.set_ylabel("1 / Inventories‑to‑Sales")
ax1.legend(); fig1.tight_layout()

# residual ACFs
fig2, axes = plt.subplots(2, 1, figsize=(10, 6))
plot_acf(mur_res,  lags=LB_LAGS, ax=axes[0]); axes[0].set_title("Murphet residual ACF")
plot_acf(prop_res, lags=LB_LAGS, ax=axes[1]); axes[1].set_title("Prophet residual ACF")
fig2.tight_layout()

# cumulative Ljung‑Box Q
fig_q, ax_q = plt.subplots(figsize=(8, 4))
lags = np.arange(1, LB_LAGS+1)
ax_q.plot(lags, acorr_ljungbox(mur_res,  lags=LB_LAGS, return_df=True)["lb_stat"],
          lw=2, label="Murphet",  color="royalblue")
ax_q.plot(lags, acorr_ljungbox(prop_res, lags=LB_LAGS, return_df=True)["lb_stat"],
          lw=2, label="Prophet", color="darkorange")
ax_q.set_xlabel("Lag"); ax_q.set_ylabel("Cumulative Ljung‑Box Q")
ax_q.set_title(f"Cumulative Ljung‑Box Q up to lag {LB_LAGS}")
ax_q.legend(); fig_q.tight_layout()

plt.show()
