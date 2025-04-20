#!/usr/bin/env python
"""
optuna_murphet_vs_prophet_diag.py
Tuned Murphet vs Prophet with rich diagnostics:
  • Rolling-origin CV (Optuna)
  • Hold-out forecast
  • Per-lag Ljung-Box table
  • Residual ACF plots
  • Cumulative Ljung-Box statistic plot
"""

# ---------------- Imports ----------------
import os, warnings, pathlib, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
import optuna
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

os.environ["STAN_NUM_THREADS"] = "4"
from murphet import fit_churn_model

# ---------------- Metrics ----------------
rmse  = lambda a, f: np.sqrt(mean_squared_error(a, f))
smape = lambda a, f: 100 * np.mean(2 * np.abs(f - a) / (np.abs(a) + np.abs(f)))
mape  = lambda a, f: 100 * np.mean(np.abs((a - f) / a))

# ---------------- Config -----------------
ROOT            = pathlib.Path(__file__).resolve().parent.parent
CSV             = ROOT / "data" / "test.csv"
INIT_YEARS      = 3
CV_HORIZON      = 3        # months predicted per fold
CV_STEP         = 2        # slide step (months)
HOLD_OUT_MONTHS = 9        # final test window
LB_LAGS         = 12       # <-> ACF length & Ljung-Box lags
SEED            = 42
TRIALS_MUR      = 200
TRIALS_PROP     = 200

# ---------------- Data -------------------
df = pd.read_csv(CSV, parse_dates=["ds"]).sort_values("ds").reset_index(drop=True)
if "y" not in df.columns:
    df["y"] = df["casual"] / df["cnt"]
df["t"] = np.arange(len(df))

t_all, y_all = df["t"].values, df["y"].values
first_test   = INIT_YEARS * 12
fold_starts  = list(range(first_test,
                          len(df) - HOLD_OUT_MONTHS - CV_HORIZON + 1,
                          CV_STEP))
if not fold_starts:
    raise SystemExit("Dataset too short for this CV setup.")

# ------------ Optuna config builder (Murphet, MAP only) ------------
def mur_cfg(trial):
    # --- seasonal structure ----------------------------------------
    add_half = trial.suggest_categorical("half_year", [0, 1])
    periods  = [12.0] if add_half == 0 else [12.0, 6.0]

    harms = [trial.suggest_int("harm_year", 1, 4)]
    if add_half:
        harms.append(trial.suggest_int("harm_half", 1, 3))

    # --- trend & prior hyper-params --------------------------------
    n_cp  = trial.suggest_int("n_cp", 0, 6)
    delta = trial.suggest_float("delta", 0.002, 0.3, log=True)

    # --- still tune adapt_delta (even though MAP ignores it) -------
    adapt = trial.suggest_float("adapt_delta", 0.85, 0.99)

    cfg = dict(
        periods         = periods,
        num_harmonics   = harms,
        n_changepoints  = n_cp,
        delta_scale     = delta,
        inference       = "map",      # <- fixed
        iter            = 3000,       # <- fixed iteration budget
        warmup          = 0,          # MAP ignores warm-up
        adapt_delta     = adapt,      # no effect for MAP, but kept for parity
        threads_per_chain = 16,       # <- always use 16 threads
        seed            = SEED,
    )
    return cfg

def prop_cfg(trial):
    return dict(
        changepoint_prior_scale = trial.suggest_float("cp_scale", 0.001, 0.5, log=True),
        seasonality_prior_scale = trial.suggest_float("sea_scale", 0.01, 20, log=True),
        yearly_seasonality=True, weekly_seasonality=False,
        daily_seasonality=False, interval_width=0.8,
    )

def cv_objective(make_cfg, is_murphet):
    def _obj(trial):
        cfg, errs = make_cfg(trial), []
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

# ------------------ 1)  Tune Murphet (fast engines) ------------------
mur_study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
t0 = time.time()
mur_study.optimize(cv_objective(mur_cfg, True),
                   n_trials=TRIALS_MUR, show_progress_bar=True)
mur_time = time.time() - t0

# best structural hyper-params found by Optuna
mur_cfg_best = mur_cfg(mur_study.best_trial)

# ------------------ 2)  Tune Prophet -------------------------------
prop_study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=SEED))
t0 = time.time()
prop_study.optimize(cv_objective(prop_cfg, False),
                    n_trials=TRIALS_PROP, show_progress_bar=True)
prop_time = time.time() - t0
prop_cfg_best = prop_cfg(prop_study.best_trial)

# ---------------------------------------------------------------
# 2-A)  Quick NUTS fine-tuning of adapt_delta (5 trials)
#       • structural params are fixed from mur_cfg_best
# ---------------------------------------------------------------
def nuts_only_cfg(trial):
    # copy the MAP-found structure
    cfg = mur_cfg_best.copy()

    # upgrade to NUTS & allow only HMC controls to vary
    cfg.update({
        "inference":      "nuts",
        "chains":         4,          # fewer chains for speed in CV
        "iter":           2000,       # 1000 warm-up + 1000 sampling
        "warmup":         1000,
        "adapt_delta":    trial.suggest_float("adapt_delta", 0.88, 0.98),
        "threads_per_chain": 4,
    })
    return cfg


print("\n=== Mini NUTS tuning (adapt_delta) ===")
nuts_study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=SEED))
nuts_study.optimize(cv_objective(nuts_only_cfg, True),
                    n_trials=11, show_progress_bar=True)

# keep the best adapt_delta
if len(nuts_study.best_trials) == 0:
    best_ad = 0.9333
    print("Mini-NUTS tuning failed; falling back to adapt_delta=0.95")
else:
    best_ad = nuts_study.best_trial.params["adapt_delta"]
print(f"Chosen adapt_delta = {best_ad:.3f}")

# ------------------ 3)  Refit Murphet with full NUTS ---------------
#   • keep the seasonal / changepoint settings from mur_cfg_best
#   • upgrade the inference engine & sampling controls
# ------------------ 3)  Re-fit with full NUTS ---------------------
best_cfg = mur_cfg_best.copy()
best_cfg.update({
    "inference":      "nuts",
    "chains":         8,
    "iter":           4000,      # 1000 warm-up + 3000 sampling
    "warmup":         1000,
    "adapt_delta":    best_ad,   # <- tuned value
    "threads_per_chain": 4,
})

print("\nRefitting Murphet with full NUTS on training set …")
train_ho, test_ho = df.iloc[:-HOLD_OUT_MONTHS], df.iloc[-HOLD_OUT_MONTHS:]

mur_fit  = fit_churn_model(t=train_ho["t"], y=train_ho["y"], **best_cfg)
mur_pred = mur_fit.predict(test_ho["t"])          # full posterior mean by default
mur_res  = train_ho["y"].values - mur_fit.predict(train_ho["t"])

# ------------------ Prophet hold-out fit ---------------------------
prop_fit = Prophet(**prop_cfg_best)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    prop_fit.fit(train_ho[["ds", "y"]])

fut = prop_fit.make_future_dataframe(HOLD_OUT_MONTHS, freq="MS").iloc[-HOLD_OUT_MONTHS:]
prop_df   = prop_fit.predict(fut)
prop_pred = prop_df["yhat"].values
prop_low, prop_upp = prop_df["yhat_lower"].values, prop_df["yhat_upper"].values
prop_res = train_ho["y"].values - prop_fit.predict(train_ho[["ds"]])["yhat"].values

# ------------------ Timing & CV summary ----------------------------
print("\n=== Best CV RMSE ===")
print(f"Murphet : {mur_study.best_value:.4f}")
print(f"Prophet : {prop_study.best_value:.4f}")

print("\n=== Optimisation timing ===")
print(f"Murphet : {mur_time:7.1f} s total ({mur_time/TRIALS_MUR:.2f} s / trial)")
print(f"Prophet : {prop_time:7.1f} s total ({prop_time/TRIALS_PROP:.2f} s / trial)")

# ----- Ljung-Box tables ----------------------
lb_mur  = acorr_ljungbox(mur_res,  lags=LB_LAGS, return_df=True)
lb_prop = acorr_ljungbox(prop_res, lags=LB_LAGS, return_df=True)

def fmt(arr): return [f"{p:.2e}" for p in arr]

print(f"\n=== Ljung-Box p-values up to lag {LB_LAGS} ===")
tbl = pd.DataFrame({
    "Lag": np.arange(1, LB_LAGS + 1),
    "Murphet_p": fmt(lb_mur["lb_pvalue"].values),
    "Prophet_p": fmt(lb_prop["lb_pvalue"].values),
})
print(tbl.head(12).to_string(index=False))

# ------------- Metrics table -----------------
metrics = pd.DataFrame({
    "Model": ["Murphet (tuned)", "Prophet (tuned)"],
    "RMSE" : [rmse(test_ho["y"], mur_pred), rmse(test_ho["y"], prop_pred)],
    "MAE"  : [mean_absolute_error(test_ho["y"], mur_pred),
              mean_absolute_error(test_ho["y"], prop_pred)],
    "SMAPE": [smape(test_ho["y"], mur_pred), smape(test_ho["y"], prop_pred)],
    "MAPE" : [mape(test_ho["y"], mur_pred),  mape(test_ho["y"], prop_pred)],
}).set_index("Model")
print(f"\n=== {HOLD_OUT_MONTHS}-month hold-out metrics ===")
print(metrics.round(4))

# ------------- Forecast plot -----------------
plt.style.use("seaborn-v0_8")
fig1, ax1 = plt.subplots(figsize=(12, 6))
ax1.plot(df["ds"], df["y"], "k.-", label="Actual", lw=1.2)
ax1.plot(train_ho["ds"], mur_fit.predict(train_ho["t"]), color="royalblue", lw=1.8, label="Murphet fit")
ax1.plot(train_ho["ds"], prop_fit.predict(train_ho[["ds"]])["yhat"], color="darkorange", lw=1.8, label="Prophet fit")
ax1.plot(test_ho["ds"], mur_pred, "o--", color="royalblue", label="Murphet forecast")
ax1.plot(test_ho["ds"], prop_pred, "o--", color="darkorange", label="Prophet forecast")
ax1.fill_between(test_ho["ds"], prop_low, prop_upp, color="darkorange", alpha=0.25, label="Prophet 80% CI")
ax1.axvline(test_ho["ds"].iloc[0], color="gray", ls="dotted")
ax1.set_ylabel("Rate (0-1)")
ax1.set_title(f"Historical fit & {HOLD_OUT_MONTHS}-month hold-out forecast (tuned models)")
ax1.legend()
fig1.tight_layout()

# ------------- Residual ACF plot ------------
fig2, axes = plt.subplots(2, 1, figsize=(10, 6))
plot_acf(mur_res, lags=LB_LAGS, ax=axes[0])
axes[0].set_title("Murphet residual ACF")
plot_acf(prop_res, lags=LB_LAGS, ax=axes[1])
axes[1].set_title("Prophet residual ACF")
plt.tight_layout()

# ------------- Cumulative Ljung-Box plot ----
lb_mur_full  = acorr_ljungbox(mur_res,  lags=LB_LAGS, return_df=True)
lb_prop_full = acorr_ljungbox(prop_res, lags=LB_LAGS, return_df=True)
lags = np.arange(1, LB_LAGS + 1)
Q_mur, Q_prop = lb_mur_full["lb_stat"].values, lb_prop_full["lb_stat"].values

fig_q, ax_q = plt.subplots(figsize=(8, 4))
ax_q.plot(lags, Q_mur,  label="Murphet", lw=2, color="royalblue")
ax_q.plot(lags, Q_prop, label="Prophet", lw=2, color="darkorange")
ax_q.set_xlabel("Lag")
ax_q.set_ylabel("Cumulative Ljung-Box Q statistic")
ax_q.set_title(f"Cumulative Ljung-Box Q up to lag {LB_LAGS}")
ax_q.legend()
fig_q.tight_layout()

# Show all figures
plt.show()