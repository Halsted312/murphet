#!/usr/bin/env python
"""
optuna_retailIR_murphet_vs_prophet.py

Murphet (beta-likelihood, MAP, Optuna-tuned) vs Prophet
on the inverted "Retailers Inventories-to-Sales ratio" (monthly, FRED).

Outputs
-------
- rolling-origin CV - best RMSE per model
- 12-month hold-out - RMSE, MAE, SMAPE, MAPE
- Ljung-Box table, ACF & cumulative-Q plots
- forecast figure
"""
import os
import warnings
import time
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
import optuna
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

os.environ["STAN_NUM_THREADS"] = "16"  # 32-core → 16 threads/chain
from murphet import fit_churn_model  # new package version

# Helper functions
rmse = lambda a, f: np.sqrt(mean_squared_error(a, f))
smape = lambda a, f: 100 * np.mean(2 * np.abs(f - a) / (np.abs(a) + np.abs(f)))
mape = lambda a, f: 100 * np.mean(np.abs((a - f) / a))

# Configuration
ROOT = pathlib.Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "examples" /  "data" / "RETAILIRNSA.csv"

INIT_YEARS = 12
CV_HORIZON = 6  # months per fold
CV_STEP = 3
HOLD_OUT_MO = 24  # final test window
LB_LAGS = 12
SEED = 42
TRIALS_MUR = 15
TRIALS_PROP = 15

# Load & prep data
df = (pd.read_csv(CSV_PATH, parse_dates=["ds"])
      .sort_values("ds")
      .reset_index(drop=True))
df["y"] = (1 / df["y"]).clip(1e-6, 1 - 1e-6)  # invert, keep (0,1)
df["t"] = np.arange(len(df))

t_all, y_all = df["t"].values, df["y"].values
first_test = INIT_YEARS * 12
fold_starts = list(range(first_test,
                         len(df) - HOLD_OUT_MO - CV_HORIZON + 1,
                         CV_STEP))
if not fold_starts:
    raise SystemExit("Dataset too short for this CV setup.")


# Optuna search spaces
def mur_cfg(trial):
    # Seasonal structure
    periods, harms = [12.0], [trial.suggest_int("harm_year", 1, 4)]
    if trial.suggest_categorical("add_half", [0, 1]):
        periods.append(6.0)
        harms.append(trial.suggest_int("harm_half", 1, 4))
    if trial.suggest_categorical("add_qtr", [0, 1]):
        periods.append(3.0)
        harms.append(trial.suggest_int("harm_qtr", 1, 4))

    # Trend / prior hyper-params
    max_cp = int(len(df) // 24)  # ≈ 1 CP / 2 years
    cfg = dict(
        likelihood="beta",
        periods=periods,
        num_harmonics=harms,
        n_changepoints=trial.suggest_int("n_cp", 1, max_cp),
        delta_scale=trial.suggest_float("delta", 0.005, 0.6, log=True),
        gamma_scale=trial.suggest_float("gamma", 1.0, 15.0),
        season_scale=trial.suggest_float("season_scale", 0.1, 2.0),

        # fast MAP engine for HPO
        inference="map",
        chains=2,
        iter=4000,
        warmup=0,
        threads_per_chain=16,
        seed=SEED,
    )
    return cfg


def prop_cfg(trial):
    return dict(
        changepoint_prior_scale=trial.suggest_float("cp_scale", 0.005, 0.4, log=True),
        seasonality_prior_scale=trial.suggest_float("sea_scale", 0.1, 15, log=True),
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.8,
    )


def cv_objective(make_cfg, is_murphet):
    def _obj(trial):
        cfg, errs = make_cfg(trial), []
        for idx in fold_starts:
            tr_end, te_end = idx, idx + CV_HORIZON
            try:
                if is_murphet:
                    mod = fit_churn_model(t=t_all[:tr_end], y=y_all[:tr_end], **cfg)
                    pred = mod.predict(t_all[tr_end:te_end])
                else:
                    mod = Prophet(**cfg)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                    mod.fit(df.iloc[:tr_end][["ds", "y"]])
                    fut = mod.make_future_dataframe(CV_HORIZON, freq="MS").iloc[-CV_HORIZON:]
                    pred = mod.predict(fut)["yhat"].values
                errs.append(rmse(y_all[tr_end:te_end], pred))
            except RuntimeError:  # any Stan failure
                return 1e6
        return float(np.mean(errs))

    return _obj


# 1) Tune Murphet (MAP)
mur_study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
t0 = time.time()
mur_study.optimize(cv_objective(mur_cfg, True),
                   n_trials=TRIALS_MUR, show_progress_bar=True)
mur_time = time.time() - t0
mur_cfg_best = mur_cfg(mur_study.best_trial)

# 2) Tune Prophet
prop_study = optuna.create_study(direction="minimize",
                                 sampler=optuna.samplers.TPESampler(seed=SEED))
t0 = time.time()
prop_study.optimize(cv_objective(prop_cfg, False),
                    n_trials=TRIALS_PROP, show_progress_bar=True)
prop_time = time.time() - t0
prop_cfg_best = prop_cfg(prop_study.best_trial)

# 3) Final MAP refit & hold-out
train_ho, test_ho = df.iloc[:-HOLD_OUT_MO], df.iloc[-HOLD_OUT_MO:]

mur_fit = fit_churn_model(t=train_ho["t"], y=train_ho["y"], **mur_cfg_best)
mur_pred = mur_fit.predict(test_ho["t"])
mur_res = train_ho["y"].values - mur_fit.predict(train_ho["t"])

prop_fit = Prophet(**prop_cfg_best)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
prop_fit.fit(train_ho[["ds", "y"]])
fut = prop_fit.make_future_dataframe(HOLD_OUT_MO, freq="MS").iloc[-HOLD_OUT_MO:]
prop_df = prop_fit.predict(fut)
prop_pred = prop_df["yhat"].values
prop_low, prop_upp = prop_df["yhat_lower"].values, prop_df["yhat_upper"].values
prop_res = train_ho["y"].values - prop_fit.predict(train_ho[["ds"]])["yhat"].values

# 4) Console summary
print("\n=== Best CV RMSE ===")
print(f"Murphet : {mur_study.best_value:.4f}")
print(f"Prophet : {prop_study.best_value:.4f}")

print("\n=== Optimisation time ===")
print(f"Murphet : {mur_time:5.1f}s ({mur_time / TRIALS_MUR:.2f}s/trial)")
print(f"Prophet : {prop_time:5.1f}s ({prop_time / TRIALS_PROP:.2f}s/trial)")

lb = lambda r: acorr_ljungbox(r, lags=LB_LAGS, return_df=True)["lb_pvalue"]
print(f"\n=== Ljung-Box p-values (first 12 lags) ===")
print(pd.concat([lb(mur_res), lb(prop_res)], axis=1,
                keys=["Murphet", "Prophet"]).head(12)
      .applymap(lambda p: f"{p:.2e}"))

metrics = pd.DataFrame({
    "Model": ["Murphet(MAP-beta)", "Prophet"],
    "RMSE": [rmse(test_ho["y"], mur_pred), rmse(test_ho["y"], prop_pred)],
    "MAE": [mean_absolute_error(test_ho["y"], mur_pred),
            mean_absolute_error(test_ho["y"], prop_pred)],
    "SMAPE": [smape(test_ho["y"], mur_pred), smape(test_ho["y"], prop_pred)],
    "MAPE": [mape(test_ho["y"], mur_pred), mape(test_ho["y"], prop_pred)],
}).set_index("Model")
print(f"\n=== {HOLD_OUT_MO}-month hold-out metrics ===")
print(metrics.round(4))


# 5) Improved plotting functions
def create_forecast_plot(data_df, train, test, mur_fit, prop_fit, mur_predictions,
                         prop_predictions, prop_lower, prop_upper, start_date=None,
                         title=None, figsize=(14, 7)):
    """Create an enhanced forecast plot with improved readability."""
    # Filter data if start_date provided
    if start_date:
        plot_df = data_df[data_df["ds"] >= pd.Timestamp(start_date)].copy()
        plot_train = train[train["ds"] >= pd.Timestamp(start_date)].copy()
        plot_test = test.copy()
    else:
        plot_df = data_df.copy()
        plot_train = train.copy()
        plot_test = test.copy()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot data with improved styling
    ax.plot(plot_df["ds"], plot_df["y"], "k.-", lw=1, ms=3, alpha=0.7, label="Actual")

    # Plot fits if we have training data in range
    if len(plot_train) > 0:
        ax.plot(plot_train["ds"], mur_fit.predict(plot_train["t"]),
                lw=2, color="royalblue", label="Murphet fit")

        prop_train_pred = prop_fit.predict(plot_train[["ds"]])["yhat"]
        ax.plot(plot_train["ds"], prop_train_pred,
                lw=2, color="darkorange", label="Prophet fit")

    # Plot forecasts with enhanced visibility
    ax.plot(plot_test["ds"], mur_predictions, "o-", ms=6, color="royalblue",
            lw=2.5, label="Murphet forecast")
    ax.plot(plot_test["ds"], prop_predictions, "o-", ms=6, color="darkorange",
            lw=2.5, label="Prophet forecast")

    # Add confidence interval
    ax.fill_between(plot_test["ds"], prop_lower, prop_upper,
                    color="darkorange", alpha=0.2, label="Prophet 80% CI")

    # Add vertical line for forecast start
    forecast_start = plot_test["ds"].iloc[0]
    ax.axvline(forecast_start, ls="--", color="gray", lw=1.5)

    # Add annotation
    y_pos = ax.get_ylim()[0] + 0.02 * (ax.get_ylim()[1] - ax.get_ylim()[0])
    ax.annotate('Forecast Start', xy=(forecast_start, y_pos),
                xytext=(10, 20), textcoords='offset points',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'),
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

    # Styling improvements
    ax.grid(True, alpha=0.3)
    ax.set_title(title or f"Retail I/S Ratio – {HOLD_OUT_MO}-month forecast",
                 fontsize=14, fontweight='bold')
    ax.set_ylabel("1 / Inventories-to-Sales", fontsize=12)

    # Date formatting
    if start_date:
        # More detailed dates for zoomed view
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    else:
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.xticks(rotation=45)

    # Add metrics to the plot
    metrics_text = (
        f"RMSE: Murphet={rmse(test['y'], mur_predictions):.4f}, "
        f"Prophet={rmse(test['y'], prop_predictions):.4f}\n"
        f"SMAPE: Murphet={smape(test['y'], mur_predictions):.2f}%, "
        f"Prophet={smape(test['y'], prop_predictions):.2f}%"
    )

    text_box = dict(boxstyle="round,pad=0.5", fc="white", ec="gray", alpha=0.8)
    ax.text(0.02, 0.02, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=text_box)

    # Improve legend
    ax.legend(loc='upper left', frameon=True, framealpha=0.9)

    fig.tight_layout()
    return fig, ax


def create_diagnostic_plots(mur_residuals, prop_residuals, lb_lags):
    """Create ACF and Ljung-Box diagnostic plots with improved styling."""
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), constrained_layout=True)

    # ACF plots
    plot_acf(mur_residuals, lags=lb_lags, ax=axes[0], alpha=0.3)
    axes[0].set_title("Murphet Residual ACF", fontsize=12)
    axes[0].grid(True, alpha=0.3)

    plot_acf(prop_residuals, lags=lb_lags, ax=axes[1], alpha=0.3)
    axes[1].set_title("Prophet Residual ACF", fontsize=12)
    axes[1].grid(True, alpha=0.3)

    # Ljung-Box plot
    lags = np.arange(1, lb_lags + 1)
    mur_lb = acorr_ljungbox(mur_residuals, lags=lb_lags, return_df=True)["lb_stat"]
    prop_lb = acorr_ljungbox(prop_residuals, lags=lb_lags, return_df=True)["lb_stat"]

    axes[2].plot(lags, mur_lb, "o-", lw=2, ms=6, color="royalblue", label="Murphet")
    axes[2].plot(lags, prop_lb, "o-", lw=2, ms=6, color="darkorange", label="Prophet")

    axes[2].set_xlabel("Lag", fontsize=12)
    axes[2].set_ylabel("Ljung-Box Q", fontsize=12)
    axes[2].set_title("Cumulative Ljung-Box Q Statistic", fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()

    # Add critical values
    from scipy import stats
    alpha = 0.05
    crit_values = [stats.chi2.ppf(1 - alpha, df=i) for i in lags]
    axes[2].plot(lags, crit_values, "k--", lw=1.5,
                 label=f"χ² critical value (α={alpha})")
    axes[2].legend()

    fig.suptitle("Residual Diagnostics", fontsize=14, fontweight='bold')
    return fig, axes


# 6) Create plots
plt.style.use('seaborn-v0_8-whitegrid')

# Full time series plot with improved styling
fig1, ax1 = create_forecast_plot(
    df, train_ho, test_ho, mur_fit, prop_fit, mur_pred, prop_pred, prop_low, prop_upp,
    title="Retail Inventories-to-Sales Ratio - Full Time Series"
)

# Zoomed plot (post-2020)
fig2, ax2 = create_forecast_plot(
    df, train_ho, test_ho, mur_fit, prop_fit, mur_pred, prop_pred, prop_low, prop_upp,
    start_date='2020-01-01',
    title="Retail Inventories-to-Sales Ratio - 2020 Onwards"
)

# Diagnostic plots
fig3, ax3 = create_diagnostic_plots(mur_res, prop_res, LB_LAGS)

plt.show()