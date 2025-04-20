#!/usr/bin/env python
"""
murphet_vs_prophet_comparison.py

Compares Murphet (beta-likelihood, MAP, Optuna-tuned) vs Prophet
on interest rate data (H0RIFSPPCM06NM) with monthly frequency.

Outputs:
- Optimized model parameters via Optuna
- Holdout metrics: RMSE, MAE, SMAPE, MAPE
- Forecast visualization
- Residual diagnostics
"""
import os
import warnings
import time
import pathlib
from typing import Dict, Any, Callable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.metrics import mean_squared_error, mean_absolute_error
from prophet import Prophet
import optuna
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf

# Use all logical cores (but limit to 1 thread on very tiny fits)
os.environ["STAN_NUM_THREADS"] = str(os.cpu_count())
from murphet import fit_churn_model  # Your custom package

# ---- Configuration parameters ----
SEED = 42
TRIALS = 25  # Optuna trials per model
INIT_MONTHS = 36  # Initial train window (≈ 2 years)
CV_HORIZON = 12  # Months predicted per CV fold
CV_STEP = 6  # Slide step
HOLD_OUT_MO = 24  # Final test window
LB_LAGS = 12  # Ljung-Box lags for residual diagnostics

# ---- Metric functions ----
rmse = lambda a, f: np.sqrt(mean_squared_error(a, f))
smape = lambda a, f: 100 * np.mean(2 * np.abs(f - a) / (np.abs(a) + np.abs(f)))
mape = lambda a, f: 100 * np.mean(np.abs((a - f) / a))


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load and prepare the dataset.

    Args:
        file_path: Path to the CSV file

    Returns:
        Prepared DataFrame with ds, y, and t columns
    """
    # Load the data
    df = pd.read_csv(file_path)

    # Rename columns to Prophet-compatible names
    df.rename(columns={"observation_date": "ds", "H0RIFSPPCM06NM": "y"}, inplace=True)

    # Convert dates to datetime
    df["ds"] = pd.to_datetime(df["ds"])

    # Convert percentage to proportion (0-1 range)
    df["y"] = df["y"] / 100.0

    # Ensure all values are in the open interval (0,1) for beta likelihood
    if not ((df["y"] > 0) & (df["y"] < 1)).all():
        df["y"] = df["y"].clip(1e-5, 1 - 1e-5)
        print("Warning: Some values were clipped to the (0,1) range.")

    # Add time index column
    df["t"] = np.arange(len(df))

    # Sort by date and reset index
    df = df.sort_values("ds").reset_index(drop=True)

    return df


def mur_cfg(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Define the Murphet model configuration space for Optuna.

    Args:
        trial: Optuna trial object

    Returns:
        Dictionary of Murphet model parameters
    """
    # Define seasonal periods and harmonics
    periods, harms = [12.0], [trial.suggest_int("harm_year", 1, 4)]

    # Optional quarterly seasonality
    if trial.suggest_categorical("add_qtr", [0, 1]):
        periods.append(3.0)
        harms.append(trial.suggest_int("harm_qtr", 1, 3))

    # Model configuration
    return dict(
        periods=periods,
        num_harmonics=harms,
        n_changepoints=trial.suggest_int("n_cp", 0, 30),
        delta_scale=trial.suggest_float("delta", 0.005, 0.5, log=True),
        gamma_scale=trial.suggest_float("gamma", 1.0, 15.0),
        season_scale=trial.suggest_float("season_scale", 0.3, 3.0),
        likelihood="beta",
        inference="map",
        chains=2,
        iter=3000,
        warmup=0,
        seed=SEED,
        threads_per_chain=4
    )


def prop_cfg(trial: optuna.Trial) -> Dict[str, Any]:
    """
    Define the Prophet model configuration space for Optuna.

    Args:
        trial: Optuna trial object

    Returns:
        Dictionary of Prophet model parameters
    """
    return dict(
        changepoint_prior_scale=trial.suggest_float("cp_scale", 0.01, 0.4, log=True),
        seasonality_prior_scale=trial.suggest_float("sea_scale", 0.1, 15, log=True),
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.8,
    )


def cv_objective(make_cfg: Callable, is_murphet: bool, t_all: np.ndarray, y_all: np.ndarray, df: pd.DataFrame,
                 fold_starts: list) -> Callable:
    """
    Create a cross-validation objective function for Optuna.

    Args:
        make_cfg: Function to create model configuration
        is_murphet: Flag indicating if this is for Murphet model
        t_all: Time indices array
        y_all: Target values array
        df: DataFrame with data
        fold_starts: List of CV fold start indices

    Returns:
        Objective function for Optuna
    """

    def _obj(trial: optuna.Trial) -> float:
        cfg, errs = make_cfg(trial), []
        for idx in fold_starts:
            tr_end, te_end = idx, idx + CV_HORIZON
            try:
                if is_murphet:
                    mod = fit_churn_model(t=t_all[:tr_end], y=y_all[:tr_end], **cfg)
                    pred = mod.predict(t_all[tr_end:te_end])
                else:
                    m = Prophet(**cfg)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        m.fit(df.iloc[:tr_end][["ds", "y"]])
                    fut = m.make_future_dataframe(CV_HORIZON, freq="MS").iloc[-CV_HORIZON:]
                    pred = m.predict(fut)["yhat"].to_numpy()
                errs.append(rmse(y_all[tr_end:te_end], pred))
            except RuntimeError:
                return 1e6  # abort trial
        return float(np.mean(errs))

    return _obj


def create_forecast_plot(df: pd.DataFrame, train: pd.DataFrame, test: pd.DataFrame,
                         mur_fit, prop_fit, mur_pred: np.ndarray, prop_pred: np.ndarray,
                         prop_low: np.ndarray, prop_upp: np.ndarray, title: str = None,
                         figsize: tuple = (12, 6)) -> tuple:
    """
    Create a comparison forecast plot for Murphet and Prophet models.

    Args:
        Various data and prediction objects

    Returns:
        Figure and axes objects
    """
    plt.style.use("seaborn-v0_8")
    fig, ax = plt.subplots(figsize=figsize)

    # Plot actual data
    ax.plot(df["ds"], df["y"], "k.-", lw=1, alpha=0.7, label="Actual")

    # Plot model fits
    ax.plot(train["ds"], mur_fit.predict(train["t"]),
            color="royalblue", lw=2, label="Murphet fit")
    ax.plot(train["ds"], prop_fit.predict(train[["ds"]])["yhat"],
            color="darkorange", lw=2, label="Prophet fit")

    # Plot forecasts
    ax.plot(test["ds"], mur_pred, "o-", color="royalblue", markersize=6,
            lw=2.5, label="Murphet forecast")
    ax.plot(test["ds"], prop_pred, "o-", color="darkorange", markersize=6,
            lw=2.5, label="Prophet forecast")

    # Add prediction interval
    ax.fill_between(test["ds"], prop_low, prop_upp,
                    color="darkorange", alpha=0.2, label="Prophet 80% CI")

    # Add vertical line for forecast start
    ax.axvline(test["ds"].iloc[0], ls="--", color="gray")

    # Add shaded region for test period
    ax.axvspan(test["ds"].iloc[0], test["ds"].iloc[-1], alpha=0.1, color='gray')

    # Set labels and title
    ax.set_title(title or f"Interest Rate - {HOLD_OUT_MO}-month hold-out", fontsize=14)
    ax.set_ylabel("Rate (proportion)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)

    # Add metrics annotation
    textstr = (f"Hold-out RMSE:\n"
               f"Murphet: {rmse(test['y'], mur_pred):.4f}\n"
               f"Prophet: {rmse(test['y'], prop_pred):.4f}\n\n"
               f"Hold-out MAPE:\n"
               f"Murphet: {mape(test['y'], mur_pred):.2f}%\n"
               f"Prophet: {mape(test['y'], prop_pred):.2f}%")
    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax.text(0.02, 0.05, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', bbox=props)

    # Add legend
    ax.legend(frameon=True, fontsize=10, loc="best")

    fig.tight_layout()
    return fig, ax


def create_residual_plot(train: pd.DataFrame, mur_fit, prop_fit, figsize: tuple = (12, 6)) -> tuple:
    """
    Create residual analysis plots for both models.

    Args:
        train: Training data
        mur_fit: Fitted Murphet model
        prop_fit: Fitted Prophet model
        figsize: Figure size

    Returns:
        Figure and axes objects
    """
    mur_pred_train = mur_fit.predict(train["t"])
    prop_pred_train = prop_fit.predict(train[["ds"]])["yhat"].to_numpy()

    mur_res = train["y"].to_numpy() - mur_pred_train
    prop_res = train["y"].to_numpy() - prop_pred_train

    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # Residual time series
    axes[0].plot(train["ds"], mur_res, 'o-', markersize=3, alpha=0.7, color='royalblue', label='Murphet')
    axes[0].plot(train["ds"], prop_res, 'o-', markersize=3, alpha=0.7, color='darkorange', label='Prophet')
    axes[0].axhline(0, color='gray', linestyle='--')
    axes[0].set_title("Residuals Over Time")
    axes[0].legend()

    # ACF plots
    plot_acf(mur_res, lags=LB_LAGS, ax=axes[1], alpha=0.5, color='royalblue', title="Murphet Residuals ACF")
    plot_acf(prop_res, lags=LB_LAGS, ax=axes[2], alpha=0.5, color='darkorange', title="Prophet Residuals ACF")

    fig.tight_layout()
    return fig, axes


def main():
    """Main execution function."""
    ROOT = pathlib.Path(__file__).resolve().parent.parent
    file_path = ROOT / "examples" / "data" / "H0RIFSPPCM06NM.csv"

    # Load and prepare data
    print(f"Loading data from {file_path}...")
    df = load_data(file_path)
    print(f"Loaded {len(df)} observations from {df['ds'].min()} to {df['ds'].max()}")

    # Extract arrays for modeling
    t_all, y_all = df["t"].to_numpy(), df["y"].to_numpy()

    # Set up CV folds
    first_test = INIT_MONTHS
    fold_starts = list(range(first_test,
                             len(df) - HOLD_OUT_MO - CV_HORIZON + 1,
                             CV_STEP))
    print(f"{len(df)} rows → {len(fold_starts)} CV folds")

    # Optimize Murphet model
    print("\nOptimizing Murphet model with Optuna...")
    sampler = optuna.samplers.TPESampler(seed=SEED)
    mur_study = optuna.create_study(direction="minimize", sampler=sampler)
    mur_study.optimize(cv_objective(mur_cfg, True, t_all, y_all, df, fold_starts),
                       n_trials=TRIALS, show_progress_bar=True)
    mur_cfg_best = mur_cfg(mur_study.best_trial)

    # Optimize Prophet model
    print("\nOptimizing Prophet model with Optuna...")
    prop_study = optuna.create_study(direction="minimize", sampler=sampler)
    prop_study.optimize(cv_objective(prop_cfg, False, t_all, y_all, df, fold_starts),
                        n_trials=TRIALS, show_progress_bar=True)
    prop_cfg_best = prop_cfg(prop_study.best_trial)

    # Print best CV RMSE
    print("\nBest CV RMSE:")
    print(f"Murphet: {mur_study.best_value:.4f}")
    print(f"Prophet: {prop_study.best_value:.4f}")

    # Split for final holdout evaluation
    train_ho, test_ho = df.iloc[:-HOLD_OUT_MO], df.iloc[-HOLD_OUT_MO:]

    # Fit and predict with Murphet
    print("\nFitting final Murphet model...")
    mur_fit = fit_churn_model(t=train_ho["t"], y=train_ho["y"], **mur_cfg_best)
    mur_pred = mur_fit.predict(test_ho["t"])

    # Fit and predict with Prophet
    print("Fitting final Prophet model...")
    prop_fit = Prophet(**prop_cfg_best)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prop_fit.fit(train_ho[["ds", "y"]])
    fut = prop_fit.make_future_dataframe(HOLD_OUT_MO, freq="MS").iloc[-HOLD_OUT_MO:]
    prop_df = prop_fit.predict(fut)
    prop_pred = prop_df["yhat"].to_numpy()
    prop_low, prop_upp = prop_df["yhat_lower"].to_numpy(), prop_df["yhat_upper"].to_numpy()

    # Calculate and display metrics
    metrics = pd.DataFrame({
        "Model": ["Murphet(MAP)", "Prophet"],
        "RMSE": [rmse(test_ho["y"], mur_pred), rmse(test_ho["y"], prop_pred)],
        "MAE": [mean_absolute_error(test_ho["y"], mur_pred),
                mean_absolute_error(test_ho["y"], prop_pred)],
        "SMAPE": [smape(test_ho["y"], mur_pred),
                  smape(test_ho["y"], prop_pred)],
        "MAPE": [mape(test_ho["y"], mur_pred),
                 mape(test_ho["y"], prop_pred)],
    }).set_index("Model")

    print("\nHoldout Metrics:")
    print(metrics.round(4))

    # Create forecast plot
    fig1, ax1 = create_forecast_plot(
        df, train_ho, test_ho, mur_fit, prop_fit, mur_pred, prop_pred, prop_low, prop_upp,
        title=f"Interest Rate Forecast - {HOLD_OUT_MO}-month holdout"
    )

    # Create residual plot
    fig2, ax2 = create_residual_plot(train_ho, mur_fit, prop_fit)

    # Save figures
    fig1.savefig("interest_rate_forecast.png", dpi=300, bbox_inches="tight")
    fig2.savefig("model_residuals.png", dpi=300, bbox_inches="tight")

    print("\nAnalysis complete! Figures saved to current directory.")
    plt.show()


if __name__ == "__main__":
    main()