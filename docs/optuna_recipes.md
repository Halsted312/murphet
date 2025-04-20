# 🔍 Murphet Optuna Recipes

> **Hyperparameter optimization recipes for time-series forecasting with Murphet**

This guide provides ready-to-use Optuna recipes for optimizing Murphet models across different data types and frequencies. Each recipe includes recommended search spaces, cross-validation setup, and complete code examples.

## 1. Core Optimization Setup

### Standard Cross-Validation Framework

```python
import optuna
import numpy as np
from murphet import fit_churn_model

# Configuration
SEED = 42
TRIALS = 50
INIT_MONTHS = 18        # Initial training window
CV_HORIZON = 6          # Prediction length per fold
CV_STEP = 3             # Window slide step

# Prepare time index and CV folds
df["t"] = np.arange(len(df))
t_all, y_all = df["t"].values, df["y"].values

first_test = INIT_MONTHS
fold_starts = list(range(first_test,
                         len(df) - HOLD_OUT_MO - CV_HORIZON + 1,
                         CV_STEP))

# RMSE helper
rmse = lambda a, f: np.sqrt(np.mean((a - f) ** 2))
```

### Standard Objective Function

```python
def cv_objective(make_cfg, is_murphet=True):
    def _obj(trial):
        cfg, errs = make_cfg(trial), []
        
        for idx in fold_starts:
            tr_end, te_end = idx, idx + CV_HORIZON
            try:
                if is_murphet:
                    # Murphet fitting & prediction
                    mod = fit_churn_model(t=t_all[:tr_end], y=y_all[:tr_end], **cfg)
                    pred = mod.predict(t_all[tr_end:te_end])
                else:
                    # Prophet alternative 
                    # (Code for Prophet would go here)
                    ...
                    
                errs.append(rmse(y_all[tr_end:te_end], pred))
            except RuntimeError:
                return 1e6   # Abort trial on errors
                
        return float(np.mean(errs))
    return _obj
```

## 2. Search Space Recipes by Data Type

### 2.1 Monthly Rate/Proportion Data (0-1 bounded)

```python
def rate_monthly_cfg(trial):
    """
    Search space for monthly rate/proportion data with Beta likelihood.
    Suitable for: churn rates, conversion rates, occupancy.
    """
    # Seasonal structure
    periods, harms = [12.0], [trial.suggest_int("harm_year", 1, 4)]
    
    # Optional quarterly component
    if trial.suggest_categorical("add_qtr", [0, 1]):
        periods.append(3.0)
        harms.append(trial.suggest_int("harm_qtr", 1, 3))
    
    return dict(
        likelihood="beta",              # Key for 0-1 bounded data
        periods=periods,
        num_harmonics=harms,
        n_changepoints=trial.suggest_int("n_cp", 2, 8),
        delta_scale=trial.suggest_float("delta_scale", 0.02, 0.5, log=True),
        gamma_scale=trial.suggest_float("gamma_scale", 1.0, 10.0),
        season_scale=trial.suggest_float("season_scale", 0.3, 2.0),
        inference="map",                # Fast for HPO
        chains=2,
        iter=4000,
        seed=SEED,
    )
```

### 2.2 Daily Rate/Proportion Data

```python
def rate_daily_cfg(trial):
    """
    Search space for daily rate/proportion data with Beta likelihood.
    Suitable for: daily conversion rates, CTR.
    """
    # Seasonal structure with potential weekly and monthly patterns
    periods, harms = [], []
    
    # Weekly seasonality
    if trial.suggest_categorical("add_weekly", [0, 1]):
        periods.append(7.0)
        harms.append(trial.suggest_int("harm_week", 2, 3))
    
    # Monthly seasonality
    if trial.suggest_categorical("add_monthly", [0, 1]):
        periods.append(30.4)  # Approximate month length
        harms.append(trial.suggest_int("harm_month", 2, 5))
    
    # Yearly seasonality
    if trial.suggest_categorical("add_yearly", [0, 1]):
        periods.append(365.25)
        harms.append(trial.suggest_int("harm_year", 3, 6))
    
    return dict(
        likelihood="beta",
        periods=periods,
        num_harmonics=harms,
        n_changepoints=trial.suggest_int("n_cp", 5, 20),  # More CPs for daily data
        delta_scale=trial.suggest_float("delta_scale", 0.05, 0.6, log=True),
        gamma_scale=trial.suggest_float("gamma_scale", 3.0, 12.0),
        season_scale=trial.suggest_float("season_scale", 0.3, 2.0),
        inference="map",
        chains=2,
        iter=4000,
        seed=SEED,
    )
```

### 2.3 Unbounded Ratio/Scale Data

```python
def unbounded_ratio_cfg(trial):
    """
    Search space for unbounded ratio data with Gaussian/Student-t likelihood.
    Suitable for: price ratios, inventory ratios, financial metrics.
    """
    # Seasonal structure
    periods, harms = [12.0], [trial.suggest_int("harm_year", 1, 4)]
    
    # Optional half-year component
    if trial.suggest_categorical("add_half", [0, 1]):
        periods.append(6.0)
        harms.append(trial.suggest_int("harm_half", 1, 3))
    
    return dict(
        likelihood="gaussian",          # Switch to unbounded Gaussian
        periods=periods,
        num_harmonics=harms,
        n_changepoints=trial.suggest_int("n_cp", 2, 10),
        delta_scale=trial.suggest_float("delta_scale", 0.01, 0.4, log=True),
        gamma_scale=trial.suggest_float("gamma_scale", 1.0, 8.0),
        season_scale=trial.suggest_float("season_scale", 0.3, 2.0),
        inference="map",
        chains=2,
        iter=4000,
        seed=SEED,
    )
```

## 3. Complete Optimization Recipes

### 3.1 Monthly Churn Rate Optimization

```python
# Full example for monthly churn rate optimization

import os, warnings
import numpy as np
import pandas as pd
import optuna
from murphet import fit_churn_model

# Configuration
SEED = 42
TRIALS = 30
INIT_MONTHS = 18
CV_HORIZON = 6
CV_STEP = 3
HOLD_OUT_MO = 6

# Load data (assuming df has 'ds' and 'y' columns, with 0<y<1)
df = pd.read_csv("churn_data.csv", parse_dates=["ds"])
df["t"] = np.arange(len(df))
t_all, y_all = df["t"].values, df["y"].values

# Set up CV folds
first_test = INIT_MONTHS
fold_starts = list(range(first_test,
                         len(df) - HOLD_OUT_MO - CV_HORIZON + 1,
                         CV_STEP))

# Define search space
def mur_cfg(trial):
    periods, harms = [12.0], [trial.suggest_int("harm_year", 1, 4)]
    if trial.suggest_categorical("add_qtr", [0, 1]):
        periods.append(3.0)
        harms.append(trial.suggest_int("harm_qtr", 1, 3))
    
    return dict(
        likelihood="beta",
        periods=periods,
        num_harmonics=harms,
        n_changepoints=trial.suggest_int("n_cp", 2, 6),
        delta_scale=trial.suggest_float("delta", 0.01, 0.4, log=True),
        gamma_scale=trial.suggest_float("gamma", 1.0, 10.0),
        season_scale=trial.suggest_float("season_scale", 0.3, 2.0),
        inference="map",
        chains=2,
        iter=4000,
        seed=SEED,
    )

# Define objective
rmse = lambda a, f: np.sqrt(np.mean((a - f) ** 2))

def objective(trial):
    cfg, errs = mur_cfg(trial), []
    for idx in fold_starts:
        tr_end, te_end = idx, idx + CV_HORIZON
        try:
            mod = fit_churn_model(t=t_all[:tr_end], y=y_all[:tr_end], **cfg)
            pred = mod.predict(t_all[tr_end:te_end])
            errs.append(rmse(y_all[tr_end:te_end], pred))
        except RuntimeError:
            return 1e6
    return float(np.mean(errs))

# Run optimization
study = optuna.create_study(direction="minimize", 
                            sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=TRIALS, show_progress_bar=True)

# Use best parameters for final model
best_params = mur_cfg(study.best_trial)
print("Best parameters:", best_params)
print("Best RMSE:", study.best_value)

# Train final model on all but holdout data
train, test = df.iloc[:-HOLD_OUT_MO], df.iloc[-HOLD_OUT_MO:]
final_model = fit_churn_model(t=train["t"], y=train["y"], **best_params)
predictions = final_model.predict(test["t"])
test_rmse = rmse(test["y"], predictions)
print(f"Holdout RMSE: {test_rmse:.4f}")
```

### 3.2 Comparing Murphet vs Prophet

```python
import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from prophet import Prophet
from murphet import fit_churn_model
from sklearn.metrics import mean_squared_error

# Configuration
SEED = 42
TRIALS_PER_MODEL = 30
INIT_MONTHS = 18
CV_HORIZON = 6
CV_STEP = 3
HOLD_OUT_MO = 6

# Metrics
rmse = lambda a, f: np.sqrt(mean_squared_error(a, f))

# Load data
df = pd.read_csv("time_series_data.csv", parse_dates=["ds"])
df["t"] = np.arange(len(df))
t_all, y_all = df["t"].values, df["y"].values

# CV setup
first_test = INIT_MONTHS
fold_starts = list(range(first_test,
                         len(df) - HOLD_OUT_MO - CV_HORIZON + 1,
                         CV_STEP))

# Murphet config
def mur_cfg(trial):
    periods, harms = [12.0], [trial.suggest_int("harm_year", 1, 4)]
    if trial.suggest_categorical("add_half", [0, 1]):
        periods.append(6.0)
        harms.append(trial.suggest_int("harm_half", 1, 3))
    
    return dict(
        likelihood="beta",
        periods=periods,
        num_harmonics=harms,
        n_changepoints=trial.suggest_int("n_cp", 2, 8),
        delta_scale=trial.suggest_float("delta", 0.01, 0.4, log=True),
        gamma_scale=trial.suggest_float("gamma", 1.0, 10.0),
        season_scale=trial.suggest_float("season_scale", 0.3, 2.0),
        inference="map",
        chains=2,
        iter=4000,
        seed=SEED,
    )

# Prophet config
def prop_cfg(trial):
    return dict(
        changepoint_prior_scale=trial.suggest_float("cp_scale", 0.01, 0.4, log=True),
        seasonality_prior_scale=trial.suggest_float("sea_scale", 0.1, 15, log=True),
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.8,
    )

# CV objective
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
            except RuntimeError:
                return 1e6
        return float(np.mean(errs))
    return _obj

# Run optimizations
sampler = optuna.samplers.TPESampler(seed=SEED)

mur_study = optuna.create_study(direction="minimize", sampler=sampler)
mur_study.optimize(cv_objective(mur_cfg, True), 
                  n_trials=TRIALS_PER_MODEL, show_progress_bar=True)

prop_study = optuna.create_study(direction="minimize", sampler=sampler)
prop_study.optimize(cv_objective(prop_cfg, False),
                   n_trials=TRIALS_PER_MODEL, show_progress_bar=True)

print("\nBest CV RMSE")
print(f"Murphet: {mur_study.best_value:.4f}")
print(f"Prophet: {prop_study.best_value:.4f}")

# Get best parameters
best_mur = mur_cfg(mur_study.best_trial)
best_prop = prop_cfg(prop_study.best_trial)

# Fit final models
train, test = df.iloc[:-HOLD_OUT_MO], df.iloc[-HOLD_OUT_MO:]

mur_fit = fit_churn_model(t=train["t"], y=train["y"], **best_mur)
mur_pred = mur_fit.predict(test["t"])

prop_fit = Prophet(**best_prop)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    prop_fit.fit(train[["ds", "y"]])
fut = prop_fit.make_future_dataframe(HOLD_OUT_MO, freq="MS").iloc[-HOLD_OUT_MO:]
prop_pred = prop_fit.predict(fut)["yhat"].values

# Evaluate holdout performance
print("\nHoldout RMSE")
print(f"Murphet: {rmse(test['y'], mur_pred):.4f}")
print(f"Prophet: {rmse(test['y'], prop_pred):.4f}")
```

## 4. Advanced Optuna Techniques

### 4.1 Early Stopping for Efficiency

```python
from optuna.pruners import MedianPruner

# Create pruner that stops trials performing worse than median
pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)

# Create study with pruner
study = optuna.create_study(
    direction="minimize", 
    sampler=optuna.samplers.TPESampler(seed=SEED),
    pruner=pruner
)

# Modify objective to support pruning
def prunable_objective(trial):
    cfg, fold_errors = mur_cfg(trial), []
    
    for i, idx in enumerate(fold_starts):
        tr_end, te_end = idx, idx + CV_HORIZON
        try:
            mod = fit_churn_model(t=t_all[:tr_end], y=y_all[:tr_end], **cfg)
            pred = mod.predict(t_all[tr_end:te_end])
            
            this_fold_error = rmse(y_all[tr_end:te_end], pred)
            fold_errors.append(this_fold_error)
            
            # Report intermediate value for pruning
            trial.report(np.mean(fold_errors), i)
            
            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        except RuntimeError:
            return 1e6
            
    return float(np.mean(fold_errors))

study.optimize(prunable_objective, n_trials=TRIALS)
```

### 4.2 Visualizing Parameter Importance

```python
import optuna.visualization as vis
import matplotlib.pyplot as plt

# After optimization complete
fig = vis.plot_param_importances(study)
plt.title("Parameter Importance for Murphet Model")
plt.tight_layout()
plt.show()

# Parallel coordinate plot
fig = vis.plot_parallel_coordinate(study)
plt.title("Parameter Relationships")
plt.tight_layout()
plt.show()
```

### 4.3 Transfer Learning Between Data Types

```python
# First optimize on related dataset
other_study = optuna.create_study(direction="minimize")
other_study.optimize(other_dataset_objective, n_trials=20)

# Extract best parameters as starting point
best_params = other_study.best_params

# Create new sampler with prior knowledge
sampler = optuna.samplers.TPESampler(
    seed=SEED,
    consider_prior=True,  # Use prior info
    prior_weight=0.8,     # How much to weight the prior
)

# Define warm-start search space
def transfer_cfg(trial):
    # Start with values close to the previous best for key parameters
    n_cp = trial.suggest_int(
        "n_cp", 
        max(1, best_params["n_cp"] - 2), 
        best_params["n_cp"] + 2
    )
    
    delta_scale = trial.suggest_float(
        "delta_scale", 
        best_params["delta_scale"] * 0.5, 
        best_params["delta_scale"] * 2.0, 
        log=True
    )
    
    # Rest of parameters...
    
    return dict(
        likelihood="beta",
        n_changepoints=n_cp,
        delta_scale=delta_scale,
        # Other parameters...
    )

# Create new study with this advanced sampler
transfer_study = optuna.create_study(direction="minimize", sampler=sampler)
transfer_study.optimize(transfer_objective, n_trials=TRIALS)
```

## 5. Parameter Impact Guidelines

| Parameter | Impact | Too Low | Too High | Notes |
|-----------|--------|---------|----------|-------|
| `n_changepoints` | Trend flexibility | Underfitting | Overfitting | Start with N/10 for monthly data |
| `delta_scale` | Changepoint magnitude | Rigid trend | Overfitting | Typically 0.01-0.4 for monthly |
| `gamma_scale` | CP transition speed | Gradual changes | Abrupt changes | 1.0-10.0 is typical range |
| `season_scale` | Seasonal strength | Damped seasonality | Seasonal overfit | 0.3-2.0 is standard |
| `num_harmonics` | Seasonal complexity | Smooth waves | Complex waves | 2-4 for yearly patterns |

## 6. Troubleshooting Optuna Optimization

### Common Issues and Solutions

1. **Slow convergence**:
   - Reduce parameter ranges
   - Increase `n_startup_trials` for TPE sampler

2. **Stan/Murphet failures during trials**:
   - Check for extreme parameter combinations
   - Adjust prior scales to be more conservative
   - Implement robust error handling in objective

3. **Poor forecasts despite good CV scores**:
   - Ensure CV setup matches forecast horizon
   - Consider time-based splits instead of rolling origin
   - Test for dataset shift in recent periods

4. **All models perform poorly**:
   - Check for data quality issues
   - Consider feature engineering or additional covariates
   - Test different seasonality configurations

### Optimization Time Management

Strategies to reduce Optuna runtime:

```python
# 1. Use MAP inference during HPO
cfg = dict(
    # Other parameters...
    inference="map",      # Fast for HPO
    iter=2000,            # Reduced iterations
)

# 2. Use faster/fewer CV folds
CV_STEP = 6               # Fewer folds
CV_HORIZON = min(horizon, 6)  # Shorter horizon if possible

# 3. Implement multiprocessing
import joblib

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=TRIALS, n_jobs=4)  # Parallel trials
```

## 7. Recommended Search Ranges by Frequency

| Parameter | Daily | Monthly | Quarterly |
|-----------|-------|---------|-----------|
| `gamma_scale` | 3.0-12.0 | 1.0-8.0 | 0.5-4.0 |
| `delta_scale` | 0.05-0.6 | 0.02-0.4 | 0.01-0.3 |
| `n_changepoints` | ≤20 | ≤10 | ≤6 |
| `season_scale` | 0.3-2.0 | 0.3-2.0 | 0.3-1.5 |

---

For additional customization options and advanced Stan parameters, see the [Stan Reference Guide](./stan_reference.md).