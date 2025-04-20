# 🚀 Murphet Quickstart Guide

This guide provides a step-by-step introduction to using Murphet for time-series forecasting, demonstrating both basic usage and hyperparameter optimization with Optuna.

## Installation

```bash
pip install murphet
```

This will install Murphet and its dependencies. If you haven't already, you may need to install CmdStan:

```python
import cmdstanpy
cmdstanpy.install_cmdstan()  # Only needed once
```

## Basic Example: Monthly Churn Rate Forecast

### 1. Prepare Your Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from murphet import fit_churn_model

# Load your data - needs 'ds' (date) and 'y' (target) columns
# For this example, we'll create synthetic churn rate data
dates = pd.date_range(start='2020-01-01', periods=36, freq='MS')
y = 0.15 + 0.03 * np.sin(np.linspace(0, 2*np.pi, 36)) + np.random.normal(0, 0.01, 36)
y = np.clip(y, 0.05, 0.95)  # Keep values in (0,1) range

df = pd.DataFrame({
    'ds': dates,
    't': np.arange(len(dates)),  # Time index
    'y': y                       # Target values (churn rate)
})
```

### 2. Fit a Simple Model

```python
# Configure and fit the model
model = fit_churn_model(
    t=df['t'],                 # Time index
    y=df['y'],                 # Target values
    likelihood="beta",         # Beta likelihood for 0-1 bounded data
    periods=12,                # Yearly seasonality
    num_harmonics=3,           # Complexity of seasonal pattern
    n_changepoints=5,          # Number of potential trend changes
    delta_scale=0.1,           # Prior scale for changepoint magnitude
    gamma_scale=5.0,           # Changepoint smoothness
    inference="map"            # Fast maximum a posteriori estimation
)

# Generate forecast for the next 6 months
future_t = np.arange(len(df), len(df) + 6)
forecast = model.predict(future_t)

# Create future dates for plotting
future_dates = pd.date_range(start=df['ds'].iloc[-1] + pd.Timedelta(days=1), 
                             periods=6, freq='MS')
```

### 3. Visualize Results

```python
plt.figure(figsize=(10, 5))
plt.plot(df['ds'], df['y'], 'k.-', label='Historical Data')
plt.plot(future_dates, forecast, 'r.-', label='Forecast')
plt.axvline(df['ds'].iloc[-1], color='gray', linestyle='--')
plt.grid(True, alpha=0.3)
plt.ylabel('Churn Rate')
plt.title('Murphet Forecast')
plt.legend()
plt.tight_layout()
plt.show()
```

## Advanced Example: Hyperparameter Optimization with Optuna

### 1. Set Up Cross-Validation Framework

```python
import optuna
from sklearn.metrics import mean_squared_error

# Configuration
SEED = 42
TRIALS = 20               # Number of optimization trials
INIT_MONTHS = 24          # Initial training window
CV_HORIZON = 3            # Prediction length per fold
CV_STEP = 3               # Window slide step

# Prepare CV folds
t_all, y_all = df['t'].values, df['y'].values
first_test = INIT_MONTHS
fold_starts = list(range(first_test, 
                         len(df) - CV_HORIZON + 1, 
                         CV_STEP))

# RMSE helper
rmse = lambda a, f: np.sqrt(mean_squared_error(a, f))
```

### 2. Define Search Space and Objective Function

```python
def create_model_config(trial):
    """Define the hyperparameter search space"""
    
    # Seasonal components
    periods, harms = [12.0], [trial.suggest_int("harm_year", 1, 4)]
    
    # Optional quarterly seasonality
    if trial.suggest_categorical("add_qtr", [0, 1]):
        periods.append(3.0)
        harms.append(trial.suggest_int("harm_qtr", 1, 3))
    
    return dict(
        likelihood="beta",          # For bounded (0,1) data
        periods=periods,            # Seasonal periods
        num_harmonics=harms,        # Fourier terms per period
        n_changepoints=trial.suggest_int("n_cp", 2, 8),
        delta_scale=trial.suggest_float("delta_scale", 0.02, 0.4, log=True),
        gamma_scale=trial.suggest_float("gamma_scale", 1.0, 10.0),
        season_scale=trial.suggest_float("season_scale", 0.3, 2.0),
        inference="map",            # Fast MAP for optimization
        chains=2,
        iter=4000,
        seed=SEED
    )

def objective(trial):
    """Cross-validation objective function"""
    
    cfg, errors = create_model_config(trial), []
    
    for idx in fold_starts:
        tr_end, te_end = idx, idx + CV_HORIZON
        try:
            # Fit model on training window
            model = fit_churn_model(
                t=t_all[:tr_end], 
                y=y_all[:tr_end], 
                **cfg
            )
            
            # Predict test window
            pred = model.predict(t_all[tr_end:te_end])
            
            # Calculate error
            errors.append(rmse(y_all[tr_end:te_end], pred))
            
        except RuntimeError:
            return 1e6  # Return large value on error
            
    return float(np.mean(errors))
```

### 3. Run Optimization

```python
# Create and run study
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=SEED)
)

study.optimize(objective, n_trials=TRIALS, show_progress_bar=True)

# Print results
print("\nBest parameters:")
for key, value in create_model_config(study.best_trial).items():
    if key not in ['chains', 'iter', 'seed', 'inference']:
        print(f"  {key}: {value}")
        
print(f"\nBest CV RMSE: {study.best_value:.6f}")
```

### 4. Final Model and Forecast

```python
# Get best parameters
best_params = create_model_config(study.best_trial)

# Train final model on all available data
final_model = fit_churn_model(
    t=df['t'],
    y=df['y'],
    **best_params
)

# Generate forecast
future_t = np.arange(len(df), len(df) + 6)
forecast = final_model.predict(future_t)

# Future dates for plotting
future_dates = pd.date_range(
    start=df['ds'].iloc[-1] + pd.Timedelta(days=1),
    periods=6, 
    freq='MS'
)

# Visualize
plt.figure(figsize=(10, 5))
plt.plot(df['ds'], df['y'], 'k.-', label='Historical Data')
plt.plot(df['ds'], final_model.predict(df['t']), 'b-', label='Model Fit')
plt.plot(future_dates, forecast, 'r.-', label='Forecast')
plt.axvline(df['ds'].iloc[-1], color='gray', linestyle='--')
plt.grid(True, alpha=0.3)
plt.ylabel('Churn Rate')
plt.title('Optimized Murphet Forecast')
plt.legend()
plt.tight_layout()
plt.show()
```

## Tips for Real-World Usage

1. **Data preparation**: Always ensure values are in (0,1) range when using `likelihood="beta"`.

2. **Model selection**: Use Optuna to find optimal hyperparameters for your specific dataset.

3. **Inference methods**:
   - Use `inference="map"` for quick exploration and optimization
   - Use `inference="advi"` for approximate Bayesian inference
   - Use `inference="nuts"` for full Bayesian sampling with uncertainty

4. **Seasonality detection**: Enable `auto_detect=True` to automatically identify seasonal periods from your data.

5. **Model validation**: Always check residuals for remaining patterns using tools like ACF plots.

For more advanced options, refer to the [Stan Reference Guide](./stan_reference.md) and [Optuna Recipes](./optuna_recipes.md).