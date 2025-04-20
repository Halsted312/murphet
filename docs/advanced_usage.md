# Advanced Usage Guide for Murphet

This document provides detailed technical information for advanced users of Murphet. It covers the mathematical foundations of the model, Stan implementation details, seasonality handling, optimization techniques, diagnostics, and examples of real-world applications.

## 1. The Model Under the Hood

Murphet implements a Bayesian structural time-series model designed specifically for forecasting rates and probabilities while preserving bounds. The core model structure builds on Prophet's foundation with several key enhancements.

### Model Diagram

The core components of Murphet can be represented as:

```
y ~ Likelihood(μ, dispersion)
μ = f_trend(t) + f_seasonality(t) + f_AR(t)
```

Where each component has carefully selected priors and parameterization.

### Logit Link vs. Gaussian Head

Murphet offers two likelihood options that fundamentally change how predictions are generated:

**Beta Likelihood (Default)**:
- Uses logistic function to map linear predictor to (0,1)
- Models `y ~ Beta(p·φ, (1-p)·φ)` where `p = logit⁻¹(μ)`
- Precision parameter `φ` controls variance (`Var[y] ≈ p·(1-p)/φ`)
- Perfect for churn rates, conversion rates, proportions

**Gaussian/Student-t Likelihood**:
- No transformation of linear predictor
- Models `y ~ Normal(μ, σ)` or `y ~ Student-t(ν, μ, σ)`
- Allows unbounded predictions
- Better for time series that may extend beyond [0,1]

### Smooth Changepoints

A key advancement over Prophet is Murphet's implementation of smooth changepoint transitions:

**Prophet's Hard Breaks**:
```
f(t) = k·t + m + Σ δⱼ·I(t > sⱼ)·(t - sⱼ)
```
Where `I()` is an indicator function causing abrupt slope changes.

**Murphet's Smooth Transitions**:
```
f(t) = k·t + m + Σ δⱼ·σ(γ·(t - sⱼ))·(t - sⱼ)
```

Where:
- `σ()` is the sigmoid/logistic function
- `γ` controls transition steepness (tunable via `gamma_scale`)
- `δⱼ` are changepoint magnitudes (regularized via `delta_scale`)

This approach prevents artificial kinks in forecasts while maintaining the flexibility to model trend changes.

## 2. Stan File Walkthrough

Understanding Murphet's Stan implementation enables advanced customization. The core files (`murphet_beta.stan` and `murphet_gauss.stan`) share similar structure but differ in likelihood specification.

### Functions Block

```stan
functions {
  real partial_sum_beta(array[] real y_slice,
                        int start, int end,
                        /* ... parameter declarations ... */) {
    // Parallelized log-likelihood computation
    real lp = 0;
    real lag = mu0;  // AR(1) state initialization
    
    for (i in 1:size(y_slice)) {
      // Deterministic trend + changepoints
      // Seasonal components
      // AR(1) disturbance
      // Heteroscedastic precision
      // Likelihood contribution
    }
    return lp;
  }
}
```

The `partial_sum_*` functions enable parallel likelihood computation using Stan's `reduce_sum` operation, significantly improving performance for large datasets.

### Data Block

```stan
data {
  int<lower=1> N;                          // Number of observations
  vector[N] t;                             // Time points
  vector[N] y;                             // Target values

  int<lower=0> num_changepoints;           // Number of changepoints
  vector[num_changepoints] s;              // Changepoint locations
  real<lower=0> delta_scale;               // Prior scale for trend changes
  real<lower=0> gamma_scale;               // Prior scale for transition steepness

  int<lower=1> num_seasons;                // Number of seasonal components
  array[num_seasons] int n_harmonics;      // Harmonics per component
  array[num_seasons] real period;          // Period lengths
  int<lower=1> total_harmonics;            // sum(n_harmonics)
  real<lower=0> season_scale;              // Prior scale for seasonality
}
```

This block defines the data structure required for model fitting. All parameters are automatically constructed by `fit_churn_model()`.

### Parameters Block

```stan
parameters {
  // Trend parameters
  real k;                                  // Base slope
  real m;                                  // Intercept
  vector[num_changepoints] delta;          // Changepoint adjustments
  real<lower=0> gamma;                     // Transition steepness

  // AR(1) parameters
  real<lower=-1,upper=1> rho;              // Persistence
  real mu0;                                // Initial state

  // Seasonality parameters
  vector[total_harmonics] A_sin;           // Sine coefficients
  vector[total_harmonics] B_cos;           // Cosine coefficients

  // Likelihood-specific parameters (differ between models)
  // Beta: log_phi0, beta_phi
  // Gaussian: log_sigma0, beta_sigma, nu
}
```

### Model Block

```stan
model {
  // Trend priors
  k      ~ normal(0, 0.5);
  m      ~ normal(0, 5);
  delta  ~ double_exponential(0, delta_scale);  // Laplace/DE prior
  gamma  ~ gamma(3, 1 / gamma_scale);

  // AR(1) priors
  rho    ~ normal(0, 0.3);  // or 0.5 for quarterly data
  mu0    ~ normal(logit(mean(y)), 1);  // or mean(y) for Gaussian

  // Seasonality priors
  A_sin  ~ normal(0, 10 * season_scale);
  B_cos  ~ normal(0, 10 * season_scale);

  // Likelihood-specific priors (differ between models)
  
  // Parallel likelihood computation
  target += reduce_sum(partial_sum_*, to_array_1d(y), 16, /* other args */);
}
```

### Parameter Computation Details

**rho, phi_i, sigma_i**

These parameters control important model behaviors:

- `rho`: AR(1) coefficient (-1 to 1) capturing autocorrelation in residuals. Higher values indicate stronger persistence in deviations.

- `phi_i = exp(log_phi0 - beta_phi * abs(mu_det))`: Heteroscedastic precision for Beta likelihood. Precision decreases when the deterministic component is far from zero.

- `sigma_i = exp(log_sigma0 + beta_sigma * abs(mu_det))`: Heteroscedastic scale for Gaussian/Student-t. Scale increases when the deterministic component is large.

## 3. Seasonality Cookbook

Murphet offers flexible seasonal modeling through Fourier series. This section provides recipes for common seasonal patterns.

### Single Period Seasonality

For simple yearly seasonality (e.g., monthly data with annual pattern):

```python
model = fit_churn_model(
    t=df['t'],
    y=df['y'],
    periods=12,                # 12-month period
    num_harmonics=3,           # 3 harmonics for flexibility
    season_scale=1.0,          # Default prior scale
    # other parameters...
)
```

For weekly seasonality with daily data:

```python
model = fit_churn_model(
    t=df['t'],
    y=df['y'],
    periods=7,                 # 7-day period
    num_harmonics=3,           # 3 harmonics
    # other parameters...
)
```

### Multiple Periods

For monthly data with both yearly and quarterly patterns:

```python
model = fit_churn_model(
    t=df['t'],
    y=df['y'],
    periods=[12.0, 3.0],       # Yearly and quarterly
    num_harmonics=[3, 2],      # 3 for yearly, 2 for quarterly
    # other parameters...
)
```

For daily data with weekly and yearly patterns:

```python
model = fit_churn_model(
    t=df['t'],
    y=df['y'],
    periods=[7.0, 365.25],     # Weekly and yearly
    num_harmonics=[3, 5],      # 3 for weekly, 5 for yearly
    # other parameters...
)
```

### Irregular Periods

For marketing calendar effects (e.g., 13×4-week retail calendar):

```python
model = fit_churn_model(
    t=df['t'],
    y=df['y'],
    periods=[28.0, 364.0],     # 4-week and 13×4-week cycle
    num_harmonics=[2, 4],      # Harmonics per period
    # other parameters...
)
```

For multiple marketing interventions at irregular intervals, consider adding explicit changepoints at campaign dates:

```python
campaign_dates = pd.to_datetime(['2023-01-15', '2023-03-20', '2023-06-01'])
campaign_t = df[df['ds'].isin(campaign_dates)]['t'].values

model = fit_churn_model(
    t=df['t'],
    y=df['y'],
    changepoints=campaign_t,   # Explicit changepoint locations
    gamma_scale=8.0,           # Sharper transitions for campaigns
    # other parameters...
)
```

## 4. Optuna Hyper-Tuning

Optimizing Murphet's hyperparameters can significantly improve forecast accuracy. This section provides guidance on effective optimization with Optuna.

### Recommended Search Spaces

```python
def murphet_search_space(trial):
    # Seasonal configuration
    periods, harms = [12.0], [trial.suggest_int("harm_year", 1, 4)]
    if trial.suggest_categorical("add_qtr", [0, 1]):
        periods.append(3.0)
        harms.append(trial.suggest_int("harm_qtr", 1, 3))
    
    return dict(
        likelihood="beta",  # or "gaussian" for unbounded data
        periods=periods,
        num_harmonics=harms,
        n_changepoints=trial.suggest_int("n_cp", 2, 8),
        delta_scale=trial.suggest_float("delta", 0.01, 0.4, log=True),
        gamma_scale=trial.suggest_float("gamma", 1.0, 10.0),
        season_scale=trial.suggest_float("season_scale", 0.3, 2.0),
        inference="map",  # Fast for optimization
        chains=2,
        iter=4000,
        seed=42,
    )
```

### Early-Stopping with Pruning

To speed up optimization, implement Optuna's pruning capabilities:

```python
from optuna.pruners import MedianPruner

# Create pruner
pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=0)

# Create study with pruner
study = optuna.create_study(
    direction="minimize",
    sampler=optuna.samplers.TPESampler(seed=42),
    pruner=pruner
)

# Implement prunable objective
def prunable_objective(trial):
    cfg, fold_errors = murphet_search_space(trial), []
    
    for i, idx in enumerate(fold_starts):
        tr_end, te_end = idx, idx + CV_HORIZON
        try:
            model = fit_churn_model(t=t_all[:tr_end], y=y_all[:tr_end], **cfg)
            pred = model.predict(t_all[tr_end:te_end])
            
            fold_error = rmse(y_all[tr_end:te_end], pred)
            fold_errors.append(fold_error)
            
            # Report for pruning
            trial.report(np.mean(fold_errors), i)
            if trial.should_prune():
                raise optuna.TrialPruned()
                
        except RuntimeError:
            return 1e6
            
    return float(np.mean(fold_errors))

# Run optimization
study.optimize(prunable_objective, n_trials=30)
```

### Warm-Starting from Previous Studies

Transfer learning between similar time series can accelerate optimization:

```python
# Get best parameters from previous study
best_params = previous_study.best_params

# Define search space informed by previous results
def transfer_search_space(trial):
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
    
    # Other parameters with tighter bounds around previous best
    
    return dict(
        likelihood="beta",
        n_changepoints=n_cp,
        delta_scale=delta_scale,
        # Other parameters...
    )

# Create new study with TPE sampler using prior information
sampler = optuna.samplers.TPESampler(seed=42, consider_prior=True)
transfer_study = optuna.create_study(direction="minimize", sampler=sampler)
transfer_study.optimize(transfer_objective, n_trials=20)
```

## 5. Diagnostics & Pitfalls

Proper model diagnostics ensure reliable forecasts. This section covers common issues and their remedies.

### Residual Analysis with ACF and Ljung-Box

Examine residual autocorrelation to validate model fit:

```python
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox

# Calculate residuals
train_residuals = train_y - model.predict(train_t)

# ACF plot
plt.figure(figsize=(10, 5))
plot_acf(train_residuals, lags=20, alpha=0.05)
plt.title("Residual Autocorrelation")
plt.tight_layout()

# Ljung-Box test
lb_test = acorr_ljungbox(train_residuals, lags=12, return_df=True)
print("Ljung-Box p-values:")
print(lb_test["lb_pvalue"])
```

Low Ljung-Box p-values (<0.05) indicate remaining autocorrelation, suggesting the model may be missing patterns in the data.

### "Too-Smooth" Forecasts

If forecasts appear overly smooth and miss pattern details:

1. Increase flexibility by widening `delta_scale` (try 0.1 to 0.3)
2. Add more changepoints via `n_changepoints` parameter
3. Increase number of harmonics for complex seasonal patterns
4. For Beta likelihood, consider decreasing heteroscedastic effect by reducing `beta_phi`

Example correction:

```python
# Original smooth model
smooth_model = fit_churn_model(
    t=df['t'], y=df['y'],
    delta_scale=0.05,        # Small delta_scale
    n_changepoints=3,        # Few changepoints
    num_harmonics=2          # Few harmonics
)

# More flexible model
flexible_model = fit_churn_model(
    t=df['t'], y=df['y'],
    delta_scale=0.2,         # Increased delta_scale
    n_changepoints=8,        # More changepoints
    num_harmonics=4          # More harmonics
)
```

### Stan Divergence Warnings & Treedepth Issues

When using MCMC inference (`inference="nuts"`), Stan may report divergences or exceed maximum treedepth:

```
Warning: There were 14 divergent transitions after warmup.
Warning: The maximum tree depth of 10 was reached.
```

To address these issues:

1. Increase adaptation target via `adapt_delta` parameter (0.95-0.99)
2. Increase maximum tree depth via `max_treedepth` (12-15)
3. For heteroscedastic models, reduce `beta_phi` or `beta_sigma` values
4. Consider standardizing your time variable (`t`) to improve numerical stability

Example with adjusted parameters:

```python
model = fit_churn_model(
    t=df['t'], y=df['y'],
    inference="nuts",
    adapt_delta=0.98,        # Higher adaptation target (default is 0.95)
    max_treedepth=12,        # Increased tree depth (default is 10)
    # Other parameters...
)
```

## 6. Examples Gallery

This section showcases real-world applications of Murphet on different datasets.

### Retail Inventories-to-Sales Ratio

The FRED Retail Inventory-to-Sales ratio dataset demonstrates Murphet's ability to capture structural changes in economic data:

```python
# Load retail I/R data
df = pd.read_csv("RETAILIRNSA.csv", parse_dates=["ds"])
df["y"] = (1 / df["y"]).clip(1e-6, 1 - 1e-6)  # Invert ratio, keep (0,1)
df["t"] = np.arange(len(df))

# Train-test split
train = df.iloc[:-12]  # All but last 12 months
test = df.iloc[-12:]   # Last 12 months

# Fit optimized model
model = fit_churn_model(
    t=train["t"], y=train["y"],
    likelihood="beta",
    periods=[12.0, 3.0],  # Yearly and quarterly seasonality
    num_harmonics=[3, 2], 
    n_changepoints=6,
    delta_scale=0.15,
    gamma_scale=5.0,
    season_scale=1.2,
    inference="nuts"      # Full Bayesian inference
)

# Generate forecast
forecast = model.predict(test["t"])

# Calculate metrics
rmse_value = np.sqrt(np.mean((test["y"] - forecast) ** 2))
print(f"Test RMSE: {rmse_value:.4f}")
```

This dataset shows how Murphet's Beta likelihood can effectively model bounded ratios and adapt to structural changes like the COVID-19 pandemic's impact on retail inventories.

### Hotel Occupancy Data

Monthly hotel occupancy rates demonstrate the importance of seasonal modeling:

```python
# Fit model to hotel occupancy data
hotel_model = fit_churn_model(
    t=hotel_df["t"], y=hotel_df["y"],
    likelihood="beta",
    periods=[12.0, 6.0],   # Yearly and half-yearly patterns
    num_harmonics=[4, 2],
    n_changepoints=5,
    inference="map"
)

# Plot results with actual data
plt.figure(figsize=(12, 6))
plt.plot(hotel_df["ds"], hotel_df["y"], "k.-", label="Actual")
plt.plot(hotel_df["ds"], hotel_model.predict(hotel_df["t"]), 
         "b-", label="Model Fit")
plt.title("Hotel Occupancy Rate - High Tariff A")
plt.ylabel("Occupancy Rate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
```

### Synthetic Churn Demo

For educational purposes, here's a synthetic churn rate example:

```python
# Generate synthetic data
np.random.seed(42)
t = np.arange(48)
seasonal = 0.05 * np.sin(2 * np.pi * t / 12)
trend = 0.01 * t / 48
noise = np.random.normal(0, 0.01, size=48)
level = 0.2  # Base churn rate

y = level + trend + seasonal + noise
y = np.clip(y, 0.05, 0.95)  # Ensure 0<y<1

# Create DataFrame
synthetic_df = pd.DataFrame({
    "ds": pd.date_range(start="2020-01-01", periods=len(t), freq="MS"),
    "t": t,
    "y": y
})

# Fit model
churn_model = fit_churn_model(
    t=synthetic_df["t"], y=synthetic_df["y"],
    likelihood="beta",
    periods=12,
    num_harmonics=2,
    n_changepoints=3,
    delta_scale=0.1,
    inference="map"
)

# Generate future predictions
future_t = np.arange(len(t), len(t) + 12)
future_ds = pd.date_range(
    start=synthetic_df["ds"].iloc[-1] + pd.DateOffset(months=1),
    periods=12,
    freq="MS"
)
future_y = churn_model.predict(future_t)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(synthetic_df["ds"], synthetic_df["y"], "ko-", label="Historical")
plt.plot(future_ds, future_y, "ro-", label="Forecast")
plt.axvline(synthetic_df["ds"].iloc[-1], ls="--", color="gray")
plt.title("Synthetic Churn Rate Forecast")
plt.ylabel("Churn Rate")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
```

## 7. API Reference Shortcuts

This section provides quick reference for Murphet's core API functions and links to comprehensive documentation.

```python
# Core fitting function
from murphet import fit_churn_model

# Model configuration
model = fit_churn_model(
    t,                      # Time index (numeric)
    y,                      # Target values (0<y<1 for beta)
    
    # Likelihood options
    likelihood="beta",      # "beta" or "gaussian"
    
    # Trend configuration
    n_changepoints=5,       # Number of changepoints
    changepoints=None,      # Explicit changepoint locations (or None)
    delta_scale=0.1,        # Regularization strength
    gamma_scale=5.0,        # Changepoint transition steepness
    
    # Seasonality configuration
    periods=12.0,           # Period length(s)
    num_harmonics=3,        # Harmonics per period
    auto_detect=False,      # Auto-detect periods via FFT
    season_scale=1.0,       # Prior scale for seasonality
    
    # Inference options
    inference="map",        # "map", "advi", or "nuts"
    chains=2,               # Number of chains (MCMC/ADVI)
    iter=4000,              # Iterations 
    warmup=0,               # Warmup iterations (MCMC)
    adapt_delta=0.95,       # Adaptation target (MCMC)
    max_treedepth=12,       # Maximum tree depth (MCMC)
    threads_per_chain=None, # Threading control
    seed=None               # Random seed
)

# Prediction
forecast = model.predict(
    t_new,                  # New time points
    method="mean_params"    # Prediction method
)

# Extract raw Stan fit
stan_fit = model.fit
```

For detailed parameter documentation, refer to the autogenerated docstrings:

```python
help(fit_churn_model)
```

For comprehensive information about Stan parameters and priors, see the [Stan Reference Guide](./stan_reference.md).

---

*This document was generated from Murphet v1.4.0 documentation. For further details and updates, check the [official repository](https://github.com/halsted312/murphet).*