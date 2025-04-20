# 📈 Murphet — Making Prophet a Beta (Pun Intended!)

[![PyPI version](https://img.shields.io/badge/pypi-v1.5.1-blue)](https://pypi.org/project/murphet/)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

> A Stan-powered time-series model that **never breaks the 0-1 bounds** while delivering more accurate forecasts than Prophet for rates and probabilities.

*Your conversion rates deserve better than to be treated like stock prices!*

---

## 🚀 Prophet, but Better for Bounded Data

Prophet is great, but it wasn't built for churn rates, conversion percentages, or hotel occupancy. **Murphet was.** 

When forecasting rates, the last thing you want is predictions that break the logical 0-1 bounds. But that's exactly what happens with vanilla Prophet when forecasting:
- Conversion rates
- Churn probabilities
- Occupancy percentages
- Click-through rates
- Any metric bounded between 0 and 1

Murphet fixes this with a **Beta likelihood** that respects these natural boundaries while providing more accurate forecasts. It's Prophet's smarter cousin - specifically designed for the data you're actually working with.

---

## 1 · Why Choose Murphet?

| Prophet's Limitation | Murphet's Solution | Your Benefit |
|----------------------|--------------------|--------------------|
| ❌ Predictions can go <0 or >1 | ✅ **Beta head** keeps everything in (0,1) | Never explain impossible forecasts to stakeholders again |
| ❌ One-size-fits-all variance model | ✅ **Heteroscedastic precision** adapts to data level | Better uncertainty intervals, especially near boundaries |
| ❌ Hard changepoints create artificial kinks | ✅ **Smooth logistic transitions** between trends | More realistic forecasts with less overfitting |
| ❌ Seasonality coefficients often explode | ✅ **Regularized Fourier terms** with sensible priors | Stable seasonal patterns even with limited data |
| ❌ Ignores autocorrelation in residuals | ✅ **Latent AR(1) structure** captures persistent patterns | Dramatically improved forecast accuracy |
| ❌ Same error structure for all predictions | ✅ **Data-adaptive variance** via smart link functions | Properly calibrated prediction intervals |

---

## 2 · Installation in Seconds

```bash
pip install murphet              # wheels include pre-compiled Stan models
```

### Requirements

* Python >= 3.8 & CmdStanPy >= 1.0 (auto-installed)
* A recent CmdStan toolchain (gcc/clang) — `cmdstanpy.install_cmdstan()` will fetch it.

---

## 3 · Quick Start (Just 10 Lines!)

```python
import pandas as pd, numpy as np
from murphet import fit_churn_model

df = pd.read_csv("churn_data.csv")        # cols: ds, y  (0<y<1)
df["ds"] = pd.to_datetime(df["ds"])
df["t"]  = np.arange(len(df))             # integer index

mod = fit_churn_model(
        t              = df["t"],
        y              = df["y"],
        periods        = 12, num_harmonics = 3,    # yearly seasonality
        n_changepoints = 4,
        likelihood     = "beta",                   # default & safest
        inference      = "nuts",                   # full posterior
      )

future_t = np.arange(len(df), len(df)+6)
fcst     = mod.predict(future_t)
```

---

## 4 · Model Architecture

<img src="docs/figs/murphet_diagram.svg" alt="Murphet Model Architecture" width="50%">

Murphet combines the best of structural time series modeling with modern Bayesian methods:

| Component | Equation | What It Gives You |
|-----------|----------|-------------------|
| Trend | *μ<sub>det</sub>(t) = k·t + m + ∑ δ<sub>j</sub> σ(γ (t − s<sub>j</sub>))* | Smooth transitions between trend regimes |
| Seasonality | Fourier blocks on raw *t* (`fmod`) | Flexible multi-periodic patterns |
| Link / saturation | *μ* → `logit⁻¹` → *p* | Automatic boundary adherence |
| Likelihoods | **Beta(p·φᵢ,(1-p)·φᵢ)**   or   **Student-t<sub>ν</sub>(μ,σᵢ)** | Properly scaled uncertainty |
| Latent error | *y\* = μ<sub>det</sub> + ρ·lag* | Capture autocorrelated patterns |

### Advanced Features

| Feature | Implementation                                                                                                            | Business Impact                   |
|---------|---------------------------------------------------------------------------------------------------------------------------|-----------------------------------|
| **AR(1)** latent error | `real<lower=-1,upper=1> rho; real mu0;` + update in `partial_sum_*`                                                       | Catch slow-moving market trends   |
| **Heteroscedastic precision** | `phi_i = exp(log_phi0 - beta_phi*abs(mu_det));` (Beta) / `sigma_i = exp(log_sigma0 + beta_sigma*abs(mu_det));` (Gaussian) | More accurate risk assessment     |
| **Heavy-tail option** | `student_t_lpdf(y \ ν, μ, σᵢ)` with `ν ~ Exp(1/30)`                                                                       | Robustness to outliers and shocks |

---

## 5 · Real-World Performance

### Hong Kong Hotel Occupancy Rates (2020-2025)
![Hotel hold-out](docs/figs/Hotel_A_holdout.png)

| 9-Month Forecast | RMSE       | Improvement |
|------------------|------------|-------------|
| **Murphet β**    | **0.0916** | **21%** better |
| Prophet (optimized) | 0.1159   | Baseline |

### U.S. Retail Inventories-to-Sales Ratio
![Retail hold-out](docs/figs/retail_IR_holdout.png)

| 24-Month Forecast | RMSE | SMAPE | Improvement |
|------------------|------|-------|-------------|
| **Murphet β**    | **0.0496** | **5.15%** | **56%** better |
| Prophet          | 0.1140 | 13.21% | Baseline |

### Residual Diagnostics
![Residual diagnostics](docs/figs/retail_diag.png)

Murphet's AR(1) and heteroscedastic components virtually eliminate autocorrelation structures that Prophet leaves behind.

---

## 6 · Choosing the Right Model Head

| Likelihood | Best For | Technical Details | Examples |
|------------|----------|-------------------|----------|
| **Beta (default)** | True proportions, rates, percentages | Logit⁻¹ link + `Beta(p·φᵢ,(1-p)·φᵢ)` likelihood | Conversion rates, CTR, churn % |
| **Gaussian / Student-t** | Approximate ratios or unbounded metrics | Identity link + `Normal/Student-t(μ,σᵢ)` | Price ratios, normalized KPIs |

Simply set `likelihood="gaussian"` to switch; all other API calls remain identical.

---

## 7 · API Reference 

| Function | Purpose | Example |
|----------|---------|---------|
| `fit_churn_model(t, y, **kwargs)` | Fit the model using MAP, ADVI, or NUTS | See quickstart |
| `model.predict(t_new)` | Generate forecasts | `forecast = model.predict(future_t)` |
| `model.fit_result` | Access raw CmdStanPy object | `draws = model.fit_result.stan_variable("rho")` |
| `model.summary()` | Get parameter summary | `print(model.summary())` |

### Key Parameters

```text
# Seasonality configuration
periods            # Length of seasonal periods (e.g., 12 for monthly data with yearly seasonality)
num_harmonics      # Number of Fourier terms per period (higher = more flexible seasonality)
season_scale       # Prior scale for seasonal components (0.3-2.0 recommended)

# Trend configuration
n_changepoints     # Number of potential trend changes (Prophet default heuristic = 0.2 * N)
delta_scale        # Prior scale for changepoint magnitudes (0.01-0.6 range)
gamma_scale        # Steepness of changepoint transitions (1.0-10.0 range)

# Inference options
likelihood         # "beta" (default) or "gaussian"
inference          # "map" (fastest), "advi" (quick uncertainty), or "nuts" (most accurate)
```

---

## 8 · Coming Soon

* Holiday regressors (Prophet style)  
* Automatic plotting functionality
* Performance optimizations for long MCMC chains
* Integration with Prophet ecosystem tools

---

## 9 · For Academic Use

If you use Murphet in academic work, please cite:

```
Murphy, S. (2025). Murphet: A Bayesian Time-Series Model for Bounded Rates.
https://github.com/halsted312/murphet
```

---

## 10 · Get Started Today

Don't let your bounded metrics be forecasted with unbounded models. Murphet brings the power of Bayesian modeling to your rates and proportions with an easy-to-use API that feels just like Prophet.

```python
# It's as simple as:
from murphet import fit_churn_model

model = fit_churn_model(t=time_index, y=bounded_values)
forecast = model.predict(future_time_points)
```

[Check out the documentation →](https://github.com/halsted312/murphet/tree/main/docs)