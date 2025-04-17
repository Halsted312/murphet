# 📈 Murphet

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.3-blue)](https://pypi.org/project/murphet/)
[![Python](https://img.shields.io/badge/python-3.7%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A Bayesian time-series model for churn rates and percentages with changepoints and seasonality. Inspired by Prophet but specifically designed for values between 0 and 1 (such as churn rates, conversion rates, or percentages).

> 💡 **"Murphet"** offers native (0-1) support with Beta likelihood, smooth changepoints, and guaranteed valid forecasts.

## ✨ Features

- **Native (0-1) Support**: No manual log-odds transformations or clipping needed
- **Beta Likelihood**: Variance adapts to mean; uncertainty widens near 0%/100%
- **Smooth Changepoints**: Logistic ramps instead of hard piece-wise linear breaks
- **Guaranteed Valid Forecasts**: Predictions & intervals never cross 0 or 1
- **Full Bayesian Inference**: Via CmdStanPy's NUTS/HMC sampler
- **Prophet-like API**: Simple `fit_churn_model → predict` workflow

## 🚀 Installation

```bash
pip install murphet
```

### Requirements

- Python 3.7 or higher
- CmdStanPy 0.10.0 or higher
- NumPy 1.19 or higher
- Pandas 1.0.0 or higher

## 🔍 Quick Start

```python
import pandas as pd
import numpy as np
from murphet.churn_model import fit_churn_model

# Load time series data (with columns 'ds' for dates and 'y' for churn rates)
df = pd.read_csv('churn_data.csv')
df['ds'] = pd.to_datetime(df['ds'])

# Create a numeric time index
df['t'] = np.arange(len(df))

# For monthly data with yearly seasonality
model = fit_churn_model(
    t=df['t'].values,
    y=df['y'].values,
    num_harmonics=2,       # Use 2 Fourier terms for seasonality
    period=12.0,           # 12 months per year
    n_changepoints=3,      # Allow 3 potential changes in trend
)

# Forecast 6 months ahead
future_t = np.arange(len(df), len(df) + 6)
predictions = model.predict(future_t)

# Display summary of fitted parameters
print(model.summary())
```

## 🔮 How Murphet Works

### Data & Domain
- Accepts only rates or percentages strictly in (0, 1)
- Uses a numeric time index `t` (integers, irregular gaps allowed)

### Trend Component
<img src="https://latex.codecogs.com/svg.latex?trend(t)=kt+m+qt^2+\sum_j\delta_j\sigma(\gamma(t-s_j))" alt="trend formula" />

- Linear slope `k`, intercept `m`, optional quadratic curvature `q`
- Changepoints (`s`) adjust the slope via smooth logistic transitions (`γ`)
- Changepoints are auto-selected at evenly spaced quantiles (≈30% of data by default)

### Seasonality Component
- Pure Fourier expansion with `num_harmonics` sine/cosine pairs on a chosen period
- Uses fmod on raw time index for numerical stability

### Likelihood & Link
- Linear predictor μ = trend + seasonal is saturated at 4, then passed through the inverse-logit
- Observations follow a Beta(p·φ, (1-p)·φ) distribution — perfect for proportions
- φ is a learned dispersion parameter → credible intervals widen when data are noisy

### Bayesian Inference
- All parameters sampled jointly via No-U-Turn HMC
- Conservative priors reduce overfitting

## 📊 Murphet vs. Prophet

| Feature | Prophet | Murphet |
|---------|---------|---------|
| **Target domain** | Any real number; manual transforms needed | Native (0-1) rates — no extra work |
| **Likelihood** | Normal/Student-t | Beta — variance adapts to mean; never < 0 or > 1 |
| **Changepoint transition** | Hard piece-wise linear | Smooth logistic ramp — less ringing |
| **Quadratic term** | Limited support | Built-in `q·t²` for gentle curvature |
| **Seasonality math** | Fourier on normalized time | Fourier on raw time via fmod |
| **Saturation/link** | Needs cap for logistic growth | Auto-saturates predictor, always uses logit link |
| **Uncertainty** | Sample quantiles of noise | Full posterior Beta variance + φ dispersion |
| **Boundary respect** | Can breach 0/1 | Guaranteed in (0-1) by construction |

### Where Murphet Excels

- KPIs expressed as rates: churn, conversion, click-through, win-loss
- Datasets where variance shrinks near 0% or 100% (Beta handles heteroskedasticity)
- Situations where abrupt jumps are rare; smooth logistic changepoints capture gradual drifts
- When credible intervals must never cross impossible bounds

### When Prophet May Still Win

- Extremely long series with numerous holiday regressors
- Non-rate data on an unbounded scale
- Very sharp structural breaks better captured by piece-wise linear segments

## ⚙️ Tunable Parameters

| Argument | Purpose | Typical Range / Tip |
|----------|---------|---------------------|
| `n_changepoints` | How many potential slope shifts | 0-10, auto ≈ 30% × N if None |
| `changepoints` | Manual changepoint positions | Sorted array of t values |
| `delta_scale` | Laplace prior scale on each δ | 0.05-0.5 (smaller ⇒ smoother trend) |
| `num_harmonics` | Fourier pairs for seasonality | 1-4 for weekly/monthly; ↑ for richer cycles |
| `period` | Seasonal period in time-index units | 7, 12, 365/day-index, etc. |
| `chains` | Parallel MCMC chains | 2-4 |
| `iter`, `warmup` | Total vs. warm-up iterations | Keep warm-up ≈ ½ for stable adaptation |
| `seed` | RNG seed for reproducibility | Any int |

## 🧠 Parameters Learned

- `k`, `m`, `q` — base trend slope, intercept, quadratic curvature
- `delta[num_changepoints]` — changepoint adjustments 
- `gamma` — steepness of each changepoint's logistic ramp
- `A_sin[num_harmonics]`, `B_cos[num_harmonics]` — Fourier coefficients
- `phi` (φ) — Beta likelihood precision (controls width of intervals)

Access any of these via:
```python
model.summary()                           # tidy dataframe (mean, sd, R-hat, etc.)
delta_draws = model.fit_result.stan_variable("delta")  # raw posterior samples
```

## 💡 Usage Tips

### Diagnosing Changepoints
Plot delta posteriors; wide HPD intervals ⇒ reduce `n_changepoints` or shrink `delta_scale`.

### Multiple Seasonalities
Fit separate models (daily + annual) and ensemble, or extend seasonality.py to stack Fourier blocks.

### Speed Tricks
```python
# For quick prototyping
model = fit_churn_model(..., chains=1, iter=800)

# Compile once and reuse
from cmdstanpy import CmdStanModel
compiled_model = CmdStanModel(stan_file=STAN_FILE, cpp_options={"STAN_THREADS": "TRUE"})
```

### Forecast Distribution (for fan charts)
```python
draws = model.fit_result.draws_pd(vars=['k','m','q','delta','A_sin','B_cos','gamma'])
# Then rebuild predictions vectorized over draws
```

## 📝 License

MIT

## 📚 Citation

If you use this package in your research, please cite:

```
Murphy, S. (2025). Murphet: A Bayesian Time-Series Model for Churn Rates with Changepoints and Seasonality. 
https://github.com/halsted312/murphet
```
