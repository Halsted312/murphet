# 📑 Murphet Stan Reference Guide

> **Technical reference for the Stan models powering Murphet's time-series forecasting**

This document provides a comprehensive reference for the Stan models that power Murphet's forecasting capabilities. Understanding these parameters will help you customize the model for specific time-series characteristics and optimize performance.

## Core Model Structure

Murphet implements two primary likelihood models:

- **`murphet_beta.stan`**: Beta likelihood for bounded (0,1) values like conversion rates
- **`murphet_gauss.stan`**: Gaussian/Student-t likelihood for unbounded values

The model has this overall structure:

```
y ~ Likelihood(μ, dispersion_params)
μ = f_trend(t) + f_seasonal(t) + f_AR(t)
```

Where each component is carefully parameterized with thoughtful priors.

## Parameter Reference

### 1. Trend Component Parameters

| Parameter | Dimensions | Purpose | Prior (Beta) | Prior (Gaussian) | Tuning Range |
|-----------|------------|---------|--------------|------------------|--------------|
| `k` | scalar | Base slope | 𝒩(0, 0.5) | 𝒩(0, 0.5) | Not typically tuned |
| `m` | scalar | Intercept | 𝒩(0, 5) | 𝒩(0, 5) | Not typically tuned |
| `delta` | [num_cp] | Changepoint adjustments | Laplace(0, δ_scale) | Laplace(0, δ_scale) | — |
| `delta_scale` | scalar | Regularization strength | — | — | 0.01-0.6 |
| `gamma` | scalar | CP transition steepness | Γ(3, 1/γ_scale) | Γ(3, 1/γ_scale) | — |
| `gamma_scale` | scalar | Prior scale for γ | — | — | 0.5-15 |

### 2. Seasonality Parameters

| Parameter | Dimensions | Purpose | Prior | Tuning Range |
|-----------|------------|---------|-------|--------------|
| `A_sin` | [total_harmonics] | Sine coefficients | 𝒩(0, 10·season_scale) | — |
| `B_cos` | [total_harmonics] | Cosine coefficients | 𝒩(0, 10·season_scale) | — |
| `season_scale` | scalar | Seasonality prior strength | — | 0.3-2.0 |
| `num_seasons` | scalar | Number of seasonal periods | — | — |
| `period` | [num_seasons] | Length of each period | — | — |
| `n_harmonics` | [num_seasons] | Fourier terms per period | — | 1-10 |

### 3. Autoregressive Component

| Parameter | Purpose | Prior (Beta) | Prior (Gaussian) | Constraints |
|-----------|---------|--------------|------------------|-------------|
| `rho` | AR(1) coefficient | 𝒩(0, 0.3)† | 𝒩(0, 0.3)† | (-1, 1) |
| `mu0` | Initial AR state | 𝒩(logit(ȳ), 1) | 𝒩(ȳ, 1) | ℝ |

† Prior SD automatically expands to 0.5 for quarterly data.

### 4. Likelihood-Specific Parameters

#### Beta Likelihood

| Parameter | Purpose | Prior | Notes |
|-----------|---------|-------|-------|
| `log_phi0` | Base precision (log scale) | 𝒩(log(20), 1) | Higher = narrower predictions |
| `beta_phi` | Heteroscedasticity strength | 𝒩(0, 0.3) | 0 = homoscedastic |

#### Gaussian/Student-t Likelihood

| Parameter | Purpose | Prior | Notes |
|-----------|---------|-------|-------|
| `log_sigma0` | Base scale (log) | 𝒩(log(sd(y)), 1) | Base noise level |
| `beta_sigma` | Heteroscedasticity strength | 𝒩(0, 0.5) | 0 = homoscedastic |
| `nu` | Student-t degrees of freedom | Exp(1/30) | ≥2; larger = more Gaussian |

## Data Block Specification

The Stan data block (automatically constructed by `fit_churn_model`):

```stan
int<lower=1>    N;                          // number of observations 
vector[N]       t;                          // time points
vector[N]       y;                          // target values: 0<y<1 for Beta, ℝ for Gaussian

int<lower=0>    num_changepoints;           // number of trend changepoints
vector[num_changepoints] s;                 // changepoint locations
real<lower=0>   delta_scale, gamma_scale;   // trend prior scales

int<lower=1>              num_seasons;      // number of seasonal components
array[num_seasons] int    n_harmonics;      // harmonics per seasonal component
array[num_seasons] real   period;           // period lengths
int<lower=1>              total_harmonics;  // sum(n_harmonics)
real<lower=0>             season_scale;     // seasonality prior scale
```

## Parameter Tuning Guidelines

### By Data Frequency

| Parameter | Daily | Monthly | Quarterly |
|-----------|-------|---------|-----------|
| `gamma_scale` | 3.0-12.0 | 1.0-8.0 | 0.5-4.0 |
| `delta_scale` | 0.05-0.6 | 0.02-0.4 | 0.01-0.3 |
| `n_changepoints` | ≤14 | ≤10 | ≤6 |
| `season_scale` | 0.3-2.0 | 0.3-2.0 | 0.3-1.5 |

### For Different Data Types

| Data Characteristic | Recommendation |
|---------------------|----------------|
| **High volatility** | ↑ `delta_scale`, ↑ Gaussian `nu` |
| **Strong seasonality** | ↑ `num_harmonics`, ↑ `season_scale` |
| **Abrupt changes** | ↑ `gamma_scale`, ↑ `n_changepoints` |
| **Smooth trends** | ↓ `delta_scale`, ↓ `gamma_scale` |
| **Stable errors** | ↓ `beta_phi`/`beta_sigma` |
| **Heteroscedastic errors** | ↑ `beta_phi`/`beta_sigma` |

## Common Gotchas & Solutions

1. **Boundary violations**: Ensure `likelihood="beta"` for 0-1 bounded data.

2. **Adaptation failures**: If Stan reports "divergences" or "treedepth warnings":
   - Start with smaller `beta_phi` (≤0.3)
   - Increase `adapt_delta` (0.95-0.99)
   - Increase `max_treedepth` (12-15)

3. **Excessive changepoints**: Consider reducing `n_changepoints` with shorter series.

4. **Overfitting seasons**: Reduce `num_harmonics` and `season_scale` for noisy data.

5. **AR(1) instability**: Keep `rho` priors reasonably tight unless you have strong evidence for autocorrelation.

## Version History

| Version | Date | Key Changes |
|---------|------|-------------|
| 1.3.0   | 2025-04-19 | Heteroscedastic φ/σ, Student-t ν, expanded AR(1) |

## Diagnostic Functions

These posterior predictive quantities are available for NUTS inference:

| Variable | Dimensions | Purpose |
|----------|------------|---------|
| `y_rep` | [N] | Posterior predictive sample (for PPC) |
| `log_lik` | [N] | Pointwise log-likelihood (for WAIC/LOO) |

## Best Practices

1. **Start simple**: Begin with default priors before customizing.

2. **Model selection**: Use cross-validation or WAIC/LOO for comparing specifications.

3. **Prior predictive checks**: For advanced users, examine the behavior of priors with `generated quantities` block.

4. **Inference method progression**: Start with MAP for exploration, then ADVI/NUTS for uncertainty.

5. **Heteroscedasticity**: Enable only after confirming residual variance pattern.

---

For practical examples and Optuna HPO recipes, see the companion documents on GitHub.