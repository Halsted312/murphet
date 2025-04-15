# Murphet

A Bayesian time-series model for churn rates and percentages with changepoints and seasonality. Inspired by Prophet but specifically designed for values between 0 and 1 (such as churn rates, conversion rates, or percentages).

## Installation

```bash
pip install murphet
```

## Requirements

- Python 3.7 or higher
- CmdStanPy 0.10.0 or higher
- NumPy 1.19 or higher
- Pandas 1.0.0 or higher

## Features

- Bayesian modeling of time series that represent rates or percentages (0-1 values)
- Flexible trend with automatic changepoint detection
- Seasonality modeling via Fourier series
- Smooth changepoint transitions
- Uncertainty intervals for forecasts
- Compatible with monthly, weekly, or any regular time interval

## Quick Start

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

## Detailed Documentation

The model combines:

1. **Trend Component**: Flexible trend with automatic changepoint detection
   - Linear + optional quadratic terms
   - Smooth transitions at changepoints 

2. **Seasonal Component**: Fourier series to model recurring patterns
   - Configurable harmonics and period
   - Handles any regular seasonality (yearly, weekly, etc.)

3. **Statistical Model**: 
   - Uses a Beta likelihood appropriate for rate/percentage data
   - Provides uncertainty intervals from the posterior distribution

## License

MIT

## Citation

If you use this package in your research, please cite:

```
Murphy, S. (2025). Murphet: A Bayesian Time-Series Model for Churn Rates with Changepoints and Seasonality. 
https://github.com/halsted312/murphet
```