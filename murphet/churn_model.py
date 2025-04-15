import os
import numpy as np
from cmdstanpy import CmdStanModel

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STAN_FILE = os.path.join(CURRENT_DIR, 'model.stan')


class ChurnProphetModel:
    def __init__(self, cmdstan_model, fit_result, changepoints, num_harmonics, period):
        self.cmdstan_model = cmdstan_model
        self.fit_result = fit_result  # CmdStanMCMC object
        self.changepoints = np.array(changepoints) if changepoints is not None else None
        self.num_harmonics = num_harmonics
        self.period = period

    def summary(self):
        """Return a summary DataFrame of the fitted model."""
        return self.fit_result.summary()

    def predict(self, t_new):
        """
        Predict churn rates for new time points.

        t_new: 1D array of future time points.
        Returns: Posterior mean predicted churn rates.
        """
        # Extract posterior samples from CmdStanPy:
        k_samples = self.fit_result.stan_variable('k')
        m_samples = self.fit_result.stan_variable('m')
        q_samples = self.fit_result.stan_variable('q')
        delta_samples = self.fit_result.stan_variable('delta')
        gamma_samples = self.fit_result.stan_variable('gamma')
        A_sin_samples = self.fit_result.stan_variable('A_sin')
        B_cos_samples = self.fit_result.stan_variable('B_cos')

        n_draws = len(k_samples)
        t_new = np.array(t_new)
        predictions = np.zeros((n_draws, len(t_new)))

        for i in range(n_draws):
            k = k_samples[i]
            m = m_samples[i]
            q = q_samples[i]
            delta = delta_samples[i]
            gamma = gamma_samples[i]
            A_sin = A_sin_samples[i]
            B_cos = B_cos_samples[i]

            for j, t_val in enumerate(t_new):
                # Compute trend:
                cp_effect = 0
                if self.changepoints is not None:
                    cp_effect = np.sum(delta * (1 / (1 + np.exp(-gamma * (t_val - self.changepoints)))))
                trend = k * t_val + m + q * (t_val ** 2) + cp_effect

                # Compute seasonal component using modulo operation:
                t_mod = t_val - np.floor(t_val / self.period) * self.period
                seas = 0
                for r in range(self.num_harmonics):
                    seas += (A_sin[r] * np.sin(2 * np.pi * (r + 1) * t_mod / self.period) +
                             B_cos[r] * np.cos(2 * np.pi * (r + 1) * t_mod / self.period))
                mu = trend + seas
                # Apply the same saturation as in Stan:
                mu_sat = min(mu, 4)
                predictions[i, j] = 1 / (1 + np.exp(-mu_sat))

        return predictions.mean(axis=0)


def fit_churn_model(
        t, y,
        n_changepoints=None,  # if None, will be computed as 30% of the data points
        changepoints=None,
        num_harmonics=3,
        period=7.0,
        delta_scale=0.1,
        chains=2,
        iter=1500,
        warmup=750,
        seed=42
):
    """
    Fit the churn model using CmdStanPy.

    Parameters:
    -----------
    t : array-like
        1D array of time indices.
    y : array-like
        Churn rates (strictly between 0 and 1).
    n_changepoints : int or None, default=None
        The number of changepoints to use. If None, defaults to approximately 30% of
        the number of data points (i.e. max(1, round(0.3 * len(t)))).
    changepoints : array-like or None, default=None
        User-specified changepoints. If None, automatically computed quantiles of t
        based on n_changepoints.
    num_harmonics : int, default=3
        Number of Fourier pairs for the seasonal component.
    period : float, default=7.0
        The seasonal period (e.g., 7 for weekly, 12 for monthly with yearly seasonality).
    delta_scale : float, default=0.1
        Scale for the Laplace prior on changepoint adjustments.
    chains : int, default=2
        Number of MCMC chains.
    iter : int, default=1500
        Total iterations per chain (including warmup).
    warmup : int, default=750
        Number of warmup (burn-in) iterations.
    seed : int, default=42
        Random seed for reproducibility.

    Returns:
    --------
    A ChurnProphetModel instance with fitted results.
    """
    t = np.array(t, dtype=float)
    y = np.array(y, dtype=float)

    # Check that y values are strictly between 0 and 1.
    if (y <= 0).any() or (y >= 1).any():
        raise ValueError("All churn rates must be strictly between 0 and 1.")

    # If n_changepoints is not provided, set it to roughly 30% of the data points.
    if n_changepoints is None:
        n_changepoints = max(1, int(round(0.3 * len(t))))

    # Automatically compute changepoints if not provided.
    if changepoints is None:
        # Compute quantiles to place n_changepoints evenly over the range of t.
        quantiles = np.linspace(0, 1, n_changepoints + 2)[1:-1]
        changepoints = np.quantile(t, quantiles)
    else:
        changepoints = np.sort(np.array(changepoints, dtype=float))
        n_changepoints = len(changepoints)

    stan_data = {
        'N': len(y),
        'y': y,
        't': t,
        'num_changepoints': n_changepoints,
        's': changepoints,
        'delta_scale': delta_scale,
        'num_harmonics': num_harmonics,
        'period': period
    }

    cmdstan_model = CmdStanModel(stan_file=STAN_FILE)
    iter_sampling = iter - warmup
    fit_result = cmdstan_model.sample(
        data=stan_data,
        chains=chains,
        parallel_chains=chains,
        iter_warmup=warmup,
        iter_sampling=iter_sampling,
        seed=seed
    )

    return ChurnProphetModel(cmdstan_model, fit_result, changepoints, num_harmonics, period)
