import os
import numpy as np
from cmdstanpy import CmdStanModel
from scipy.special import expit

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STAN_FILE = os.path.join(CURRENT_DIR, 'improved_model.stan')


class ChurnProphetModel:
    def __init__(self, cmdstan_model, fit_result, changepoints, num_harmonics, period):
        self.cmdstan_model = cmdstan_model
        self.fit_result = fit_result
        self.changepoints = np.array(changepoints) if changepoints is not None else None
        self.num_harmonics = num_harmonics
        self.period = period

        # Extract and cache parameter means for faster prediction
        self.k_mean = np.mean(self.fit_result.stan_variable('k'))
        self.m_mean = np.mean(self.fit_result.stan_variable('m'))
        self.q_mean = np.mean(self.fit_result.stan_variable('q'))
        self.delta_mean = np.mean(self.fit_result.stan_variable('delta'), axis=0)
        self.gamma_mean = np.mean(self.fit_result.stan_variable('gamma'))
        self.A_sin_mean = np.mean(self.fit_result.stan_variable('A_sin'), axis=0)
        self.B_cos_mean = np.mean(self.fit_result.stan_variable('B_cos'), axis=0)

    def summary(self):
        """Return a summary DataFrame of the fitted model."""
        return self.fit_result.summary()

    def predict(self, t_new, method='full_posterior'):
        """
        Predict churn rates for new time points.

        Args:
            t_new: 1D array of future time points
            method: 'full_posterior' or 'mean_params'
                   'full_posterior': uses all posterior samples (slower but more accurate)
                   'mean_params': uses posterior means (faster)

        Returns: Posterior predicted churn rates
        """
        t_new = np.array(t_new)

        if method == 'mean_params':
            # Fast prediction using posterior means
            predictions = np.zeros(len(t_new))

            for j, t_val in enumerate(t_new):
                # Compute trend
                cp_effect = 0
                if self.changepoints is not None:
                    logistic_terms = expit(self.gamma_mean * (t_val - self.changepoints))
                    cp_effect = np.sum(self.delta_mean * logistic_terms)

                trend = self.k_mean * t_val + self.m_mean + self.q_mean * (t_val ** 2) + cp_effect

                # Compute seasonal component
                t_mod = t_val - np.floor(t_val / self.period) * self.period
                seas = 0
                for r in range(self.num_harmonics):
                    angle = 2 * np.pi * (r + 1) * t_mod / self.period
                    seas += self.A_sin_mean[r] * np.sin(angle) + self.B_cos_mean[r] * np.cos(angle)

                mu = trend + seas
                mu_sat = min(mu, 4)
                predictions[j] = expit(mu_sat)

            return predictions

        else:  # Full posterior
            # Extract posterior samples
            k_samples = self.fit_result.stan_variable('k')
            m_samples = self.fit_result.stan_variable('m')
            q_samples = self.fit_result.stan_variable('q')
            delta_samples = self.fit_result.stan_variable('delta')
            gamma_samples = self.fit_result.stan_variable('gamma')
            A_sin_samples = self.fit_result.stan_variable('A_sin')
            B_cos_samples = self.fit_result.stan_variable('B_cos')

            n_draws = len(k_samples)
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
                    # Compute trend
                    cp_effect = 0
                    if self.changepoints is not None:
                        cp_effect = np.sum(delta * expit(gamma * (t_val - self.changepoints)))

                    trend = k * t_val + m + q * (t_val ** 2) + cp_effect

                    # Compute seasonal component
                    t_mod = t_val - np.floor(t_val / self.period) * self.period
                    seas = 0
                    for r in range(self.num_harmonics):
                        angle = 2 * np.pi * (r + 1) * t_mod / self.period
                        seas += A_sin[r] * np.sin(angle) + B_cos[r] * np.cos(angle)

                    mu = trend + seas
                    mu_sat = min(mu, 4)
                    predictions[i, j] = expit(mu_sat)

            return predictions.mean(axis=0)


def fit_churn_model(
        t, y,
        n_changepoints=None,
        changepoints=None,
        num_harmonics=3,
        period=12.0,  # Changed default from 7.0 to 12.0 for monthly data
        delta_scale=0.05,  # Reduced from 0.1
        chains=4,  # Increased from 2
        iter=2000,  # Increased from 1500
        warmup=1000,  # Increased from 750
        adapt_delta=0.95,  # Added control
        max_treedepth=15,  # Added control
        seed=42
):
    """
    Fit an improved Bayesian churn model with smooth changepoints and seasonality.
    """
    t = np.array(t, dtype=float)
    y = np.array(y, dtype=float)

    # Validate y is in (0,1)
    if np.any(y <= 0) or np.any(y >= 1):
        # Apply a small buffer to zeros and ones
        y = np.clip(y, 0.001, 0.999)
        print("Warning: Values ≤0 or ≥1 detected and clipped to (0.001, 0.999)")

    # Set n_changepoints if not provided
    if n_changepoints is None:
        n_changepoints = max(1, int(round(0.2 * len(t))))  # Reduced from 0.3

    # Compute changepoints if not provided
    if changepoints is None:
        # Place changepoints at quantiles with a buffer from the edges
        quantiles = np.linspace(0.1, 0.9, n_changepoints + 2)[1:-1]
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
        adapt_delta=adapt_delta,
        max_treedepth=max_treedepth,
        seed=seed,
        show_progress=True
    )

    # Print diagnostic summary
    print(fit_result.diagnose())

    return ChurnProphetModel(cmdstan_model, fit_result, changepoints, num_harmonics, period)