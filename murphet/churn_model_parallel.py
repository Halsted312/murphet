# churn_model_parallel.py  ✧  2025‑04‑18
# ---------------------------------------------------------------
# Multithreaded Murphet backend with optional MAP / ADVI inference
# ---------------------------------------------------------------
import os
import multiprocessing as _mp
import numpy as np
from cmdstanpy import CmdStanModel, CmdStanMLE, CmdStanVB, CmdStanMCMC
from scipy.special import expit

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
STAN_FILE   = os.path.join(CURRENT_DIR, "improved_model_parallel.stan")

# ──────────────────────────────────────────────────────────────────
# 0) Configure Stan thread pool once
# ──────────────────────────────────────────────────────────────────
def _set_num_threads(n):
    n_cpu = _mp.cpu_count()
    n = max(1, min(n, n_cpu))            # cap to available cores
    os.environ["STAN_NUM_THREADS"] = str(n)
    return n

DEFAULT_THREADS = _set_num_threads(int(os.getenv("STAN_NUM_THREADS", 8)))

# ──────────────────────────────────────────────────────────────────
# 1) Predictor that handles MCMC, VB, and MAP results
# ──────────────────────────────────────────────────────────────────
class ChurnProphetModel:
    def __init__(self, cmdstan_model, fit_result,
                 changepoints, num_harmonics, period):
        self.cmdstan_model  = cmdstan_model
        self.fit_result     = fit_result
        self.changepoints   = (np.array(changepoints)
                               if changepoints is not None else None)
        self.num_harmonics  = num_harmonics
        self.period         = period

        # ─ Extract posterior means OR point estimates ──────────────
        if isinstance(fit_result, CmdStanMCMC) or isinstance(fit_result, CmdStanVB):
            self.k_mean     = np.mean(fit_result.stan_variable("k"))
            self.m_mean     = np.mean(fit_result.stan_variable("m"))
            self.q_mean     = np.mean(fit_result.stan_variable("q"))
            self.delta_mean = np.mean(fit_result.stan_variable("delta"), axis=0)
            self.gamma_mean = np.mean(fit_result.stan_variable("gamma"))
            self.A_sin_mean = np.mean(fit_result.stan_variable("A_sin"), axis=0)
            self.B_cos_mean = np.mean(fit_result.stan_variable("B_cos"), axis=0)

        elif isinstance(fit_result, CmdStanMLE):           # MAP
            p = fit_result.optimized_params_dict

            self.k_mean     = p["k"]
            self.m_mean     = p["m"]
            self.q_mean     = p["q"]
            self.gamma_mean = p["gamma"]

            self.delta_mean = np.array(
                [p[f"delta[{i+1}]"] for i in range(len(changepoints))]
            )
            self.A_sin_mean = np.array(
                [p[f"A_sin[{i+1}]"] for i in range(num_harmonics)]
            )
            self.B_cos_mean = np.array(
                [p[f"B_cos[{i+1}]"] for i in range(num_harmonics)]
            )
        else:
            raise TypeError("Unsupported CmdStan result type.")

    # ----------------------------------------------------------------
    def summary(self):
        return (self.fit_result.summary() if hasattr(self.fit_result, "summary")
                else self.fit_result.optimized_params_pd)

    # ----------------------------------------------------------------
    def predict(self, t_new, method="mean_params"):
        if method != "mean_params":
            raise NotImplementedError("Only mean_params supported for MAP/VB.")
        t_new = np.asarray(t_new, dtype=float)
        preds = np.empty(len(t_new))

        for j, t in enumerate(t_new):
            cp = 0.0
            if self.changepoints is not None:
                cp = np.sum(self.delta_mean *
                            expit(self.gamma_mean * (t - self.changepoints)))

            trend = self.k_mean * t + self.m_mean + self.q_mean * t**2 + cp

            t_mod = t - np.floor(t / self.period) * self.period
            seas  = 0.0
            for r in range(self.num_harmonics):
                ang  = 2 * np.pi * (r + 1) * t_mod / self.period
                seas += (self.A_sin_mean[r] * np.sin(ang) +
                         self.B_cos_mean[r] * np.cos(ang))

            preds[j] = expit(min(trend + seas, 4))
        return preds

# ──────────────────────────────────────────────────────────────────
# 2) Fit function
# ──────────────────────────────────────────────────────────────────
def fit_churn_model(
        t, y,
        n_changepoints=None, changepoints=None,
        num_harmonics=3, period=12.0,
        delta_scale=0.05,
        chains=4, iter=2000, warmup=1000,
        adapt_delta=0.95, max_treedepth=12,
        threads_per_chain=DEFAULT_THREADS,
        inference="nuts",          # {"nuts","map","advi"}
        seed=42):

    # ─ Data prep ─
    t = np.asarray(t, dtype=float)
    y = np.clip(np.asarray(y, dtype=float), 0.001, 0.999)

    if n_changepoints is None:
        n_changepoints = max(1, int(round(0.2 * len(t))))

    if changepoints is None:
        qs = np.linspace(0.1, 0.9, n_changepoints + 2)[1:-1]
        changepoints = np.quantile(t, qs)
    else:
        changepoints = np.sort(np.asarray(changepoints, dtype=float))
        n_changepoints = len(changepoints)

    stan_data = {
        "N": len(y), "y": y, "t": t,
        "num_changepoints": n_changepoints, "s": changepoints,
        "delta_scale": delta_scale,
        "num_harmonics": num_harmonics, "period": period
    }

    # ─ Threads ─
    threads_per_chain = _set_num_threads(threads_per_chain)

    # ─ Compile ─
    cmdstan_model = CmdStanModel(
        stan_file=STAN_FILE,
        cpp_options={"STAN_THREADS": "TRUE"}
    )

    # ─ Inference branch ─
    if inference == "map":
        fit_result = cmdstan_model.optimize(
            data=stan_data, algorithm="lbfgs", iter=10_000, seed=seed)

    elif inference == "advi":
        fit_result = cmdstan_model.variational(
            data=stan_data, algorithm="meanfield",
            iter=20_000, output_samples=400, seed=seed)

    elif inference == "nuts":
        fit_result = cmdstan_model.sample(
            data=stan_data,
            chains=chains, parallel_chains=chains,
            threads_per_chain=threads_per_chain,
            iter_warmup=warmup, iter_sampling=iter - warmup,
            adapt_delta=adapt_delta, max_treedepth=max_treedepth,
            seed=seed, show_progress=True)
    else:
        raise ValueError("inference must be 'nuts', 'map', or 'advi'")

    # ─ Diagnostics (only NUTS has .diagnose()) ─
    if isinstance(fit_result, CmdStanMCMC):
        print(fit_result.diagnose())

    return ChurnProphetModel(cmdstan_model, fit_result,
                             changepoints, num_harmonics, period)
