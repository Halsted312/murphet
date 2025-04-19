"""
Unified Murphet wrapper (parallel by default).

* Single source of truth  →  easier maintenance
* Works with NUTS, MAP, or ADVI
* Auto‑detects available cores, falls back gracefully
"""

from __future__ import annotations
import os
import multiprocessing as _mp
from typing import Literal, Sequence

import numpy as np
from cmdstanpy import (
    CmdStanModel, CmdStanMCMC, CmdStanMLE, CmdStanVB, CmdStanGQ
)
from scipy.special import expit

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_STAN_FILE   = os.path.join(_CURRENT_DIR, "murphet_model.stan")

# ────────────────────────────────────────────────────────────────────
# 0)  Helper – compile **once** per interpreter session
# ────────────────────────────────────────────────────────────────────
_COMPILED_MODEL: CmdStanModel | None = None


def _get_compiled_model() -> CmdStanModel:
    global _COMPILED_MODEL
    if _COMPILED_MODEL is None:
        _COMPILED_MODEL = CmdStanModel(
            stan_file=_STAN_FILE,
            cpp_options={"STAN_THREADS": "TRUE"}
        )
    return _COMPILED_MODEL


# ────────────────────────────────────────────────────────────────────
# 1)  Predictor object
# ────────────────────────────────────────────────────────────────────
class ChurnProphetModel:
    """Light‑weight predict/summary façade over CmdStan results."""

    def __init__(
        self,
        fit_result: CmdStanMCMC | CmdStanMLE | CmdStanVB | CmdStanGQ,
        changepoints: np.ndarray,
        num_harmonics: int,
        period: float,
    ):
        self.fit_result   = fit_result
        self.changepoints = changepoints
        self.num_harmonics = num_harmonics
        self.period        = period

        # Posterior/point means for fast predictions ----------------
        if isinstance(fit_result, CmdStanMCMC) or isinstance(
            fit_result, CmdStanVB
        ):
            k = np.mean(fit_result.stan_variable("k"))
            m = np.mean(fit_result.stan_variable("m"))
            q = np.mean(fit_result.stan_variable("q"))
            delta = np.mean(fit_result.stan_variable("delta"), axis=0)
            gamma = np.mean(fit_result.stan_variable("gamma"))
            A_sin = np.mean(fit_result.stan_variable("A_sin"), axis=0)
            B_cos = np.mean(fit_result.stan_variable("B_cos"), axis=0)
        elif isinstance(fit_result, CmdStanMLE):
            p = fit_result.optimized_params_dict
            k, m, q, gamma = p["k"], p["m"], p["q"], p["gamma"]
            delta = np.array([p[f"delta[{i+1}]"] for i in range(len(changepoints))])
            A_sin = np.array([p[f"A_sin[{i+1}]"] for i in range(num_harmonics)])
            B_cos = np.array([p[f"B_cos[{i+1}]"] for i in range(num_harmonics)])
        else:
            raise TypeError("Unsupported CmdStan result type.")

        self._k = k
        self._m = m
        self._q = q
        self._delta = delta
        self._gamma = gamma
        self._A_sin = A_sin
        self._B_cos = B_cos

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def summary(self):
        return (
            self.fit_result.summary()
            if hasattr(self.fit_result, "summary")
            else self.fit_result.optimized_params_pd
        )

    # ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    def predict(
        self,
        t_new: Sequence[float] | np.ndarray,
        method: Literal["mean_params"] = "mean_params",
    ) -> np.ndarray:
        """
        Currently only supports 'mean_params' (fast deterministic).
        """
        if method != "mean_params":
            raise NotImplementedError("Only method='mean_params' is implemented.")

        t_new = np.asarray(t_new, dtype=float)
        preds = np.empty(len(t_new))

        for j, t in enumerate(t_new):
            cp_effect = (
                np.sum(self._delta * expit(self._gamma * (t - self.changepoints)))
                if self.changepoints.size
                else 0.0
            )
            trend = self._k * t + self._m + self._q * t ** 2 + cp_effect

            # Seasonality
            t_mod = t - np.floor(t / self.period) * self.period
            seas = 0.0
            for r in range(self.num_harmonics):
                ang = 2 * np.pi * (r + 1) * t_mod / self.period
                seas += self._A_sin[r] * np.sin(ang) + self._B_cos[r] * np.cos(ang)

            preds[j] = expit(min(trend + seas, 4))

        return preds


# ────────────────────────────────────────────────────────────────────
# 2)  Public fit function
# ────────────────────────────────────────────────────────────────────
def fit_churn_model(
    *,
    t: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    n_changepoints: int | None = None,
    changepoints: Sequence[float] | np.ndarray | None = None,
    num_harmonics: int = 3,
    period: float = 12.0,
    delta_scale: float = 0.05,
    inference: Literal["nuts", "map", "advi"] = "nuts",
    chains: int = 4,
    iter: int = 2000,
    warmup: int = 1000,
    adapt_delta: float = 0.95,
    max_treedepth: int = 12,
    threads_per_chain: int | None = None,
    seed: int | None = None,
):
    """
    Fit Murphet to a (0,1) time‑series.

    Parameters
    ----------
    t, y : numeric sequences
        *y* **must** be strictly between 0 and 1. A ValueError is raised otherwise.
    inference : {"nuts", "map", "advi"}
        • "nuts": full‑Bayes No‑U‑Turn sampler (default)  
        • "map" : LBFGS optimisation (fast, no uncertainty)  
        • "advi": mean‑field ADVI (posterior approx.)
    threads_per_chain : int, optional
        `None` → use `min(4, cpu_count())`; force 1 if len(y) < 32.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    # --- strict domain check -------------------------------------------------
    if np.any(y <= 0) or np.any(y >= 1):
        raise ValueError(
            "All target values must be strictly between 0 and 1. "
            "Found min={:.4f}, max={:.4f}".format(y.min(), y.max())
        )

    # changepoints ------------------------------------------------------------
    if n_changepoints is None and changepoints is None:
        n_changepoints = max(1, int(round(0.2 * len(t))))
    if changepoints is None:
        qs = np.linspace(0.1, 0.9, n_changepoints + 2)[1:-1]
        changepoints = np.quantile(t, qs)
    else:
        changepoints = np.sort(np.asarray(changepoints, dtype=float))
        n_changepoints = len(changepoints)

    # threads / grainsize -----------------------------------------------------
    if threads_per_chain is None:
        threads_per_chain = min(_mp.cpu_count(), 4)
    if len(y) < 32:                       # avoid overhead on tiny datasets
        threads_per_chain = 1

    os.environ["STAN_NUM_THREADS"] = str(max(1, threads_per_chain))

    # ------------------------------------------------------------------------
    stan_data = {
        "N": len(y),
        "y": y,
        "t": t,
        "num_changepoints": n_changepoints,
        "s": changepoints,
        "delta_scale": delta_scale,
        "num_harmonics": num_harmonics,
        "period": period,
    }

    # Compile / reuse cached model -------------------------------------------
    model = _get_compiled_model()

    # Inference branch --------------------------------------------------------
    if inference == "map":
        fit_res = model.optimize(
            data=stan_data,
            algorithm="lbfgs",
            iter=10_000,
            seed=seed,
        )
    elif inference == "advi":
        fit_res = model.variational(
            data=stan_data,
            algorithm="meanfield",
            iter=20_000,
            output_samples=400,
            seed=seed,
        )
    elif inference == "nuts":
        fit_res = model.sample(
            data=stan_data,
            chains=chains,
            parallel_chains=chains,
            threads_per_chain=threads_per_chain,
            iter_warmup=warmup,
            iter_sampling=iter - warmup,
            adapt_delta=adapt_delta,
            max_treedepth=max_treedepth,
            seed=seed,
            show_progress=True,
        )
        # Optional diagnose print
        print(fit_res.diagnose())
    else:
        raise ValueError("inference must be 'nuts', 'map', or 'advi'.")

    return ChurnProphetModel(
        fit_res, changepoints, num_harmonics=num_harmonics, period=period
    )
