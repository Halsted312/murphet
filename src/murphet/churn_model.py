"""
Murphet – multi‑season logistic‑beta model
======================================================================
  · Thread‑parallel Stan backend (reduce_sum)
  · Optional auto‑detection of dominant periods via FFT
  · Fast MAP / ADVI for HPO + full NUTS for final fit

Public API
----------
    fit_churn_model(...)          ->  ChurnProphetModel
    ChurnProphetModel.predict(...) (vectorised, fast)
"""
from __future__ import annotations

import os, warnings, multiprocessing as _mp
from typing import Sequence, Literal, overload
import numpy as np
from scipy.special import expit
from cmdstanpy import (
    CmdStanModel, CmdStanMCMC, CmdStanMLE, CmdStanVB, CmdStanGQ
)

# ────────────────────────────────────────────────────────────────
# 0)  Compile‑once Stan model cache
# ────────────────────────────────────────────────────────────────
_DIR        = os.path.dirname(os.path.abspath(__file__))
_STAN_FILE  = os.path.join(_DIR, "murphet_model.stan")
_COMPILED: CmdStanModel | None = None


def _get_model() -> CmdStanModel:
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = CmdStanModel(
            stan_file=_STAN_FILE,
            cpp_options={"STAN_THREADS": "TRUE"},
        )
    return _COMPILED


# ────────────────────────────────────────────────────────────────
# 1)  FFT helper – suggest dominant periods
# ────────────────────────────────────────────────────────────────
def _suggest_periods(y: np.ndarray,
                     top_n: int = 2,
                     max_period: int = 365) -> list[float]:
    """Return up to *top_n* candidate periods (coarse)."""
    if y.size < 8:
        return []
    power = np.abs(np.fft.rfft(y - y.mean()))**2
    freqs = np.fft.rfftfreq(y.size, d=1.0)
    idx   = np.argsort(power[1:])[::-1] + 1            # skip DC
    out: list[float] = []
    for i in idx:
        if freqs[i] == 0:
            continue
        p = 1 / freqs[i]
        if p <= max_period:
            out.append(float(p))
            if len(out) >= top_n:
                break
    return out


# ────────────────────────────────────────────────────────────────
# 2)  Predictor façade (posterior‑mean parameters)
# ────────────────────────────────────────────────────────────────
class ChurnProphetModel:
    """Light‑weight predictor using mean posterior / MAP params."""

    def __init__(
        self,
        fit_result: CmdStanMCMC | CmdStanMLE | CmdStanVB | CmdStanGQ,
        changepoints: np.ndarray,
        periods: list[float],
        num_harmonics: list[int],
    ):
        self.changepoints = changepoints
        self.periods      = periods
        self.num_harm     = num_harmonics
        self._tot_harm    = sum(num_harmonics)

        # helper
        has_var = lambda v: v in fit_result.metadata.stan_vars

        # posterior means / point estimates ----------------------------------
        if isinstance(fit_result, (CmdStanMCMC, CmdStanVB)):
            m = lambda v: np.mean(fit_result.stan_variable(v), axis=0)
            self._k, self._m, self._q = m("k"), m("m"), m("q")
            self._gamma               = m("gamma")
            self._A, self._B          = m("A_sin"), m("B_cos")
            self._delta = m("delta") if has_var("delta") else np.zeros(0)
        elif isinstance(fit_result, CmdStanMLE):
            p  = fit_result.optimized_params_dict
            self._k, self._m, self._q = p["k"], p["m"], p["q"]
            self._gamma               = p["gamma"]
            self._delta               = (
                np.array([p[f"delta[{i+1}]"] for i in range(len(changepoints))])
                if changepoints.size else np.zeros(0)
            )
            self._A = np.array([p[f"A_sin[{i+1}]"] for i in range(self._tot_harm)])
            self._B = np.array([p[f"B_cos[{i+1}]"] for i in range(self._tot_harm)])
        else:
            raise TypeError("Unsupported CmdStan result type.")
        self.fit_result = fit_result

    # ------------------------------------------------------------
    def predict(self,
                t_new: Sequence[float] | np.ndarray,
                method: Literal["mean_params"] = "mean_params"
               ) -> np.ndarray:
        if method != "mean_params":
            raise NotImplementedError
        t_new = np.asarray(t_new, float)
        out   = np.empty_like(t_new)
        for j, t in enumerate(t_new):
            cp = (np.sum(self._delta * expit(self._gamma*(t - self.changepoints)))
                  if self._delta.size else 0.0)
            trend = self._k*t + self._m + self._q*t**2 + cp
            pos, seas = 0, 0.0
            for p, h in zip(self.periods, self.num_harm):
                tmod = t % p
                for k in range(1, h+1):
                    ang  = 2*np.pi*k*tmod/p
                    seas += self._A[pos]*np.sin(ang) + self._B[pos]*np.cos(ang)
                    pos  += 1
            seas = np.clip(seas, -5.0, 5.0)        # guard against blow‑ups
            out[j] = expit(trend + seas)
        return out


# ────────────────────────────────────────────────────────────────
# 3)  Public fit function
# ────────────────────────────────────────────────────────────────
@overload
def fit_churn_model(*,
                    t: Sequence[float] | np.ndarray,
                    y: Sequence[float] | np.ndarray,
                    **kwargs) -> ChurnProphetModel: ...


def fit_churn_model(
    *,
    t: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    # changepoints
    n_changepoints: int | None = None,
    changepoints: Sequence[float] | np.ndarray | None = None,
    # seasonality
    periods: float | Sequence[float] = 12.0,
    num_harmonics: int | Sequence[int] = 3,
    auto_detect: bool = False,
    season_scale: float = 0.15,           # NEW  ← λ_seas
    # priors / inference
    delta_scale: float = 0.05,
    inference: Literal["nuts", "map", "advi"] = "nuts",
    chains: int = 4,
    iter: int = 2000,
    warmup: int = 1000,
    adapt_delta: float = 0.95,
    max_treedepth: int = 12,
    threads_per_chain: int | None = None,
    seed: int | None = None,
) -> ChurnProphetModel:

    # ---------- input validation ----------------------------------
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    if np.any((y <= 0) | (y >= 1)):
        raise ValueError("Target must satisfy 0 < y < 1.")

    # ---------- seasonality lists ---------------------------------
    if auto_detect and (periods is None or not periods):
        periods = _suggest_periods(y) or [12.0]
    periods = [float(p) for p in (periods if isinstance(periods, (list, tuple, np.ndarray))
                                  else [periods])]
    if isinstance(num_harmonics, (int, float)):
        num_harmonics = [int(num_harmonics)] * len(periods)
    else:
        if len(num_harmonics) != len(periods):
            raise ValueError("num_harmonics length mismatch.")
        num_harmonics = [int(h) for h in num_harmonics]

    # ---------- changepoints -------------------------------------
    if changepoints is None:
        if n_changepoints is None:
            n_changepoints = max(1, int(round(0.2*len(t))))
        qs = np.linspace(0.1, 0.9, n_changepoints+2)[1:-1]
        changepoints = np.quantile(t, qs)
    else:
        changepoints = np.sort(np.asarray(changepoints, float))
        n_changepoints = changepoints.size

    # ---------- threading ----------------------------------------
    if threads_per_chain is None:
        threads_per_chain = min(_mp.cpu_count(), 4)
    if len(y) < 32:
        threads_per_chain = 1
    os.environ["STAN_NUM_THREADS"] = str(threads_per_chain)

    # ---------- Stan data ----------------------------------------
    stan_data = dict(
        N=len(y), y=y, t=t,
        num_changepoints=n_changepoints, s=changepoints,
        delta_scale=delta_scale,
        num_seasons=len(periods),
        n_harmonics=num_harmonics,            # new key (matches Stan)
        period=periods,
        total_harmonics=int(sum(num_harmonics)),
        season_scale=season_scale,            # passes λ_seas to Stan
    )

    model = _get_model()

    # ---------- inference routes ---------------------------------
    if inference == "map":
        fit_res = model.optimize(data=stan_data, algorithm="lbfgs",
                                 iter=10000, seed=seed)

    elif inference == "advi":
        try:
            fit_res = model.variational(data=stan_data, algorithm="meanfield",
                                        iter=iter, draws=400,
                                        grad_samples=20, elbo_samples=20,
                                        tol_rel_obj=2e-3, seed=seed)
            if fit_res.num_draws < 1:
                raise RuntimeError
        except Exception:
            warnings.warn("ADVI failed – falling back to MAP.")
            fit_res = model.optimize(data=stan_data, algorithm="lbfgs",
                                     iter=10000, seed=seed)

    elif inference == "nuts":
        fit_res = model.sample(
            data=stan_data, chains=chains, parallel_chains=chains,
            iter_warmup=warmup, iter_sampling=iter - warmup,
            adapt_delta=adapt_delta, max_treedepth=max_treedepth,
            threads_per_chain=threads_per_chain, seed=seed,
            show_progress=True)
    else:
        raise ValueError("inference must be 'map', 'advi' or 'nuts'.")

    return ChurnProphetModel(
        fit_res,
        changepoints=np.asarray(changepoints, float),
        periods=periods,
        num_harmonics=num_harmonics,
    )
