"""
Murphet – multi‑season time‑series model (Prophet‑compatible core)
------------------------------------------------------------------
  · optional Gaussian *or* Beta likelihood
  · piece‑wise‑linear trend  (+ CPs ≈ Prophet)
  · weak Normal seasonal priors (σ ≈ 10)
"""
from __future__ import annotations
import os, warnings, multiprocessing as _mp
from typing import Sequence, Literal, overload

import numpy as np
from scipy.special import expit
from cmdstanpy import CmdStanModel, CmdStanMCMC, CmdStanMLE, CmdStanVB, CmdStanGQ

# ────────────────────────────────────────────────────────────────
# 0)  compile‑once Stan cache      – pick file by likelihood
# ────────────────────────────────────────────────────────────────
_DIR = os.path.dirname(os.path.abspath(__file__))
_STAN_BETA  = os.path.join(_DIR, "murphet_beta.stan")
_STAN_GAUSS = os.path.join(_DIR, "murphet_gauss.stan")
_COMPILED: dict[str, CmdStanModel] = {}          # keyed by "beta"/"gaussian"


def _get_model(kind: Literal["beta", "gaussian"]) -> CmdStanModel:
    if kind not in _COMPILED:
        _COMPILED[kind] = CmdStanModel(
            stan_file=_STAN_BETA if kind == "beta" else _STAN_GAUSS,
            cpp_options={"STAN_THREADS": "TRUE"},
        )
    return _COMPILED[kind]


# ────────────────────────────────────────────────────────────────
# 1)  tiny helper – FFT periodogram (unchanged)
# ────────────────────────────────────────────────────────────────
def _suggest_periods(y: np.ndarray,
                     top_n: int = 2,
                     max_period: int = 365) -> list[float]:
    if y.size < 8:
        return []
    power = np.abs(np.fft.rfft(y - y.mean()))**2
    freqs = np.fft.rfftfreq(y.size, d=1.0)
    idx   = np.argsort(power[1:])[::-1] + 1
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
# 2)  Predictor façade  (posterior / MAP means)
# ────────────────────────────────────────────────────────────────
class ChurnProphetModel:
    """
    Fast vectorised predictions from posterior‑mean parameters.
    """

    def __init__(self,
                 fit: CmdStanMCMC | CmdStanMLE | CmdStanVB | CmdStanGQ,
                 changepoints: np.ndarray,
                 periods: list[float],
                 n_harm: list[int],
                 likelihood: Literal["beta", "gaussian"]):
        self.lik          = likelihood
        self.changepoints = changepoints
        self.periods      = periods
        self.n_harm       = n_harm
        self._H           = sum(n_harm)

        get = lambda v: np.mean(fit.stan_variable(v), axis=0) if hasattr(fit, "stan_variable") \
                        else fit.optimized_params_dict[v]

        self._k      = get("k")
        self._m      = get("m")
        self._gamma  = get("gamma")
        self._delta  = get("delta") if "delta" in fit.metadata.stan_vars else np.zeros(0)
        self._A      = get("A_sin")
        self._B      = get("B_cos")
        self._sigma  = get("sigma") if likelihood == "gaussian" else None  # only Gauss head
        self.fit     = fit

    # ------------------------------------------------------------------
    def predict(self,
                t_new: Sequence[float] | np.ndarray,
                method: Literal["mean_params"] = "mean_params") -> np.ndarray:

        if method != "mean_params":
            raise NotImplementedError
        t_new = np.asarray(t_new, float)
        out   = np.empty_like(t_new)

        for j, t in enumerate(t_new):
            # piece‑wise‑linear trend (Prophet style)
            cp   = np.sum(self._delta * expit(self._gamma * (t - self.changepoints))
                          ) if self._delta.size else 0.0
            mu_t = self._k * t + self._m + cp      # <- **no quadratic**

            # Fourier season blocks
            pos, seas = 0, 0.0
            for p, h in zip(self.periods, self.n_harm):
                tau = t % p
                for k in range(1, h + 1):
                    ang  = 2 * np.pi * k * tau / p
                    seas += self._A[pos]*np.sin(ang) + self._B[pos]*np.cos(ang)
                    pos  += 1
            mu_t += seas

            out[j] = expit(mu_t) if self.lik == "beta" else mu_t

        return out


# ────────────────────────────────────────────────────────────────
# 3)  public fit function  (only the *signature* below may differ)
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
        likelihood: Literal["beta", "gaussian"] = "beta",
        # changepoints
        n_changepoints: int | None = None,
        changepoints: Sequence[float] | np.ndarray | None = None,
        # seasonality
        periods: float | Sequence[float] = 12.0,
        num_harmonics: int | Sequence[int] = 3,
        auto_detect: bool = False,
        season_scale: float = 1.0,          # weak N(0,10) → scale=1
        # trend priors
        delta_scale: float = 0.05,
        gamma_scale: float = 3.0,
        # inference
        inference: Literal["map", "advi", "nuts"] = "map",
        chains: int = 2,
        iter: int = 4000,
        warmup: int = 0,
        adapt_delta: float = 0.95,
        max_treedepth: int = 12,
        threads_per_chain: int | None = None,
        seed: int | None = None,
) -> ChurnProphetModel:
    # ── 0) input / range checks ──────────────────────────────────
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    if likelihood == "beta" and (np.any(y <= 0) or np.any(y >= 1)):
        raise ValueError("Beta head requires 0 < y < 1. "
                         "For un‑bound ratios pick likelihood='gaussian'.")

    # ── 1) seasonality coercion ──────────────────────────────────
    if auto_detect and (periods is None or not periods):
        periods = _suggest_periods(y) or [12.0]
    periods = [float(p) for p in (periods if isinstance(periods, (list, tuple, np.ndarray))
                                  else [periods])]
    if isinstance(num_harmonics, (int, float)):
        num_harmonics = [int(num_harmonics)] * len(periods)
    elif len(num_harmonics) != len(periods):
        raise ValueError("len(num_harmonics) must match len(periods)")

    # ── 2) changepoints  (Prophet heuristic) ─────────────────────
    if changepoints is None:
        if n_changepoints is None:
            n_changepoints = max(1, int(0.2 * len(t)))
        qs = np.linspace(0.1, 0.9, n_changepoints + 2)[1:-1]
        changepoints = np.quantile(t, qs)
    else:
        changepoints = np.sort(np.asarray(changepoints, float))
        n_changepoints = changepoints.size

    # ── 3) threading ─────────────────────────────────────────────
    threads_per_chain = threads_per_chain or min(_mp.cpu_count(), 4)
    os.environ["STAN_NUM_THREADS"] = str(threads_per_chain)

    # ── 4) Stan data dict ────────────────────────────────────────
    stan_data = dict(
        N=len(y), y=y, t=t,
        num_changepoints=n_changepoints, s=changepoints,
        delta_scale=delta_scale, gamma_scale=gamma_scale,
        num_seasons=len(periods), n_harmonics=num_harmonics,
        period=periods, total_harmonics=int(sum(num_harmonics)),
        season_scale=season_scale,
    )

    # ── 5) choose & fit backend ──────────────────────────────────
    model = _get_model(likelihood)

    if inference == "map":
        fit = model.optimize(data=stan_data, algorithm="lbfgs",
                             iter=10000, seed=seed)

    elif inference == "advi":
        try:
            fit = model.variational(data=stan_data, algorithm="meanfield",
                                    iter=iter, draws=400,
                                    grad_samples=20, elbo_samples=20,
                                    tol_rel_obj=2e-3, seed=seed)
            if fit.num_draws < 1:
                raise RuntimeError
        except Exception:
            warnings.warn("ADVI failed – falling back to MAP.")
            fit = model.optimize(data=stan_data, algorithm="lbfgs",
                                 iter=10000, seed=seed)

    elif inference == "nuts":
        fit = model.sample(
            data=stan_data, chains=chains, parallel_chains=chains,
            iter_warmup=warmup, iter_sampling=iter - warmup,
            adapt_delta=adapt_delta, max_treedepth=max_treedepth,
            threads_per_chain=threads_per_chain, seed=seed,
            show_progress=True,
        )
    else:
        raise ValueError("inference must be 'map', 'advi' or 'nuts'.")

    # ── 6) wrap predictor & return ───────────────────────────────
    return ChurnProphetModel(
        fit, changepoints=np.asarray(changepoints, float),
        periods=periods, n_harm=num_harmonics,
        likelihood=likelihood,
    )
