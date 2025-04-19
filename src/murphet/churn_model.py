"""
Murphet – unified, multi‑season wrapper  (parallel by default)

Public API:
    fit_churn_model(...)
    ChurnProphetModel.predict(...)
"""

from __future__ import annotations
import os, warnings, multiprocessing as _mp
from typing import Sequence, Literal, overload, List
import numpy as np

from cmdstanpy import (
    CmdStanModel, CmdStanMCMC, CmdStanMLE, CmdStanVB, CmdStanGQ
)
from scipy.special import expit


# ────────────────────────────────────────────────────────────────────
# 0)  Compile‑once model cache
# ────────────────────────────────────────────────────────────────────
_DIR       = os.path.dirname(os.path.abspath(__file__))
_STAN_FILE = os.path.join(_DIR, "murphet_model.stan")
_COMPILED: CmdStanModel | None = None


def _get_model() -> CmdStanModel:
    global _COMPILED
    if _COMPILED is None:
        _COMPILED = CmdStanModel(
            stan_file=_STAN_FILE,
            cpp_options={"STAN_THREADS": "TRUE"},
        )
    return _COMPILED


# ────────────────────────────────────────────────────────────────────
# 1)  Tiny FFT helper  → suggest dominant periods
# ────────────────────────────────────────────────────────────────────
def _suggest_periods(y: np.ndarray, top_n: int = 2, max_period: int = 365) -> list[float]:
    """Return up to `top_n` candidate periods using a simple FFT periodogram."""
    y = y - np.mean(y)
    n = len(y)
    if n < 8:                                          # nothing sensible here
        return []

    # FFT power spectrum
    power = np.abs(np.fft.rfft(y)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0)

    # Skip DC component (freq[0]==0)
    ix_sorted = np.argsort(power[1:])[::-1] + 1
    periods = []
    for ix in ix_sorted:
        if freqs[ix] == 0:
            continue
        p = 1 / freqs[ix]
        if p <= max_period:
            periods.append(float(p))
        if len(periods) >= top_n:
            break
    return periods


# ────────────────────────────────────────────────────────────────────
# 2)  Predictor façade
# ────────────────────────────────────────────────────────────────────
class ChurnProphetModel:
    """Fast prediction via posterior/point means."""

    def __init__(
        self,
        fit_result: CmdStanMCMC | CmdStanMLE | CmdStanVB | CmdStanGQ,
        changepoints: np.ndarray,
        periods: list[float],
        num_harmonics: list[int],
    ):
        self.fit_result   = fit_result
        self.changepoints = changepoints
        self.periods      = periods
        self.num_harmonics = num_harmonics
        self.num_seasons   = len(periods)

        # Flattened A/B vectors lengths ------------------------------
        self._total_harm = sum(num_harmonics)

        # —— Extract posterior means (or point estimates) —————————
        if isinstance(fit_result, (CmdStanMCMC, CmdStanVB)):
            self._k      = np.mean(fit_result.stan_variable("k"))
            self._m      = np.mean(fit_result.stan_variable("m"))
            self._q      = np.mean(fit_result.stan_variable("q"))
            self._delta  = np.mean(fit_result.stan_variable("delta"), axis=0)
            self._gamma  = np.mean(fit_result.stan_variable("gamma"))
            self._A_sin  = np.mean(fit_result.stan_variable("A_sin"), axis=0)
            self._B_cos  = np.mean(fit_result.stan_variable("B_cos"), axis=0)
        elif isinstance(fit_result, CmdStanMLE):
            pdict        = fit_result.optimized_params_dict
            self._k      = pdict["k"]
            self._m      = pdict["m"]
            self._q      = pdict["q"]
            self._gamma  = pdict["gamma"]
            self._delta  = np.array([pdict[f"delta[{i+1}]"] for i in range(len(changepoints))])
            self._A_sin  = np.array([pdict[f"A_sin[{i+1}]"] for i in range(self._total_harm)])
            self._B_cos  = np.array([pdict[f"B_cos[{i+1}]"] for i in range(self._total_harm)])
        else:
            raise TypeError("Unsupported CmdStan result type.")

    # ----------------------------------------------------------------
    def summary(self):
        return (
            self.fit_result.summary()
            if hasattr(self.fit_result, "summary")
            else self.fit_result.optimized_params_pd
        )

    # ----------------------------------------------------------------
    def predict(
        self,
        t_new: Sequence[float] | np.ndarray,
        method: Literal["mean_params"] = "mean_params",
    ) -> np.ndarray:
        if method != "mean_params":
            raise NotImplementedError("Only mean_params is implemented.")
        t_new = np.asarray(t_new, dtype=float)
        preds = np.empty(len(t_new))

        for j, t in enumerate(t_new):
            # —— Trend ——
            cp_effect = (
                np.sum(self._delta * expit(self._gamma * (t - self.changepoints)))
                if self.changepoints.size
                else 0.0
            )
            trend = self._k * t + self._m + self._q * t**2 + cp_effect

            # —— Seasonality ——
            pos  = 0
            seas = 0.0
            for p, h in zip(self.periods, self.num_harmonics):
                tmod = t - np.floor(t / p) * p
                for k in range(1, h + 1):
                    ang = 2 * np.pi * k * tmod / p
                    seas += (
                        self._A_sin[pos] * np.sin(ang)
                        + self._B_cos[pos] * np.cos(ang)
                    )
                    pos += 1

            preds[j] = expit(min(trend + seas, 4))

        return preds


# ────────────────────────────────────────────────────────────────────
# 3)  Public fit function
# ────────────────────────────────────────────────────────────────────
@overload
def fit_churn_model(
    *,
    t: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    periods: float | Sequence[float] = 12.0,
    num_harmonics: int | Sequence[int] = 3,
    auto_detect: bool = False,
    **kwargs,
) -> ChurnProphetModel: ...


def fit_churn_model(
    *,
    t: Sequence[float] | np.ndarray,
    y: Sequence[float] | np.ndarray,
    n_changepoints: int | None = None,
    changepoints: Sequence[float] | np.ndarray | None = None,
    # NEW ---------------------------------------------------------------------
    periods: float | Sequence[float] = 12.0,
    num_harmonics: int | Sequence[int] = 3,
    auto_detect: bool = False,
    # Existing ----------------------------------------------------------------
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
    Fit Murphet with *one or more* seasonalities.

    Parameters
    ----------
    periods         : scalar or list[float]
        Seasonal period(s) in **time‑index units** (e.g. 7, 30.4, 365.25).
    num_harmonics   : scalar or list[int]
        Number of Fourier pairs per period. If scalar ⇒ replicated.
    auto_detect     : bool
        If True and `periods` was *omitted* (or None), pick up to two dominant
        periods via an FFT periodogram.  (Requires at least 8 observations.)
    """
    # ---------- Convert inputs to numpy --------------------------------------
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)

    if np.any(y <= 0) or np.any(y >= 1):
        raise ValueError(
            f"Target values must be 0<y<1.  Found min={y.min():.4f}, max={y.max():.4f}"
        )

    # ---------- Seasonality argument handling --------------------------------
    # 1) periods argument coercion
    if auto_detect and (periods is None or (isinstance(periods, (list, tuple, np.ndarray)) and len(periods) == 0)):
        periods = _suggest_periods(y, top_n=2)
        if not periods:
            warnings.warn(
                "auto_detect=True but no significant periods found; "
                "falling back to default period=12."
            )
            periods = [12.0]
    if isinstance(periods, (int, float)):
        periods = [float(periods)]
    else:
        periods = [float(p) for p in periods]

    # 2) num_harmonics coercion
    if isinstance(num_harmonics, (int, float)):
        num_harmonics = [int(num_harmonics)] * len(periods)
    else:
        if len(num_harmonics) != len(periods):
            raise ValueError("num_harmonics length must match periods length.")
        num_harmonics = [int(h) for h in num_harmonics]

    num_seasons      = len(periods)
    total_harmonics  = int(sum(num_harmonics))

    # ---------- Changepoints --------------------------------------------------
    if n_changepoints is None and changepoints is None:
        n_changepoints = max(1, int(round(0.2 * len(t))))
    if changepoints is None:
        qs = np.linspace(0.1, 0.9, n_changepoints + 2)[1:-1]
        changepoints = np.quantile(t, qs)
    else:
        changepoints = np.sort(np.asarray(changepoints, dtype=float))
        n_changepoints = len(changepoints)

    # ---------- Threads -------------------------------------------------------
    if threads_per_chain is None:
        threads_per_chain = min(_mp.cpu_count(), 4)
    if len(y) < 32:
        threads_per_chain = 1
    os.environ["STAN_NUM_THREADS"] = str(max(1, threads_per_chain))

    # ---------- Stan data dict ------------------------------------------------
    stan_data = {
        # basic
        "N": len(y),
        "y": y,
        "t": t,
        # trend / cp
        "num_changepoints": n_changepoints,
        "s": changepoints,
        "delta_scale": delta_scale,
        # seasonality
        "num_seasons": num_seasons,
        "num_harmonics": num_harmonics,
        "period": periods,
        "total_harmonics": total_harmonics,
    }

    # ---------- Compile model -------------------------------------------------
    model = _get_model()

    # ---------- Inference branch ---------------------------------------------
    if inference == "map":
        fit_res = model.optimize(
            data=stan_data, algorithm="lbfgs", iter=10_000, seed=seed
        )
    elif inference == "advi":
        fit_res = model.variational(
            data=stan_data, algorithm="meanfield",
            iter=20_000, output_samples=400, seed=seed
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
        print(fit_res.diagnose())
    else:
        raise ValueError("inference must be 'nuts', 'map', or 'advi'.")

    return ChurnProphetModel(
        fit_res,
        changepoints=np.asarray(changepoints, dtype=float),
        periods=periods,
        num_harmonics=num_harmonics,
    )
