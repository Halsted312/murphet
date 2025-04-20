"""
Murphet – multi‑season time‑series model (Prophet‑compatible core)
------------------------------------------------------------------
  · optional Gaussian *or* Beta likelihood
  · piece‑wise‑linear trend   (+ smooth CPs  ≈ Prophet)
  · weak Normal seasonal priors (σ≈10)       – no horseshoe
"""
from __future__ import annotations
import os, warnings, multiprocessing as _mp
from typing import Sequence, Literal, overload
import numpy as np
from scipy.special import expit
from cmdstanpy import CmdStanModel, CmdStanMCMC, CmdStanMLE, CmdStanVB, CmdStanGQ
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"The default behavior of CmdStanMLE\.stan_variable\(\) will change",
    category=UserWarning,
)

# ────────────────────────────────────────────────────────────────
# 0)  compile‑once Stan cache  (one exe per likelihood)
# ────────────────────────────────────────────────────────────────
_DIR         = os.path.dirname(os.path.abspath(__file__))
_STAN_BETA   = os.path.join(_DIR, "murphet_beta.stan")
_STAN_GAUSS  = os.path.join(_DIR, "murphet_gauss.stan")
_COMPILED: dict[str, CmdStanModel] = {}          # key ∈ {"beta","gaussian"}


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
    pwr   = np.abs(np.fft.rfft(y - y.mean()))**2
    freqs = np.fft.rfftfreq(y.size, 1.0)
    idx   = np.argsort(pwr[1:])[::-1] + 1
    out: list[float] = []
    for i in idx:
        if freqs[i] == 0:
            continue
        prd = 1 / freqs[i]
        if prd <= max_period:
            out.append(float(prd))
            if len(out) >= top_n:
                break
    return out

# ────────────────────────────────────────────────────────────────
# 2)  Predictor façade (posterior / MAP means)   ✅ final version
# ────────────────────────────────────────────────────────────────
class ChurnProphetModel:
    """Fast vectorised predictor built from posterior / MAP means."""

    def __init__(
        self,
        fit: CmdStanMCMC | CmdStanMLE | CmdStanVB | CmdStanGQ,
        changepoints: np.ndarray,
        periods: list[float],
        n_harm: list[int],
        likelihood: Literal["beta", "gaussian"],
    ):
        self.lik      = likelihood
        self.s        = changepoints
        self.periods  = periods
        self.n_harm   = n_harm
        self._H       = sum(n_harm)

        # ---------------- helper extractors -----------------------
        is_mle = isinstance(fit, CmdStanMLE)  # <── add

        def _scalar(var: str) -> float:
            if is_mle:  # MLE path  → use dict
                return float(fit.optimized_params_dict[var])
            arr = np.asarray(fit.stan_variable(var))  # MCMC/VB
            return float(arr.mean())

        def _vector(var: str) -> np.ndarray:
            """
            Return a 1‑D numpy array for Stan variable *var*
            – works for both MCMC/VB (draws) and MAP/MLE (scalar dict).
            """
            if is_mle:                         # -------  MAP / MLE path
                if var == "delta":
                    n_cp = len(changepoints)
                    return np.array(
                        [fit.optimized_params_dict[f"delta[{i+1}]"] for i in range(n_cp)],
                        dtype=float,
                    )

                if var in ("A_sin", "B_cos"):
                    return np.array(
                        [fit.optimized_params_dict[f"{var}[{i+1}]"] for i in range(self._H)],
                        dtype=float,
                    )

                # vectors that are actually stored whole (rare)
                if var in fit.optimized_params_dict:
                    return np.asarray(fit.optimized_params_dict[var], float)

                raise KeyError(f"{var} not found in optimised params dict")

            # -----------------------------  MCMC / VB path
            arr = np.asarray(fit.stan_variable(var), float)      # draws × dim
            return arr.mean(axis=0) if arr.ndim == 2 else arr


        has = lambda v: hasattr(fit, "metadata") and v in fit.metadata.stan_vars

        # ---- AR(1) disturbance (optional) ------------------------
        self._rho = _scalar("rho")  if has("rho")  else 0.0
        self._mu0 = _scalar("mu0")  if has("mu0")  else 0.0

        # ---- trend ----------------------------------------------
        self._k     = _scalar("k")
        self._m     = _scalar("m")        # intercept only – no quadratic
        self._gamma = _scalar("gamma")
        self._delta = _vector("delta") if has("delta") else np.zeros(0)

        # ---- seasonality ----------------------------------------
        self._A = _vector("A_sin")
        self._B = _vector("B_cos")
        assert len(self._A) == self._H == len(self._B), "Fourier length mismatch"

        # ---- Gaussian head --------------------------------------
        self._sigma = _scalar("sigma") if likelihood == "gaussian" and has("sigma") else None
        self.fit    = fit

    # ------------------------------------------------------------------
    def predict(
            self,
            t_new: Sequence[float] | np.ndarray,
            method: Literal["mean_params", "sample"] = "mean_params",
    ) -> np.ndarray:
        """
        Generate predictions for new time points.

        Parameters:
            t_new: New time points to predict
            method: Either "mean_params" (use posterior means) or "sample" (draw from posterior)

        Returns:
            Array of predictions for each time point
        """
        t_new = np.asarray(t_new, float)

        # Mean parameters approach (existing implementation)
        if method == "mean_params":
            out = np.empty_like(t_new)
            lag_state = self._mu0  # AR(1) initial state
            for j, t in enumerate(t_new):
                # ---- piece‑wise‑linear trend ---------------------------------
                cp = np.sum(self._delta * expit(self._gamma * (t - self.s))
                            ) if self._delta.size else 0.0
                mu = self._k * t + self._m + cp

                # ---- additive Fourier seasonality ----------------------------
                pos = 0
                for P, H in zip(self.periods, self.n_harm):
                    tau = t % P
                    for h in range(1, H + 1):
                        ang = 2 * np.pi * h * tau / P
                        mu += self._A[pos] * np.sin(ang) + self._B[pos] * np.cos(ang)
                        pos += 1

                # ---- AR(1) disturbance  --------------------------------------
                mu += self._rho * lag_state
                lag_state = mu

                # ---- link function -------------------------------------------
                out[j] = expit(mu) if self.lik == "beta" else mu

            return out

        # Sample from posterior approach
        elif method == "sample":
            # Verify we have MCMC samples
            if not isinstance(self.fit, CmdStanMCMC):
                raise ValueError("Sampling requires NUTS inference (inference='nuts')")

            # Select a random posterior sample
            n_samples = len(self.fit.stan_variable("k"))
            sample_idx = np.random.randint(0, n_samples)

            # Extract parameters for this sample
            k = float(self.fit.stan_variable("k")[sample_idx])
            m = float(self.fit.stan_variable("m")[sample_idx])
            gamma = float(self.fit.stan_variable("gamma")[sample_idx])

            # Extract delta (changepoint adjustment) if it exists
            if self.s.size > 0:
                delta = self.fit.stan_variable("delta")[sample_idx]
            else:
                delta = np.zeros(0)

            # Extract seasonal components
            A_sin = self.fit.stan_variable("A_sin")[sample_idx]
            B_cos = self.fit.stan_variable("B_cos")[sample_idx]

            # Extract AR(1) components
            try:
                rho = float(self.fit.stan_variable("rho")[sample_idx])
            except:
                rho = 0.0

            try:
                mu0 = float(self.fit.stan_variable("mu0")[sample_idx])
            except:
                mu0 = 0.0

            # Generate predictions with this sample
            out = np.empty_like(t_new)
            lag_state = mu0  # AR(1) initial state

            for j, t in enumerate(t_new):
                # ---- piece‑wise‑linear trend ---------------------------------
                cp = np.sum(delta * expit(gamma * (t - self.s))
                            ) if delta.size else 0.0
                mu = k * t + m + cp

                # ---- additive Fourier seasonality ----------------------------
                pos = 0
                for P, H in zip(self.periods, self.n_harm):
                    tau = t % P
                    for h in range(1, H + 1):
                        ang = 2 * np.pi * h * tau / P
                        mu += A_sin[pos] * np.sin(ang) + B_cos[pos] * np.cos(ang)
                        pos += 1

                # ---- AR(1) disturbance  --------------------------------------
                mu += rho * lag_state
                lag_state = mu

                # ---- link function -------------------------------------------
                out[j] = expit(mu) if self.lik == "beta" else mu

            return out

        else:
            raise ValueError(f"Unknown method: {method}. Use 'mean_params' or 'sample'")
# ────────────────────────────────────────────────────────────────
# 3)  public fit function
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
        # changepoints --------------------------------------------------
        n_changepoints: int | None = None,
        changepoints: Sequence[float] | np.ndarray | None = None,
        # seasonality ---------------------------------------------------
        periods: float | Sequence[float] = 12.0,
        num_harmonics: int | Sequence[int] = 3,
        auto_detect: bool = False,
        season_scale: float = 1.0,
        # trend priors --------------------------------------------------
        delta_scale: float = 0.05,
        gamma_scale: float = 3.0,
        # inference -----------------------------------------------------
        inference: Literal["map", "advi", "nuts"] = "map",
        chains: int = 2,
        iter: int = 4000,
        warmup: int = 0,
        adapt_delta: float = 0.95,
        max_treedepth: int = 12,
        threads_per_chain: int | None = None,
        seed: int | None = None,

) -> ChurnProphetModel:

    # ── checks & data coercion ─────────────────────────────────────
    t = np.asarray(t, float)
    y = np.asarray(y, float)

    # First check if data is completely outside bounds (error condition)
    if likelihood == "beta" and (np.any(y <= 0) or np.any(y >= 1)):
        raise ValueError("Beta likelihood requires 0 < y < 1.")

    # Then add safety margin for beta likelihood
    if likelihood == "beta":
        # Add small epsilon to ensure values aren't too close to boundaries
        epsilon = 1e-5
        y = np.clip(y, epsilon, 1 - epsilon)

    # ── seasonality lists ─────────────────────────────────────────
    if auto_detect and (periods is None or not periods):
        periods = _suggest_periods(y) or [12.0]
    periods = [float(p) for p in (periods if isinstance(periods, (list, tuple, np.ndarray))
                                  else [periods])]
    if isinstance(num_harmonics, (int, float)):
        num_harmonics = [int(num_harmonics)] * len(periods)
    elif len(num_harmonics) != len(periods):
        raise ValueError("len(num_harmonics) must match len(periods)")
    num_harmonics = [int(h) for h in num_harmonics]

    # ── changepoints (Prophet heuristic) ──────────────────────────
    if changepoints is None:
        if n_changepoints is None:
            n_changepoints = max(1, int(0.2 * len(t)))
        qs = np.linspace(0.1, 0.9, n_changepoints + 2)[1:-1]
        changepoints = np.quantile(t, qs)
    else:
        changepoints = np.sort(np.asarray(changepoints, float))
        n_changepoints = changepoints.size

    # ── threading & env var ───────────────────────────────────────
    threads_per_chain = threads_per_chain or min(_mp.cpu_count(), 4)
    os.environ["STAN_NUM_THREADS"] = str(threads_per_chain)

    # ── Stan data dict ────────────────────────────────────────────
    stan_data = dict(
        N=len(y), y=y, t=t,
        num_changepoints=n_changepoints, s=changepoints,
        delta_scale=delta_scale, gamma_scale=gamma_scale,
        num_seasons=len(periods), n_harmonics=num_harmonics,
        period=periods, total_harmonics=int(sum(num_harmonics)),
        season_scale=season_scale,
    )

    model = _get_model(likelihood)

    # ── inference routes ─────────────────────────────────────────
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
            warnings.warn("ADVI failed – falling back to MAP.")
            fit = model.optimize(data=stan_data, algorithm="lbfgs",
                                 iter=10000, seed=seed)

    elif inference == "nuts":
        fit = model.sample(
            data=stan_data, chains=chains, parallel_chains=chains,
            iter_warmup=warmup, iter_sampling=iter - warmup,
            adapt_delta=adapt_delta, max_treedepth=max_treedepth,
            threads_per_chain=threads_per_chain, seed=seed,
            show_progress=True)
    else:
        raise ValueError("inference must be 'map', 'advi' or 'nuts'.")

    return ChurnProphetModel(
        fit, changepoints=np.asarray(changepoints, float),
        periods=periods, n_harm=num_harmonics,
        likelihood=likelihood,
    )
