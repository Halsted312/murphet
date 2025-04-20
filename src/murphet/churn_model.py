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
os.environ["CMDSTAN_PRINT"] = "FALSE"
warnings.filterwarnings(
    "ignore",
    message=r"The default behavior of CmdStanMLE\.stan_variable\(\) will change",
    category=UserWarning,
)
import sys
import io
import os

# Add this near the top of your module, after imports
import logging

# Optional: If you want to suppress ALL cmdstanpy logs (even warnings)
logging.getLogger("cmdstanpy").setLevel(logging.CRITICAL)  # Only critical errors


# Create a class to temporarily redirect stdout/stderr
class SuppressOutput:
    """Context manager to suppress stdout and stderr."""

    def __init__(self, suppress_stdout=True, suppress_stderr=True):
        self.suppress_stdout = suppress_stdout
        self.suppress_stderr = suppress_stderr
        self.original_stdout = None
        self.original_stderr = None

    def __enter__(self):
        # Save original stdout/stderr
        if self.suppress_stdout:
            self.original_stdout = sys.stdout
            sys.stdout = io.StringIO()

        if self.suppress_stderr:
            self.original_stderr = sys.stderr
            sys.stderr = io.StringIO()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore original stdout/stderr
        if self.suppress_stdout and self.original_stdout:
            sys.stdout = self.original_stdout

        if self.suppress_stderr and self.original_stderr:
            sys.stderr = self.original_stderr

# Utility function for concise error messages
def extract_key_error(error_message):
    """Extract only the most relevant part of a CmdStan error."""
    if isinstance(error_message, Exception):
        error_message = str(error_message)

    # Look for common error patterns
    lines = error_message.split("\n")

    # First check for "Failed with error" lines
    for line in lines:
        if "Failed with error" in line:
            return line.strip()

    # Look for Exception lines
    for line in lines:
        if "Exception:" in line:
            return line.strip()

    # Fall back to the first couple of lines if it's a long message
    if len(lines) > 3 and len(error_message) > 300:
        return lines[0].strip() + " [...]"

    # Return original if we couldn't simplify
    return error_message

def _ensure_clean_environment():
    """Set up clean environment for Stan operations with permission handling."""
    import tempfile
    import os
    import stat

    # Create a fresh temporary directory with explicit permissions
    try:
        temp_dir = tempfile.mkdtemp(prefix="murphet_")
        # Ensure directory has full permissions (read/write/execute)
        os.chmod(temp_dir, stat.S_IRWXU)
    except Exception as e:
        # Fall back to user's home directory if temp creation fails
        import os.path
        home_dir = os.path.expanduser("~")
        temp_dir = os.path.join(home_dir, ".murphet_tmp")

        # Create the directory if it doesn't exist
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir, exist_ok=True)
            os.chmod(temp_dir, stat.S_IRWXU)

    # Set environment variables
    os.environ["TMPDIR"] = temp_dir
    os.environ["TMP"] = temp_dir

    return temp_dir


# Initialize clean environment at module load time
_TEMP_DIR = _ensure_clean_environment()

# ────────────────────────────────────────────────────────────────
# 0)  compile‑once Stan cache  (one exe per likelihood)
# ────────────────────────────────────────────────────────────────
_DIR         = os.path.dirname(os.path.abspath(__file__))
_STAN_BETA   = os.path.join(_DIR, "murphet_beta.stan")
_STAN_GAUSS  = os.path.join(_DIR, "murphet_gauss.stan")
_COMPILED: dict[str, CmdStanModel] = {}          # key ∈ {"beta","gaussian"}


def _ensure_model_directory():
    """Create a persistent directory for Stan models that will survive across sessions."""
    import os
    import os.path
    import stat

    # Create directory in user's home directory (this has better permission stability)
    home_dir = os.path.expanduser("~")
    model_dir = os.path.join(home_dir, ".murphet_models")

    # Create it if it doesn't exist
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    # Ensure it has proper permissions
    try:
        os.chmod(model_dir, stat.S_IRWXU)  # Read, write, execute for user
    except:
        pass

    return model_dir


# Get a persistent model directory
_MODEL_DIR = _ensure_model_directory()


# Then replace your _get_model function with this:
def _get_model(kind: Literal["beta", "gaussian"]) -> CmdStanModel:
    """Get compiled Stan model with persistent storage to avoid 'No such file' errors."""
    import os
    import stat
    import shutil
    import time
    import hashlib

    # Check if model already compiled
    if kind in _COMPILED:
        return _COMPILED[kind]

    # Prepare for compilation with multiple attempts
    max_attempts = 5
    last_error = None
    stan_file = _STAN_BETA if kind == "beta" else _STAN_GAUSS

    # Generate a hash of the stan file content for versioning
    try:
        with open(stan_file, 'rb') as f:
            file_content = f.read()
            model_hash = hashlib.md5(file_content).hexdigest()[:8]
    except:
        model_hash = "default"

    # Create a stable path for the model in our persistent directory
    model_basename = f"murphet_{kind}_{model_hash}"
    model_path = os.path.join(_MODEL_DIR, f"{model_basename}.stan")

    # Copy the stan file to our persistent directory
    try:
        shutil.copy2(stan_file, model_path)
        os.chmod(model_path, stat.S_IRUSR | stat.S_IWUSR)  # Read/write for user
    except Exception as e:
        print(f"Warning: Could not copy Stan file: {e}")

    # Try to compile the model
    for attempt in range(max_attempts):
        try:
            # Set environment variables to use our persistent directory
            os.environ["TMPDIR"] = _MODEL_DIR
            os.environ["TMP"] = _MODEL_DIR

            # Reduce threads with each attempt to avoid resource issues
            threads = max(1, min(4, 4 // (attempt + 1)))
            os.environ["STAN_NUM_THREADS"] = str(threads)

            # Try to compile the model with suppressed output
            with SuppressOutput():
                _COMPILED[kind] = CmdStanModel(
                    stan_file=model_path,
                    cpp_options={"STAN_THREADS": "TRUE"},
                    model_name=model_basename,
                    compile=True,  # Force compilation
                )

            # Check if the executable was actually created
            exe_file = _COMPILED[kind].exe_file
            if not os.path.exists(exe_file):
                raise FileNotFoundError(f"Compiled executable {exe_file} not found")

            # Make sure executable has execute permission
            os.chmod(exe_file, stat.S_IRWXU)  # Read, write, execute for user

            # Success!
            return _COMPILED[kind]

        except Exception as e:
            last_error = e

            # Clean up and try again
            try:
                # Remove any partial compilation artifacts
                if kind in _COMPILED:
                    del _COMPILED[kind]
            except:
                pass

            # Wait longer between attempts
            time.sleep(2 * (attempt + 1))

    # All attempts failed
    raise RuntimeError(
        f"Failed to compile Stan model '{kind}' after {max_attempts} attempts. Last error: {str(last_error)}")


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

# -----------------------------------------------------------------
# 2· misc tiny helpers
# -----------------------------------------------------------------
def _trim_changepoints(d: dict):
    n = int(d["num_changepoints"])
    if len(d["s"]) != n:
        d["s"] = d["s"][:n]

def _clip_simple(d: dict):                       #  <<< NEW
    """Ultra‑wide but safe hyper‑prior clamps for the simple fallback."""
    _trim_changepoints(d)
    d["delta_scale"]  = float(np.clip(d["delta_scale"], 1e-5, 0.90))
    d["gamma_scale"]  = float(np.clip(d["gamma_scale"], 1e-3, 10.0))
    d["season_scale"] = float(np.clip(d["season_scale"], 1e-4, 10.0))

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
        Generate predictions for new time points with robust error handling.

        Parameters:
            t_new: New time points to predict
            method: Either "mean_params" (use posterior means) or "sample" (draw from posterior)

        Returns:
            Array of predictions for each time point
        """
        # Input validation
        try:
            t_new = np.asarray(t_new, float)

            # Check for invalid values
            if np.any(np.isnan(t_new)) or np.any(np.isinf(t_new)):
                raise ValueError("New time points contain NaN or infinity values")
        except Exception as e:
            raise ValueError(f"Invalid input for t_new: {str(e)}")

        # Mean parameters approach
        if method == "mean_params":
            try:
                out = np.empty_like(t_new)
                lag_state = self._mu0  # AR(1) initial state

                for j, t in enumerate(t_new):
                    # Piece-wise-linear trend with safety checks
                    try:
                        cp = np.sum(self._delta * expit(self._gamma * (t - self.s))
                                    ) if self._delta.size else 0.0
                    except Exception:
                        # Fallback if calculation fails
                        cp = 0.0

                    mu = self._k * t + self._m + cp

                    # Additive Fourier seasonality with error protection
                    pos = 0
                    try:
                        for P, H in zip(self.periods, self.n_harm):
                            tau = t % P
                            for h in range(1, H + 1):
                                ang = 2 * np.pi * h * tau / P
                                mu += self._A[pos] * np.sin(ang) + self._B[pos] * np.cos(ang)
                                pos += 1
                    except Exception:
                        # If seasonality fails, continue without it
                        warnings.warn("Error in seasonal component calculation, using trend only")

                    # AR(1) disturbance
                    mu += self._rho * lag_state
                    lag_state = mu

                    # Link function with safety bounds for beta
                    if self.lik == "beta":
                        # Ensure predictions stay in valid range
                        out[j] = min(max(expit(mu), 1e-5), 1 - 1e-5)
                    else:
                        out[j] = mu

                return out

            except Exception as e:
                # Fallback to simple linear trend if everything else fails
                warnings.warn(f"Error during prediction: {str(e)}. Falling back to simple model.")
                simple_trend = self._k * t_new + self._m

                if self.lik == "beta":
                    # Ensure valid range for beta
                    return np.clip(expit(simple_trend), 1e-5, 1 - 1e-5)
                else:
                    return simple_trend

        # Sample from posterior approach
        elif method == "sample":
            try:
                # Verify we have MCMC samples
                if not isinstance(self.fit, CmdStanMCMC):
                    raise ValueError("Sampling requires NUTS inference (inference='nuts')")

                # Select a random posterior sample
                n_samples = len(self.fit.stan_variable("k"))
                sample_idx = np.random.randint(0, n_samples)

                # Extract parameters with error handling
                try:
                    k = float(self.fit.stan_variable("k")[sample_idx])
                    m = float(self.fit.stan_variable("m")[sample_idx])
                    gamma = float(self.fit.stan_variable("gamma")[sample_idx])
                except Exception:
                    # Fallback to mean parameters if extraction fails
                    warnings.warn("Failed to extract individual parameters, using posterior means")
                    return self.predict(t_new, method="mean_params")

                # Extract delta (changepoint adjustment) if it exists
                try:
                    if self.s.size > 0:
                        delta = self.fit.stan_variable("delta")[sample_idx]
                    else:
                        delta = np.zeros(0)
                except Exception:
                    delta = np.zeros(0)

                # Extract seasonal components
                try:
                    A_sin = self.fit.stan_variable("A_sin")[sample_idx]
                    B_cos = self.fit.stan_variable("B_cos")[sample_idx]
                except Exception:
                    # Fallback to mean seasonal components
                    A_sin = self._A
                    B_cos = self._B

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
                    # Piece-wise-linear trend
                    try:
                        cp = np.sum(delta * expit(gamma * (t - self.s))
                                    ) if delta.size else 0.0
                    except Exception:
                        cp = 0.0

                    mu = k * t + m + cp

                    # Additive Fourier seasonality
                    pos = 0
                    try:
                        for P, H in zip(self.periods, self.n_harm):
                            tau = t % P
                            for h in range(1, H + 1):
                                ang = 2 * np.pi * h * tau / P
                                mu += A_sin[pos] * np.sin(ang) + B_cos[pos] * np.cos(ang)
                                pos += 1
                    except Exception:
                        pass  # Continue without seasonality if it fails

                    # AR(1) disturbance
                    mu += rho * lag_state
                    lag_state = mu

                    # Link function with safety bounds
                    if self.lik == "beta":
                        out[j] = min(max(expit(mu), 1e-5), 1 - 1e-5)
                    else:
                        out[j] = mu

                return out

            except Exception as e:
                # Fallback to mean parameters if sampling fails
                warnings.warn(f"Error during sampling: {str(e)}. Falling back to mean parameters.")
                return self.predict(t_new, method="mean_params")

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

    # Add after the data clipping code and before the seasonality lists
    # ── parameter validation and constraints ───────────────────────
    if np.any(np.isnan(t)) or np.any(np.isnan(y)) or np.any(np.isinf(t)) or np.any(np.isinf(y)):
        raise ValueError("Input data contains NaN or infinity values")

    # Cap potentially problematic parameter values
    if delta_scale <= 0:
        delta_scale = 0.05
        warnings.warn("delta_scale must be positive. Using default value 0.05.")
    else:
        delta_scale = min(delta_scale, 0.77)  # Cap at a reasonable maximum

    if gamma_scale <= 0:
        gamma_scale = 3.0
        warnings.warn("gamma_scale must be positive. Using default value 3.0.")
    else:
        gamma_scale = min(max(gamma_scale, 1.0), 17.0)  # Keep in reasonable range

    if season_scale <= 0:
        season_scale = 1.0
        warnings.warn("season_scale must be positive. Using default value 1.0.")
    else:
        season_scale = min(max(season_scale, 0.1), 8.0)  # Keep in reasonable range





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

    def _smart_changepoints(t: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
        """Return k changepoint locations biased toward high curvature."""
        # 1) smooth the series to kill noise
        from scipy.ndimage import gaussian_filter1d
        y_s = gaussian_filter1d(y, sigma=max(1, len(y) // 60))

        # 2) absolute 2nd difference as “importance” score
        curv = np.abs(np.diff(y_s, n=2, prepend=[y_s[0], y_s[0]]))

        # 3) convert to CDF → pick equally‑spaced quantiles in *curvature space*
        cdf = np.cumsum(curv)
        cdf = cdf / cdf[-1]
        qs = np.linspace(0.05, 0.95, k)  # slightly tighter edges
        idx = np.searchsorted(cdf, qs)
        return t[idx]

    # ── changepoints (Prophet heuristic) ──────────────────────────
    if changepoints is None:
        if n_changepoints is None:
            n_changepoints = max(1, int(0.2 * len(t)))
        changepoints = _smart_changepoints(t, y, n_changepoints)
    else:
        changepoints = np.sort(np.asarray(changepoints, float))
        n_changepoints = changepoints.size

    # ── threading & env var ───────────────────────────────────────
    threads_per_chain = threads_per_chain or min(_mp.cpu_count(), 1)
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

    # -----------------------------------------------------------------
    # 3·  Calm initial‑value helper  (place once, just above inference)
    # -----------------------------------------------------------------
    def _zero_init(sd: dict) -> dict:
        """Return a single‑chain dict of near‑zero initials for LBFGS/BFGS."""
        return dict(
            k=0.0,
            m=0.0,
            delta=np.zeros(sd["num_changepoints"]),
            gamma=1.0,
            rho=0.0,
            mu0=0.0,
            A_sin=np.zeros(sd["total_harmonics"]),
            B_cos=np.zeros(sd["total_harmonics"]),
            log_phi0=np.log(10.0),
            beta_phi=0.1,
        )

    # -----------------------------------------------------------------
    # 4·  inference routes
    # -----------------------------------------------------------------
    data_used = stan_data  # ← points to the dict currently sent to Stan

    # ------------------------------------------------------------------#
    #  MAP  (primary)                                                   #
    # ------------------------------------------------------------------#
    if inference == "map":
        try:  # ── primary attempt ───────────
            data_used = stan_data
            with SuppressOutput():
                fit = model.optimize(
                    data=data_used,
                    algorithm="lbfgs",
                    iter=10_000,
                    seed=seed,
                    inits=_zero_init(data_used),
                    show_console=False,
                )

        except Exception as err:
            warnings.warn(
                f"Optimization failed: {extract_key_error(err)}.  Trying BFGS rescue."
            )

            # -------- 1st rescue – slightly safer hyper‑scales ----------
            data_used = stan_data_safe = stan_data.copy()
            data_used["delta_scale"] = min(stan_data["delta_scale"], 0.10)
            data_used["gamma_scale"] = max(min(stan_data["gamma_scale"], 8.0), 3.0)

            try:
                with SuppressOutput():
                    fit = model.optimize(
                        data=data_used,
                        algorithm="bfgs",
                        iter=5_000,
                        seed=seed or 42,
                        inits=_zero_init(data_used),
                        show_console=False,
                    )

            except Exception as err2:
                warnings.warn(
                    f"BFGS rescue also failed: {extract_key_error(err2)}.  "
                    "Falling back to an even simpler model."
                )

                # -------- 2nd rescue – ultra‑simple model ---------------
                data_used = stan_data_simple = stan_data.copy()
                data_used.update(
                    delta_scale=0.10,
                    gamma_scale=3.0,
                    season_scale=1.0,
                    num_changepoints=min(stan_data_simple["num_changepoints"], 3),
                )
                _clip_simple(data_used)

                with SuppressOutput():
                    fit = model.optimize(
                        data=data_used,
                        algorithm="bfgs",
                        iter=2_000,
                        seed=seed or 42,
                        inits=_zero_init(data_used),
                        show_console=False,
                    )

    # ------------------------------------------------------------------#
    #  ADVI  (variational)                                              #
    # ------------------------------------------------------------------#
    elif inference == "advi":
        try:  # ── primary attempt ───────────
            data_used = stan_data
            with SuppressOutput():
                fit = model.variational(
                    data=data_used,
                    algorithm="meanfield",
                    iter=iter,
                    draws=400,
                    grad_samples=20,
                    elbo_samples=20,
                    tol_rel_obj=2e-3,
                    seed=seed,
                    show_console=False,
                )
            if fit.num_draws < 1:
                raise RuntimeError("ADVI returned no draws")

        except Exception as err:
            warnings.warn(
                f"ADVI failed: {extract_key_error(err)}.  Switching to MAP fallback."
            )

            # -------- ADVI → MAP rescue --------------------------------
            data_used = stan_data_safe = stan_data.copy()
            data_used["delta_scale"] = min(stan_data["delta_scale"], 0.30)
            data_used["gamma_scale"] = max(min(stan_data["gamma_scale"], 8.0), 3.0)

            try:
                with SuppressOutput():
                    fit = model.optimize(
                        data=data_used,
                        algorithm="lbfgs",
                        iter=5_000,
                        seed=seed or 42,
                        inits=_zero_init(data_used),
                        show_console=False,
                    )

            except Exception as err2:
                warnings.warn(
                    f"MAP fallback also failed: {extract_key_error(err2)}.  "
                    "Trying very simple model."
                )

                data_used = stan_data_simple = stan_data.copy()
                data_used.update(
                    delta_scale=0.10,
                    gamma_scale=3.0,
                    season_scale=1.0,
                    num_changepoints=min(stan_data_simple["num_changepoints"], 3),
                )
                _clip_simple(data_used)

                with SuppressOutput():
                    fit = model.optimize(
                        data=data_used,
                        algorithm="bfgs",
                        iter=2_000,
                        seed=seed or 42,
                        inits=_zero_init(data_used),
                        show_console=False,
                    )

    # ------------------------------------------------------------------#
    #  NUTS  (full HMC)                                                 #
    # ------------------------------------------------------------------#
    elif inference == "nuts":
        try:  # ── primary attempt ───────────
            data_used = stan_data
            with SuppressOutput(stdout=True, stderr=True):
                fit = model.sample(
                    data=data_used,
                    chains=chains,
                    parallel_chains=chains,
                    iter_warmup=warmup,
                    iter_sampling=iter - warmup,
                    adapt_delta=adapt_delta,
                    max_treedepth=max_treedepth,
                    threads_per_chain=threads_per_chain,
                    seed=seed,
                    show_progress=False,
                    show_console=False,
                )

        except Exception as err:
            warnings.warn(
                f"NUTS sampling failed: {extract_key_error(err)}.  Falling back to ADVI."
            )

            # -------- NUTS → ADVI rescue -------------------------------
            try:
                data_used = stan_data
                with SuppressOutput():
                    fit = model.variational(
                        data=data_used,
                        algorithm="meanfield",
                        iter=iter,
                        draws=400,
                        grad_samples=20,
                        elbo_samples=20,
                        tol_rel_obj=2e-3,
                        seed=seed,
                        show_console=False,
                    )
                if fit.num_draws < 1:
                    raise RuntimeError("ADVI returned no draws")

            except Exception as err2:
                warnings.warn(
                    f"ADVI fallback also failed: {extract_key_error(err2)}.  "
                    "Trying MAP as a last resort."
                )

                # -------- ADVI → MAP rescue ----------------------------
                try:
                    data_used = stan_data_safe = stan_data.copy()
                    with SuppressOutput():
                        fit = model.optimize(
                            data=data_used,
                            algorithm="lbfgs",
                            iter=5_000,
                            seed=seed or 42,
                            inits=_zero_init(data_used),
                            show_console=False,
                        )

                except Exception as err3:
                    warnings.warn(
                        f"LBFGS failed too: {extract_key_error(err3)}.  "
                        "Using the simplest model possible."
                    )

                    data_used = stan_data_simple = stan_data.copy()
                    data_used.update(
                        delta_scale=0.15,
                        gamma_scale=3.0,
                        season_scale=1.0,
                        num_changepoints=min(stan_data_simple["num_changepoints"], 2),
                        n_harmonics=[1] * len(stan_data_simple["period"]),
                    )
                    data_used["total_harmonics"] = sum(data_used["n_harmonics"])
                    _clip_simple(data_used)

                    with SuppressOutput():
                        fit = model.optimize(
                            data=data_used,
                            algorithm="bfgs",
                            iter=2_000,
                            seed=seed or 42,
                            inits=_zero_init(data_used),
                            show_console=False,
                        )

    # ------------------------------------------------------------------#
    #  Invalid keyword                                                  #
    # ------------------------------------------------------------------#
    else:
        raise ValueError("inference must be 'map', 'advi' or 'nuts'.")

    # -----------------------------------------------------------------
    # 5·  build predictor  — use `data_used["s"]`
    # -----------------------------------------------------------------
    return ChurnProphetModel(
        fit,
        changepoints=np.asarray(data_used["s"], float),  # ← always matches Stan
        periods=periods,
        n_harm=num_harmonics,
        likelihood=likelihood,
    )