// ─────────────────────────────────────────────────────────────
//  Murphet  –  multi‑season model        (Gaussian likelihood)
//  • piece‑wise‑linear trend  +  AR(1) disturbance
//  • weak‑Normal seasonality  (σ ≈ 10 · season_scale)
// ─────────────────────────────────────────────────────────────
functions {
  real partial_sum_gauss(array[]   real  y_slice,
                         int                start,
                         int                end,
                         vector             t,
                         real               k,
                         real               m,
                         vector             delta,
                         real               gamma,
                         real               rho,        // NEW
                         real               mu0,        // NEW
                         vector             A_sin,
                         vector             B_cos,
                         real<lower=0>      sigma,
                         int                num_cp,
                         vector             s,
                         int                num_seasons,
                         array[] int        n_harm,
                         array[] real       period) {

    real lp  = 0;
    real lag = mu0;                       // AR(1) initial state

    for (i in 1:size(y_slice)) {
      int idx = start + i - 1;

      // --- piece‑wise‑linear deterministic part -----------------
      real cp = 0;
      for (j in 1:num_cp)
        cp += delta[j] * inv_logit(gamma * (t[idx] - s[j]));
      real mu_det = k * t[idx] + m + cp;

      // --- additive seasonality --------------------------------
      real seas = 0;
      int  pos  = 1;
      for (b in 1:num_seasons) {
        real tau = fmod(t[idx], period[b]);
        for (h in 1:n_harm[b]) {
          real ang = 2 * pi() * h * tau / period[b];
          seas    += A_sin[pos] * sin(ang)
                   + B_cos[pos] * cos(ang);
          pos     += 1;
        }
      }
      mu_det += seas;

      // --- AR(1) disturbance -----------------------------------
      real mu = mu_det + rho * lag;
      lag     = mu;

      // --- Gaussian likelihood ---------------------------------
      lp += normal_lpdf(y_slice[i] | mu, sigma);
    }
    return lp;
  }
}

// ─────────────────────────── data ────────────────────────────
data {
  int<lower=1> N;
  vector[N] y;                    // already on original scale
  vector[N] t;

  int<lower=0>   num_changepoints;
  vector[num_changepoints] s;
  real<lower=0> delta_scale;
  real<lower=0> gamma_scale;

  int<lower=1>              num_seasons;
  array[num_seasons] int<lower=1>  n_harmonics;
  array[num_seasons] real<lower=0> period;
  int<lower=1>              total_harmonics;
  real<lower=0>             season_scale;
}

// ─────────────────────── parameters ───────────────────────────
parameters {
  // trend
  real k;
  real m;
  vector[num_changepoints] delta;
  real<lower=0> gamma;

  // AR(1) disturbance
  real<lower=-1,upper=1> rho;
  real mu0;

  // seasonality
  vector[total_harmonics] A_sin;
  vector[total_harmonics] B_cos;

  // observation noise
  real<lower=0.001> sigma;
}

// ────────────────────────── model ─────────────────────────────
model {
  // ---- priors: trend & CPs -----------------------------------
  k      ~ normal(0, 0.5);
  m      ~ normal(0, 5);
  delta  ~ double_exponential(0, delta_scale);
  gamma  ~ gamma(3, 1 / gamma_scale);

  // ---- priors: AR(1) -----------------------------------------
  rho  ~ normal(0, 0.3);
  mu0  ~ normal(mean(y), 1);

  // ---- priors: seasonality -----------------------------------
  A_sin ~ normal(0, 10 * season_scale);
  B_cos ~ normal(0, 10 * season_scale);

  // ---- priors: obs‑noise -------------------------------------
  sigma ~ student_t(3, 0, 1);        // heavy‑tailed for robustness

  // ---- parallel likelihood -----------------------------------
  target += reduce_sum(
              partial_sum_gauss,
              to_array_1d(y), 16,
              t, k, m, delta, gamma, rho, mu0,
              A_sin, B_cos, sigma,
              num_changepoints, s,
              num_seasons, n_harmonics, period);
}
