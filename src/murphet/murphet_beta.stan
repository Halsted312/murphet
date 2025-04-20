// ─────────────────────────────────────────────────────────────
//  Murphet – multi‑season model       (Beta likelihood head)
//  • piece‑wise‑linear trend  +  AR(1) disturbance
//  • weak‑Normal seasonality  (σ ≈ 10 · season_scale)
//  • heteroscedastic precision:  φᵢ declines when |μ_det| is large
// ─────────────────────────────────────────────────────────────
functions {
  /**  parallel log‑likelihood over a slice, beta head */
  real partial_sum_beta(array[] real   y_slice,
                        int             start,
                        int             end,
                        vector          t,
                        real            k,
                        real            m,
                        vector          delta,
                        real            gamma,
                        real            rho,
                        real            mu0,
                        vector          A_sin,
                        vector          B_cos,
                        real            log_phi0,   // NEW
                        real            beta_phi,   // NEW
                        int             num_cp,
                        vector          s,
                        int             num_seasons,
                        array[] int     n_harm,
                        array[] real    period) {

    real lp  = 0;
    real lag = mu0;                     // AR(1) state

    for (i in 1:size(y_slice)) {
      int idx = start + i - 1;

      // ── deterministic trend + CPs ────────────────────────────
      real cp = 0;
      for (j in 1:num_cp)
        cp += delta[j] * inv_logit(gamma * (t[idx] - s[j]));
      real mu_det = k * t[idx] + m + cp;

      // ── additive Fourier seasonality ─────────────────────────
      int pos = 1;
      for (b in 1:num_seasons) {
        real tau = fmod(t[idx], period[b]);
        for (h in 1:n_harm[b]) {
          real ang  = 2 * pi() * h * tau / period[b];
          mu_det   += A_sin[pos] * sin(ang) + B_cos[pos] * cos(ang);
          pos      += 1;
        }
      }

      // ── AR(1) disturbance  y* = μ_det + ρ·lag  ───────────────
      real mu  = mu_det + rho * lag;
      lag      = mu;

      // ── heteroscedastic precision  φᵢ  ───────────────────────
      real phi_i = exp(log_phi0 - beta_phi * abs(mu_det));
      // ── Beta likelihood  yᵢ ~ Beta(p·φᵢ, (1‑p)·φᵢ) ───────────
      real p   = inv_logit(mu);
      lp      += beta_lpdf(y_slice[i] | p * phi_i, (1 - p) * phi_i);
    }
    return lp;
  }
}

// ─────────────────────────── data ────────────────────────────
data {
  int<lower=1> N;
  vector<lower=0,upper=1>[N] y;
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

// ───────────────────── parameters  ───────────────────────────
parameters {
  // trend
  real k;
  real m;
  vector[num_changepoints] delta;
  real<lower=0> gamma;

  // AR(1)
  real<lower=-1,upper=1> rho;
  real                   mu0;

  // seasonality
  vector[total_harmonics] A_sin;
  vector[total_harmonics] B_cos;

  // heteroscedastic precision
  real log_phi0;            // unconstrained
  real<lower=0> beta_phi;   // strength (0 = homoscedastic)
}

// ───────────────────────── model  ────────────────────────────
model {
  // trend priors
  k      ~ normal(0, 0.5);
  m      ~ normal(0, 5);
  delta  ~ double_exponential(0, delta_scale);
  gamma  ~ gamma(3, 1 / gamma_scale);

  // AR(1) priors (ρ prior widened for quarterly macro data)
  rho  ~ normal(0, 0.5);
  mu0  ~ normal(logit(mean(y)), 1);

  // seasonality
  A_sin ~ normal(0, 10 * season_scale);
  B_cos ~ normal(0, 10 * season_scale);

  // heteroscedastic φ priors
  log_phi0 ~ normal(log(20), 1);     // centre near φ ≈ 20
  beta_phi ~ normal(0, 0.5);         // shrink towards homoscedastic

  // likelihood (parallel)
  target += reduce_sum(
              partial_sum_beta,
              to_array_1d(y), 16,
              t, k, m, delta, gamma, rho, mu0,
              A_sin, B_cos,
              log_phi0, beta_phi,
              num_changepoints, s,
              num_seasons, n_harmonics, period);
}
