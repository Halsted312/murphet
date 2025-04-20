// ─────────────────────────────────────────────────────────────
//  Murphet – multi‑season model     (Gaussian / Student‑t head)
//  • piece‑wise‑linear trend  +  AR(1) disturbance
//  • weak‑Normal seasonality  (σ ≈ 10 · season_scale)
//  • heteroscedastic scale    σᵢ = exp(log_σ0 + β_σ·|μ_det|)
//  • heavy‑tail option        Student‑t(ν)     (ν learned)
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
                         real               rho,
                         real               mu0,
                         vector             A_sin,
                         vector             B_cos,
                         real               log_sigma0,
                         real               beta_sigma,
                         real<lower=2>      nu,          // dof for Student‑t
                         int                num_cp,
                         vector             s,
                         int                num_seasons,
                         array[] int        n_harm,
                         array[] real       period) {

    real lp  = 0;
    real lag = mu0;                       // AR(1) latent state

    for (i in 1:size(y_slice)) {
      int idx = start + i - 1;

      // ── deterministic part: trend + CPs ─────────────────────
      real cp = 0;
      for (j in 1:num_cp)
        cp += delta[j] * inv_logit(gamma * (t[idx] - s[j]));
      real mu_det = k * t[idx] + m + cp;

      // ── additive seasonality  (Fourier blocks) ─────────────
      int pos = 1;
      for (b in 1:num_seasons) {
        real tau = fmod(t[idx], period[b]);
        for (h in 1:n_harm[b]) {
          real ang = 2 * pi() * h * tau / period[b];
          mu_det  += A_sin[pos] * sin(ang) + B_cos[pos] * cos(ang);
          pos     += 1;
        }
      }

      // ── AR(1) disturbance  y* = μ_det + ρ·lag  ─────────────
      real mu = mu_det + rho * lag;
      lag     = mu;

      // ── heteroscedastic σᵢ  (log‑linear in |μ_det|) ────────
      real sigma_i_raw = exp(log_sigma0 + beta_sigma * abs(mu_det));
      real sigma_i = fmax(sigma_i_raw, 1e-5);  // Safety check for minimum scale

      // ── Student‑t likelihood  (ν → large ⇒ Gaussian) ───────
      lp += student_t_lpdf(y_slice[i] | nu, mu, sigma_i);
    }
    return lp;
  }
}

// ─────────────────────────── data ────────────────────────────
data {
  int<lower=1> N;
  vector[N] y;                    // original scale
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

  // AR(1)
  real<lower=-1,upper=1> rho;
  real                   mu0;

  // seasonality
  vector[total_harmonics] A_sin;
  vector[total_harmonics] B_cos;

  // heteroscedastic scale hyper‑params
  real log_sigma0;
  real<lower=0> beta_sigma;

  // heavy‑tail dof
  real<lower=2> nu;
}

// ────────────────────────── model ─────────────────────────────
model {
  // priors: trend & CPs
  k      ~ normal(0, 0.5);
  m      ~ normal(0, 5);
  delta  ~ double_exponential(0, delta_scale);
  gamma  ~ gamma(3, 1 / gamma_scale);

  // priors: AR(1)
  rho  ~ normal(0, 0.5);          // quarterly data ⇒ wider prior
  mu0  ~ normal(mean(y), 1);

  // priors: seasonality
  A_sin ~ normal(0, 10 * season_scale);
  B_cos ~ normal(0, 10 * season_scale);

  // priors: heteroscedastic scale
  log_sigma0 ~ normal(log(sd(y)), 1);
  beta_sigma ~ normal(0, 0.5);    // shrink towards homoscedastic

  // priors: Student‑t dof  (fat tails → small ν)
  nu ~ exponential(1 / 30);       // mode 2, median ≈ 21

  // parallel likelihood
  target += reduce_sum(
              partial_sum_gauss,
              to_array_1d(y), 16,
              t, k, m, delta, gamma, rho, mu0,
              A_sin, B_cos,
              log_sigma0, beta_sigma, nu,
              num_changepoints, s,
              num_seasons, n_harmonics, period);
}