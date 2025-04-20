// ─────────────────────────────────────────────────────────────────────────────
//  Murphet – multi‑season Stan model  (v2: hierarchical seasonality shrinkage)
// ─────────────────────────────────────────────────────────────────────────────
functions {
  real partial_sum_beta(array[] real y_slice,
                        int start, int end,
                        vector t,
                        real k, real m, real q,
                        vector delta, real gamma,
                        vector A_sin, vector B_cos,
                        real phi,           real tau,          // NEW tau
                        int   num_cp,       vector s,
                        int   num_seasons,  array[] int n_harm,
                        array[] real period) {

    real lp = 0;
    for (i in 1:size(y_slice)) {
      int idx = start + i - 1;

      // —— Trend w. smooth changepoints ————————————————————————
      real cp = 0;
      for (j in 1:num_cp)
        cp += delta[j] * inv_logit(gamma * (t[idx] - s[j]));
      real trend = k * t[idx] + m + q * square(t[idx]) + cp;

      // —— Seasonality (multiple Fourier blocks) ————————————————
      real seas = 0;
      int pos   = 1;
      for (b in 1:num_seasons) {
        real tmod = fmod(t[idx], period[b]);
        for (h in 1:n_harm[b]) {
          real ang = 2 * pi() * h * tmod / period[b];
          seas += A_sin[pos] * sin(ang) + B_cos[pos] * cos(ang);
          pos  += 1;
        }
      }

      // —— Likelihood (soft seasonal scale τ) ————————————————
      real p = inv_logit(trend + seas / tau);
      // beta_lpdf needs 0<p<1, add small ε
      p = fmin(fmax(p, 1e-6), 1 - 1e-6);
      lp += beta_lpdf(y_slice[i] | p * phi, (1 - p) * phi);
    }
    return lp;
  }
}

data {
  int<lower=1> N;
  vector<lower=0, upper=1>[N] y;
  vector[N] t;

  // Trend / changepoints
  int<lower=0>   num_changepoints;
  vector[num_changepoints] s;
  real<lower=0> delta_scale;
  real<lower=0> gamma_scale;

  // Seasonality meta
  int<lower=1>              num_seasons;
  array[num_seasons] int<lower=1>  n_harmonics;
  array[num_seasons] real<lower=0> period;
  int<lower=1>              total_harmonics;
  real<lower=0>             season_scale;
}

parameters {
  // trend
  real k;
  real m;
  real q;
  vector[num_changepoints] delta;
  real<lower=0> gamma;

  // seasonal amplitudes (hierarchical)
  vector[total_harmonics]  A_sin_raw;
  vector[total_harmonics]  B_cos_raw;
  vector<lower=0>[total_harmonics] sigma_h;   // local scales

  real<lower=0.1> phi;    // beta precision
  real<lower=0>   tau;    // soft seasonality scale
}

transformed parameters {
  vector[total_harmonics] A_sin = A_sin_raw .* sigma_h * season_scale;
  vector[total_harmonics] B_cos = B_cos_raw .* sigma_h * season_scale;
}

model {
  // ——— Priors ————————————————————————————————————————————————
  k          ~ normal(0, 0.1);
  m          ~ normal(logit(0.06), 0.5);
  q          ~ normal(0, 0.01);
  delta      ~ double_exponential(0, delta_scale);
  gamma      ~ gamma(3, 1 / gamma_scale);

  // Horseshoe‑like shrinkage for each Fourier coeff
  sigma_h    ~ cauchy(0, 1);
  A_sin_raw  ~ normal(0, 1);
  B_cos_raw  ~ normal(0, 1);

  phi        ~ lognormal(log(20), 0.4);
  tau        ~ normal(1, 0.5);      // centred near 1; pushes seas/τ into ±4

  // ——— Parallel likelihood ————————————————————————————————
  target += reduce_sum(partial_sum_beta,
                       to_array_1d(y),
                       16,
                       t, k, m, q, delta, gamma,
                       A_sin, B_cos, phi, tau,
                       num_changepoints, s,
                       num_seasons, n_harmonics, period);
}
