// ─────────────────────────────────────────────────────────────
//  Murphet – multi‑season model    (Beta likelihood head)
//  piece‑wise‑linear trend + weak‑normal seasonality
// ─────────────────────────────────────────────────────────────
functions {
  real partial_sum_beta(array[] real y_slice,
                        int start, int end,
                        vector t,
                        real k, real m,
                        vector delta, real gamma,
                        vector A_sin, vector B_cos,
                        real phi,
                        int   num_cp,       vector s,
                        int   num_seasons,  array[] int n_harm,
                        array[] real period) {

    real lp = 0;
    for (i in 1:size(y_slice)) {
      int idx = start + i - 1;

      // --- piece‑wise‑linear trend --------------------------------
      real cp = 0;
      for (j in 1:num_cp)
        cp += delta[j] * inv_logit(gamma * (t[idx] - s[j]));
      real mu = k * t[idx] + m + cp;

      // --- additive seasonality ----------------------------------
      int pos = 1;
      for (b in 1:num_seasons) {
        real tau = fmod(t[idx], period[b]);
        for (h in 1:n_harm[b]) {
          real ang = 2 * pi() * h * tau / period[b];
          mu      += A_sin[pos] * sin(ang) + B_cos[pos] * cos(ang);
          pos     += 1;
        }
      }

      real p = inv_logit(mu);          // map to (0,1)
      lp    += beta_lpdf(y_slice[i] | p * phi, (1 - p) * phi);
    }
    return lp;
  }
}

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

parameters {
  real k;
  real m;
  vector[num_changepoints] delta;
  real<lower=0> gamma;

  vector[total_harmonics] A_sin;
  vector[total_harmonics] B_cos;

  real<lower=0.1> phi;          // beta precision
}

model {
  // --- Priors (Prophet‑style) -----------------------------------
  k        ~ normal(0, 0.5);
  m        ~ normal(0, 5);
  delta    ~ double_exponential(0, delta_scale);
  gamma    ~ gamma(3, 1 / gamma_scale);

  A_sin    ~ normal(0, 10 * season_scale);
  B_cos    ~ normal(0, 10 * season_scale);

  phi      ~ lognormal(log(20), 0.4);

  // --- parallel likelihood --------------------------------------
  target += reduce_sum(partial_sum_beta,
                       to_array_1d(y), 16,
                       t, k, m, delta, gamma,
                       A_sin, B_cos, phi,
                       num_changepoints, s,
                       num_seasons, n_harmonics, period);
}
