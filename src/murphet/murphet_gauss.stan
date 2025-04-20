// ────────────────────────────────────────────────────────────────
//  Murphet – multi‑season model  (Gaussian likelihood)
//  * piece‑wise‑linear trend   (Prophet‑style)
//  * weak Normal seasonal priors σ≈10
// ────────────────────────────────────────────────────────────────
functions {
  real partial_sum_gauss(array[] real y_slice,
                         int start, int end,
                         vector t,
                         real k, real m,
                         vector delta, real gamma,
                         vector A_sin, vector B_cos,
                         real sigma,                      // ← NEW
                         int   num_cp,       vector s,
                         int   num_seasons,  array[] int n_harm,
                         array[] real period) {

    real lp = 0;
    for (i in 1:size(y_slice)) {
      int idx = start + i - 1;

      // —— piece‑wise‑linear trend ————————————————
      real cp = 0;
      for (j in 1:num_cp)
        cp += delta[j] * inv_logit(gamma * (t[idx] - s[j]));
      real trend = k * t[idx] + m + cp;

      // —— additive seasonality ————————————————
      real seas = 0;
      int  pos  = 1;
      for (b in 1:num_seasons) {
        real tmod = fmod(t[idx], period[b]);
        for (h in 1:n_harm[b]) {
          real ang = 2 * pi() * h * tmod / period[b];
          seas   += A_sin[pos] * sin(ang) + B_cos[pos] * cos(ang);
          pos    += 1;
        }
      }

      // —— Gaussian log‑likelihood ————————————————
      lp += normal_lpdf(y_slice[i] | trend + seas, sigma);
    }
    return lp;
  }
}

data {
  int<lower=1> N;
  vector[N] y;                               // ratios can be outside (0,1)
  vector[N] t;

  // trend / changepoints
  int<lower=0>   num_changepoints;
  vector[num_changepoints] s;
  real<lower=0>  delta_scale;
  real<lower=0>  gamma_scale;

  // seasonality meta
  int<lower=1>              num_seasons;
  array[num_seasons] int<lower=1>  n_harmonics;
  array[num_seasons] real<lower=0> period;
  int<lower=1>              total_harmonics;
}

parameters {
  // trend
  real k;
  real m;
  vector[num_changepoints] delta;
  real<lower=0> gamma;

  // seasonality
  vector[total_harmonics] A_sin;
  vector[total_harmonics] B_cos;

  // observation noise
  real<lower=0.001> sigma;
}

model {
  // —— Priors (very close to Prophet) ——————————
  k         ~ normal(0, 0.5);
  m         ~ normal(0, 5);
  delta     ~ double_exponential(0, delta_scale);
  gamma     ~ gamma(3, 1 / gamma_scale);
  A_sin     ~ normal(0, 10);
  B_cos     ~ normal(0, 10);
  sigma     ~ student_t(3, 0, 1);           // weak but heavy‑tailed

  // —— Likelihood (parallel reduce_sum) ————————
  target += reduce_sum(partial_sum_gauss,
                       to_array_1d(y), 16,
                       t, k, m, delta, gamma,
                       A_sin, B_cos, sigma,
                       num_changepoints, s,
                       num_seasons, n_harmonics, period);
}
