// ─────────────────────────────────────────────────────────────────────────────
//  Murphet – multi‑season Stan model  (parallel‑safe)
//  Save as: murphet/murphet/murphet_model.stan
// ─────────────────────────────────────────────────────────────────────────────
functions {
  // Partial log‑likelihood for reduce_sum ------------------------------------
  real _beta_partial_sum(array[] real y_slice,
                         int start, int end,
                         vector t,
                         real k, real m, real q,
                         vector delta, real gamma,
                         vector A_sin, vector B_cos,
                         real phi,
                         int   num_cp,         vector s,
                         int   num_seasons,    array[] int num_harm,
                         array[] real period) {

    real lp = 0;

    for (i in 1:size(y_slice)) {
      int idx = start + i - 1;

      // —— Trend with smooth changepoints ————————————————
      real cp = 0;
      for (j in 1:num_cp)
        cp += delta[j] * inv_logit(gamma * (t[idx] - s[j]));
      real trend = k * t[idx] + m + q * square(t[idx]) + cp;

      // —— Seasonality (multiple Fourier blocks) ————————————
      real seas = 0;
      real tmod;
      int  pos  = 1;                         // flat index in A/B vectors

      for (s_ix in 1:num_seasons) {
        tmod = fmod(t[idx], period[s_ix]);

        for (h in 1:num_harm[s_ix]) {
          real ang = 2 * pi() * h * tmod / period[s_ix];
          seas    += A_sin[pos] * sin(ang)
                   + B_cos[pos] * cos(ang);
          pos += 1;
        }
      }

      // —— Likelihood ————————————————————————————————
      real p = inv_logit(fmin(trend + seas, 4));
      lp += beta_lpdf(y_slice[i] | p * phi, (1 - p) * phi);
    }
    return lp;
  }
}
// ─────────────────────────────────────────────────────────────────────────────
data {
  int<lower=1> N;
  vector<lower=0, upper=1>[N] y;
  vector[N] t;

  // Trend controls
  int<lower=0>   num_changepoints;
  vector[num_changepoints] s;
  real<lower=0> delta_scale;

  // NEW — multi‑seasonality controls
  int<lower=1>              num_seasons;
  array[num_seasons] int<lower=1>  num_harmonics;
  array[num_seasons] real<lower=0> period;
  int<lower=1>              total_harmonics;          // Σ num_harmonics
}
parameters {
  real k;           real m;          real q;
  vector[num_changepoints] delta;
  real<lower=0> gamma;

  // Flattened Fourier coefficients
  vector[total_harmonics] A_sin;
  vector[total_harmonics] B_cos;

  real<lower=0.1> phi;
}
model {
  // ─ Priors ────────────────────────────────────────────────────────────────
  k      ~ normal(0, 0.1);
  m      ~ normal(logit(0.06), 0.5);
  q      ~ normal(0, 0.01);
  delta  ~ double_exponential(0, delta_scale);
  gamma  ~ gamma(3, 1);

  A_sin  ~ normal(0, 0.5);
  B_cos  ~ normal(0, 0.5);
  phi    ~ lognormal(log(20), 0.3);

  // ─ Parallelised likelihood  (grainsize 16) ───────────────────────────────
  int grainsize = 16;
  target += reduce_sum(
              _beta_partial_sum,
              to_array_1d(y),
              grainsize,
              t, k, m, q, delta, gamma,
              A_sin, B_cos, phi,
              num_changepoints, s,
              num_seasons, num_harmonics, period);
}
