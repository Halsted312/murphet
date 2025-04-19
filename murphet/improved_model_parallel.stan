// improved_model_parallel.stan  ✧  2025‑04‑18  (fixed name)

// ──────────────────────────────────────────────────────────────
// 1.  User‑defined partial log‑likelihood for reduce_sum
//     NOTE: name **must NOT** end with _lpdf / _lpmf / _lcdf
// ──────────────────────────────────────────────────────────────
functions {
  /* partial log‑likelihood for reduce_sum ------------------------- */
  real beta_partial_sum(array[] real y_slice,        // ① sliced ARRAY first
                        int start, int end,          // ② start,end second/third
                        vector t,
                        real k, real m, real q,
                        vector delta, real gamma,
                        vector A_sin, vector B_cos,
                        real phi,
                        int num_cp, vector s,
                        int num_harm, real period) {

    real lp = 0;
    for (i in 1:size(y_slice)) {
      int idx = start + i - 1;          // global index
      real cp = 0;
      for (j in 1:num_cp)
        cp += delta[j] * inv_logit(gamma * (t[idx] - s[j]));

      real trend = k * t[idx] + m + q * square(t[idx]) + cp;

      real seas  = 0;
      real tmod  = fmod(t[idx], period);
      for (r in 1:num_harm) {
        real angle = 2 * pi() * r * tmod / period;
        seas += A_sin[r] * sin(angle) + B_cos[r] * cos(angle);
      }

      real p = inv_logit(fmin(trend + seas, 4));
      lp += beta_lpdf(y_slice[i] | p * phi, (1 - p) * phi);
    }
    return lp;
  }
}


// ──────────────────────────────────────────────────────────────
// 2.  Data, parameters, model (unchanged except for reduce_sum call)
// ──────────────────────────────────────────────────────────────
data {
  int<lower=1> N;
  vector<lower=0, upper=1>[N] y;
  vector[N] t;
  int<lower=0> num_changepoints;
  vector[num_changepoints] s;
  real<lower=0> delta_scale;
  int<lower=0> num_harmonics;
  real<lower=0> period;
}

parameters {
  real k; real m; real q;
  vector[num_changepoints] delta;
  real<lower=0> gamma;
  vector[num_harmonics] A_sin;
  vector[num_harmonics] B_cos;
  real<lower=0.1> phi;
}

model {
  // ─ Priors (identical) ─
  k     ~ normal(0, 0.1);
  m     ~ normal(logit(0.06), 0.5);
  q     ~ normal(0, 0.01);
  delta ~ double_exponential(0, delta_scale);
  gamma ~ gamma(3, 1);
  A_sin ~ normal(0, 0.5);
  B_cos ~ normal(0, 0.5);
  phi ~ lognormal(log(20), 0.3);   // mean ≈ 20, SD ≈ 6


    // ─ Parallelised likelihood ───────────────────────────────────────

    int grainsize = 16;
    target += reduce_sum(
                beta_partial_sum,
                to_array_1d(y),          // ← pass ARRAY, not vector
                grainsize,
                t, k, m, q, delta, gamma,
                A_sin, B_cos, phi,
                num_changepoints, s,
                num_harmonics, period);
}