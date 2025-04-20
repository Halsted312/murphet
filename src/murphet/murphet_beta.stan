// ─────────────────────────────────────────────────────────────
//  Murphet  –  multi‑season model         (Beta likelihood head)
//  • piece‑wise‑linear trend  +  AR(1) disturbance
//  • weak‑Normal seasonality  (σ ≈ 10 · season_scale)
// ─────────────────────────────────────────────────────────────
functions {
  /**  parallel log‑likelihood over y_slice          (beta head)
   *
   *   Arguments after `y_slice` & slice‑indices are passed through
   *   reduce_sum() from the model block.
   */
  real partial_sum_beta(array[]   real  y_slice,
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
                        real               phi,
                        int                num_cp,
                        vector             s,
                        int                num_seasons,
                        array[] int        n_harm,
                        array[] real       period) {

    real lp  = 0;
    real lag = mu0;                 // initialise latent AR(1) state

    for (i in 1:size(y_slice)) {
      int idx = start + i - 1;

      // ------- piece‑wise‑linear trend --------------------------
      real cp = 0;
      for (j in 1:num_cp)
        cp += delta[j] * inv_logit(gamma * (t[idx] - s[j]));
      real mu_det = k * t[idx] + m + cp;

      // ------- additive seasonality -----------------------------
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

      // ------- AR(1) disturbance -------------------------------
      real mu = mu_det + rho * lag;
      lag     = mu;                 // propagate

      // ------- Beta likelihood  y ~ Beta(p·φ, (1‑p)·φ) ----------
      real p  = inv_logit(mu);      // ensure 0<p<1
      lp     += beta_lpdf(y_slice[i] | p * phi, (1 - p) * phi);
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

  // beta precision
  real<lower=0.1> phi;
}

// ────────────────────────── model ─────────────────────────────
model {
  // ---- priors: trend & CPs -----------------------------------
  k      ~ normal(0, 0.5);
  m      ~ normal(0, 5);
  delta  ~ double_exponential(0, delta_scale);
  gamma  ~ gamma(3, 1 / gamma_scale);

  // ---- priors: AR(1) -----------------------------------------
  rho  ~ normal(0, 0.3);                      // mild persistence
  mu0  ~ normal(logit(mean(y)), 1);           // centre at mean rate

  // ---- priors: seasonality -----------------------------------
  A_sin ~ normal(0, 10 * season_scale);
  B_cos ~ normal(0, 10 * season_scale);

  // ---- priors: precision -------------------------------------
  phi   ~ lognormal(log(20), 0.4);

  // ---- parallel log‑likelihood -------------------------------
  target += reduce_sum(
              partial_sum_beta,
              to_array_1d(y), 16,
              t, k, m, delta, gamma, rho, mu0,
              A_sin, B_cos, phi,
              num_changepoints, s,
              num_seasons, n_harmonics, period);
}
