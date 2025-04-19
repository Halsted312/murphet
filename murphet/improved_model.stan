// improved_model.stan
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
  real k;
  real m;
  real q;
  vector[num_changepoints] delta;
  real<lower=0.01, upper=100> gamma;  // Bounded away from 0

  vector[num_harmonics] A_sin;
  vector[num_harmonics] B_cos;

  real<lower=0.1> phi;  // Lower bounded to prevent zeros
}

transformed parameters {
  vector[N] trend;
  vector[N] seasonal;
  vector[N] mu;
  vector<lower=0.001, upper=0.999>[N] p;  // Force reasonable bounds

  for (i in 1:N) {
    real cp_effect = 0;
    for (j in 1:num_changepoints) {
      cp_effect += delta[j] * inv_logit(gamma * (t[i] - s[j]));
    }
    trend[i] = k * t[i] + m + q * square(t[i]) + cp_effect;

    real t_mod = fmod(t[i], period);
    real seas = 0;
    for (r in 1:num_harmonics) {
      seas += A_sin[r] * sin(2 * pi() * r * t_mod / period)
            + B_cos[r] * cos(2 * pi() * r * t_mod / period);
    }
    seasonal[i] = seas;

    mu[i] = trend[i] + seasonal[i];
    // Force p to stay away from exact 0 or 1
    p[i] = fmax(fmin(inv_logit(fmin(mu[i], 4)), 0.999), 0.001);
  }
}

model {
  // More informative priors
  k ~ normal(0, 0.1);  // Tighter prior on slope
  m ~ normal(logit(0.06), 0.5);  // Centered around typical churn rate
  q ~ normal(0, 0.01);  // Tighter quadratic
  delta ~ double_exponential(0, delta_scale);
  gamma ~ gamma(3, 1);  // Slightly higher mean

  // Tighter priors on seasonality
  A_sin ~ normal(0, 0.5);
  B_cos ~ normal(0, 0.5);

  // Stronger prior on dispersion
  phi ~ gamma(5, 0.1);  // Mean of 50, more concentration

  // Direct likelihood calculation to avoid potential issues
  for (i in 1:N) {
    target += beta_lpdf(y[i] | p[i] * phi, (1 - p[i]) * phi);
  }
}