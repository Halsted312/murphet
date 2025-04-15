data {
  int<lower=1> N;                       // Number of data points
  vector<lower=0, upper=1>[N] y;          // Churn rates between 0 and 1
  vector[N] t;                          // Time index (e.g., months)

  // Changepoint data:
  int<lower=0> num_changepoints;          // Number of changepoints
  vector[num_changepoints] s;             // Changepoint times (should lie within t)
  real<lower=0> delta_scale;              // Scale for the Laplace prior on changepoint adjustments

  // Seasonality:
  int<lower=0> num_harmonics;             // Number of Fourier pairs for seasonality
  real<lower=0> period;                   // Seasonal period (e.g., 12 for monthly data with yearly seasonality)
}

parameters {
  // Trend components:
  real k;                               // Base slope
  real m;                               // Intercept
  real q;                               // Quadratic term for slight curvature
  vector[num_changepoints] delta;       // Changepoint adjustments on the trend
  real<lower=0> gamma;                  // Controls steepness of the smooth changepoint transition

  // Seasonality components:
  vector[num_harmonics] A_sin;          // Fourier coefficients for sine terms
  vector[num_harmonics] B_cos;          // Fourier coefficients for cosine terms

  // Likelihood dispersion:
  real<lower=0> phi;                    // Precision for the Beta likelihood
}

transformed parameters {
  vector[N] trend;       // Trend component (linear + quadratic + changepoints)
  vector[N] seasonal;    // Seasonal component computed from the period
  vector[N] mu;          // Total linear predictor
  for (i in 1:N) {
    real cp_effect = 0;
    // Smooth changepoint effects using a logistic function:
    for (j in 1:num_changepoints) {
      cp_effect += delta[j] * inv_logit(gamma * (t[i] - s[j]));
    }
    trend[i] = k * t[i] + m + q * square(t[i]) + cp_effect;

    // Compute seasonality using the remainder of t modulo period:
    real t_mod = fmod(t[i], period);
    real seas = 0;
    for (r in 1:num_harmonics) {
      seas += A_sin[r] * sin(2 * pi() * r * t_mod / period)
            + B_cos[r] * cos(2 * pi() * r * t_mod / period);
    }
    seasonal[i] = seas;

    // Combine trend and seasonality:
    mu[i] = trend[i] + seasonal[i];
  }
}

model {
  // Tighter (but not too strict) priors on trend components:
  k ~ normal(0, 0.5);
  m ~ normal(0, 1);
  q ~ normal(0, 0.05);                   // Expect a very small quadratic effect
  delta ~ double_exponential(0, delta_scale);
  gamma ~ gamma(2, 2);                   // Mean ~1

  // Priors for seasonality parameters:
  A_sin ~ normal(0, 1);
  B_cos ~ normal(0, 1);

  // Prior for the Beta dispersion:
  phi ~ gamma(2, 0.1);

  // Likelihood:
  for (i in 1:N) {
    // Optionally saturate mu to avoid extreme values:
    real mu_sat = fmin(mu[i], 4);
    real p_i = inv_logit(mu_sat);
    y[i] ~ beta(p_i * phi, (1 - p_i) * phi);
  }
}
