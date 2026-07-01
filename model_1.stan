data {
  int<lower=1> N;
  int<lower=1> J;         // number of bean origins
  int<lower=1> K;         // number of producer locations
  int<lower=1> C;         // number of rating categories

  array[N] int<lower=1, upper=J> origin;
  array[N] int<lower=1, upper=K> producer;
  array[N] int<lower=1, upper=C> rating;

  // for each bean origin j, which producer location does it come from
  array[J] int<lower=1, upper=K> origin_location;
}

parameters {
  ordered[C - 1] c;
  real mu_global;

  // direct producer location -> rating effect
  real<lower=0> sigma_producer;
  vector[K] gamma_raw;

  // producer location -> bean origin mean effect (separate from gamma!)
  real<lower=0> sigma_lambda;
  vector[K] lambda_raw;

  // residual bean origin effect after accounting for location
  real<lower=0> sigma_origin;
  vector[J] alpha_raw;
}

transformed parameters {
  // direct effect of producer location on ratings
  vector[K] gamma = gamma_raw * sigma_producer;

  // typical bean quality for origins sourced from each location
  vector[K] lambda = lambda_raw * sigma_lambda;

  // origin effect = location-level mean + residual origin noise
  vector[J] alpha;
  for (j in 1:J)
    alpha[j] = lambda[origin_location[j]] + alpha_raw[j] * sigma_origin;
}

model {
  // hyperpriors
  mu_global      ~ normal(0, 1);
  sigma_producer ~ normal(0, 1);   // HalfNormal due to <lower=0>
  sigma_lambda   ~ normal(0, 1);
  sigma_origin   ~ normal(0, 1);

  // non-centered parameterisation
  gamma_raw  ~ std_normal();
  lambda_raw ~ std_normal();
  alpha_raw  ~ std_normal();

  c ~ normal(0, 2.5);

  // likelihood
  vector[N] phi;
  for (i in 1:N)
    phi[i] = mu_global + alpha[origin[i]] + gamma[producer[i]];

  rating ~ ordered_logistic(phi, c);
}

generated quantities {
  array[N] int rating_pred;
  vector[N] log_lik;
  for (i in 1:N) {
    real phi_i = mu_global + alpha[origin[i]] + gamma[producer[i]];
    rating_pred[i] = ordered_logistic_rng(phi_i, c);
    log_lik[i]     = ordered_logistic_lpmf(rating[i] | phi_i, c);
  }
}