data {
  int<lower=1> N;                   // Number of observations
  int<lower=1> J;                   // Number of origins
  int<lower=1> K;                   // Number of producers
  int<lower=1> C;                   // Number of rating categories (e.g., 17)
  
  array[N] int<lower=1, upper=J> origin;
  array[N] int<lower=1, upper=K> producer;
  array[N] int<lower=1, upper=C> rating;
}

parameters {
  // Cutpoints for the ordered logistic
  ordered[C - 1] c;
  
  // Hierarchical standard deviations
  real<lower=0> sigma_origin;
  real<lower=0> sigma_producer;
  
  // Raw effects (for non-centered parameterization to help the sampler)
  vector[J] alpha_raw;
  vector[K] gamma_raw;
}

transformed parameters {
  // Non-centered parameterization (prevents divergent transitions!)
  vector[J] alpha = alpha_raw * sigma_origin;
  vector[K] gamma = gamma_raw * sigma_producer;
}

model {
  // Priors
  sigma_origin ~ std_normal();
  sigma_producer ~ std_normal();
  sigma_origin   ~ normal(0, 1);   // with <lower=0> constraint this is HalfNormal(1)
  sigma_producer ~ normal(0, 1);
  c ~ normal(0, 2);
  
  // Likelihood
  vector[N] phi;
  for (i in 1:N) {
    phi[i] = alpha[origin[i]] + gamma[producer[i]];
  }
  rating ~ ordered_logistic(phi, c);
}

generated quantities {
  array[N] int rating_pred;
  vector[N] log_lik;
  for (i in 1:N) {
    real phi_pred = alpha[origin[i]] + gamma[producer[i]];
    rating_pred[i] = ordered_logistic_rng(phi_pred, c);
    log_lik[i] = ordered_logistic_lpmf(rating[i] | phi_pred, c);
  }
}