// model_producer.stan
data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> C;
  array[N] int<lower=1, upper=K> producer;
  array[N] int<lower=1, upper=C> rating;
}

parameters {
  ordered[C - 1] c;
  real mu_global;
  real<lower=0> sigma_producer;
  vector[K] gamma_raw;
}

transformed parameters {
  vector[K] gamma = gamma_raw * sigma_producer;
}

model {
  mu_global      ~ normal(0, 1);
  sigma_producer ~ normal(0, 1);
  gamma_raw      ~ std_normal();
  c              ~ normal(0, 2.5);

  vector[N] phi;
  for (i in 1:N)
    phi[i] = mu_global + gamma[producer[i]];
  rating ~ ordered_logistic(phi, c);
}

generated quantities {
  array[N] int rating_pred;
  vector[N] log_lik;
  for (i in 1:N) {
    real phi_i = mu_global + gamma[producer[i]];
    rating_pred[i] = ordered_logistic_rng(phi_i, c);
    log_lik[i]     = ordered_logistic_lpmf(rating[i] | phi_i, c);
  }
}