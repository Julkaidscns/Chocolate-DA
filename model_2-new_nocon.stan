data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1> K;
  int<lower=1> C;

  array[N] int<lower=1, upper=J> origin;
  array[N] int<lower=1, upper=K> producer;
  array[N] int<lower=1, upper=C> rating;

  vector[N] cocoa_logit;
  array[N] int<lower=0, upper=1> has_vanilla;
}

parameters {
  ordered[C - 1] c;
  real mu_global;

  real<lower=0> sigma_origin;
  real<lower=0> sigma_producer;

  vector[J] alpha_raw;
  vector[K] gamma_raw;

  real beta_cocoa;
  real beta_vanilla;
}

transformed parameters {
  vector[J] alpha = alpha_raw * sigma_origin;
  vector[K] gamma = gamma_raw * sigma_producer;
}

model {
  mu_global      ~ normal(0, 1);
  sigma_origin   ~ normal(0, 1);
  sigma_producer ~ normal(0, 1);

  alpha_raw ~ std_normal();
  gamma_raw ~ std_normal();

  c ~ normal(0, 2.5);

  beta_cocoa   ~ normal(0, 1);
  beta_vanilla ~ normal(0, 1);

  vector[N] phi;
  for (i in 1:N)
    phi[i] = mu_global
             + alpha[origin[i]]
             + gamma[producer[i]]
             + beta_cocoa * cocoa_logit[i]
             + beta_vanilla * has_vanilla[i];

  rating ~ ordered_logistic(phi, c);
}

generated quantities {
  array[N] int rating_pred;
  vector[N] log_lik;
  for (i in 1:N) {
    real phi_i = mu_global
                 + alpha[origin[i]]
                 + gamma[producer[i]]
                 + beta_cocoa * cocoa_logit[i]
                 + beta_vanilla * has_vanilla[i];
    rating_pred[i] = ordered_logistic_rng(phi_i, c);
    log_lik[i]     = ordered_logistic_lpmf(rating[i] | phi_i, c);
  }
}