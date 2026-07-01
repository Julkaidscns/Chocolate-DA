// model_2.stan
data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1> K;
  int<lower=1> C;

  array[N] int<lower=1, upper=J> origin;
  array[N] int<lower=1, upper=K> producer;
  array[N] int<lower=1, upper=C> rating;
  array[J] int<lower=1, upper=K> origin_location;

  vector[N] cocoa_logit;     // logit(cocoa%), normalized
  vector[N] n_ingredients;   // normalized ingredient count
}

parameters {
  ordered[C - 1] c;
  real mu_global;

  real<lower=0> sigma_producer;
  vector[K] gamma_raw;

  real<lower=0> sigma_lambda;
  vector[K] lambda_raw;

  real<lower=0> sigma_origin;
  vector[J] alpha_raw;

  real beta_cocoa;
  real beta_ning;
}

transformed parameters {
  vector[K] gamma = gamma_raw * sigma_producer;
  vector[K] lambda = lambda_raw * sigma_lambda;

  vector[J] alpha;
  for (j in 1:J)
    alpha[j] = lambda[origin_location[j]] + alpha_raw[j] * sigma_origin;
}

model {
  // Group-level hyperpriors — same as model 1
  mu_global      ~ normal(0, 1);
  sigma_producer ~ normal(0, 1);
  sigma_lambda   ~ normal(0, 1);
  sigma_origin   ~ normal(0, 1);

  gamma_raw  ~ std_normal();
  lambda_raw ~ std_normal();
  alpha_raw  ~ std_normal();

  c ~ normal(0, 2.5);

  // Priors for bar-level predictors
  beta_cocoa ~ normal(0, 1);
  beta_ning  ~ normal(0, 1);

  // Likelihood
  vector[N] phi;
  for (i in 1:N)
    phi[i] = mu_global + alpha[origin[i]] + gamma[producer[i]]
             + beta_cocoa * cocoa_logit[i]
             + beta_ning  * n_ingredients[i];

  rating ~ ordered_logistic(phi, c);
}

generated quantities {
  array[N] int rating_pred;
  vector[N] log_lik;
  for (i in 1:N) {
    real phi_i = mu_global + alpha[origin[i]] + gamma[producer[i]]
                 + beta_cocoa * cocoa_logit[i]
                 + beta_ning  * n_ingredients[i];
    rating_pred[i] = ordered_logistic_rng(phi_i, c);
    log_lik[i]     = ordered_logistic_lpmf(rating[i] | phi_i, c);
  }
}