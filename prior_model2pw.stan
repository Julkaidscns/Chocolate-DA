// prior_model2.stan
data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1> K;
  int<lower=1> C;
  array[N] int<lower=1, upper=J> origin;
  array[N] int<lower=1, upper=K> producer;
  array[J] int<lower=1, upper=K> origin_location;
  vector[N] cocoa_logit;
  vector[N] n_ingredients;
}

generated quantities {
  vector[C - 1] c;
  {
    vector[C - 1] c_raw;
    for (k in 1:(C - 1))
      c_raw[k] = normal_rng(0, 2.5);
    c = sort_asc(c_raw);
  }

  real mu_global      = normal_rng(0, 1);
  real sigma_origin   = abs(normal_rng(0, 1));
  real sigma_producer = abs(normal_rng(0, 1));
  real sigma_lambda   = abs(normal_rng(0, 1));
  real beta_cocoa     = normal_rng(0, 1);
  real beta_ning      = normal_rng(0, 1);

  vector[K] lambda_raw;
  for (k in 1:K) lambda_raw[k] = normal_rng(0, 1);
  vector[K] lambda = lambda_raw * sigma_lambda;

  vector[J] alpha;
  for (j in 1:J)
    alpha[j] = normal_rng(lambda[origin_location[j]], sigma_origin);

  vector[K] gamma;
  for (k in 1:K) gamma[k] = normal_rng(0, sigma_producer);

  array[N] int rating_pred;
  for (i in 1:N) {
    real phi_i = mu_global + alpha[origin[i]] + gamma[producer[i]]
                 + beta_cocoa * cocoa_logit[i]
                 + beta_ning  * n_ingredients[i];
    rating_pred[i] = ordered_logistic_rng(phi_i, c);
  }
}