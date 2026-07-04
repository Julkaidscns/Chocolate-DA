data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1> K;
  int<lower=1> C;
  array[N] int<lower=1, upper=J> origin;
  array[N] int<lower=1, upper=K> producer;
  vector[N] cocoa_logit;
  array[N] int<lower=0, upper=1> has_vanilla;
}

generated quantities {
  vector[C - 1] c;
  {
    vector[C - 1] c_raw;
    for (k in 1:(C - 1)) c_raw[k] = normal_rng(0, 2.5);
    c = sort_asc(c_raw);
  }

  real mu_global      = normal_rng(0, 1);
  real sigma_origin   = abs(normal_rng(0, 1));
  real sigma_producer = abs(normal_rng(0, 1));

  real mu_beta_cocoa    = normal_rng(0, 1);
  real sigma_beta_cocoa = abs(normal_rng(0, 0.5));
  real beta_vanilla     = normal_rng(0, 1);

  vector[J] alpha;
  for (j in 1:J)
    alpha[j] = normal_rng(0, sigma_origin);

  vector[K] gamma;
  vector[K] beta_cocoa;
  for (k in 1:K) {
    gamma[k]      = normal_rng(0, sigma_producer);
    beta_cocoa[k] = normal_rng(mu_beta_cocoa, sigma_beta_cocoa);
  }

  array[N] int rating_pred;
  for (i in 1:N) {
    real phi_i = mu_global
                 + alpha[origin[i]]
                 + gamma[producer[i]]
                 + beta_cocoa[producer[i]] * cocoa_logit[i]
                 + beta_vanilla * has_vanilla[i];
    rating_pred[i] = ordered_logistic_rng(phi_i, c);
  }
}
