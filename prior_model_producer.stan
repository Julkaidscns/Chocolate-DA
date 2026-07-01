// prior_model_producer.stan
data {
  int<lower=1> N;
  int<lower=1> K;
  int<lower=1> C;
  array[N] int<lower=1, upper=K> producer;
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
  real sigma_producer = abs(normal_rng(0, 1));

  vector[K] gamma_raw;
  for (k in 1:K) gamma_raw[k] = normal_rng(0, 1);
  vector[K] gamma = gamma_raw * sigma_producer;

  array[N] int rating_pred;
  for (i in 1:N) {
    real phi_i = mu_global + gamma[producer[i]];
    rating_pred[i] = ordered_logistic_rng(phi_i, c);
  }
}