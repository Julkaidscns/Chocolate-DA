data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1> K;
  array[N] int<lower=1, upper=J> country;
  array[N] int<lower=1, upper=K> location;
}
generated quantities {
  real mu_global = normal_rng(3, 0.5);
  real<lower=1e-6> sigma_origin   = abs(normal_rng(0, 0.3));
  real<lower=1e-6> sigma_producer = abs(normal_rng(0, 0.3));
  real<lower=1e-6> sigma          = exponential_rng(2);

  vector[J] a_raw;
  vector[K] b_raw;
  vector[J] a;
  vector[K] b;
  array[N] real rating_pred;

  for (j in 1:J) {
    a_raw[j] = normal_rng(0, 1);
    a[j] = a_raw[j] * sigma_origin;
  }
  for (k in 1:K) {
    b_raw[k] = normal_rng(0, 1);
    b[k] = b_raw[k] * sigma_producer;
  }

  for (i in 1:N)
    rating_pred[i] = normal_rng(mu_global + a[country[i]] + b[location[i]], sigma);
}