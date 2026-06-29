data {
  int<lower=1> N;
  int<lower=1> J;
  int<lower=1> K;
  array[N] int<lower=1, upper=J> country;
  array[N] int<lower=1, upper=K> location;
  vector[N] rating;
}
parameters {
  real mu_global;
  vector[J] a_raw;
  vector[K] b_raw;
  real<lower=1e-6> sigma_origin;
  real<lower=1e-6> sigma_producer;
  real<lower=1e-6> sigma;
}
transformed parameters {
  vector[J] a = a_raw * sigma_origin;
  vector[K] b = b_raw * sigma_producer;
}
model {
  mu_global      ~ normal(3, 0.5);
  sigma_origin   ~ normal(0, 0.3);
  sigma_producer ~ normal(0, 0.3);
  sigma          ~ exponential(2);

  a_raw ~ std_normal();
  b_raw ~ std_normal();

  for (i in 1:N)
    rating[i] ~ normal(mu_global + a[country[i]] + b[location[i]], sigma);
}
generated quantities {
  array[N] real rating_pred;
  array[N] real log_lik;
  for (i in 1:N) {
    rating_pred[i] = normal_rng(mu_global + a[country[i]] + b[location[i]], sigma);
    log_lik[i] = normal_lpdf(rating[i] | mu_global + a[country[i]] + b[location[i]], sigma);
  }
}