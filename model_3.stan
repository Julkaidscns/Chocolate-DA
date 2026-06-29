data {
  int<lower=0> N;
  int<lower=1> J;
  int<lower=1> K;
  array[N] int<lower=1, upper=J> country;
  array[N] int<lower=1, upper=K> location;
  vector[N] cocoa_pct_std;
  vector[N] n_ingredients;
  vector[N] has_vanilla;
  vector[N] year_std;
  array[N] real<lower=1, upper=5> rating;
}
parameters {
  real mu_global;
  vector[J] a_raw;
  vector[K] b_raw;
  real<lower=1e-6> sigma_origin;
  real<lower=1e-6> sigma_producer;
  real beta_cocoa1;
  real beta_cocoa2;
  real beta_ingredients;
  real beta_vanilla;
  real beta_year;
  real<lower=1e-6> sigma;
}
transformed parameters {
  vector[J] a = a_raw * sigma_origin;
  vector[K] b = b_raw * sigma_producer;
}
model {
  mu_global        ~ normal(3, 0.5);
  sigma_origin     ~ normal(0, 0.3);
  sigma_producer   ~ normal(0, 0.3);
  beta_cocoa1      ~ normal(0, 1);
  beta_cocoa2      ~ normal(0, 1);
  beta_ingredients ~ normal(0, 0.5);
  beta_vanilla     ~ normal(0, 0.5);
  beta_year        ~ normal(0, 0.3);
  sigma            ~ normal(0, 0.5);

  a_raw ~ std_normal();
  b_raw ~ std_normal();

  for (i in 1:N) {
    real eta = mu_global + a[country[i]] + b[location[i]]
             + beta_cocoa1 * cocoa_pct_std[i]
             + beta_cocoa2 * square(cocoa_pct_std[i])
             + beta_ingredients * n_ingredients[i]
             + beta_vanilla * has_vanilla[i]
             + beta_year * year_std[i];
    rating[i] ~ normal(eta, sigma);
  }
}
generated quantities {
  array[N] real rating_pred;
  array[N] real log_lik;
  for (i in 1:N) {
    real eta = mu_global + a[country[i]] + b[location[i]]
             + beta_cocoa1 * cocoa_pct_std[i]
             + beta_cocoa2 * square(cocoa_pct_std[i])
             + beta_ingredients * n_ingredients[i]
             + beta_vanilla * has_vanilla[i]
             + beta_year * year_std[i];
    rating_pred[i] = normal_rng(eta, sigma);
    log_lik[i]     = normal_lpdf(rating[i] | eta, sigma);
  }
}