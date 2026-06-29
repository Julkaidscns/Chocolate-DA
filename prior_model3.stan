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
}
generated quantities {
  real mu_global = normal_rng(3, 0.5);
  real<lower=1e-6> sigma_origin   = abs(normal_rng(0, 0.3));
  real<lower=1e-6> sigma_producer = abs(normal_rng(0, 0.3));
  real beta_cocoa1      = normal_rng(0, 1);
  real beta_cocoa2      = normal_rng(0, 1);
  real beta_ingredients = normal_rng(0, 0.5);
  real beta_vanilla     = normal_rng(0, 0.5);
  real beta_year        = normal_rng(0, 0.3);
  real<lower=1e-6> sigma = abs(normal_rng(0, 0.5));

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

  for (i in 1:N) {
    real eta = mu_global + a[country[i]] + b[location[i]]
             + beta_cocoa1 * cocoa_pct_std[i]
             + beta_cocoa2 * square(cocoa_pct_std[i])
             + beta_ingredients * n_ingredients[i]
             + beta_vanilla * has_vanilla[i]
             + beta_year * year_std[i];
    rating_pred[i] = normal_rng(eta, sigma);
  }
}