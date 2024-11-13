//
// Gaussian prior on regression coeffs (logistic regression)
//

data {

  // number of unpenalized columns in model matrix
  int U;

  // number of observations
  int N;

  // prior standard deviation for the unpenalised variables
  real <lower=0> scale_u;

  // design matrix
  matrix[N, U] X;

  // binary response variable
  array[N] int<lower=0, upper=1> y;
}

parameters {

  // unpenalized regression parameters
  vector[U] beta_u;
  vector[U] beta_u_prior;
}

model {

  // unpenalized coefficients including intercept
  beta_u ~ cauchy(0, scale_u);
  beta_u_prior ~ cauchy(0, scale_u);

  // likelihood
  y ~ bernoulli_logit_glm(X, 0, beta_u);
}

generated quantities {
  // prior predictive
  array[N] int<lower=0, upper=1> y_prior_pred = bernoulli_logit_glm_rng(X, 0*to_vector(y), beta_u_prior);

  // posterior predictive
  array[N] int<lower=0, upper=1> y_post_pred = bernoulli_logit_glm_rng(X, 0*to_vector(y), beta_u);
}
