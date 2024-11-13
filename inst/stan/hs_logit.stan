//
// Hierarchical shrinkage prior on regression coeffs (logistic regression)
//
// This implements the regularized horseshoe prior according to the simple
// parametrization presented in:
//
// Piironen, Vehtari (2017), Sparsity information and regularization in the
// horseshoe and other shrinkage priors, Electronic Journal of Statistics

functions {
#include /chunks/hs.fun
}

data {

  // number of columns in model matrix
  int P;

  // number of unpenalized columns in model matrix
  int U;

  // number of observations
  int N;

  // design matrix
  matrix[N, P] X;

  // binary response variable
  array[N] int<lower=0, upper=1> y;

  // prior standard deviation for the unpenalised variables
  real<lower=0> scale_u;

  // whether the regularized horseshoe should be used
  int<lower=0, upper=1> regularized;

  // degrees of freedom for the half-t priors on lambda
  real<lower=1> nu;

  // scale for the half-t prior on tau
  real<lower=0> global_scale;

  // degrees of freedom for the half-t prior on tau
  real<lower=1> global_df;

  // slab scale for the regularized horseshoe
  real<lower=0> slab_scale;

  // slab degrees of freedom for the regularized horseshoe
  real<lower=0> slab_df;

}

parameters {

  // unpenalized regression parameters
  vector[U] beta_u;
  vector[U] beta_u_prior;

  // global shrinkage parameter
  real<lower=0> tau;
  real<lower=0> tau_prior;

  // local shrinkage parameter
  vector<lower=0>[P-U] lambda;
  vector<lower=0>[P-U] lambda_prior;

  // auxiliary variables
  vector[P-U] z;
  vector[P-U] z_prior;
  real<lower=0> c2;
  real<lower=0> c2_prior;
}

transformed parameters {

  // penalized regression parameters
  vector[P-U] beta_p;
  vector[P-U] beta_p_prior;

  if (regularized) {
    beta_p = reg_hs(z, lambda, tau, slab_scale^2 * c2);
    beta_p_prior = reg_hs(z_prior, lambda_prior, tau_prior, slab_scale^2 * c2_prior);
  } else {
    beta_p = hs(z, lambda, tau);
    beta_p_prior = hs(z_prior, lambda_prior, tau_prior);
  }
}

model {

  // regression coefficients
  vector[P] beta = append_row(beta_u, beta_p);
  vector[P] beta_prior = append_row(beta_u_prior, beta_p_prior);

  // half t-priors for lambdas and tau
  z ~ std_normal();
  z_prior ~ std_normal();
  lambda ~ student_t(nu, 0, 1);
  lambda_prior ~ student_t(nu, 0, 1);
  tau ~ student_t(global_df, 0, global_scale);
  tau_prior ~ student_t(global_df, 0, global_scale);

  // inverse-gamma prior for c^2
  c2 ~ inv_gamma(0.5 * slab_df, 0.5 * slab_df);
  c2_prior ~ inv_gamma(0.5 * slab_df, 0.5 * slab_df);

  // unpenalized coefficients including intercept
  beta_u ~ cauchy(0, scale_u);
  beta_u_prior ~ cauchy(0, scale_u);

  // likelihood
  y ~ bernoulli_logit_glm(X, 0, beta);
}

generated quantities {
  // prior predictive
  array[N] int<lower=0, upper=1> y_prior_pred = bernoulli_logit_glm_rng(X, 0*to_vector(y), beta_u_prior);

  // posterior predictive
  array[N] int<lower=0, upper=1> y_post_pred = bernoulli_logit_glm_rng(X, 0*to_vector(y), beta_u);
}
