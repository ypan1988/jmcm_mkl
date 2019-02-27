#include <iostream>

#include "slab/matrix.h"
#include "slab/stats.h"
using namespace slab;

#include "jmcm_fit.h"
#include "mcd.h"

int main() {

  // Load the cattle data and covariate matrices
  vec m = zeros<vec>(30);
  m = 11;

  vec Y;
  mat X,Z,W;

  Y.load("Y.smat");
  X.load("X.smat");
  Z.load("Z.smat");
  W.load("W.smat");

  // Initialize beta, lambda, gamma
  lmfit xyfit(X, Y);
  vec beta0 = xyfit.coefficients();

  vec res = Y - matmul(X, beta0);
  lmfit zrfit(Z, log(pow(res, 2)));
  vec lambda0 = zrfit.coefficients();
  vec gamma0 = zeros<vec>(W.n_cols());

  beta0.print("bta0 = ");
  lambda0.print("lmd0 = ");
  gamma0.print("gma = ");

  vec theta0 = join_vecs(beta0, lambda0, gamma0);

  // Model cattle data using jmcm-mcd
  JmcmFit<jmcm::MCD> mcdfit(m, Y, X, Z, W, theta0, Y, true);
  mcdfit.Optimize();

  return 0;
}