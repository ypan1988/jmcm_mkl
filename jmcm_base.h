//  jmcm_base.h: base class for three joint mean-covariance models (MCD/ACD/HPC)
//  This file is part of jmcm_mkl.
//
//  Copyright (C) 2015-2018 Yi Pan <ypan1988@gmail.com>
//
//  This program is free software; you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation; either version 2 of the License, or
//  (at your option) any later version.
//
//  This program is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//
//  A copy of the GNU General Public License is available at
//  https://www.R-project.org/Licenses/

#ifndef JMCM_MKL_JMCM_BASE_H_
#define JMCM_MKL_JMCM_BASE_H_

#include "slab/matrix.h"

namespace jmcm {

class JmcmBase {
 public:
  JmcmBase() = delete;
  JmcmBase(const JmcmBase&) = delete;
  virtual ~JmcmBase() = default;

  JmcmBase(const slab::vec& m, const slab::vec& Y, const slab::mat& X,
           const slab::mat& Z, const slab::mat& W, const slab::uword method_id);

  slab::uword get_method_id() const { return method_id_; }

  slab::vec get_m() const { return m_; }
  slab::vec get_Y() const { return Y_; }
  slab::mat get_X() const { return X_; }
  slab::mat get_Z() const { return Z_; }
  slab::mat get_W() const { return W_; }

  slab::uword get_m(slab::uword i) const;
  slab::vec get_Y(slab::uword i) const;
  slab::mat get_X(slab::uword i) const;
  slab::mat get_Z(slab::uword i) const;
  slab::mat get_W(slab::uword i) const;

  slab::vec get_theta() const { return theta_; }
  slab::vec get_beta() const { return beta_; }
  slab::vec get_lambda() const { return lambda_; }
  slab::vec get_gamma() const { return gamma_; }
  slab::uword get_free_param() const { return free_param_; }

  void set_theta(const slab::vec& x);
  void set_beta(const slab::vec& x);
  void set_lambda(const slab::vec& x);
  void set_gamma(const slab::vec& x);
  void set_lmdgma(const slab::vec& x);
  void set_free_param(slab::uword n) { free_param_ = n; }

  // virtual void UpdateBeta() {}
  void UpdateBeta();
  virtual void UpdateLambda(const slab::vec&) {}
  virtual void UpdateGamma() {}
  virtual void UpdateLambdaGamma(const slab::vec&) {}

  virtual slab::mat get_D(slab::uword i) const = 0;
  virtual slab::mat get_T(slab::uword i) const = 0;
  virtual slab::vec get_mu(slab::uword i) const = 0;
  virtual slab::mat get_Sigma(slab::uword i) const = 0;
  virtual slab::mat get_Sigma_inv(slab::uword i) const = 0;
  virtual slab::vec get_Resid(slab::uword i) const = 0;

  virtual double operator()(const slab::vec& x) = 0;
  virtual void Gradient(const slab::vec& x, slab::vec& grad) = 0;
  virtual void UpdateJmcm(const slab::vec& x) = 0;

  void set_mean(const slab::vec& mean) {
    cov_only_ = true;
    mean_ = mean;
  }

 protected:
  slab::vec m_, Y_;
  slab::mat X_, Z_, W_;
  slab::uword method_id_;

  slab::vec theta_, beta_, lambda_, gamma_, lmdgma_;
  slab::vec Xbta_, Zlmd_, Wgma_, Resid_;

  // free_param_ == 0  ---- beta + lambda + gamma
  // free_param_ == 1  ---- beta
  // free_param_ == 2  ---- lambda
  // free_param_ == 3  ---- gamma
  // free_param_ == 23 -----lambda + gamma
  slab::uword free_param_;

  bool cov_only_;
  slab::vec mean_;
};

inline JmcmBase::JmcmBase(const slab::vec& m, const slab::vec& Y,
                          const slab::mat& X, const slab::mat& Z,
                          const slab::mat& W, const slab::uword method_id)
    : m_(m),
      Y_(Y),
      X_(X),
      Z_(Z),
      W_(W),
      method_id_(method_id),
      free_param_(0),
      cov_only_(false),
      mean_(Y) {
  slab::uword N = Y_.n_rows();
  slab::uword n_bta = X_.n_cols();
  slab::uword n_lmd = Z_.n_cols();
  slab::uword n_gma = W_.n_cols();

  theta_ = slab::zeros<slab::vec>(n_bta + n_lmd + n_gma);
  beta_ = slab::zeros<slab::vec>(n_bta);
  lambda_ = slab::zeros<slab::vec>(n_lmd);
  gamma_ = slab::zeros<slab::vec>(n_gma);
  lmdgma_ = slab::zeros<slab::vec>(n_lmd + n_gma);

  Xbta_ = slab::zeros<slab::vec>(N);
  Zlmd_ = slab::zeros<slab::vec>(N);
  Wgma_ = slab::zeros<slab::vec>(W_.n_rows());
  Resid_ = slab::zeros<slab::vec>(N);
}

inline slab::uword JmcmBase::get_m(slab::uword i) const { return m_(i); }

inline slab::vec JmcmBase::get_Y(slab::uword i) const {
  slab::vec Yi;
  if (i == 0)
    Yi = Y_.subvec(0, m_(0) - 1);
  else {
    int index = slab::sum(m_.subvec(0, i - 1));
    Yi = Y_.subvec(index, index + m_(i) - 1);
  }
  return Yi;
}

inline slab::mat JmcmBase::get_X(slab::uword i) const {
  slab::mat Xi;
  if (i == 0)
    Xi = X_.rows(0, m_(0) - 1);
  else {
    int index = slab::sum(m_.subvec(0, i - 1));
    Xi = X_.rows(index, index + m_(i) - 1);
  }
  return Xi;
}

inline slab::mat JmcmBase::get_Z(slab::uword i) const {
  slab::mat Zi;
  if (i == 0)
    Zi = Z_.rows(0, m_(0) - 1);
  else {
    int index = slab::sum(m_.subvec(0, i - 1));
    Zi = Z_.rows(index, index + m_(i) - 1);
  }
  return Zi;
}

inline slab::mat JmcmBase::get_W(slab::uword i) const {
  slab::mat Wi;
  if (m_(i) != 1) {
    if (i == 0) {
      int first_index = 0;
      int last_index = m_(0) * (m_(0) - 1) / 2 - 1;
      Wi = W_.rows(first_index, last_index);
    } else {
      int first_index = 0;
      for (slab::uword idx = 0; idx != i; ++idx) {
        first_index += m_(idx) * (m_(idx) - 1) / 2;
      }
      int last_index = first_index + m_(i) * (m_(i) - 1) / 2 - 1;

      Wi = W_.rows(first_index, last_index);
    }
  }

  return Wi;
}

inline void JmcmBase::set_theta(const slab::vec& x) {
  slab::uword fp2 = free_param_;
  free_param_ = 0;
  UpdateJmcm(x);
  free_param_ = fp2;
}

inline void JmcmBase::set_beta(const slab::vec& x) {
  slab::uword fp2 = free_param_;
  free_param_ = 1;
  UpdateJmcm(x);
  free_param_ = fp2;
}

inline void JmcmBase::set_lambda(const slab::vec& x) {
  slab::uword fp2 = free_param_;
  free_param_ = 2;
  UpdateJmcm(x);
  free_param_ = fp2;
}

inline void JmcmBase::set_gamma(const slab::vec& x) {
  slab::uword fp2 = free_param_;
  free_param_ = 3;
  UpdateJmcm(x);
  free_param_ = fp2;
}

inline void JmcmBase::set_lmdgma(const slab::vec& x) {
  slab::uword fp2 = free_param_;
  free_param_ = 23;
  UpdateJmcm(x);
  free_param_ = fp2;
}

inline void JmcmBase::UpdateBeta() {
  slab::uword i, n_sub = m_.size(), n_bta = X_.n_cols();
  slab::mat XSX = slab::zeros<slab::mat>(n_bta, n_bta);
  slab::vec XSY = slab::zeros<slab::vec>(n_bta);

  for (i = 0; i < n_sub; ++i) {
    slab::mat Xi = get_X(i);
    slab::vec Yi = get_Y(i);
    slab::mat Sigmai_inv = get_Sigma_inv(i);

    XSX += slab::matmul_n(Xi.t(), Sigmai_inv, Xi) ;
    XSY += slab::matmul_n(Xi.t(), Sigmai_inv, Yi);
  }

  slab::vec beta = slab::matmul(XSX.i(), XSY);

  slab::uword fp2 = free_param_;
  free_param_ = 1;
  UpdateJmcm(beta);  // template method
  free_param_ = fp2;
}

}  // namespace jmcm

#endif