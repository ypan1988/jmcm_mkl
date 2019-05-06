//  mcd.h: joint mean-covariance models based on modified Cholesky
//         decomposition (MCD) of the covariance matrix
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

#ifndef JMCM_MCD_H_
#define JMCM_MCD_H_

#include <cstddef>

#include <algorithm>  // std::equal

#include "jmcm_base.h"

namespace jmcm {

class MCD : public JmcmBase {
 public:
  MCD() = delete;
  MCD(const MCD&) = delete;
  ~MCD() = default;

  MCD(const slab::vec& m, const slab::vec& Y, const slab::mat& X,
      const slab::mat& Z, const slab::mat& W);

  void UpdateLambda(const slab::vec& x) override;
  void UpdateGamma() override;

  slab::mat get_D(slab::uword i) const override;
  slab::mat get_T(slab::uword i) const override;
  slab::vec get_mu(slab::uword i) const override;
  slab::mat get_Sigma(slab::uword i) const override;
  slab::mat get_Sigma_inv(slab::uword i) const override;
  slab::vec get_Resid(slab::uword i) const override;

  void get_D(slab::uword i, slab::mat& Di) const;
  void get_T(slab::uword i, slab::mat& Ti) const;
  void get_Sigma_inv(slab::uword i, slab::mat& Sigmai_inv) const;
  void get_Resid(slab::uword i, slab::vec& ri) const;

  double operator()(const slab::vec& x) override;
  void Gradient(const slab::vec& x, slab::vec& grad) override;
  void Grad1(slab::vec& grad1);
  void Grad2(slab::vec& grad2);
  void Grad3(slab::vec& grad3);

  void UpdateJmcm(const slab::vec& x) override;
  void UpdateParam(const slab::vec& x);
  void UpdateModel();

 private:
  slab::mat G_;
  slab::vec TResid_;

  slab::mat get_G(slab::uword i) const;
  slab::vec get_TResid(slab::uword i) const;
  void get_G(slab::uword i, slab::mat& Gi) const;
  void get_TResid(slab::uword i, slab::vec& Tiri) const;
  void UpdateG();
  void UpdateTResid();

};  // class MCD

inline MCD::MCD(const slab::vec& m, const slab::vec& Y, const slab::mat& X,
                const slab::mat& Z, const slab::mat& W)
    : JmcmBase(m, Y, X, Z, W, 0) {
  slab::uword N = Y_.n_rows();
  slab::uword n_gma = W_.n_cols();

  G_ = slab::zeros<slab::mat>(N, n_gma);
  TResid_ = slab::zeros<slab::vec>(N);
}

inline void MCD::UpdateLambda(const slab::vec& x) { set_lambda(x); }

inline void MCD::UpdateGamma() {
  slab::uword i, n_sub = m_.size(), n_gma = W_.n_cols();
  slab::mat GDG = slab::zeros<slab::mat>(n_gma, n_gma);
  slab::vec GDr = slab::zeros<slab::vec>(n_gma);

  for (i = 0; i < n_sub; ++i) {
    slab::mat Gi;
    get_G(i, Gi);
    slab::vec ri;
    get_Resid(i, ri);
    slab::mat Di;
    get_D(i, Di);
    slab::mat Di_inv = slab::diagmat(slab::pow(Di.diag(), -1));

    GDG += slab::matmul_n(Gi.t(), Di_inv, Gi);
    GDr += slab::matmul_n(Gi.t(), Di_inv, ri);
  }

  slab::vec gamma = slab::matmul(GDG.i(), GDr);

  set_gamma(gamma);
}

inline slab::mat MCD::get_D(slab::uword i) const {
  slab::mat Di = slab::eye<slab::mat>(m_(i), m_(i));
  if (i == 0)
    Di = slab::diagmat(slab::exp(Zlmd_.subvec(0, m_(0) - 1)));
  else {
    slab::uword index = slab::sum(m_.subvec(0, i - 1));
    Di = slab::diagmat(slab::exp(Zlmd_.subvec(index, index + m_(i) - 1)));
  }
  return Di;
}

inline void MCD::get_D(slab::uword i, slab::mat& Di) const {
  Di = slab::eye<slab::mat>(m_(i), m_(i));
  if (i == 0)
    Di = slab::diagmat(slab::exp(Zlmd_.subvec(0, m_(0) - 1)));
  else {
    slab::uword index = slab::sum(m_.subvec(0, i - 1));
    Di = slab::diagmat(slab::exp(Zlmd_.subvec(index, index + m_(i) - 1)));
  }
}

inline slab::mat MCD::get_T(slab::uword i) const {
  slab::mat Ti = slab::eye<slab::mat>(m_(i), m_(i));
  if (m_(i) != 1) {
    if (i == 0) {
      slab::uword first_index = 0;
      slab::uword last_index = m_(0) * (m_(0) - 1) / 2 - 1;

      //Ti = pan::ltrimat(m_(0), -Wgma_.subvec(first_index, last_index));

      slab::vec Ti_elem = -Wgma_.subvec(first_index, last_index);
      slab::unit_ltri_mat Ti_tmp(m_(i), Ti_elem.std_vec());
      Ti = Ti_tmp;
    } else {
      slab::uword first_index = 0;
      for (slab::uword idx = 0; idx != i; ++idx) {
        first_index += m_(idx) * (m_(idx) - 1) / 2;
      }
      slab::uword last_index = first_index + m_(i) * (m_(i) - 1) / 2 - 1;

      //Ti = pan::ltrimat(m_(i), -Wgma_.subvec(first_index, last_index));
      slab::vec Ti_elem = -Wgma_.subvec(first_index, last_index);
      slab::unit_ltri_mat Ti_tmp(m_(i), Ti_elem.std_vec());
      Ti = Ti_tmp;

    }
  }
  return Ti;
}

inline void MCD::get_T(slab::uword i, slab::mat& Ti) const {
  Ti = slab::eye<slab::mat>(m_(i), m_(i));
  if (m_(i) != 1) {
    if (i == 0) {
      slab::uword first_index = 0;
      slab::uword last_index = m_(0) * (m_(0) - 1) / 2 - 1;

      //Ti = pan::ltrimat(m_(0), -Wgma_.subvec(first_index, last_index));

      slab::vec Ti_elem = -Wgma_.subvec(first_index, last_index);
      slab::unit_ltri_mat Ti_tmp(m_(i), Ti_elem.std_vec());
      Ti = Ti_tmp;

    } else {
      slab::uword first_index = 0;
      for (slab::uword idx = 0; idx != i; ++idx) {
        first_index += m_(idx) * (m_(idx) - 1) / 2;
      }
      slab::uword last_index = first_index + m_(i) * (m_(i) - 1) / 2 - 1;

      //Ti = pan::ltrimat(m_(i), -Wgma_.subvec(first_index, last_index));

      slab::vec Ti_elem = -Wgma_.subvec(first_index, last_index);
      slab::unit_ltri_mat Ti_tmp(m_(i), Ti_elem.std_vec());
      Ti = Ti_tmp;

    }
  }
}

inline slab::vec MCD::get_mu(slab::uword i) const {
  slab::vec mui;
  if (i == 0)
    mui = Xbta_.subvec(0, m_(0) - 1);
  else {
    slab::uword index = slab::sum(m_.subvec(0, i - 1));
    mui = Xbta_.subvec(index, index + m_(i) - 1);
  }
  return mui;
}

inline slab::mat MCD::get_Sigma(slab::uword i) const {
  slab::mat Ti = get_T(i);
  slab::mat Ti_inv = slab::pinv(Ti);
  slab::mat Di = get_D(i);

  return slab::matmul_n(Ti_inv, Di, Ti_inv.t());
}

inline slab::mat MCD::get_Sigma_inv(slab::uword i) const {
  slab::mat Ti = get_T(i);
  slab::mat Di = get_D(i);
  slab::mat Di_inv = slab::diagmat(slab::pow(Di.diag(), -1));

  return slab::matmul_n(Ti.t(), Di_inv, Ti);
}

inline void MCD::get_Sigma_inv(slab::uword i, slab::mat& Sigmai_inv) const {
  slab::mat Ti;
  get_T(i, Ti);
  slab::mat Di;
  get_D(i, Di);
  slab::mat Di_inv = slab::diagmat(slab::pow(Di.diag(), -1));

  Sigmai_inv = slab::matmul_n(Ti.t(), Di_inv, Ti);
}

inline slab::vec MCD::get_Resid(slab::uword i) const {
  slab::vec ri;
  if (i == 0)
    ri = Resid_.subvec(0, m_(0) - 1);
  else {
    slab::uword index = slab::sum(m_.subvec(0, i - 1));
    ri = Resid_.subvec(index, index + m_(i) - 1);
  }
  return ri;
}

inline void MCD::get_Resid(slab::uword i, slab::vec& ri) const {
  if (i == 0)
    ri = Resid_.subvec(0, m_(0) - 1);
  else {
    slab::uword index = slab::sum(m_.subvec(0, i - 1));
    ri = Resid_.subvec(index, index + m_(i) - 1);
  }
}

inline double MCD::operator()(const slab::vec& x) {
  UpdateJmcm(x);

  slab::uword i, n_sub = m_.size();
  double result = 0.0;
  for (i = 0; i < n_sub; ++i) {
    slab::vec ri;
    get_Resid(i, ri);
    slab::mat Sigmai_inv;
    get_Sigma_inv(i, Sigmai_inv);
    result += slab::as_scalar(slab::matmul_n(ri.t(), Sigmai_inv, ri));
  }

  result += slab::sum(slab::log(slab::exp(Zlmd_)));
  return result;
}

inline void MCD::Gradient(const slab::vec& x, slab::vec& grad) {
  UpdateJmcm(x);

  slab::uword n_bta = X_.n_cols(), n_lmd = Z_.n_cols(), n_gma = W_.n_cols();

  slab::vec grad1, grad2, grad3;

  switch (free_param_) {
    case 0:

      Grad1(grad1);
      Grad2(grad2);
      Grad3(grad3);

      grad = slab::zeros<slab::vec>(theta_.n_rows());
      grad.subvec(0, n_bta - 1) = grad1;
      grad.subvec(n_bta, n_bta + n_lmd - 1) = grad2;
      grad.subvec(n_bta + n_lmd, n_bta + n_lmd + n_gma - 1) = grad3;

      break;

    case 1:
      Grad1(grad);
      break;

    case 2:
      Grad2(grad);
      break;

    case 3:
      Grad3(grad);
      break;

    default:
    apue::err_msg("Wrong value for free_param_");
  }
}

inline void MCD::Grad1(slab::vec& grad1) {
  slab::uword i, n_sub = m_.size(), n_bta = X_.n_cols();
  grad1 = slab::zeros<slab::vec>(n_bta);

  for (i = 0; i < n_sub; ++i) {
    slab::mat Xi = get_X(i);
    slab::vec ri;
    get_Resid(i, ri);
    slab::mat Sigmai_inv;
    get_Sigma_inv(i, Sigmai_inv);
    grad1 += slab::matmul_n(Xi.t(), Sigmai_inv, ri);
  }

  grad1 *= -2;
}

inline void MCD::Grad2(slab::vec& grad2) {
  slab::uword i, n_sub = m_.size(), n_lmd = Z_.n_cols();
  grad2 = slab::zeros<slab::vec>(n_lmd);

  for (i = 0; i < n_sub; ++i) {
    slab::vec one = slab::ones<slab::vec>(m_(i));
    slab::mat Zi = get_Z(i);

    slab::mat Di;
    get_D(i, Di);
    slab::mat Di_inv = slab::diagmat(slab::pow(Di.diag(), -1));

    slab::vec ei = slab::pow(get_TResid(i), 2);

    grad2 += 0.5 * slab::matmul(Zi.t(), (slab::matmul(Di_inv, ei) - one));
  }

  grad2 *= -2;
}

inline void MCD::Grad3(slab::vec& grad3) {
  slab::uword i, n_sub = m_.size(), n_gma = W_.n_cols();
  grad3 = slab::zeros<slab::vec>(n_gma);

  for (i = 0; i < n_sub; ++i) {
    slab::mat Gi;
    get_G(i, Gi);

    slab::mat Di;
    get_D(i, Di);
    slab::mat Di_inv = slab::diagmat(slab::pow(Di.diag(), -1));

    slab::vec Tiri;
    get_TResid(i, Tiri);

    grad3 += slab::matmul_n(Gi.t(), Di_inv, Tiri);
  }

  grad3 *= -2;
}

inline void MCD::UpdateJmcm(const slab::vec& x) {
  slab::uword debug = 0;
  bool update = true;

  switch (free_param_) {
    case 0:
      if (std::equal(x.begin(), x.end(), theta_.begin())) update = false;
      break;

    case 1:
      if (std::equal(x.begin(), x.end(), beta_.begin())) update = false;
      break;

    case 2:
      if (std::equal(x.begin(), x.end(), lambda_.begin())) update = false;
      break;

    case 3:
      if (std::equal(x.begin(), x.end(), gamma_.begin())) update = false;
      break;

    default:
      std::cout << "Wrong value for free_param_" << std::endl;
  }

  if (update) {
    UpdateParam(x);
    UpdateModel();
  } else {
    if (debug) std::cout << "Hey, I did save some time!:)" << std::endl;
  }
}

inline void MCD::UpdateParam(const slab::vec& x) {
  slab::uword n_bta = X_.n_cols();
  slab::uword n_lmd = Z_.n_cols();
  slab::uword n_gma = W_.n_cols();

  switch (free_param_) {
    case 0:
      theta_ = x;
      beta_ = x.rows(0, n_bta - 1);
      lambda_ = x.rows(n_bta, n_bta + n_lmd - 1);
      gamma_ = x.rows(n_bta + n_lmd, n_bta + n_lmd + n_gma - 1);
      break;

    case 1:
      theta_.rows(0, n_bta - 1) = x;
      beta_ = x;
      break;

    case 2:
      theta_.rows(n_bta, n_bta + n_lmd - 1) = x;
      lambda_ = x;
      break;

    case 3:
      theta_.rows(n_bta + n_lmd, n_bta + n_lmd + n_gma - 1) = x;
      gamma_ = x;
      break;

    default:
      std::cout << "Wrong value for free_param_" << std::endl;
  }
}

inline void MCD::UpdateModel() {
  switch (free_param_) {
    case 0:
      if (cov_only_)
        Xbta_ = mean_;
      else
        Xbta_ = slab::matmul(X_, beta_);

      Zlmd_ = slab::matmul(Z_, lambda_);
      Wgma_ = slab::matmul(W_, gamma_);
      Resid_ = Y_ - Xbta_;

      UpdateG();
      UpdateTResid();

      break;

    case 1:
      if (cov_only_)
        Xbta_ = mean_;
      else
        Xbta_ = slab::matmul(X_, beta_);

      Resid_ = Y_ - Xbta_;

      UpdateG();
      UpdateTResid();

      break;

    case 2:
      Zlmd_ = slab::matmul(Z_, lambda_);

      break;

    case 3:
      Wgma_ = slab::matmul(W_, gamma_);

      UpdateTResid();

      break;

    default:
    std::cout << "Wrong value for free_param_" << std::endl;
  }
}

inline slab::mat MCD::get_G(slab::uword i) const {
  slab::mat Gi;
  if (i == 0)
    Gi = G_.rows(0, m_(0) - 1);
  else {
    slab::uword index = slab::sum(m_.subvec(0, i - 1));
    Gi = G_.rows(index, index + m_(i) - 1);
  }
  return Gi;
}

inline void MCD::get_G(slab::uword i, slab::mat& Gi) const {
  if (i == 0)
    Gi = G_.rows(0, m_(0) - 1);
  else {
    slab::uword index = slab::sum(m_.subvec(0, i - 1));
    Gi = G_.rows(index, index + m_(i) - 1);
  }
}

inline slab::vec MCD::get_TResid(slab::uword i) const {
  slab::vec Tiri;
  if (i == 0)
    Tiri = TResid_.subvec(0, m_(0) - 1);
  else {
    slab::uword index = slab::sum(m_.subvec(0, i - 1));
    Tiri = TResid_.subvec(index, index + m_(i) - 1);
  }
  return Tiri;
}

inline void MCD::get_TResid(slab::uword i, slab::vec& Tiri) const {
  if (i == 0)
    Tiri = TResid_.subvec(0, m_(0) - 1);
  else {
    slab::uword index = slab::sum(m_.subvec(0, i - 1));
    Tiri = TResid_.subvec(index, index + m_(i) - 1);
  }
}

inline void MCD::UpdateG() {
  slab::uword i, j, n_sub = m_.size();

  for (i = 0; i < n_sub; ++i) {
    slab::mat Gi = slab::zeros<slab::mat>(m_(i), W_.n_cols());

    slab::mat Wi = get_W(i);
    slab::vec ri;
    get_Resid(i, ri);
    for (j = 1; j != m_(i); ++j) {
      slab::uword index = 0;
      if (j == 1)
        index = 0;
      else {
        for (slab::uword idx = 1; idx < j; ++idx) index += idx;
      }
      //Gi.row(j) = slab::matmul(ri.subvec(0, j - 1).t(), Wi.rows(index, index + j - 1));
      Gi.row(j) = slab::matmul(Wi.rows(index, index + j - 1).t(), ri.subvec(0, j - 1));
    }
    if (i == 0)
      G_.rows(0, m_(0) - 1) = Gi;
    else {
      slab::uword index = slab::sum(m_.subvec(0, i - 1));
      G_.rows(index, index + m_(i) - 1) = Gi;
    }
  }
}

inline void MCD::UpdateTResid() {
  slab::uword i, n_sub = m_.size();

  for (i = 0; i < n_sub; ++i) {
    slab::vec ri;
    get_Resid(i, ri);
    slab::mat Ti;
    get_T(i, Ti);
    slab::mat Tiri = slab::matmul(Ti, ri);
    if (i == 0)
      TResid_.subvec(0, m_(0) - 1) = Tiri;
    else {
      slab::uword index = slab::sum(m_.subvec(0, i - 1));
      TResid_.subvec(index, index + m_(i) - 1) = Tiri;
    }
  }
}

}  // namespace jmcm

#endif  // JMCM_MCD_H_
