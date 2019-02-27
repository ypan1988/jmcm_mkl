//  jmcm_base.h: model fitting for three joint mean-covariance models
//               (MCD/ACD/HPC)
//  This file is part of jmcm.
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

#ifndef JMCM_MKL_JMCM_FIT_H
#define JMCM_MKL_JMCM_FIT_H

#include <slab/matrix.h>

#include "bfgs.h"

template <typename JMCM>
class JmcmFit {
 public:
  JmcmFit() = delete;
  JmcmFit(const JmcmFit&) = delete;
  ~JmcmFit() = default;

  JmcmFit(const slab::vec& m, const slab::vec& Y, const slab::mat& X,
          const slab::mat& Z, const slab::mat& W, slab::vec start,
          slab::vec mean, bool trace = false, bool profile = true,
          bool errormsg = false, bool covonly = false,
          std::string optim_method = "default")
      : jmcm_(m, Y, X, Z, W),
        start_(start),
        mean_(mean),
        trace_(trace),
        profile_(profile),
        errormsg_(errormsg),
        covonly_(covonly),
        optim_method_(optim_method) {
    method_id_ = jmcm_.get_method_id();
    f_min_ = 0.0;
    n_iters_ = 0;
  }

  slab::vec Optimize();
  double get_f_min() const { return f_min_; }
  slab::uword get_n_iters() const { return n_iters_; }

 private:
  JMCM jmcm_;
  slab::uword method_id_;
  slab::vec start_, mean_;
  bool trace_, profile_, errormsg_, covonly_;
  std::string optim_method_;

  double f_min_;
  slab::uword n_iters_;
};

template <typename JMCM>
slab::vec JmcmFit<JMCM>::Optimize() {
  int n_bta = jmcm_.get_X().n_cols();
  int n_lmd = jmcm_.get_Z().n_cols();
  int n_gma = jmcm_.get_W().n_cols();

  if (covonly_) {
    if ((jmcm_.get_Y().n_rows() != mean_.n_rows()) && errormsg_)
      std::cout << "The size of the responses Y does not match the size of "
          "the given mean"
                  << std::endl;
    jmcm_.set_mean(mean_);
  }

  pan::BFGS<JMCM> bfgs;
  pan::LineSearch<JMCM> linesearch;
  linesearch.set_message(errormsg_);

  //roptim::Roptim<JMCM> optim;

//  if (optim_method_ == "default") {
//  } else if (optim_method_ == "Nelder-Mead" || optim_method_ == "BFGS" ||
//      optim_method_ == "CG" || optim_method_ == "L-BFGS-B") {
//    optim.set_method(optim_method_);
//  }

  slab::vec x = start_;

  if (profile_) {
    bfgs.set_trace(trace_);
    bfgs.set_message(errormsg_);

    //optim.control.trace = trace_;

    // Maximum number of iterations
    const int kIterMax = 200;

    // Machine precision
    const double kEpsilon = std::numeric_limits<double>::epsilon();

    // Convergence criterion on x values
    const double kTolX = 4 * kEpsilon;

    // Scaled maximum step length allowed in line searches
    const double kScaStepMax = 100;

    const double grad_tol = 1e-6;

    const int n_pars = x.n_rows();  // number of parameters

    double f = jmcm_(x);

    slab::vec grad;
    jmcm_.Gradient(x, grad);

    // Initialize the inverse Hessian to a unit matrix
    slab::mat hess_inv = slab::eye<slab::mat>(n_pars, n_pars);

    // Initialize Newton Step
    //slab::vec p = -hess_inv * grad;
    slab::vec p = -slab::matmul(hess_inv, grad);

    // Calculate the maximum step length
    double sum = sqrt(slab::dot(x, x));
    const double kStepMax = kScaStepMax * std::max(sum, double(n_pars));

    // Main loop over the iterations
    for (int iter = 0; iter != kIterMax; ++iter) {
      n_iters_ = iter;

      slab::vec x2 = x;  // Save the old point

      linesearch.GetStep(jmcm_, x, p, kStepMax);

      f = jmcm_(x);  // Update function value
      p = x - x2;    // Update line direction
      x2 = x;
      f_min_ = f;

      if (trace_) {
        std::cout << std::setw(5) << iter << ": " << std::setw(10) << jmcm_(x)
                    << ": ";
        x.t().print();
      }

      // Test for convergence on Delta x
      double test = 0.0;
      for (int i = 0; i != n_pars; ++i) {
        double temp = std::abs(p(i)) / std::max(std::abs(x(i)), 1.0);
        if (temp > test) test = temp;
      }

      if (test < kTolX) {
        break;
      }

      slab::vec grad2 = grad;   // Save the old gradient
      jmcm_.Gradient(x, grad);  // Get the new gradient

      // Test for convergence on zero gradient
      test = 0.0;
      double den = std::max(f, 1.0);
      for (int i = 0; i != n_pars; ++i) {
        double temp = std::abs(grad(i)) * std::max(std::abs(x(i)), 1.0) / den;
        if (temp > test) test = temp;
      }

      if (test < grad_tol) {
        break;
      }

      if (!covonly_) jmcm_.UpdateBeta();

      if (method_id_ == 0) {
        slab::vec lmd = x.rows(n_bta, n_bta + n_lmd - 1);

        if (trace_) {
          std::cout << "--------------------------------------------------"
                      << "\n Updating Innovation Variance Parameters..."
                      << std::endl;
        }

        jmcm_.set_free_param(2);
        if (optim_method_ == "default")
          bfgs.Optimize(jmcm_, lmd);
        else ;
          //optim.minimize(jmcm_, lmd);
        jmcm_.set_free_param(0);

        if (trace_) {
          std::cout << "--------------------------------------------------"
                      << std::endl;
        }

        jmcm_.UpdateLambda(lmd);
        jmcm_.UpdateGamma();

      } else if (method_id_ == 1 || method_id_ == 2) {
        slab::vec lmdgma = x.rows(n_bta, n_bta + n_lmd + n_gma - 1);

        if (trace_) {
          switch (method_id_) {
            case 1: {
              std::cout
                  << "--------------------------------------------------"
                  << "\n Updating Innovation Variance Parameters"
                  << " and Moving Average Parameters..." << std::endl;
              break;
            }
            case 2: {
              std::cout
                  << "--------------------------------------------------"
                  << "\n Updating Variance Parameters"
                  << " and Angle Parameters..." << std::endl;
              break;
            }
            default: {}
          }
        }
        jmcm_.set_free_param(23);
        if (optim_method_ == "default")
          bfgs.Optimize(jmcm_, lmdgma);
        else ;
          // optim.minimize(jmcm_, lmdgma);
        jmcm_.set_free_param(0);
        if (trace_) {
          std::cout << "--------------------------------------------------"
                      << std::endl;
        }

        jmcm_.UpdateLambdaGamma(lmdgma);
      }

      slab::vec xnew = jmcm_.get_theta();

      p = xnew - x;
    }
  } else {
    if (optim_method_ == "default") {
      bfgs.set_trace(trace_);
      bfgs.set_message(errormsg_);
      bfgs.Optimize(jmcm_, x);
      f_min_ = bfgs.f_min();
      n_iters_ = bfgs.n_iters();
    } else {
      // optim.control.trace = trace_;
      // optim.minimize(jmcm_, x);
      // f_min_ = optim.value();
    }
  }

  return x;
}

#endif //JMCM_MKL_JMCM_FIT_H
