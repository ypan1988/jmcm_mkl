//
// Created by Yi Pan (Institute of Cancer and Genomic Sciences) on 24/07/2018.
//

#ifndef MKL_CMMR_BFGS_H
#define MKL_CMMR_BFGS_H
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>

#include "slab/matrix.h"
#include "linesearch.h"

namespace pan {

template<typename T>
class BFGS : public LineSearch<T> {
 public:
  BFGS();
  ~BFGS();

  void set_trace(bool trace) { trace_ = trace; }
  // void set_message(bool message) { LineSearch<T>::set_message(message); }
  void Optimize(T &func, slab::vec &x, const double grad_tol = 1e-6);
  int n_iters() const;
  double f_min() const;

 private:
  bool trace_;
  // bool message_;
  int n_iters_;
  double f_min_;
};  // class BFGS

#include "bfgs_impl.h"

}

#endif //MKL_CMMR_BFGS_H
