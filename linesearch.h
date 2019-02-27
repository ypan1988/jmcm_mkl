//
// Created by Yi Pan (Institute of Cancer and Genomic Sciences) on 20/07/2018.
//

#ifndef MKL_CMMR_LINESEARCH_H
#define MKL_CMMR_LINESEARCH_H

#include <algorithm>
#include <cmath>
#include <limits>

#include <slab/matrix.h>

namespace pan {

template <typename T>
class LineSearch {
 public:
  LineSearch();   // Constructor
  ~LineSearch();  // Destructor

  void GetStep(T &func, slab::vec &x, slab::vec &p, const double kStepMax);

  void set_message(bool message) { message_ = message; }

 protected:
  bool message_;
  bool IsInfOrNaN(double x);
};  // class LineSearch

#include "linesearch_impl.h"

}  // namespace pan

#endif //MKL_CMMR_LINESEARCH_H
