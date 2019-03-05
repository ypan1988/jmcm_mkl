# jmcm_mkl: Joint Mean-Covariance Models using Intel(R) Math Kernel Library 

## Prerequisites

    CMake >= 3.6.0
    Intel Math Kernel Library (Intel MKL) or OpenBLAS
   
## Installation on Ubuntu / macOS
+ Step 1: install package matrix
  ```sh
  git clone git@github.com:ypan1988/matrix.git
  cd matrix
  mkdir build && cd build
  cmake ..
  make
  cd ../..  # go back to the directory where you install the package
  ```

+ Step 2: install package stats
  ```sh
  git clone git@github.com:ypan1988/stats.git
  cd stats
  mkdir build && cd build
  cmake ..
  make
  cd ../..  # go back to the directory where you install the package
  ```

+ Step 3: install package jmcm_mkl
  ```sh
  git clone git@github.com:ypan1988/jmcm_mkl.git
  cd jmcm_mkl
  mkdir build && cd build
  cmake ..
  make
  ./main
  ```



