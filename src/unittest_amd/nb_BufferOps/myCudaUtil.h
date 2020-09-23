#ifndef _MY_UTIL_H_
#define _MY_UTIL_H_

#include <stdexcept>
#include <sstream>
#include <iostream>

#include <cuda_runtime.h>

// Here flag can be a constant, variable or function call
#define MY_CUDA_CHECK(flag)                                                                                                      \
     do  {                                                                                                                      \
         cudaError_t _tmpVal;                                                                                                    \
         if ( (_tmpVal = flag) != cudaSuccess) {                                                                                 \
               std::ostringstream ostr;                                                                                         \
               ostr << "CUDA Function Failed (" <<  __FILE__ << "," <<  __LINE__ << ") " <<  cudaGetErrorString(_tmpVal);         \
               std::cerr << ostr.str() << std::endl;                                                                            \
               throw std::runtime_error(ostr.str());                                                                            \
          }                                                                                                                     \
     }                                                                                                                          \
     while (0)

#endif 


