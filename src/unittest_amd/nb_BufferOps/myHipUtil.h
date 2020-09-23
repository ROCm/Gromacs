#ifndef _MY_UTIL_H_
#define _MY_UTIL_H_

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <sstream>
#include <iostream>

// Here flag can be a constant, variable or function call
#define MY_HIP_CHECK(flag)                                                                                                      \
     do  {                                                                                                                      \
         hipError_t _tmpVal;                                                                                                    \
         if ( (_tmpVal = flag) != hipSuccess) {                                                                                 \
               std::ostringstream ostr;                                                                                         \
               ostr << "HIP Function Failed (" <<  __FILE__ << "," <<  __LINE__ << ") " <<  hipGetErrorString(_tmpVal);         \
               std::cerr << ostr.str() << std::endl;                                                                            \
               throw std::runtime_error(ostr.str());                                                                            \
          }                                                                                                                     \
     }                                                                                                                          \
     while (0)

#endif 


