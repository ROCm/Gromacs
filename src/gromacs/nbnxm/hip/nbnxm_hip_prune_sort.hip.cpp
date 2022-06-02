#include "gmxpre.h"

#include "nbnxm_hip_prune_sort.hpp"

#ifndef FUNCTION_DECLARATION_ONLY
/* Instantiate external template functions */
template __global__ void
nbnxn_kernel_sort_j_hip<false>(const NBAtomDataGpu, const NBParamGpu, const Nbnxm::gpu_plist, int, int);
template __global__ void
nbnxn_kernel_sort_j_hip<true>(const NBAtomDataGpu, const NBParamGpu, const Nbnxm::gpu_plist, int, int);
#endif
