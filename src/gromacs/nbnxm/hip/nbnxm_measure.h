#include "gmxpre.h"

#include "gromacs/gpu_utils/cudautils_hip.h"

#include "nbnxm_cuda_kernel_utils_hip.h"
#include "nbnxm_cuda_types.h"

__global__ void nbnxm_measure(const cu_atomdata_t atdat, const NBParamGpu nbparam, const Nbnxm::gpu_plist plist, bool bCalcFshift);
