/opt/rocm/bin/hipcc nbnxm_cuda_kernel.hip.cpp -o nbnxm.out
/opt/rocm/bin/hipcc nbnxm_cuda_kernel-atomicAddNoRet.hip.cpp -o nbnxm-atomicAddNoRet.out
/opt/rocm/bin/hipcc nbnxm_cuda_kernel_adh.hip.cpp -o nbnxm_adh.out
