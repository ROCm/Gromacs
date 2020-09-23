/opt/rocm/bin/hipcc nbnxm_cuda_kernel.hip.cpp -o nbnxm.out
/opt/rocm/bin/hipcc nbnxm_cuda_kernel_adh-atomicInt.hip.cpp -o nbnxm_adh.out
