#!/bin/bash

cp -f ../input.txt .

#nb
cp -f inputData/adh/* .
HIP_VISIBLE_DEVICES=1 rocprof -i input.txt --stats -o ../result/adh/nbnxm-atomicAddNoRet-0.csv ./nbnxm_cuda_kernel_adh-atomicAddNoRet.hip.out 0
HIP_VISIBLE_DEVICES=1 rocprof -i input.txt --stats -o ../result/adh/nbnxm-atomicAddNoRet-1.csv ./nbnxm_cuda_kernel_adh-atomicAddNoRet.hip.out 1
HIP_VISIBLE_DEVICES=1 rocprof -i input.txt --stats -o ../result/adh/nbnxm-0.csv ./nbnxm_cuda_kernel_adh.hip.out 0
HIP_VISIBLE_DEVICES=1 rocprof -i input.txt --stats -o ../result/adh/nbnxm-1.csv ./nbnxm_cuda_kernel_adh.hip.out 1
HIP_VISIBLE_DEVICES=1 rocprof -i input.txt --stats -o ../result/adh/nbnxm-atomicInt-0.csv ./nbnxm_cuda_kernel_adh-atomicInt.hip.out 0
HIP_VISIBLE_DEVICES=1 rocprof -i input.txt --stats -o ../result/adh/nbnxm-atomicInt-1.csv ./nbnxm_cuda_kernel_adh-atomicInt.hip.out 1
