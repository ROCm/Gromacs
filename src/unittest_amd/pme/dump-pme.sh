#!/bin/bash

cp -f inputData/adh/* .
HIP_VISIBLE_DEVICES=1 rocprof -i input.txt --stats -o ../result/adh/spread-0.csv ./spread.out 95561 0
HIP_VISIBLE_DEVICES=1 rocprof -i input.txt --stats -o ../result/adh/spread-1.csv ./spread.out 95561 1
HIP_VISIBLE_DEVICES=1 rocprof -i input.txt --stats -o ../result/adh/spread-atomicAddInt-0.csv ./spread-atomicAddInt.out 95561 0
HIP_VISIBLE_DEVICES=1 rocprof -i input.txt --stats -o ../result/adh/spread-atomicAddInt-1.csv ./spread-atomicAddInt.out 95561 1
HIP_VISIBLE_DEVICES=1 rocprof -i input.txt --stats -o ../result/adh/spread-atomicAddNoRet-0.csv ./spread-atomicAddNoRet.out 95561 0
HIP_VISIBLE_DEVICES=1 rocprof -i input.txt --stats -o ../result/adh/spread-atomicAddNoRet-1.csv ./spread-atomicAddNoRet.out 95561 1

HIP_VISIBLE_DEVICES=1 rocprof -i input.txt --stats -o ../result/adh/gather-0.csv ./gather.out 95561 0
HIP_VISIBLE_DEVICES=1 rocprof -i input.txt --stats -o ../result/adh/gather-1.csv ./gather.out 95561 1

HIP_VISIBLE_DEVICES=1 rocprof -i input.txt --stats -o ../result/adh/solve-0.csv ./solve.out 0
HIP_VISIBLE_DEVICES=1 rocprof -i input.txt --stats -o ../result/adh/solve-1.csv ./solve.out 1
HIP_VISIBLE_DEVICES=1 rocprof -i input.txt --stats -o ../result/adh/solve-atomicAddInt-0.csv ./solve-atomicAddInt.out 0
HIP_VISIBLE_DEVICES=1 rocprof -i input.txt --stats -o ../result/adh/solve-atomicAddInt-1.csv ./solve-atomicAddInt.out 1
HIP_VISIBLE_DEVICES=1 rocprof -i input.txt --stats -o ../result/adh/solve-atomicAddNoRet-0.csv ./solve-atomicAddNoRet.out 0
HIP_VISIBLE_DEVICES=1 rocprof -i input.txt --stats -o ../result/adh/solve-atomicAddNoRet-1.csv ./solve-atomicAddNoRet.out 1

