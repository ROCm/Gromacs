#!/bin/bash

cp -f ../input.txt .

#nb
HIP_VISIBLE_DEVICES=1 rocprof -i input.txt --stats -o ../result/adh/bonded-0-0.csv ./bondedCheck_hip.out inputData/adh/ 0 0
HIP_VISIBLE_DEVICES=1 rocprof -i input.txt --stats -o ../result/adh/bonded-1-1.csv ./bondedCheck_hip.out inputData/adh/ 1 1
HIP_VISIBLE_DEVICES=1 rocprof -i input.txt --stats -o ../result/adh/bonded-atomicAddNoRet-0-0.csv ./bondedCheck_hip-atomicAddNoRet.out inputData/adh/ 0 0
HIP_VISIBLE_DEVICES=1 rocprof -i input.txt --stats -o ../result/adh/bonded-atomicAddNoRet-1-1.csv ./bondedCheck_hip-atomicAddNoRet.out inputData/adh/ 1 1

