////GROMACS Unit-Test 
//Scope: single GPU

//In Gromacs/build folder
HIP_VISIBLE_DEVICES=0 make check -j$(nproc)  2>&1 |tee out.log






////Regression test
//Scope: single GPU or intra-node but 2 GPU only
//In Gromacs folder
tar zxvf regressiontests-2020.3.tar.gz
cd regressiontests-2020.3
HIP_VISIBLE_DEVICES=0,1 ./gmxtest.pl all 2>&1 | tee regression_threadmpi.log

//Scope: intranode mpi with ucx
//In Gromacs folder
tar zxvf regressiontests-2020.3.tar.gz
cd regressiontests-2020.3
sed -i 's/-wdir/-mca pml ucx -x UCX_TLS=sm,rocm -wdir/g' gmxtest.pl
HIP_VISIBLE_DEVICES=0,1 ./gmxtest.pl all -np 2 2>&1 | tee regression_intrampi.log
