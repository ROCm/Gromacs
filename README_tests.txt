////GROMACS Unit-Test 
//Scope: single GPU

//In Gromacs/build folder
HIP_VISIBLE_DEVICES=0 make check -j$(nproc)  2>&1 |tee out.log






////Regression test
//Scope: single GPU or intra-node but 2 GPU only
//In Gromacs folder
tar zxvf regressiontests-2020.tar.gz
cd regressiontests-2020
HIP_VISIBLE_DEVICES=0,1 ./gmxtest.pl all 2>&1 | tee regression_threadmpi.log

//Scope: intranode mpi with ucx
//In Gromacs folder
tar zxvf regressiontests-2020.tar.gz
cd regressiontests-2020
sed -i 's/-wdir/-mca pml ucx -x UCX_TLS=rc,sm,rocm_cpy,rocm_gdr,rocm_ipc -wdir/g' gmxtest.pl
HIP_VISIBLE_DEVICES=0,1 ./gmxtest.pl all -np 2 2>&1 | tee regression_intrampi.log
