


 The data under this directory is got by running the adh_dodec benchmark as follows

     #> export GMX_BONDED_KERNEL_CHECK=1 
     #> gmx mdrun -s topol.tpr -pin on -ntmpi 2 -npme 1 -ntomp 4  -nsteps 5000 -v -tunepme -pme gpu -nb gpu  -bonded gpu  -gpu_id 0,1



