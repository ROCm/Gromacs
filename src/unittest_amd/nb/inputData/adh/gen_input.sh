INPUTFILE=$1

grep basic ${INPUTFILE} | awk -F ":" '{print $2}' > basic.txt
grep xq ${INPUTFILE} | awk -F ":" '{print $2}' > xq.txt
grep "e_el\|e_lj" ${INPUTFILE} | awk -F ":" '{print $2}' > energy.txt
grep shift_vec ${INPUTFILE} | awk -F ":" '{print $2}' > shift_vec.txt
grep lj_comb ${INPUTFILE} | awk -F ":" '{print $2}' > lj_comb.txt
grep atom_types ${INPUTFILE} | awk -F ":" '{print $2}' > atom_types.txt
grep "sci\[" ${INPUTFILE} | awk -F ":" '{print $2}' > nbnxn_sci_t.txt
grep "cj4\[" ${INPUTFILE} | awk -F ":" '{print $2}' > nbnxn_cj4_t.txt
grep "excl\[" ${INPUTFILE} | awk -F ":" '{print $2}' > nbnxn_excl_t.txt
grep "nbp" ${INPUTFILE} | awk -F ":" '{print $2}' > nbp.txt
