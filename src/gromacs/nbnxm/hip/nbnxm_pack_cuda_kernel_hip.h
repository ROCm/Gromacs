/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2012,2013,2014,2015,2016 by the GROMACS development team.
 * Copyright (c) 2017,2018,2019,2020, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */

/*! \internal \file
 *  \brief
 *  CUDA non-bonded kernel used through preprocessor-based code generation
 *  of multiple kernel flavors, see nbnxn_cuda_kernels.cuh.
 *
 *  NOTE: No include fence as it is meant to be included multiple times.
 *
 *  \author Szilárd Páll <pall.szilard@gmail.com>
 *  \author Berk Hess <hess@kth.se>
 *  \ingroup module_nbnxm
 */

#include "gromacs/gpu_utils/hip_arch_utils_hip.h"
#include "gromacs/gpu_utils/cuda_kernel_utils_hip.h"
#include "gromacs/math/utilities.h"
#include "gromacs/pbcutil/ishift.h"
/* Note that floating-point constants in CUDA code should be suffixed
 * with f (e.g. 0.5f), to stop the compiler producing intermediate
 * code that is in double precision.
 */

#if defined EL_EWALD_ANA || defined EL_EWALD_TAB
/* Note: convenience macro, needs to be undef-ed at the end of the file. */
#    define EL_EWALD_ANY
#endif

#if defined LJ_EWALD_COMB_GEOM || defined LJ_EWALD_COMB_LB
/* Note: convenience macro, needs to be undef-ed at the end of the file. */
#    define LJ_EWALD
#endif

#if defined EL_EWALD_ANY || defined EL_RF || defined LJ_EWALD \
        || (defined EL_CUTOFF && defined CALC_ENERGIES)
/* Macro to control the calculation of exclusion forces in the kernel
 * We do that with Ewald (elec/vdw) and RF. Cut-off only has exclusion
 * energy terms.
 *
 * Note: convenience macro, needs to be undef-ed at the end of the file.
 */
#    define EXCLUSION_FORCES
#endif

#if defined LJ_COMB_GEOM || defined LJ_COMB_LB
#    define LJ_COMB
#endif

/*
   Kernel launch parameters:
    - #blocks   = #pair lists, blockId = pair list Id
    - #threads  = NTHREAD_Z * c_clSize^2
    - shmem     = see nbnxn_cuda.cu:calc_shmem_required_nonbonded()

    Each thread calculates an i force-component taking one pair of i-j atoms.
 */

/**@{*/
/*! \brief Compute capability dependent definition of kernel launch configuration parameters.
 *
 * NTHREAD_Z controls the number of j-clusters processed concurrently on NTHREAD_Z
 * warp-pairs per block.
 *
 * - On CC 3.0-3.5, and >=5.0 NTHREAD_Z == 1, translating to 64 th/block with 16
 * blocks/multiproc, is the fastest even though this setup gives low occupancy
 * (except on 6.0).
 * NTHREAD_Z > 1 results in excessive register spilling unless the minimum blocks
 * per multiprocessor is reduced proportionally to get the original number of max
 * threads in flight (and slightly lower performance).
 * - On CC 3.7 there are enough registers to double the number of threads; using
 * NTHREADS_Z == 2 is fastest with 16 blocks (TODO: test with RF and other kernels
 * with low-register use).
 *
 * Note that the current kernel implementation only supports NTHREAD_Z > 1 with
 * shuffle-based reduction, hence CC >= 3.0.
 *
 *
 * NOTEs on Volta / CUDA 9 extensions:
 *
 * - While active thread masks are required for the warp collectives
 *   (we use any and shfl), the kernel is designed such that all conditions
 *   (other than the inner-most distance check) including loop trip counts
 *   are warp-synchronous. Therefore, we don't need ballot to compute the
 *   active masks as these are all full-warp masks.
 *
 * - TODO: reconsider the use of __syncwarp(): its only role is currently to prevent
 *   WAR hazard due to the cj preload; we should try to replace it with direct
 *   loads (which may be faster given the improved L1 on Volta).
 */

/* Kernel launch bounds for different compute capabilities. The value of NTHREAD_Z
 * determines the number of threads per block and it is chosen such that
 * 16 blocks/multiprocessor can be kept in flight.
 * - CC 3.0,3.5, and >=5.0: NTHREAD_Z=1, (64, 16) bounds
 * - CC 3.7:                NTHREAD_Z=2, (128, 16) bounds
 *
 * Note: convenience macros, need to be undef-ed at the end of the file.
 */

#ifndef NTHREAD_Z_VALUE
#    define NTHREAD_Z 1
#else
#    define NTHREAD_Z NTHREAD_Z_VALUE
#endif

#define MIN_BLOCKS_PER_MP (16)
#define THREADS_PER_BLOCK (c_clSize * c_clSize * NTHREAD_Z)

__launch_bounds__(THREADS_PER_BLOCK)
#ifdef PRUNE_NBL
#    ifdef CALC_ENERGIES
#       if NTHREAD_Z == 4
            __global__ void NB_KERNEL_FUNC_NAME(nbnxn_pack_kernel, _VF_prune_cuda_dimZ_4)
#       else
            __global__ void NB_KERNEL_FUNC_NAME(nbnxn_pack_kernel, _VF_prune_cuda)
#       endif
#    else
#       if NTHREAD_Z == 4
            __global__ void NB_KERNEL_FUNC_NAME(nbnxn_pack_kernel, _F_prune_cuda_dimZ_4)
#       else
            __global__ void NB_KERNEL_FUNC_NAME(nbnxn_pack_kernel, _F_prune_cuda)
#       endif
#    endif /* CALC_ENERGIES */
#else
#    ifdef CALC_ENERGIES
#       if NTHREAD_Z == 4
            __global__ void NB_KERNEL_FUNC_NAME(nbnxn_pack_kernel, _VF_cuda_dimZ_4)
#       else
            __global__ void NB_KERNEL_FUNC_NAME(nbnxn_pack_kernel, _VF_cuda)
#       endif
#    else
#       if NTHREAD_Z == 4
            __global__ void NB_KERNEL_FUNC_NAME(nbnxn_pack_kernel, _F_cuda_dimZ_4)
#       else
            __global__ void NB_KERNEL_FUNC_NAME(nbnxn_pack_kernel, _F_cuda)
#       endif
#    endif /* CALC_ENERGIES */
#endif     /* PRUNE_NBL */
                (const cu_atomdata_t atdat, const NBParamGpu nbparam, const Nbnxm::gpu_plist plist, bool bCalcFshift)
#ifdef FUNCTION_DECLARATION_ONLY
                        ; /* Only do function declaration, omit the function body. */
#else
{
    /* convenience variables */
    const nbnxn_sci_t* pl_sci = plist.sci;
#    ifndef PRUNE_NBL
    const
#    endif
            nbnxn_cj4_t* pl_cj4      = plist.cj4;
    const nbnxn_excl_t*  excl        = plist.excl;
#    ifndef LJ_COMB
    const int*           atom_types  = atdat.atom_types;
    int                  ntypes      = atdat.ntypes;
#    else
    const float2* lj_comb = atdat.lj_comb;
    float2        ljcp_i, ljcp_j;
#    endif
    const float4*        xq          = atdat.xq;
    float3*              f           = atdat.f;
    const float3*        shift_vec   = atdat.shift_vec;
    float                rcoulomb_sq = nbparam.rcoulomb_sq;
#    ifdef VDW_CUTOFF_CHECK
    float                rvdw_sq     = nbparam.rvdw_sq;
    float                vdw_in_range;
#    endif
#    ifdef LJ_EWALD
    float                lje_coeff2, lje_coeff6_6;
#    endif
#    ifdef EL_RF
    float                two_k_rf    = nbparam.two_k_rf;
#    endif
#    ifdef EL_EWALD_ANA
    float                beta2       = nbparam.ewald_beta * nbparam.ewald_beta;
    float                beta3       = nbparam.ewald_beta * nbparam.ewald_beta * nbparam.ewald_beta;
#    endif
#    ifdef PRUNE_NBL
    float                rlist_sq    = nbparam.rlistOuter_sq;
#    endif

    unsigned int bidx  = blockIdx.x;

#    ifdef CALC_ENERGIES
#        ifdef EL_EWALD_ANY
    float                beta        = nbparam.ewald_beta;
    float                ewald_shift = nbparam.sh_ewald;
#        else
    float c_rf = nbparam.c_rf;
#        endif /* EL_EWALD_ANY */
    float*               e_lj        = atdat.e_lj + bidx % c_clEnergySize + 1;
    float*               e_el        = atdat.e_el + bidx % c_clEnergySize + 1;
#    endif     /* CALC_ENERGIES */

    /* thread/block/warp id-s */
    unsigned int tidxi = threadIdx.x;
    unsigned int tidxj = threadIdx.y % 4;
    unsigned int jm_stride  = threadIdx.y / 4;
    unsigned int tidx  = threadIdx.y * blockDim.x + threadIdx.x;
#    if NTHREAD_Z == 1
    unsigned int tidxz = 0;
#    else
    unsigned int  tidxz = threadIdx.z;
#    endif

    int          sci, ci, cj, ai, aj, cij4_start, cij4_end;
#    ifndef LJ_COMB
    int          typei, typej;
#    endif
    int          i, jm, j4, wexcl_idx;
    float        qi, qj_f, r2, inv_r, inv_r2;
#    if !defined LJ_COMB_LB || defined CALC_ENERGIES
    float        inv_r6, c6, c12;
#    endif
#    ifdef LJ_COMB_LB
    float        sigma, epsilon;
#    endif
    float        int_bit, F_invr;
#    ifdef CALC_ENERGIES
    float        E_lj, E_el;
#    endif
#    if defined CALC_ENERGIES || defined LJ_POT_SWITCH
    float        E_lj_p;
#    endif
    unsigned int wexcl, imask, mask_ji, wexcl_;
    float4       xqbuf;
    float3       xi, xj, rv, f_ij, fcj_buf;
    float3       fci_buf[c_nbnxnGpuNumClusterPerSupercluster]; /* i force buffer */
    nbnxn_sci_t  nb_sci;

    int2 aj_int2;
#    ifndef LJ_COMB
    int2 typej_int2;
#    else
    float2        ljcp_j_x;
    float2        ljcp_j_y;
#    endif

#    ifdef LJ_COMB_LB
    float2        sigma_f2, epsilon_f2;
#    endif

#    if defined CALC_ENERGIES || defined LJ_POT_SWITCH
    float2        E_lj_p_f2;
#    endif

#    ifdef VDW_CUTOFF_CHECK
    float2                vdw_in_range_f2;
#    endif

    float2 xj_x_f2;
    float2 xj_y_f2;
    float2 xj_z_f2;

    float2 qj_f_f2;
    float2 r2_f2;
    float2 int_bit_f2;
    float2 c6_f2;
    float2 c12_f2;
    float2 inv_r_f2;
    float2 inv_r2_f2;
    float2 inv_r6_f2;
    float2 F_invr_f2;

    float2 rv_x_f2;
    float2 rv_y_f2;
    float2 rv_z_f2;

    float2 f_ij_x_f2;
    float2 f_ij_y_f2;
    float2 f_ij_z_f2;

    float2 fcj_buf_x_f2;
    float2 fcj_buf_y_f2;
    float2 fcj_buf_z_f2;

    /*! i-cluster interaction mask for a super-cluster with all c_nbnxnGpuNumClusterPerSupercluster=8 bits set */
    const unsigned superClInteractionMask = ((1U << c_nbnxnGpuNumClusterPerSupercluster) - 1U);

    /*********************************************************************
     * Set up shared memory pointers.
     * sm_nextSlotPtr should always be updated to point to the "next slot",
     * that is past the last point where data has been stored.
     */
    HIP_DYNAMIC_SHARED( char, sm_dynamicShmem)
    char*                  sm_nextSlotPtr = sm_dynamicShmem;
    static_assert(sizeof(char) == 1,
                  "The shared memory offset calculation assumes that char is 1 byte");

    /* shmem buffer for i x+q pre-loading */
    float4* xqib = (float4*)sm_nextSlotPtr;
    sm_nextSlotPtr += (c_nbnxnGpuNumClusterPerSupercluster * c_clSize * sizeof(*xqib));

    /* shmem buffer for cj, for each warp separately */
    int* cjs = (int*)(sm_nextSlotPtr);
    /* the cjs buffer's use expects a base pointer offset for pairs of warps in the j-concurrent execution */
    cjs += tidxz * c_nbnxnGpuClusterpairSplit * c_nbnxnGpuJgroupSize;
    sm_nextSlotPtr += (NTHREAD_Z * c_nbnxnGpuClusterpairSplit * c_nbnxnGpuJgroupSize * sizeof(*cjs));

#    ifndef LJ_COMB
    /* shmem buffer for i atom-type pre-loading */
    int* atib = (int*)sm_nextSlotPtr;
    sm_nextSlotPtr += (c_nbnxnGpuNumClusterPerSupercluster * c_clSize * sizeof(*atib));
#    else
    /* shmem buffer for i-atom LJ combination rule parameters */
    float2* ljcpib = (float2*)sm_nextSlotPtr;
    sm_nextSlotPtr += (c_nbnxnGpuNumClusterPerSupercluster * c_clSize * sizeof(*ljcpib));
#    endif
    /*********************************************************************/

    nb_sci     = pl_sci[bidx];         /* my i super-cluster's index = current bidx */
    sci        = nb_sci.sci;           /* super-cluster */
    cij4_start = nb_sci.cj4_ind_start; /* first ...*/
    cij4_end   = nb_sci.cj4_ind_end;   /* and last index of j clusters */

    if (tidxz == 0)
    {
        /* Pre-load i-atom x and q into shared memory */
        ci = sci * c_nbnxnGpuNumClusterPerSupercluster + threadIdx.y;
        ai = ci * c_clSize + tidxi;

        float* shiftptr = (float*)&shift_vec[nb_sci.shift];
        xqbuf = xq[ai] + make_float4(LDG(shiftptr), LDG(shiftptr + 1), LDG(shiftptr + 2), 0.0f);
        xqbuf.w *= nbparam.epsfac;
        xqib[threadIdx.y * c_clSize + tidxi] = xqbuf;

#    ifndef LJ_COMB
        /* Pre-load the i-atom types into shared memory */
        atib[threadIdx.y * c_clSize + tidxi] = atom_types[ai];
#    else
        /* Pre-load the LJ combination parameters into shared memory */
        ljcpib[threadIdx.y * c_clSize + tidxi] = lj_comb[ai];
#    endif
    }
    __syncthreads();

    for (i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; i++)
    {
        fci_buf[i] = make_float3(0.0f);
    }

#    ifdef LJ_EWALD
    /* TODO: we are trading registers with flops by keeping lje_coeff-s, try re-calculating it later */
    lje_coeff2   = nbparam.ewaldcoeff_lj * nbparam.ewaldcoeff_lj;
    lje_coeff6_6 = lje_coeff2 * lje_coeff2 * lje_coeff2 * c_oneSixth;
#    endif


#    ifdef CALC_ENERGIES
    E_lj         = 0.0f;
    E_el         = 0.0f;

#        ifdef EXCLUSION_FORCES /* Ewald or RF */
    if (nb_sci.shift == CENTRAL && pl_cj4[cij4_start].cj[0] == sci * c_nbnxnGpuNumClusterPerSupercluster)
    {
        /* we have the diagonal: add the charge and LJ self interaction energy term */
        for (i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; i++)
        {
#            if defined EL_EWALD_ANY || defined EL_RF || defined EL_CUTOFF
            qi = xqib[i * c_clSize + tidxi].w;
            E_el += qi * qi;
#            endif

#            ifdef LJ_EWALD
#                if DISABLE_CUDA_TEXTURES
            E_lj += LDG(&nbparam.nbfp[atom_types[(sci * c_nbnxnGpuNumClusterPerSupercluster + i) * c_clSize + tidxi]
                                      * (ntypes + 1) * 2]);
#                else
            E_lj += tex1Dfetch<float>(
                    nbparam.nbfp_texobj,
                    atom_types[(sci * c_nbnxnGpuNumClusterPerSupercluster + i) * c_clSize + tidxi]
                            * (ntypes + 1) * 2);
#                endif
#            endif
        }

        /* divide the self term(s) equally over the j-threads, then multiply with the coefficients. */
#            ifdef LJ_EWALD
        E_lj /= c_clSize * NTHREAD_Z;
        E_lj *= 0.5f * c_oneSixth * lje_coeff6_6;
#            endif

#            if defined EL_EWALD_ANY || defined EL_RF || defined EL_CUTOFF
        /* Correct for epsfac^2 due to adding qi^2 */
        E_el /= nbparam.epsfac * c_clSize * NTHREAD_Z;
#                if defined EL_RF || defined EL_CUTOFF
        E_el *= -0.5f * c_rf;
#                else
        E_el *= -beta * M_FLOAT_1_SQRTPI; /* last factor 1/sqrt(pi) */
#                endif
#            endif /* EL_EWALD_ANY || defined EL_RF || defined EL_CUTOFF */
    }
#        endif     /* EXCLUSION_FORCES */

#    endif /* CALC_ENERGIES */

#    ifdef EXCLUSION_FORCES
     int2 nonSelfInteraction_int2 = {!(nb_sci.shift == CENTRAL & tidxj <= tidxi), !(nb_sci.shift == CENTRAL & (tidxj+4) <= tidxi)};
#    endif

//#    ifdef EXCLUSION_FORCES
//    const int nonSelfInteraction = !(nb_sci.shift == CENTRAL & tidxj <= tidxi);
//#    endif

    /* loop over the j clusters = seen by any of the atoms in the current super-cluster;
     * The loop stride NTHREAD_Z ensures that consecutive warps-pairs are assigned
     * consecutive j4's entries.
     */
    for (j4 = cij4_start + tidxz; j4 < cij4_end; j4 += NTHREAD_Z)
    {
        wexcl_idx = pl_cj4[j4].imei[0].excl_ind;
        imask     = pl_cj4[j4].imei[0].imask;

        tidx      = tidxj * blockDim.x + tidxi;
        wexcl     = excl[wexcl_idx].pair[(tidx) & (warpSize - 1)];
	tidx      = (tidxj+4) * blockDim.x + tidxi;
	wexcl_    = excl[wexcl_idx].pair[(tidx) & (warpSize - 1)];

#    ifndef PRUNE_NBL
        if (imask)
#    endif
        {
            /* Pre-load cj into shared memory on both warps separately */
            if (tidxi == 0)
            {
                cjs[tidxj] = pl_cj4[j4].cj[tidxj];
            }
//            __syncwarp(c_fullWarpMask); //cm todo
              __all(1);

            /* Unrolling this loop
               - with pruning leads to register spilling;
               - on Kepler and later it is much slower;
               Tested with up to nvcc 7.5 */
#    if !defined PRUNE_NBL
#        pragma unroll 2
#    endif
            for (jm = jm_stride; jm < c_nbnxnGpuJgroupSize; jm += 2)
            {
                if (imask & (superClInteractionMask << (jm * c_nbnxnGpuNumClusterPerSupercluster)))
                {


                    mask_ji = (1U << (jm * c_nbnxnGpuNumClusterPerSupercluster));

                    cj = cjs[jm];
                    //aj = cj * c_clSize + tidxj;
		    aj_int2 = {cj * c_clSize + tidxj, cj * c_clSize + tidxj +4};

                    /* load j atom data */
                    xqbuf = xq[aj_int2.x];
		    xj_x_f2.x = xqbuf.x;
		    xj_y_f2.x = xqbuf.y;
		    xj_z_f2.x = xqbuf.z;
		    qj_f_f2.x = xqbuf.w;

                    xqbuf = xq[aj_int2.y];
		    xj_x_f2.y = xqbuf.x;
		    xj_y_f2.y = xqbuf.y;
		    xj_z_f2.y = xqbuf.z;
		    qj_f_f2.y = xqbuf.w;
                    //xj    = make_float3(xqbuf.x, xqbuf.y, xqbuf.z);
		    //xj_x_f2 = {xq[aj_int2.x].x, xq[aj_int2.y].x};
		    //xj_y_f2 = {xq[aj_int2.x].y, xq[aj_int2.y].y};
		    //xj_z_f2 = {xq[aj_int2.x].z, xq[aj_int2.y].z};

                    //qj_f  = xqbuf.w;
		    //qj_f_f2 = {xq[aj_int2.x].w, xq[aj_int2.y].w};
#    ifndef LJ_COMB
                    //typej = atom_types[aj];
		    typej_int2 = {atom_types[aj_int2.x], atom_types[aj_int2.y]};
#    else
                    ljcp_j = lj_comb[aj_int2.x];
		    ljcp_j_x.x = ljcp_j.x;
		    ljcp_j_y.x = ljcp_j.y;

                    ljcp_j = lj_comb[aj_int2.y];
		    ljcp_j_x.y = ljcp_j.x;
		    ljcp_j_y.y = ljcp_j.y;

#    endif

                    fcj_buf = make_float3(0.0f);
		    fcj_buf_x_f2 = {0.0f, 0.0f};
		    fcj_buf_y_f2 = {0.0f, 0.0f};
		    fcj_buf_z_f2 = {0.0f, 0.0f};

#    if !defined PRUNE_NBL
#        pragma unroll 8
#    endif
                    for (i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; i++)
                    {
                        if (imask & mask_ji)
                        {
                            ci = sci * c_nbnxnGpuNumClusterPerSupercluster + i; /* i cluster index */

                            /* all threads load an atom from i cluster ci into shmem! */
                            xqbuf = xqib[i * c_clSize + tidxi];
                            xi    = make_float3(xqbuf.x, xqbuf.y, xqbuf.z);

                            /* distance between i and j atoms */
                            //rv = xi - xj;
                            //r2 = norm2(rv);
			    rv_x_f2 = {xi.x - xj_x_f2.x, xi.x - xj_x_f2.y};
			    rv_y_f2 = {xi.y - xj_y_f2.x, xi.y - xj_y_f2.y};
			    rv_z_f2 = {xi.z - xj_z_f2.x, xi.z - xj_z_f2.y};

			    r2_f2 = rv_x_f2 * rv_x_f2 + rv_y_f2 * rv_y_f2 + rv_z_f2 * rv_z_f2;

#    ifdef PRUNE_NBL
                            /* If _none_ of the atoms pairs are in cutoff range,
                               the bit corresponding to the current
                               cluster-pair in imask gets set to 0. */
                            if (!__any(r2 < rlist_sq))
                            {
                                imask &= ~mask_ji;
                            }
#    endif

                            //int_bit = (wexcl & mask_ji) ? 1.0f : 0.0f;
			    int_bit_f2.x = (wexcl & mask_ji) ? 1.0f : 0.0f;
			    int_bit_f2.y = (wexcl_ & mask_ji) ? 1.0f : 0.0f;

                            /* cutoff & exclusion check */
#    ifdef EXCLUSION_FORCES
                            if ((r2_f2.x < rcoulomb_sq) * (nonSelfInteraction_int2.x | (ci != cj)) ||
                                (r2_f2.y < rcoulomb_sq) * (nonSelfInteraction_int2.y | (ci != cj)))
#    else
                            if ((r2 < rcoulomb_sq) * int_bit)
#    endif
                            {
                                /* load the rest of the i-atom parameters */
                                qi = xqbuf.w;

				if ( !((r2_f2.x < rcoulomb_sq) * (nonSelfInteraction_int2.x | (ci != cj))) ) {
					r2_f2.x = r2_f2.y;
					rv_x_f2.x = 0.0f;
                                        rv_y_f2.x = 0.0f;
                                        rv_z_f2.x = 0.0f;
				}
				else if ( !((r2_f2.y < rcoulomb_sq) * (nonSelfInteraction_int2.y | (ci != cj))) ) {
					r2_f2.y = r2_f2.x;
					rv_x_f2.y = 0.0f;
                                        rv_y_f2.y = 0.0f;
                                        rv_z_f2.y = 0.0f;
				}
#    ifndef LJ_COMB
                                /* LJ 6*C6 and 12*C12 */
                                typei = atib[i * c_clSize + tidxi];
                                //fetch_nbfp_c6_c12(c6, c12, nbparam, ntypes * typei + typej);
				int2 index = {ntypes * typei + typej_int2.x, ntypes * typei + typej_int2.y};
                                fetch_nbfp_c6_c12(c6_f2, c12_f2, nbparam, index);
#    else
                                ljcp_i       = ljcpib[i * c_clSize + tidxi];
#        ifdef LJ_COMB_GEOM
                                c6_f2           = ljcp_i.x * ljcp_j_x;
                                c12_f2          = ljcp_i.y * ljcp_j_y;
#        else
                                /* LJ 2^(1/6)*sigma and 12*epsilon */
                                sigma_f2   = ljcp_i.x + ljcp_j_x;
                                epsilon_f2 = ljcp_i.y * ljcp_j_y;
#            if defined CALC_ENERGIES || defined LJ_FORCE_SWITCH || defined LJ_POT_SWITCH
                                convert_sigma_epsilon_to_c6_c12(sigma_f2, epsilon_f2, &c6_f2, &c12_f2);
#            endif
#        endif /* LJ_COMB_GEOM */
#    endif     /* LJ_COMB */

                                // Ensure distance do not become so small that r^-12 overflows
                                //r2 = fmax(r2, c_nbnxnMinDistanceSquared);

                                //inv_r  = __frsqrt_rn(r2);
                                //inv_r2 = inv_r * inv_r;
				r2_f2 = {fmax(r2_f2.x, c_nbnxnMinDistanceSquared), fmax(r2_f2.y, c_nbnxnMinDistanceSquared)};
				inv_r_f2 = {__frsqrt_rn(r2_f2.x), __frsqrt_rn(r2_f2.y)};
				inv_r2_f2 = inv_r_f2 * inv_r_f2;

#    if !defined LJ_COMB_LB || defined CALC_ENERGIES
                                //inv_r6 = inv_r2 * inv_r2 * inv_r2;
				inv_r6_f2 = inv_r2_f2 * inv_r2_f2 * inv_r2_f2;
#        ifdef EXCLUSION_FORCES
                                /* We could mask inv_r2, but with Ewald
                                 * masking both inv_r6 and F_invr is faster */
                                //inv_r6 *= int_bit;
				inv_r6_f2 *= int_bit_f2;
#        endif /* EXCLUSION_FORCES */

                                //F_invr = inv_r6 * (c12 * inv_r6 - c6) * inv_r2;
				F_invr_f2 = inv_r6_f2 * (c12_f2 * inv_r6_f2 - c6_f2) * inv_r2_f2;
#        if defined CALC_ENERGIES || defined LJ_POT_SWITCH
                                E_lj_p_f2 = int_bit_f2
                                         * (c12_f2 * (inv_r6_f2 * inv_r6_f2 + nbparam.repulsion_shift.cpot) * c_oneTwelveth
                                            - c6_f2 * (inv_r6_f2 + nbparam.dispersion_shift.cpot) * c_oneSixth);
#        endif
#    else /* !LJ_COMB_LB || CALC_ENERGIES */
                                float2 sig_r  = sigma_f2 * inv_r_f2;
                                float2 sig_r2 = sig_r * sig_r;
                                float2 sig_r6 = sig_r2 * sig_r2 * sig_r2;
#        ifdef EXCLUSION_FORCES
                                sig_r6 *= int_bit_f2;
#        endif /* EXCLUSION_FORCES */

                                F_invr_f2 = epsilon_f2 * sig_r6 * (sig_r6 - 1.0f) * inv_r2_f2;
#    endif     /* !LJ_COMB_LB || CALC_ENERGIES */

#    ifdef LJ_FORCE_SWITCH
#        ifdef CALC_ENERGIES
                                calculate_force_switch_F_E(nbparam, c6_f2, c12_f2, inv_r_f2, r2_f2, &F_invr_f2, &E_lj_p_f2);
#        else
                                calculate_force_switch_F(nbparam, c6_f2, c12_f2, inv_r_f2, r2_f2, &F_invr_f2);
#        endif /* CALC_ENERGIES */
#    endif     /* LJ_FORCE_SWITCH */


#    ifdef LJ_EWALD
#        ifdef LJ_EWALD_COMB_GEOM
#            ifdef CALC_ENERGIES
                                calculate_lj_ewald_comb_geom_F_E(nbparam, typei, typej_int2, r2_f2, inv_r2_f2,
                                                                 lje_coeff2, lje_coeff6_6, int_bit_f2,
                                                                 &F_invr_f2, &E_lj_p_f2);
#            else
                                calculate_lj_ewald_comb_geom_F(nbparam, typei, typej_int2, r2_f2, inv_r2_f2,
                                                               lje_coeff2, lje_coeff6_6, &F_invr_f2);
#            endif /* CALC_ENERGIES */
#        elif defined LJ_EWALD_COMB_LB
                                calculate_lj_ewald_comb_LB_F_E(nbparam, typei, typej_int2, r2_f2, inv_r2_f2,
                                                               lje_coeff2, lje_coeff6_6,
#            ifdef CALC_ENERGIES
                                                               int_bit_f2, &F_invr_f2, &E_lj_p_f2
#            else
                                                               0, &F_invr_f2, nullptr
#            endif /* CALC_ENERGIES */
                                );
#        endif     /* LJ_EWALD_COMB_GEOM */
#    endif         /* LJ_EWALD */

#    ifdef LJ_POT_SWITCH
#        ifdef CALC_ENERGIES
                                calculate_potential_switch_F_E(nbparam, inv_r, r2, &F_invr, &E_lj_p);
#        else
                                calculate_potential_switch_F(nbparam, inv_r, r2, &F_invr, &E_lj_p);
#        endif /* CALC_ENERGIES */
#    endif     /* LJ_POT_SWITCH */

#    ifdef VDW_CUTOFF_CHECK
                                /* Separate VDW cut-off check to enable twin-range cut-offs
                                 * (rvdw < rcoulomb <= rlist)
                                 */
                                vdw_in_range_f2.x = (r2_f2.x < rvdw_sq) ? 1.0f : 0.0f;
                                vdw_in_range_f2.y = (r2_f2.y < rvdw_sq) ? 1.0f : 0.0f;
                                F_invr_f2 *= vdw_in_range_f2;
#        ifdef CALC_ENERGIES
                                E_lj_p_f2 *= vdw_in_range_f2;
#        endif
#    endif /* VDW_CUTOFF_CHECK */

#    ifdef CALC_ENERGIES
                                E_lj += (E_lj_p_f2.x + E_lj_p_f2.y);
#    endif


#    ifdef EL_CUTOFF
#        ifdef EXCLUSION_FORCES
                                F_invr_f2 += qi * qj_f_f2 * int_bit_f2 * inv_r2_f2 * inv_r2_f2;
#        else
                                F_invr_f2 += qi * qj_f_f2 * inv_r2_f2 * inv_r2_f2;
#        endif
#    endif
#    ifdef EL_RF
                                F_invr_f2 += qi * qj_f_f2 * (int_bit_f2 * inv_r2_f2 * inv_r2_f2 - two_k_rf);
#    endif
#    if defined   EL_EWALD_ANA
                                F_invr_f2 += qi * qj_f_f2
                                          * (int_bit_f2 * inv_r2_f2 * inv_r_f2 + pmecorrF(beta2 * r2) * beta3);
#    elif defined EL_EWALD_TAB
                                F_invr_f2 += qi * qj_f_f2
                                          * (int_bit_f2 * inv_r2_f2
                                             - interpolate_coulomb_force_r(nbparam, r2_f2 * inv_r_f2))
                                          * inv_r_f2;
#    endif /* EL_EWALD_ANA/TAB */

#    ifdef CALC_ENERGIES
#        ifdef EL_CUTOFF
                                E_el += (qi * qj_f_f2.x * (int_bit_f2.x * inv_r_f2.x - c_rf) + qi * qj_f_f2.y * (int_bit_f2.y * inv_r_f2.y - c_rf));
#        endif
#        ifdef EL_RF
                                E_el += qi * qj_f * (int_bit * inv_r + 0.5f * two_k_rf * r2 - c_rf);
#        endif
#        ifdef EL_EWALD_ANY
                                /* 1.0f - erff is faster than erfcf */
                                E_el += qi * qj_f
                                        * (inv_r * (int_bit - erff(r2 * inv_r * beta)) - int_bit * ewald_shift);
#        endif /* EL_EWALD_ANY */
#    endif
                                f_ij_x_f2 = rv_x_f2 * F_invr_f2;
                                f_ij_y_f2 = rv_y_f2 * F_invr_f2;
                                f_ij_z_f2 = rv_z_f2 * F_invr_f2;

                                /* accumulate j forces in registers */
                                //fcj_buf = fcj_buf - f_ij;
				fcj_buf_x_f2 = fcj_buf_x_f2 - f_ij_x_f2;
				fcj_buf_y_f2 = fcj_buf_y_f2 - f_ij_y_f2;
				fcj_buf_z_f2 = fcj_buf_z_f2 - f_ij_z_f2;

                                /* accumulate i forces in registers */
                                //fci_buf[i] = fci_buf[i] + f_ij;
                                fci_buf[i].x = fci_buf[i].x + f_ij_x_f2.x + f_ij_x_f2.y;
                                fci_buf[i].y = fci_buf[i].y + f_ij_y_f2.x + f_ij_y_f2.y;
                                fci_buf[i].z = fci_buf[i].z + f_ij_z_f2.x + f_ij_z_f2.y;
                            }
                        }
                        /* shift the mask bit by 1 */
                        mask_ji += mask_ji;
                    }

                    /* reduce j forces */
		    fcj_buf = make_float3(fcj_buf_x_f2.x, fcj_buf_y_f2.x, fcj_buf_z_f2.x);
                    reduce_force_j_warp_shfl(fcj_buf, f, tidxi, aj_int2.x, c_fullWarpMask);
		    fcj_buf = make_float3(fcj_buf_x_f2.y, fcj_buf_y_f2.y, fcj_buf_z_f2.y);
                    reduce_force_j_warp_shfl(fcj_buf, f, tidxi, aj_int2.y, c_fullWarpMask);
                }
            }
#    ifdef PRUNE_NBL
            /* Update the imask with the new one which does not contain the
               out of range clusters anymore. */
            pl_cj4[j4].imei[0].imask = imask;
#    endif
        }
        // avoid shared memory WAR hazards between loop iterations
        //__syncwarp(c_fullWarpMask);
	__all(1);
    }

    /* skip central shifts when summing shift forces */
    if (nb_sci.shift == CENTRAL)
    {
        bCalcFshift = false;
    }

    float fshift_buf = 0.0f;

    /* reduce i forces */
#    if !defined PRUNE_NBL
#        pragma unroll 8
#    endif
    for (i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; i++)
    {
        ai = (sci * c_nbnxnGpuNumClusterPerSupercluster + i) * c_clSize + tidxi;
        reduce_force_i_warp_shfl(fci_buf[i], f, &fshift_buf, bCalcFshift, threadIdx.y, ai, c_fullWarpMask);
    }

    /* add up local shift forces into global mem, tidxj indexes x,y,z */
    if ( bCalcFshift)
    {
        #pragma unroll
        for (unsigned int offset = (c_clSize >> 1); offset > 0; offset >>= 1)
        {
            fshift_buf += __shfl_down(fshift_buf, offset);
        }

        if( tidxi == 0 && tidxj < 3 )
        {
            atomicAdd(&(atdat.fshift[nb_sci.shift + SHIFTS * (bidx % c_clShiftSize)].x) + tidxj, fshift_buf);
        }
    }

#    ifdef CALC_ENERGIES
    /* reduce the energies over warps and store into global memory */
    reduce_energy_warp_shfl(E_lj, E_el, e_lj, e_el, tidx, c_fullWarpMask);
#    endif
}
#endif /* FUNCTION_DECLARATION_ONLY */

#undef NTHREAD_Z
#undef MIN_BLOCKS_PER_MP
#undef THREADS_PER_BLOCK

#undef EL_EWALD_ANY
#undef EXCLUSION_FORCES
#undef LJ_EWALD

#undef LJ_COMB
