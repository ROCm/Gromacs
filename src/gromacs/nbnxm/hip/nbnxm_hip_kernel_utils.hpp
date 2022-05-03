/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2012,2013,2014,2016,2017 by the GROMACS development team.
 * Copyright (c) 2018,2019,2020,2021, by the GROMACS development team, led by
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
 *  Utility constant and function declaration for the HIP non-bonded kernels.
 *  This header should be included once at the top level, just before the
 *  kernels are included (has to be preceded by nbnxn_hip_types.h).
 *
 *  \author Szilárd Páll <pall.szilard@gmail.com>
 *  \ingroup module_nbnxm
 */

#ifndef NBNXM_HIP_KERNEL_UTILS_HPP
#define NBNXM_HIP_KERNEL_UTILS_HPP

#include <assert.h>

/* Note that floating-point constants in HIP code should be suffixed
 * with f (e.g. 0.5f), to stop the compiler producing intermediate
 * code that is in double precision.
 */

#include "gromacs/gpu_utils/hip_arch_utils.hpp"
#include "gromacs/gpu_utils/hip_kernel_utils.hpp"
#include "gromacs/gpu_utils/vectype_ops.hpp"
#include "gromacs/gpu_utils/typecasts.hpp"

#include "nbnxm_hip_types.h"

constexpr int c_subWarp = 64 / c_nbnxnGpuClusterpairSplit;
/*! \brief Log of the i and j cluster size.
 *  change this together with c_clSize !*/
static const int __device__ c_clSizeLog2 = 3;
/*! \brief Square of cluster size. */
static const int __device__ c_clSizeSq = c_clSize * c_clSize;
/*! \brief j-cluster size after split (4 in the current implementation). */
static const int __device__ c_splitClSize = c_clSize / c_nbnxnGpuClusterpairSplit;
/*! \brief Stride in the force accumualation buffer */
static const int __device__ c_fbufStride = c_clSizeSq;
/*! \brief i-cluster interaction mask for a super-cluster with all c_nbnxnGpuNumClusterPerSupercluster=8 bits set */
static const unsigned __device__ superClInteractionMask =
        ((1U << c_nbnxnGpuNumClusterPerSupercluster) - 1U);

static const float __device__ c_oneSixth    = 0.16666667F;
static const float __device__ c_oneTwelveth = 0.08333333F;

template<class T, int dpp_ctrl, int row_mask = 0xf, int bank_mask = 0xf, bool bound_ctrl = true>
__device__ inline
T warp_move_dpp(const T& input) {
    constexpr int words_no = (sizeof(T) + sizeof(int) - 1) / sizeof(int);

    struct V { int words[words_no]; };
    V a = __builtin_bit_cast(V, input);
    #pragma unroll
    for (int i = 0; i < words_no; i++) {
        a.words[i] = __builtin_amdgcn_update_dpp(
          0, a.words[i],
          dpp_ctrl, row_mask, bank_mask, bound_ctrl
        );
    }

    return __builtin_bit_cast(T, a);
}

__device__ __forceinline__ int __nb_any(int predicate,int widx)
{
    if (c_subWarp == warpSize)
    {
        return __any(predicate);
    }
    else
    {
        return (int)(__ballot(predicate) >> (widx * c_subWarp));
    }
}

static __forceinline__ __device__
void float3_reduce_final(float3* input_ptr, const unsigned int size)
{
    const unsigned int flat_id = threadIdx.x;

    float3 input;
    input.x = atomicExch(&(input_ptr[size * (flat_id + 1)].x)    , 0.0f);
    input.y = atomicExch(&(input_ptr[size * (flat_id + 1)].x) + 1, 0.0f);
    input.z = atomicExch(&(input_ptr[size * (flat_id + 1)].x) + 2, 0.0f);

    #pragma unroll
    for(unsigned int offset = 1; offset < warpSize; offset *= 2)
    {
        input.x = input.x + __shfl_down(input.x, offset);
        input.y = input.y + __shfl_down(input.y, offset);
        input.z = input.z + __shfl_down(input.z, offset);
    }

    if( flat_id == 0 || flat_id == warpSize )
    {
        atomicAdd(&(input_ptr[0].x)    , input.x);
        atomicAdd(&(input_ptr[0].x) + 1, input.y);
        atomicAdd(&(input_ptr[0].x) + 2, input.z);
    }
}

static __forceinline__ __device__
void energy_reduce_final(float* e_lj_ptr, float* e_el_ptr)
{
    const unsigned int flat_id = threadIdx.x;

    float E_lj = atomicExch(e_lj_ptr + (flat_id + 1), 0.0f);
    float E_el = atomicExch(e_el_ptr + (flat_id + 1), 0.0f);

    #pragma unroll
    for(unsigned int offset = 1; offset < warpSize; offset *= 2)
    {
        E_lj += __shfl_down(E_lj, offset);
        E_el += __shfl_down(E_el, offset);
    }

    if( flat_id == 0 || flat_id == warpSize )
    {
        atomicAdd(e_lj_ptr, E_lj);
        atomicAdd(e_el_ptr, E_el);
    }
}

template<
    unsigned int BlockSize
>
__launch_bounds__(BlockSize) __global__
void nbnxn_kernel_sum_up(
    NBAtomDataGpu atdat,
    int size,
    bool computeEnergy,
    bool computeVirial)
{
    unsigned int bidx = blockIdx.x;

    // Sum up fshifts
    if(computeVirial)
    {
        float3* values_ptr = asFloat3(atdat.fShift) + bidx;
        float3_reduce_final(values_ptr, size);
    }

    // Sum up energies
    if(computeEnergy && bidx == 0)
    {
        float* e_lj_ptr = atdat.eLJ;
        float* e_el_ptr = atdat.eElec;

        energy_reduce_final(e_lj_ptr, e_el_ptr);
    }
}

template<
    unsigned int BlockSize,
    unsigned int ItemsPerThread
>
__launch_bounds__(BlockSize) __global__
void nbnxn_kernel_bucket_sci_sort(
    Nbnxm::gpu_plist plist)
{
    int size = plist.nsci;

    const unsigned int flat_id      = threadIdx.x;
    const unsigned int block_id     = blockIdx.x;
    const unsigned int block_offset = blockIdx.x * BlockSize * ItemsPerThread;

    const nbnxn_sci_t* pl_sci = plist.sci;
    nbnxn_sci_t* pl_sci_sort  = plist.sci_sorted;
    const int* pl_sci_count   = plist.sci_count;
    int* pl_sci_offset        = plist.sci_offset;

    int sci_count[ItemsPerThread];
    int sci_offset[ItemsPerThread];
    nbnxn_sci_t sci[ItemsPerThread];

    #pragma unroll
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        if( size > (block_offset + ItemsPerThread * flat_id + i) )
        {
            sci[i]        = pl_sci[block_offset + ItemsPerThread * flat_id + i];
            sci_count[i]  = pl_sci_count[block_offset + ItemsPerThread * flat_id + i];
        }
    }

    #pragma unroll
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        if( size > (block_offset + ItemsPerThread * flat_id + i) )
            sci_offset[i] = atomicAdd(&pl_sci_offset[c_sciHistogramSize - sci_count[i] - 1], 1);
    }

    #pragma unroll
    for(unsigned int i = 0; i < ItemsPerThread; i++)
    {
        if( size > (block_offset + ItemsPerThread * flat_id + i) )
            pl_sci_sort[sci_offset[i]] = sci[i];
    }
}

/*! Convert LJ sigma,epsilon parameters to C6,C12. */
static __forceinline__ __device__ void
convert_sigma_epsilon_to_c6_c12(const float sigma, const float epsilon, float* c6, float* c12)
{
    float sigma2, sigma6;

    sigma2 = sigma * sigma;
    sigma6 = sigma2 * sigma2 * sigma2;
    *c6    = epsilon * sigma6;
    *c12   = *c6 * sigma6;
}

static __forceinline__ __device__ void
                       convert_sigma_epsilon_to_c6_c12(const float2 sigma, const float2 epsilon, float2* c6, float2* c12)
{
    float2 sigma2, sigma6;

    sigma2 = sigma * sigma;
    sigma6 = sigma2 * sigma2 * sigma2;
    *c6    = epsilon * sigma6;
    *c12   = *c6 * sigma6;
}
/*! Apply force switch,  force + energy version. */
static __forceinline__ __device__ void
calculate_force_switch_F(const NBParamGpu nbparam, float c6, float c12, float inv_r, float r2, float* F_invr)
{
    float r, r_switch;

    /* force switch constants */
    float disp_shift_V2 = nbparam.dispersion_shift.c2;
    float disp_shift_V3 = nbparam.dispersion_shift.c3;
    float repu_shift_V2 = nbparam.repulsion_shift.c2;
    float repu_shift_V3 = nbparam.repulsion_shift.c3;

    r        = r2 * inv_r;
    r_switch = r - nbparam.rvdw_switch;
    r_switch = r_switch >= 0.0F ? r_switch : 0.0F;

    *F_invr += -c6 * (disp_shift_V2 + disp_shift_V3 * r_switch) * r_switch * r_switch * inv_r
               + c12 * (repu_shift_V2 + repu_shift_V3 * r_switch) * r_switch * r_switch * inv_r;
}

// For float2 type
static __forceinline__ __device__ void
                       calculate_force_switch_F(const NBParamGpu nbparam, float2 c6, float2 c12, float2 inv_r, float2 r2, float2* F_invr)
{
    float2 r, r_switch;

    /* force switch constants */
    float2 disp_shift_V2 = {nbparam.dispersion_shift.c2, nbparam.dispersion_shift.c2};
    float2 disp_shift_V3 = {nbparam.dispersion_shift.c3, nbparam.dispersion_shift.c3};
    float2 repu_shift_V2 = {nbparam.repulsion_shift.c2, nbparam.repulsion_shift.c2};
    float2 repu_shift_V3 = {nbparam.repulsion_shift.c3, nbparam.repulsion_shift.c3};

    r        = r2 * inv_r;
    r_switch = {r.x - nbparam.rvdw_switch, r.y - nbparam.rvdw_switch};
    r_switch.x = r_switch.x >= 0.0f ? r_switch.x : 0.0f;
    r_switch.y = r_switch.y >= 0.0f ? r_switch.y : 0.0f;

    *F_invr += -c6 * (disp_shift_V2 + disp_shift_V3 * r_switch) * r_switch * r_switch * inv_r
               + c12 * (repu_shift_V2 + repu_shift_V3 * r_switch) * r_switch * r_switch * inv_r;
}
/*! Apply force switch, force-only version. */
static __forceinline__ __device__ void calculate_force_switch_F_E(const NBParamGpu nbparam,
                                                                  float            c6,
                                                                  float            c12,
                                                                  float            inv_r,
                                                                  float            r2,
                                                                  float*           F_invr,
                                                                  float*           E_lj)
{
    float r, r_switch;

    /* force switch constants */
    float disp_shift_V2 = nbparam.dispersion_shift.c2;
    float disp_shift_V3 = nbparam.dispersion_shift.c3;
    float repu_shift_V2 = nbparam.repulsion_shift.c2;
    float repu_shift_V3 = nbparam.repulsion_shift.c3;

    float disp_shift_F2 = nbparam.dispersion_shift.c2 / 3;
    float disp_shift_F3 = nbparam.dispersion_shift.c3 / 4;
    float repu_shift_F2 = nbparam.repulsion_shift.c2 / 3;
    float repu_shift_F3 = nbparam.repulsion_shift.c3 / 4;

    r        = r2 * inv_r;
    r_switch = r - nbparam.rvdw_switch;
    r_switch = r_switch >= 0.0F ? r_switch : 0.0F;

    *F_invr += -c6 * (disp_shift_V2 + disp_shift_V3 * r_switch) * r_switch * r_switch * inv_r
               + c12 * (repu_shift_V2 + repu_shift_V3 * r_switch) * r_switch * r_switch * inv_r;
    *E_lj += c6 * (disp_shift_F2 + disp_shift_F3 * r_switch) * r_switch * r_switch * r_switch
             - c12 * (repu_shift_F2 + repu_shift_F3 * r_switch) * r_switch * r_switch * r_switch;
}

static __forceinline__ __device__ void calculate_force_switch_F_E(const NBParamGpu nbparam,
                                                                  float2            c6,
                                                                  float2            c12,
                                                                  float2            inv_r,
                                                                  float2            r2,
                                                                  float2*           F_invr,
                                                                  float2*           E_lj)
{
    float2 r, r_switch;

    /* force switch constants */
    float disp_shift_V2 = nbparam.dispersion_shift.c2;
    float disp_shift_V3 = nbparam.dispersion_shift.c3;
    float repu_shift_V2 = nbparam.repulsion_shift.c2;
    float repu_shift_V3 = nbparam.repulsion_shift.c3;

    float disp_shift_F2 = nbparam.dispersion_shift.c2 / 3;
    float disp_shift_F3 = nbparam.dispersion_shift.c3 / 4;
    float repu_shift_F2 = nbparam.repulsion_shift.c2 / 3;
    float repu_shift_F3 = nbparam.repulsion_shift.c3 / 4;

    r        = r2 * inv_r;
    r_switch = r - nbparam.rvdw_switch;
    r_switch.x = r_switch.x >= 0.0f ? r_switch.x : 0.0f;
    r_switch.y = r_switch.y >= 0.0f ? r_switch.y : 0.0f;

    *F_invr += -c6 * (disp_shift_V2 + disp_shift_V3 * r_switch) * r_switch * r_switch * inv_r
               + c12 * (repu_shift_V2 + repu_shift_V3 * r_switch) * r_switch * r_switch * inv_r;
    *E_lj += c6 * (disp_shift_F2 + disp_shift_F3 * r_switch) * r_switch * r_switch * r_switch
             - c12 * (repu_shift_F2 + repu_shift_F3 * r_switch) * r_switch * r_switch * r_switch;
}
/*! Apply potential switch, force-only version. */
static __forceinline__ __device__ void calculate_potential_switch_F(const NBParamGpu& nbparam,
                                                                    float             inv_r,
                                                                    float             r2,
                                                                    float*            F_invr,
                                                                    const float*      E_lj)
{
    float r, r_switch;
    float sw, dsw;

    /* potential switch constants */
    float switch_V3 = nbparam.vdw_switch.c3;
    float switch_V4 = nbparam.vdw_switch.c4;
    float switch_V5 = nbparam.vdw_switch.c5;
    float switch_F2 = 3 * nbparam.vdw_switch.c3;
    float switch_F3 = 4 * nbparam.vdw_switch.c4;
    float switch_F4 = 5 * nbparam.vdw_switch.c5;

    r        = r2 * inv_r;
    r_switch = r - nbparam.rvdw_switch;

    /* Unlike in the F+E kernel, conditional is faster here */
    if (r_switch > 0.0F)
    {
        sw  = 1.0F + (switch_V3 + (switch_V4 + switch_V5 * r_switch) * r_switch) * r_switch * r_switch * r_switch;
        dsw = (switch_F2 + (switch_F3 + switch_F4 * r_switch) * r_switch) * r_switch * r_switch;

        *F_invr = (*F_invr) * sw - inv_r * (*E_lj) * dsw;
    }
}

/*! Apply potential switch, force + energy version. */
static __forceinline__ __device__ void
calculate_potential_switch_F_E(const NBParamGpu nbparam, float inv_r, float r2, float* F_invr, float* E_lj)
{
    float r, r_switch;
    float sw, dsw;

    /* potential switch constants */
    float switch_V3 = nbparam.vdw_switch.c3;
    float switch_V4 = nbparam.vdw_switch.c4;
    float switch_V5 = nbparam.vdw_switch.c5;
    float switch_F2 = 3 * nbparam.vdw_switch.c3;
    float switch_F3 = 4 * nbparam.vdw_switch.c4;
    float switch_F4 = 5 * nbparam.vdw_switch.c5;

    r        = r2 * inv_r;
    r_switch = r - nbparam.rvdw_switch;
    r_switch = r_switch >= 0.0F ? r_switch : 0.0F;

    /* Unlike in the F-only kernel, masking is faster here */
    sw  = 1.0F + (switch_V3 + (switch_V4 + switch_V5 * r_switch) * r_switch) * r_switch * r_switch * r_switch;
    dsw = (switch_F2 + (switch_F3 + switch_F4 * r_switch) * r_switch) * r_switch * r_switch;

    *F_invr = (*F_invr) * sw - inv_r * (*E_lj) * dsw;
    *E_lj *= sw;
}


/*! \brief Fetch C6 grid contribution coefficients and return the product of these.
 *
 *  Depending on what is supported, it fetches parameters either
 *  using direct load, texture objects, or texrefs.
 */
static __forceinline__ __device__ float calculate_lj_ewald_c6grid(const NBParamGpu nbparam, int typei, int typej)
{
#    if DISABLE_HIP_TEXTURES
    float c6_i = LDG(&nbparam.nbfp_comb[typei]).x;
    float c6_j = LDG(&nbparam.nbfp_comb[typej]).x;
#    else
    float c6_i = tex1Dfetch<float2>(nbparam.nbfp_comb_texobj, typei).x;
    float c6_j = tex1Dfetch<float2>(nbparam.nbfp_comb_texobj, typej).x;
#    endif /* DISABLE_HIP_TEXTURES */
    return c6_i * c6_j;
}


/*! Calculate LJ-PME grid force contribution with
 *  geometric combination rule.
 */
static __forceinline__ __device__ void calculate_lj_ewald_comb_geom_F(const NBParamGpu nbparam,
                                                                      int              typei,
                                                                      int              typej,
                                                                      float            r2,
                                                                      float            inv_r2,
                                                                      float            lje_coeff2,
                                                                      float            lje_coeff6_6,
                                                                      float*           F_invr)
{
    float c6grid, inv_r6_nm, cr2, expmcr2, poly;

    c6grid = calculate_lj_ewald_c6grid(nbparam, typei, typej);

    /* Recalculate inv_r6 without exclusion mask */
    inv_r6_nm = inv_r2 * inv_r2 * inv_r2;
    cr2       = lje_coeff2 * r2;
    expmcr2   = __expf(-cr2);
    poly      = 1.0F + cr2 + 0.5F * cr2 * cr2;

    /* Subtract the grid force from the total LJ force */
    *F_invr += c6grid * (inv_r6_nm - expmcr2 * (inv_r6_nm * poly + lje_coeff6_6)) * inv_r2;
}


/*! Calculate LJ-PME grid force + energy contribution with
 *  geometric combination rule.
 */
static __forceinline__ __device__ void calculate_lj_ewald_comb_geom_F_E(const NBParamGpu nbparam,
                                                                        int              typei,
                                                                        int              typej,
                                                                        float            r2,
                                                                        float            inv_r2,
                                                                        float            lje_coeff2,
                                                                        float  lje_coeff6_6,
                                                                        float  int_bit,
                                                                        float* F_invr,
                                                                        float* E_lj)
{
    float c6grid, inv_r6_nm, cr2, expmcr2, poly, sh_mask;

    c6grid = calculate_lj_ewald_c6grid(nbparam, typei, typej);

    /* Recalculate inv_r6 without exclusion mask */
    inv_r6_nm = inv_r2 * inv_r2 * inv_r2;
    cr2       = lje_coeff2 * r2;
    expmcr2   = __expf(-cr2);
    poly      = 1.0F + cr2 + 0.5F * cr2 * cr2;

    /* Subtract the grid force from the total LJ force */
    *F_invr += c6grid * (inv_r6_nm - expmcr2 * (inv_r6_nm * poly + lje_coeff6_6)) * inv_r2;

    /* Shift should be applied only to real LJ pairs */
    sh_mask = nbparam.sh_lj_ewald * int_bit;
    *E_lj += c_oneSixth * c6grid * (inv_r6_nm * (1.0F - expmcr2 * poly) + sh_mask);
}

/*! Fetch per-type LJ parameters.
 *
 *  Depending on what is supported, it fetches parameters either
 *  using direct load, texture objects, or texrefs.
 */
static __forceinline__ __device__ float2 fetch_nbfp_comb_c6_c12(const NBParamGpu nbparam, int type)
{
#    if DISABLE_HIP_TEXTURES
    return LDG(&nbparam.nbfp_comb[type]);
#    else
    return tex1Dfetch<float2>(nbparam.nbfp_comb_texobj, type);
#    endif /* DISABLE_HIP_TEXTURES */
}


/*! Calculate LJ-PME grid force + energy contribution (if E_lj != nullptr) with
 *  Lorentz-Berthelot combination rule.
 *  We use a single F+E kernel with conditional because the performance impact
 *  of this is pretty small and LB on the CPU is anyway very slow.
 */
static __forceinline__ __device__ void calculate_lj_ewald_comb_LB_F_E(const NBParamGpu nbparam,
                                                                      int              typei,
                                                                      int              typej,
                                                                      float            r2,
                                                                      float            inv_r2,
                                                                      float            lje_coeff2,
                                                                      float            lje_coeff6_6,
                                                                      float            int_bit,
                                                                      float*           F_invr,
                                                                      float*           E_lj)
{
    float c6grid, inv_r6_nm, cr2, expmcr2, poly;
    float sigma, sigma2, epsilon;

    /* sigma and epsilon are scaled to give 6*C6 */
    float2 c6c12_i = fetch_nbfp_comb_c6_c12(nbparam, typei);
    float2 c6c12_j = fetch_nbfp_comb_c6_c12(nbparam, typej);

    sigma   = c6c12_i.x + c6c12_j.x;
    epsilon = c6c12_i.y * c6c12_j.y;

    sigma2 = sigma * sigma;
    c6grid = epsilon * sigma2 * sigma2 * sigma2;

    /* Recalculate inv_r6 without exclusion mask */
    inv_r6_nm = inv_r2 * inv_r2 * inv_r2;
    cr2       = lje_coeff2 * r2;
    expmcr2   = __expf(-cr2);
    poly      = 1.0F + cr2 + 0.5F * cr2 * cr2;

    /* Subtract the grid force from the total LJ force */
    *F_invr += c6grid * (inv_r6_nm - expmcr2 * (inv_r6_nm * poly + lje_coeff6_6)) * inv_r2;

    if (E_lj != nullptr)
    {
        float sh_mask;

        /* Shift should be applied only to real LJ pairs */
        sh_mask = nbparam.sh_lj_ewald * int_bit;
        *E_lj += c_oneSixth * c6grid * (inv_r6_nm * (1.0F - expmcr2 * poly) + sh_mask);
    }
}


/*! Fetch two consecutive values from the Ewald correction F*r table.
 *
 *  Depending on what is supported, it fetches parameters either
 *  using direct load, texture objects, or texrefs.
 */
static __forceinline__ __device__ float2 fetch_coulomb_force_r(const NBParamGpu nbparam, int index)
{
    float2 d;

#    if DISABLE_HIP_TEXTURES
    /* Can't do 8-byte fetch because some of the addresses will be misaligned. */
    d.x = LDG(&nbparam.coulomb_tab[index]);
    d.y = LDG(&nbparam.coulomb_tab[index + 1]);
#    else
    d.x   = tex1Dfetch<float>(nbparam.coulomb_tab_texobj, index);
    d.y   = tex1Dfetch<float>(nbparam.coulomb_tab_texobj, index + 1);
#    endif // DISABLE_HIP_TEXTURES

    return d;
}

/*! Linear interpolation using exactly two FMA operations.
 *
 *  Implements numeric equivalent of: (1-t)*d0 + t*d1
 *  Note that HIP does not have fnms, otherwise we'd use
 *  fma(t, d1, fnms(t, d0, d0)
 *  but input modifiers are designed for this and are fast.
 */
template<typename T>
__forceinline__ __host__ __device__ T lerp(T d0, T d1, T t)
{
    return fma(t, d1, fma(-t, d0, d0));
}

__forceinline__ __device__ float flerp(float d0, float d1, float t)
{
    return __fmaf_rn(t, d1, __fmaf_rn(-t, d0, d0));
}

/*! Interpolate Ewald coulomb force correction using the F*r table.
 */
static __forceinline__ __device__ float interpolate_coulomb_force_r(const NBParamGpu nbparam, float r)
{
    float normalized = nbparam.coulomb_tab_scale * r;
    int   index      = static_cast<int>(normalized);
    float fraction   = normalized - index;

    float2 d01 = fetch_coulomb_force_r(nbparam, index);

    return flerp(d01.x, d01.y, fraction);
}

/*! Fetch C6 and C12 from the parameter table.
 *
 *  Depending on what is supported, it fetches parameters either
 *  using direct load, texture objects, or texrefs.
 */
// NOLINTNEXTLINE(google-runtime-references)
static __forceinline__ __device__ void fetch_nbfp_c6_c12(float& c6, float& c12, const NBParamGpu nbparam, int baseIndex)
{
    float2 c6c12;
#    if DISABLE_HIP_TEXTURES
    c6c12 = LDG(&nbparam.nbfp[baseIndex]);
#    else
    c6c12 = tex1Dfetch<float2>(nbparam.nbfp_texobj, baseIndex);
#    endif // DISABLE_HIP_TEXTURES
    c6  = c6c12.x;
    c12 = c6c12.y;
}


/*! Calculate analytical Ewald correction term. */
static __forceinline__ __device__ float pmecorrF(float z2)
{
    constexpr float FN6 = -1.7357322914161492954e-8f;
    constexpr float FN5 = 1.4703624142580877519e-6f;
    constexpr float FN4 = -0.000053401640219807709149f;
    constexpr float FN3 = 0.0010054721316683106153f;
    constexpr float FN2 = -0.019278317264888380590f;
    constexpr float FN1 = 0.069670166153766424023f;
    constexpr float FN0 = -0.75225204789749321333f;

    constexpr float FD4 = 0.0011193462567257629232f;
    constexpr float FD3 = 0.014866955030185295499f;
    constexpr float FD2 = 0.11583842382862377919f;
    constexpr float FD1 = 0.50736591960530292870f;
    constexpr float FD0 = 1.0f;

    float z4;
    float polyFN0, polyFN1, polyFD0, polyFD1;

    z4 = z2 * z2;

    polyFD0 = FD4 * z4 + FD2;
    polyFD1 = FD3 * z4 + FD1;
    polyFD0 = polyFD0 * z4 + FD0;
    polyFD0 = polyFD1 * z2 + polyFD0;

    polyFD0 = 1.0F / polyFD0;

    polyFN0 = FN6 * z4 + FN4;
    polyFN1 = FN5 * z4 + FN3;
    polyFN0 = polyFN0 * z4 + FN2;
    polyFN1 = polyFN1 * z4 + FN1;
    polyFN0 = polyFN0 * z4 + FN0;
    polyFN0 = polyFN1 * z2 + polyFN0;

    return polyFN0 * polyFD0;
}

/*! Final j-force reduction; this implementation only with power of two
 *  array sizes.
 */
static __forceinline__ __device__ void
reduce_force_j_warp_shfl(float3 f, float3* fout, int tidxi, int aidx)
{
    /*for (int offset = c_clSize >> 1; offset > 0; offset >>= 1)
    {
        f.x += __shfl_down(f.x, offset);
        f.y += __shfl_down(f.y, offset);
        f.z += __shfl_down(f.z, offset);
    }*/

    f.x += warp_move_dpp<float, 0xb1>(f.x);
    f.y += warp_move_dpp<float, 0xb1>(f.y);
    f.z += warp_move_dpp<float, 0xb1>(f.z);

    f.x += warp_move_dpp<float, 0x4e>(f.x);
    f.y += warp_move_dpp<float, 0x4e>(f.y);
    f.z += warp_move_dpp<float, 0x4e>(f.z);

    f.x += warp_move_dpp<float, 0x114>(f.x);
    f.y += warp_move_dpp<float, 0x114>(f.y);
    f.z += warp_move_dpp<float, 0x114>(f.z);

    //if (tidxi == 0)
    if (tidxi == c_clSize - 1)
    {
        atomicAdd((&fout[aidx].x), f.x);
        atomicAdd((&fout[aidx].y), f.y);
        atomicAdd((&fout[aidx].z), f.z);
    }
}

/*! Final i-force reduction; this implementation works only with power of two
 *  array sizes.
 */
static __forceinline__ __device__ void reduce_force_i_warp_shfl(float3             fin,
                                                                float3*            fout,
                                                                float3&            fshift_buf,
                                                                bool               bCalcFshift,
                                                                int                tidxj,
                                                                int                aidx)
{
    #pragma unroll
    for (int offset = warpSize >> 1; offset >= c_clSize; offset >>= 1)
    {
        fin.x += __shfl_down(fin.x, offset);
        fin.y += __shfl_down(fin.y, offset);
        fin.z += __shfl_down(fin.z, offset);
    }

    if (tidxj % (warpSize / c_clSize) == 0)
    {
        atomicAdd((&fout[aidx].x), fin.x);
        atomicAdd((&fout[aidx].y), fin.y);
        atomicAdd((&fout[aidx].z), fin.z);

        if (bCalcFshift)
        {
            fshift_buf.x += fin.x;
            fshift_buf.y += fin.y;
            fshift_buf.z += fin.z;
        }
    }
}

/*! Energy reduction; this implementation works only with power of two
 *  array sizes.
 */
static __forceinline__ __device__ void
reduce_energy_warp_shfl(float E_lj, float E_el, float* e_lj, float* e_el, int tidx)
{
    /*for (int offset = c_subWarp >> 1; offset > 0; offset >>= 1)
    {
        E_lj += __shfl_down(E_lj, offset);
        E_el += __shfl_down(E_el, offset);
    }*/

    if(c_subWarp > 1)
    {
        E_lj += warp_move_dpp<float, 0xb1>(E_lj);
        E_el += warp_move_dpp<float, 0xb1>(E_el);
    }

    if(c_subWarp > 2)
    {
        E_lj += warp_move_dpp<float, 0x4e>(E_lj);
        E_el += warp_move_dpp<float, 0x4e>(E_el);
    }

    if(c_subWarp > 4)
    {
        E_lj += warp_move_dpp<float, 0x114>(E_lj);
        E_el += warp_move_dpp<float, 0x114>(E_el);
    }

    if(c_subWarp > 8)
    {
        E_lj += warp_move_dpp<float, 0x118>(E_lj);
        E_el += warp_move_dpp<float, 0x118>(E_el);
    }

    if(c_subWarp > 16)
    {
#ifndef __gfx1030__
        E_lj += warp_move_dpp<float, 0x142>(E_lj);
        E_el += warp_move_dpp<float, 0x142>(E_el);
#else
        E_lj += __shfl(E_lj, 15, warpSize);
        E_el += __shfl(E_el, 15, warpSize);
#endif
    }

#ifndef __gfx1030__
    if(c_subWarp > 32)
    {

        E_lj += warp_move_dpp<float, 0x143>(E_lj);
        E_el += warp_move_dpp<float, 0x143>(E_el);
    }

    /* The last thread in the subWarp writes the reduced energies */
    if ((tidx & (c_subWarp - 1)) == (c_subWarp - 1))
    {
        atomicAdd(e_lj, E_lj);
        atomicAdd(e_el, E_el);
    }
#else
    if(c_subWarp > 32)
    {
        if ((tidx & (c_subWarp - 1)) == (c_subWarp - 1))
        {
            atomicAdd(e_lj, E_lj);
            atomicAdd(e_el, E_el);
        }
    }
    else
    {
        if ((tidx & (warpSize - 1)) == (warpSize - 1))
        {
            atomicAdd(e_lj, E_lj);
            atomicAdd(e_el, E_el);
        }
    }
    return;
#endif
}

#endif /* NBNXN_HIP_KERNEL_UTILS_HPP */
