#include "hip/hip_runtime.h"
/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2018,2019, by the GROMACS development team, led by
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
 *
 * \brief Implements CUDA bonded functionality
 *
 * \author Jon Vincent <jvincent@nvidia.com>
 * \author Magnus Lundborg <lundborg.magnus@gmail.com>
 * \author Berk Hess <hess@kth.se>
 * \author Szilárd Páll <pall.szilard@gmail.com>
 * \author Alan Gray <alang@nvidia.com>
 * \author Mark Abraham <mark.j.abraham@gmail.com>
 *
 * \ingroup module_listed_forces
 */

#include "gmxpre.h"

#if ! __HIP_DEVICE_COMPILE__
#include <cassert>
#endif

//#include <math_constants.h>
#include <cmath>

#include "gromacs/gpu_utils/cudautils.hip.h"
#include "gromacs/gpu_utils/gpu_vec.hip.h"
#include "gromacs/listed_forces/gpubonded.h"
#include "gromacs/math/units.h"
#include "gromacs/mdlib/force_flags.h"
#include "gromacs/mdtypes/forcerec.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/pbcutil/pbc_aiuc_cuda.cuh"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/gpu_utils/hip_arch_utils.h"
#include "gpubonded_impl.h"

#if defined(_MSVC)
#    include <limits>
#endif

// CUDA threads per block
#ifdef __gfx908__
    #define TPB_BONDED 256
#else
    #define TPB_BONDED 64
#endif

template<typename T>
struct fixed_array
{
    T values[gmx::numFTypesOnGpu];

    fixed_array(T vs[gmx::numFTypesOnGpu]) {
        for (int i = 0; i < gmx::numFTypesOnGpu; i++) {
            values[i] = vs[i];
        }
    }

    __device__ __forceinline__
    T operator[](int index) const {
        // return values[index];

        switch (index) {
            case 0: return values[0];
            case 1: return values[1];
            case 2: return values[2];
            case 3: return values[3];
            case 4: return values[4];
            case 5: return values[5];
            case 6: return values[6];
            default: return values[7];
        }
    }
};

#if ((HIP_VERSION_MAJOR >= 3) && (HIP_VERSION_MINOR > 3)) || (HIP_VERSION_MAJOR >= 4)
    #define hipGlobalAtomicAdd(a, b) atomicAddNoRet(a, b);
    #define hipLocalAtomicAdd(a, b) atomicAddLocalNoRet(a, b);
#else
    #define hipGlobalAtomicAdd(a, b) atomicAdd(a, b);
    #define hipLocalAtomicAdd(a, b) atomicAdd(a, b);
#endif

__device__ __forceinline__ float hipHeadSegmentedSum(float &input, const bool &flag)
{

    uint64_t warp_flags = __ballot(flag);

    warp_flags >>= 1;
    uint32_t lane_id = __lane_id();

    warp_flags &= uint64_t(-1) ^ ((uint64_t(1) << lane_id) - 1U);
    warp_flags >>= (lane_id / warp_size) * warp_size;
    warp_flags |= uint64_t(1) << (warp_size - 1U);
    uint32_t valid_items = __lastbit_u32_u64(warp_flags) + 1U;

    float output = input;
    float value = 0.0f;
    #pragma unroll
    for(unsigned int offset = 1; offset < warp_size; offset *= 2)
    {
        value = __shfl_down(output, offset, warp_size);
        lane_id = __lane_id() & (warp_size - 1);
        if (lane_id + offset < valid_items)
        {
            output += value;
        }
    }
    return output;
}

__device__
void atomicAddLocalNoRet(float* dst, float x)
{
    // atomicAddNoRet(dst, x);
    __asm__ volatile("ds_add_f32 %0, %1" : : "v"((__local float *)dst), "v"(x));
}

const uint32_t cacheline_size_bytes = 128;

/*-------------------------------- HIP kernels -------------------------------- */
/*------------------------------------------------------------------------------*/

#define HIP_DEG2RAD_F (HIP_PI_F / 180.0f)

/*---------------- BONDED CUDA kernels--------------*/

/* Harmonic */
__device__ __forceinline__ static void
           harmonic_gpu(const float kA, const float xA, const float x, float* V, float* F)
{
    constexpr float half = 0.5f;
    float           dx, dx2;

    dx  = x - xA;
    dx2 = dx * dx;

    *F = -kA * dx;
    *V = half * kA * dx2;
}

template<bool calcVir, bool calcEner>
__device__ void bonds_gpu(const int       i,
                          float*          vtot_loc,
                          const int       numBonds,
                          const t_iatom   d_forceatoms[],
                          const t_iparams d_forceparams[],
                          const float4    gm_xq[],
                          fvec            gm_f[],
                          fvec            sm_fShiftLoc[],
                          const PbcAiuc   pbcAiuc)
{
    if (i < numBonds)
    {
        int3 bondData = *(int3*)(d_forceatoms + 3 * i);
        int  type     = bondData.x;
        int  ai       = bondData.y;
        int  aj       = bondData.z;

        /* dx = xi - xj, corrected for periodic boundary conditions. */
        fvec dx;
        int  ki = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[aj], dx);

        float dr2 = iprod_gpu(dx, dx);
        float dr  = sqrt(dr2);

        float vbond;
        float fbond;
        harmonic_gpu(d_forceparams[type].harmonic.krA, d_forceparams[type].harmonic.rA, dr, &vbond, &fbond);

        if (calcEner)
        {
            *vtot_loc += vbond;
        }

        if (dr2 != 0.0f)
        {
            fbond *= rsqrtf(dr2);

#pragma unroll
            for (int m = 0; m < DIM; m++)
            {
                float fij = fbond * dx[m];
                hipGlobalAtomicAdd(&gm_f[ai][m], fij);
                hipGlobalAtomicAdd(&gm_f[aj][m], -fij);
                if (calcVir && ki != CENTRAL)
                {
                    hipLocalAtomicAdd(&sm_fShiftLoc[ki][m], fij);
                    hipLocalAtomicAdd(&sm_fShiftLoc[CENTRAL][m], -fij);
                }
            }
        }
    }
}

template<bool returnShift>
__device__ __forceinline__ static float bond_angle_gpu(const float4   xi,
                                                       const float4   xj,
                                                       const float4   xk,
                                                       const PbcAiuc& pbcAiuc,
                                                       fvec           r_ij,
                                                       fvec           r_kj,
                                                       float*         costh,
                                                       int*           t1,
                                                       int*           t2)
/* Return value is the angle between the bonds i-j and j-k */
{
    *t1 = pbcDxAiuc<returnShift>(pbcAiuc, xi, xj, r_ij);
    *t2 = pbcDxAiuc<returnShift>(pbcAiuc, xk, xj, r_kj);

    *costh   = cos_angle_gpu(r_ij, r_kj);
    float th = acosf(*costh);

    return th;
}

template<bool calcVir, bool calcEner>
__device__ void angles_gpu(const int       i,
                           float*          vtot_loc,
                           const int       numBonds,
                           const t_iatom   d_forceatoms[],
                           const t_iparams d_forceparams[],
                           const float4    gm_xq[],
                           fvec            gm_f[],
                           fvec            sm_fShiftLoc[],
                           const PbcAiuc   pbcAiuc)
{
    if (i < numBonds)
    {
        int4 angleData = *(int4*)(d_forceatoms + 4 * i);
        int  type      = angleData.x;
        int  ai        = angleData.y;
        int  aj        = angleData.z;
        int  ak        = angleData.w;

        fvec  r_ij;
        fvec  r_kj;
        float cos_theta;
        int   t1;
        int   t2;
        float theta = bond_angle_gpu<calcVir>(gm_xq[ai], gm_xq[aj], gm_xq[ak], pbcAiuc, r_ij, r_kj,
                                              &cos_theta, &t1, &t2);

        float va;
        float dVdt;
        harmonic_gpu(d_forceparams[type].harmonic.krA,
                     d_forceparams[type].harmonic.rA * HIP_DEG2RAD_F, theta, &va, &dVdt);

        if (calcEner)
        {
            *vtot_loc += va;
        }

        float cos_theta2 = cos_theta * cos_theta;
        if (cos_theta2 < 1.0f)
        {
            float st    = dVdt * rsqrtf(1.0f - cos_theta2);
            float sth   = st * cos_theta;
            float nrij2 = iprod_gpu(r_ij, r_ij);
            float nrkj2 = iprod_gpu(r_kj, r_kj);

            float nrij_1 = rsqrtf(nrij2);
            float nrkj_1 = rsqrtf(nrkj2);

            float cik = st * nrij_1 * nrkj_1;
            float cii = sth * nrij_1 * nrij_1;
            float ckk = sth * nrkj_1 * nrkj_1;

            fvec f_i;
            fvec f_k;
            fvec f_j;
#pragma unroll
            for (int m = 0; m < DIM; m++)
            {
                f_i[m] = -(cik * r_kj[m] - cii * r_ij[m]);
                f_k[m] = -(cik * r_ij[m] - ckk * r_kj[m]);
                f_j[m] = -f_i[m] - f_k[m];
                hipGlobalAtomicAdd(&gm_f[ai][m], f_i[m]);
                hipGlobalAtomicAdd(&gm_f[aj][m], f_j[m]);
                hipGlobalAtomicAdd(&gm_f[ak][m], f_k[m]);
                if (calcVir)
                {
                    hipLocalAtomicAdd(&sm_fShiftLoc[t1][m], f_i[m]);
                    hipLocalAtomicAdd(&sm_fShiftLoc[CENTRAL][m], f_j[m]);
                    hipLocalAtomicAdd(&sm_fShiftLoc[t2][m], f_k[m]);
                }
            }
        }
    }
}

template<bool calcVir, bool calcEner>
__device__ void urey_bradley_gpu(const int       i,
                                 float*          vtot_loc,
                                 const int       numBonds,
                                 const t_iatom   d_forceatoms[],
                                 const t_iparams d_forceparams[],
                                 const float4    gm_xq[],
                                 fvec            gm_f[],
                                 fvec            sm_fShiftLoc[],
                                 const PbcAiuc   pbcAiuc)
{
    if (i < numBonds)
    {
        int4 ubData = *(int4*)(d_forceatoms + 4 * i);
        int  type   = ubData.x;
        int  ai     = ubData.y;
        int  aj     = ubData.z;
        int  ak     = ubData.w;

        float th0A = d_forceparams[type].u_b.thetaA * HIP_DEG2RAD_F;
        float kthA = d_forceparams[type].u_b.kthetaA;
        float r13A = d_forceparams[type].u_b.r13A;
        float kUBA = d_forceparams[type].u_b.kUBA;

        fvec  r_ij;
        fvec  r_kj;
        float cos_theta;
        int   t1;
        int   t2;
        float theta = bond_angle_gpu<calcVir>(gm_xq[ai], gm_xq[aj], gm_xq[ak], pbcAiuc, r_ij, r_kj,
                                              &cos_theta, &t1, &t2);

        float va;
        float dVdt;
        harmonic_gpu(kthA, th0A, theta, &va, &dVdt);

        if (calcEner)
        {
            *vtot_loc += va;
        }

        fvec r_ik;
        int  ki = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[ak], r_ik);

        float dr2 = iprod_gpu(r_ik, r_ik);
        float dr  = dr2 * rsqrtf(dr2);

        float vbond;
        float fbond;
        harmonic_gpu(kUBA, r13A, dr, &vbond, &fbond);

        float cos_theta2 = cos_theta * cos_theta;
        if (cos_theta2 < 1.0f)
        {
            float st  = dVdt * rsqrtf(1.0f - cos_theta2);
            float sth = st * cos_theta;

            float nrkj2 = iprod_gpu(r_kj, r_kj);
            float nrij2 = iprod_gpu(r_ij, r_ij);

            float cik = st * rsqrtf(nrkj2 * nrij2);
            float cii = sth / nrij2;
            float ckk = sth / nrkj2;

            fvec f_i;
            fvec f_j;
            fvec f_k;
#pragma unroll
            for (int m = 0; m < DIM; m++)
            {
                f_i[m] = -(cik * r_kj[m] - cii * r_ij[m]);
                f_k[m] = -(cik * r_ij[m] - ckk * r_kj[m]);
                f_j[m] = -f_i[m] - f_k[m];

                hipGlobalAtomicAdd(&gm_f[ai][m], f_i[m]);
                hipGlobalAtomicAdd(&gm_f[aj][m], f_j[m]);
                hipGlobalAtomicAdd(&gm_f[ak][m], f_k[m]);
                if (calcVir)
                {
                    hipLocalAtomicAdd(&sm_fShiftLoc[t1][m], f_i[m]);
                    hipLocalAtomicAdd(&sm_fShiftLoc[CENTRAL][m], f_j[m]);
                    hipLocalAtomicAdd(&sm_fShiftLoc[t2][m], f_k[m]);
                }
            }
        }

        /* Time for the bond calculations */
        if (dr2 != 0.0f)
        {
            if (calcEner)
            {
                *vtot_loc += vbond;
            }

            fbond *= rsqrtf(dr2);

#pragma unroll
            for (int m = 0; m < DIM; m++)
            {
                float fik = fbond * r_ik[m];
                hipGlobalAtomicAdd(&gm_f[ai][m], fik);
                hipGlobalAtomicAdd(&gm_f[ak][m], -fik);

                if (calcVir && ki != CENTRAL)
                {
                    hipLocalAtomicAdd(&sm_fShiftLoc[ki][m], fik);
                    hipLocalAtomicAdd(&sm_fShiftLoc[CENTRAL][m], -fik);
                }
            }
        }
    }
}

template<bool returnShift, typename T>
__device__ __forceinline__ static float dih_angle_gpu(const T        xi,
                                                      const T        xj,
                                                      const T        xk,
                                                      const T        xl,
                                                      const PbcAiuc& pbcAiuc,
                                                      fvec           r_ij,
                                                      fvec           r_kj,
                                                      fvec           r_kl,
                                                      fvec           m,
                                                      fvec           n,
                                                      int*           t1,
                                                      int*           t2,
                                                      int*           t3)
{
    *t1 = pbcDxAiuc<returnShift>(pbcAiuc, xi, xj, r_ij);
    *t2 = pbcDxAiuc<returnShift>(pbcAiuc, xk, xj, r_kj);
    *t3 = pbcDxAiuc<returnShift>(pbcAiuc, xk, xl, r_kl);

    cprod_gpu(r_ij, r_kj, m);
    cprod_gpu(r_kj, r_kl, n);
    float phi  = gmx_angle_gpu(m, n);
    float ipr  = iprod_gpu(r_ij, n);
    float sign = (ipr < 0.0f) ? -1.0f : 1.0f;
    phi        = sign * phi;

    return phi;
}


__device__ __forceinline__ static void
           dopdihs_gpu(const float cpA, const float phiA, const int mult, const float phi, float* v, float* f)
{
    float mdphi, sdphi;

    mdphi = mult * phi - phiA * HIP_DEG2RAD_F;
    sdphi = sinf(mdphi);
    *v    = cpA * (1.0f + cosf(mdphi));
    *f    = -cpA * mult * sdphi;
}

template<bool calcVir>
__device__ static void do_dih_fup_gpu(const int      i,
                                      const int      j,
                                      const int      k,
                                      const int      l,
                                      const float    ddphi,
                                      const fvec     r_ij,
                                      const fvec     r_kj,
                                      const fvec     r_kl,
                                      const fvec     m,
                                      const fvec     n,
                                      fvec           gm_f[],
                                      fvec           sm_fShiftLoc[],
                                      const PbcAiuc& pbcAiuc,
                                      const float4   gm_xq[],
                                      const int      t1,
                                      const int      t2,
                                      const int gmx_unused t3)
{
    float iprm  = iprod_gpu(m, m);
    float iprn  = iprod_gpu(n, n);
    float nrkj2 = iprod_gpu(r_kj, r_kj);
    float toler = nrkj2 * GMX_REAL_EPS;
    if ((iprm > toler) && (iprn > toler))
    {
        float nrkj_1 = rsqrtf(nrkj2); // replacing std::invsqrt call
        float nrkj_2 = nrkj_1 * nrkj_1;
        float nrkj   = nrkj2 * nrkj_1;
        float a      = -ddphi * nrkj / iprm;
        fvec  f_i;
        svmul_gpu(a, m, f_i);
        float b = ddphi * nrkj / iprn;
        fvec  f_l;
        svmul_gpu(b, n, f_l);
        float p = iprod_gpu(r_ij, r_kj);
        p *= nrkj_2;
        float q = iprod_gpu(r_kl, r_kj);
        q *= nrkj_2;
        fvec uvec;
        svmul_gpu(p, f_i, uvec);
        fvec vvec;
        svmul_gpu(q, f_l, vvec);
        fvec svec;
        fvec_sub_gpu(uvec, vvec, svec);
        fvec f_j;
        fvec_sub_gpu(f_i, svec, f_j);
        fvec f_k;
        fvec_add_gpu(f_l, svec, f_k);


        unsigned long long int b_ = __ballot(1);
        const int prev_lane_i = __shfl_up(i, 1);
        const int prev_lane_j = __shfl_up(j, 1);
        const int prev_lane_k = __shfl_up(k, 1);
        const int prev_lane_l = __shfl_up(l, 1);
        const bool headi = threadIdx.x % warp_size == 0 || i != prev_lane_i;
        const bool headj = threadIdx.x % warp_size == 0 || j != prev_lane_j;
        const bool headk = threadIdx.x % warp_size == 0 || k != prev_lane_k;
        const bool headl = threadIdx.x % warp_size == 0 || l != prev_lane_l;
        #pragma unroll
        for (int m = 0; (m < DIM); m++)
        {
            if (b_ == ~(unsigned long long int)0)
            {
                const float sumfi = hipHeadSegmentedSum(f_i[m], headi);
                const float sumfj = hipHeadSegmentedSum(f_j[m], headj);
                const float sumfk = hipHeadSegmentedSum(f_k[m], headk);
                const float sumfl = hipHeadSegmentedSum(f_l[m], headl);

                if (headi)
                {
                    hipGlobalAtomicAdd(&gm_f[i][m], sumfi);
                }

                if (headj)
                {
                    hipGlobalAtomicAdd(&gm_f[j][m], -sumfj);
                }

                if (headk)
                {
                    hipGlobalAtomicAdd(&gm_f[k][m], -sumfk);
                }

                if (headl)
                {
                    hipGlobalAtomicAdd(&gm_f[l][m], sumfl);
                }
            }
            else
            {
                hipGlobalAtomicAdd(&gm_f[i][m], f_i[m]);
                hipGlobalAtomicAdd(&gm_f[j][m], -f_j[m]);
                hipGlobalAtomicAdd(&gm_f[k][m], -f_k[m]);
                hipGlobalAtomicAdd(&gm_f[l][m], f_l[m]);
            }
        }


        if (calcVir)
        {
            fvec dx_jl;
            int  t3 = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[l], gm_xq[j], dx_jl);

#pragma unroll
            for (int m = 0; (m < DIM); m++)
            {
                hipLocalAtomicAdd(&sm_fShiftLoc[t1][m], f_i[m]);
                hipLocalAtomicAdd(&sm_fShiftLoc[CENTRAL][m], -f_j[m]);
                hipLocalAtomicAdd(&sm_fShiftLoc[t2][m], -f_k[m]);
                hipLocalAtomicAdd(&sm_fShiftLoc[t3][m], f_l[m]);
            }
        }
    }
}

template<bool calcVir, bool calcEner>
__device__ void pdihs_gpu(const int       i,
                          float*          vtot_loc,
                          const int       numBonds,
                          const t_iatom   d_forceatoms[],
                          const t_iparams d_forceparams[],
                          const float4    gm_xq[],
                          fvec            gm_f[],
                          fvec            sm_fShiftLoc[],
                          const PbcAiuc   pbcAiuc)
{
    if (i < numBonds)
    {
        int type = d_forceatoms[5 * i];
        int ai   = d_forceatoms[5 * i + 1];
        int aj   = d_forceatoms[5 * i + 2];
        int ak   = d_forceatoms[5 * i + 3];
        int al   = d_forceatoms[5 * i + 4];

        fvec  r_ij;
        fvec  r_kj;
        fvec  r_kl;
        fvec  m;
        fvec  n;
        int   t1;
        int   t2;
        int   t3;
        float phi = dih_angle_gpu<calcVir>(gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc,
                                           r_ij, r_kj, r_kl, m, n, &t1, &t2, &t3);

        float vpd;
        float ddphi;
        dopdihs_gpu(d_forceparams[type].pdihs.cpA, d_forceparams[type].pdihs.phiA,
                    d_forceparams[type].pdihs.mult, phi, &vpd, &ddphi);

        if (calcEner)
        {
            *vtot_loc += vpd;
        }

        do_dih_fup_gpu<calcVir>(ai, aj, ak, al, ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc,
                                pbcAiuc, gm_xq, t1, t2, t3);
    }
}

template<bool calcVir, bool calcEner>
__device__ void rbdihs_gpu(const int       i,
                           float*          vtot_loc,
                           const int       numBonds,
                           const t_iatom   d_forceatoms[],
                           const t_iparams d_forceparams[],
                           const float4    gm_xq[],
                           fvec            gm_f[],
                           fvec            sm_fShiftLoc[],
                           const PbcAiuc   pbcAiuc)
{
    constexpr float c0 = 0.0f, c1 = 1.0f, c2 = 2.0f, c3 = 3.0f, c4 = 4.0f, c5 = 5.0f;

    if (i < numBonds)
    {
        int type = d_forceatoms[5 * i];
        int ai   = d_forceatoms[5 * i + 1];
        int aj   = d_forceatoms[5 * i + 2];
        int ak   = d_forceatoms[5 * i + 3];
        int al   = d_forceatoms[5 * i + 4];

        fvec  r_ij;
        fvec  r_kj;
        fvec  r_kl;
        fvec  m;
        fvec  n;
        int   t1;
        int   t2;
        int   t3;
        float phi = dih_angle_gpu<calcVir>(gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc,
                                           r_ij, r_kj, r_kl, m, n, &t1, &t2, &t3);

        /* Change to polymer convention */
        if (phi < c0)
        {
            phi += HIP_PI_F;
        }
        else
        {
            phi -= HIP_PI_F;
        }
        float cos_phi = cosf(phi);
        /* Beware of accuracy loss, cannot use 1-sqrt(cos^2) ! */
        float sin_phi = sinf(phi);

        float parm[NR_RBDIHS];
        for (int j = 0; j < NR_RBDIHS; j++)
        {
            parm[j] = d_forceparams[type].rbdihs.rbcA[j];
        }
        /* Calculate cosine powers */
        /* Calculate the energy */
        /* Calculate the derivative */
        float v      = parm[0];
        float ddphi  = c0;
        float cosfac = c1;

        float rbp = parm[1];
        ddphi += rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[2];
        ddphi += c2 * rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[3];
        ddphi += c3 * rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[4];
        ddphi += c4 * rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
        }
        rbp = parm[5];
        ddphi += c5 * rbp * cosfac;
        cosfac *= cos_phi;
        if (calcEner)
        {
            v += cosfac * rbp;
        }

        ddphi = -ddphi * sin_phi;

        do_dih_fup_gpu<calcVir>(ai, aj, ak, al, ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc,
                                pbcAiuc, gm_xq, t1, t2, t3);
        if (calcEner)
        {
            *vtot_loc += v;
        }
    }
}

__device__ __forceinline__ static void make_dp_periodic_gpu(float* dp)
{
    /* dp cannot be outside (-pi,pi) */
    if (*dp >= HIP_PI_F)
    {
        *dp -= 2.0f * HIP_PI_F;
    }
    else if (*dp < -HIP_PI_F)
    {
        *dp += 2.0f * HIP_PI_F;
    }
}

template<bool calcVir, bool calcEner>
__device__ void idihs_gpu(const int       i,
                          float*          vtot_loc,
                          const int       numBonds,
                          const t_iatom   d_forceatoms[],
                          const t_iparams d_forceparams[],
                          const float4    gm_xq[],
                          fvec            gm_f[],
                          fvec            sm_fShiftLoc[],
                          const PbcAiuc   pbcAiuc)
{
    if (i < numBonds)
    {
        int type = d_forceatoms[5 * i];
        int ai   = d_forceatoms[5 * i + 1];
        int aj   = d_forceatoms[5 * i + 2];
        int ak   = d_forceatoms[5 * i + 3];
        int al   = d_forceatoms[5 * i + 4];

        fvec  r_ij;
        fvec  r_kj;
        fvec  r_kl;
        fvec  m;
        fvec  n;
        int   t1;
        int   t2;
        int   t3;
        float phi = dih_angle_gpu<calcVir>(gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc,
                                           r_ij, r_kj, r_kl, m, n, &t1, &t2, &t3);

        /* phi can jump if phi0 is close to Pi/-Pi, which will cause huge
         * force changes if we just apply a normal harmonic.
         * Instead, we first calculate phi-phi0 and take it modulo (-Pi,Pi).
         * This means we will never have the periodicity problem, unless
         * the dihedral is Pi away from phiO, which is very unlikely due to
         * the potential.
         */
        float kA = d_forceparams[type].harmonic.krA;
        float pA = d_forceparams[type].harmonic.rA;

        float phi0 = pA * HIP_DEG2RAD_F;

        float dp = phi - phi0;

        make_dp_periodic_gpu(&dp);

        float ddphi = -kA * dp;

        do_dih_fup_gpu<calcVir>(ai, aj, ak, al, -ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc,
                                pbcAiuc, gm_xq, t1, t2, t3);

        if (calcEner)
        {
            *vtot_loc += -0.5f * ddphi * dp;
        }
    }
}

template<bool calcVir, bool calcEner>
__device__ void pairs_gpu(const int       i,
                          const int       numBonds,
                          const t_iatom   d_forceatoms[],
                          const t_iparams iparams[],
                          const float4    gm_xq[],
                          fvec            gm_f[],
                          fvec            sm_fShiftLoc[],
                          const PbcAiuc   pbcAiuc,
                          const float     scale_factor,
                          float*          vtotVdw_loc,
                          float*          vtotElec_loc)
{
    if (i < numBonds)
    {
        int3 pairData = *(int3*)(d_forceatoms + 3 * i);
        int  type     = pairData.x;
        int  ai       = pairData.y;
        int  aj       = pairData.z;

        float qq  = gm_xq[ai].w * gm_xq[aj].w;
        float c6  = iparams[type].lj14.c6A;
        float c12 = iparams[type].lj14.c12A;

        /* Do we need to apply full periodic boundary conditions? */
        fvec dr;
        int  fshift_index = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[aj], dr);

        float r2    = norm2_gpu(dr);
        float rinv  = rsqrtf(r2);
        float rinv3 = rinv * rinv;
        float rinv6 = rinv3 * rinv3 * rinv3;

        /* Calculate the Coulomb force * r */
        float velec = scale_factor * qq * rinv;

        /* Calculate the LJ force * r and add it to the Coulomb part */
        float fr = (12.0f * c12 * rinv6 - 6.0f * c6) * rinv6 + velec;

        float finvr = fr * rinv3;
        fvec  f;
        svmul_gpu(finvr, dr, f);

        /* Add the forces */
#pragma unroll
        for (int m = 0; m < DIM; m++)
        {
            hipGlobalAtomicAdd(&gm_f[ai][m], f[m]);
            hipGlobalAtomicAdd(&gm_f[aj][m], -f[m]);
            if (calcVir && fshift_index != CENTRAL)
            {
                hipLocalAtomicAdd(&sm_fShiftLoc[fshift_index][m], f[m]);
                hipLocalAtomicAdd(&sm_fShiftLoc[CENTRAL][m], -f[m]);
            }
        }

        if (calcEner)
        {
            *vtotVdw_loc += (c12 * rinv6 - c6) * rinv6;
            *vtotElec_loc += velec;
        }
    }
}

namespace gmx
{
    template<bool calcVir, bool calcEner>
    __global__ void exec_kernel_gpu(
        //! Periodic boundary data
        PbcAiuc pbcAiuc,
        //! Scale factor
        float scaleFactor,
        //! The bonded types on GPU
        const fixed_array<int> fTypesOnGpu,
        //! The number of interaction atom (iatom) elements for every function type
        const fixed_array<int> numFTypeIAtoms,
        //! The number of bonds for every function type
        const fixed_array<int> numFTypeBonds,
        //! The start index in the range of each interaction type
        const fixed_array<int> fTypeRangeStart,
        //! The end index in the range of each interaction type
        const fixed_array<int> fTypeRangeEnd,
        //! Force parameters (on GPU)
        t_iparams* d_forceParams,
        //! Coordinates before the timestep (on GPU)
        const float4* d_xq,
        //! Forces on atoms (on GPU)
        fvec* d_f,
        //! Force shifts on atoms (on GPU)
        fvec* d_fShift,
        //! Total Energy (on GPU)
        float* d_vTot,
        //! Interaction list atoms (on GPU)
        const fixed_array<t_iatom*> d_iatoms)

    {
        assert(blockDim.y == 1 && blockDim.z == 1);
        const int  tid          = blockIdx.x * blockDim.x + threadIdx.x;
        float      vtot_loc     = 0;
        float      vtotVdw_loc  = 0;
        float      vtotElec_loc = 0;
        __shared__ fvec sm_fShiftLoc[SHIFTS];

        // two extra elements, one for F_LJ14 and one for F_COUL14
        constexpr uint32_t atomic_shared_mem_size = numFTypesOnGpu + 2;
        constexpr uint32_t F_LJ14_shared_index    = numFTypesOnGpu;
        constexpr uint32_t F_COUL14_shared_index  = numFTypesOnGpu + 1;

        // __shared__ float d_vTot_shared[atomic_shared_mem_size];
        // if (calcEner)
        // {
        //     if(threadIdx.x < atomic_shared_mem_size)
        //     {
        //         d_vTot_shared[threadIdx.x] = 0.0f;
        //     }
        //     __syncthreads();
        // }

        if (calcVir)
        {
            if (threadIdx.x < SHIFTS)
            {
                sm_fShiftLoc[threadIdx.x][XX] = 0.0f;
                sm_fShiftLoc[threadIdx.x][YY] = 0.0f;
                sm_fShiftLoc[threadIdx.x][ZZ] = 0.0f;
            }
            __syncthreads();
        }

        int  fType;
        int fType_shared_index = -1;
    #pragma unroll
        for (int j = 0; j < numFTypesOnGpu; j++)
        {
            if (tid >= fTypeRangeStart[j] && tid <= fTypeRangeEnd[j])
            {
                const int      numBonds = numFTypeBonds[j];
                int            fTypeTid = tid - fTypeRangeStart[j];
                const t_iatom* iatoms   = d_iatoms[j];
                fType                   = fTypesOnGpu[j];
                fType_shared_index      = j;

                switch (fType)
                {
                    case F_BONDS:
                        bonds_gpu<calcVir, calcEner>(fTypeTid, &vtot_loc, numBonds, iatoms,
                                                     d_forceParams, d_xq,
                                                     d_f, sm_fShiftLoc, pbcAiuc);
                        break;
                    case F_ANGLES:
                        angles_gpu<calcVir, calcEner>(
                                fTypeTid, &vtot_loc, numBonds, iatoms, d_forceParams,
                                d_xq, d_f, sm_fShiftLoc, pbcAiuc);
                        break;
                    case F_UREY_BRADLEY:
                        urey_bradley_gpu<calcVir, calcEner>(
                                fTypeTid, &vtot_loc, numBonds, iatoms, d_forceParams,
                                d_xq, d_f, sm_fShiftLoc, pbcAiuc);
                        break;
                    case F_PDIHS:
                    case F_PIDIHS:
                        pdihs_gpu<calcVir, calcEner>(fTypeTid, &vtot_loc, numBonds, iatoms,
                                                     d_forceParams, d_xq,
                                                     d_f, sm_fShiftLoc, pbcAiuc);
                        break;
                    case F_RBDIHS:
                        rbdihs_gpu<calcVir, calcEner>(
                                fTypeTid, &vtot_loc, numBonds, iatoms, d_forceParams,
                                d_xq, d_f, sm_fShiftLoc, pbcAiuc);
                        break;
                    case F_IDIHS:
                        idihs_gpu<calcVir, calcEner>(fTypeTid, &vtot_loc, numBonds, iatoms,
                                                     d_forceParams, d_xq,
                                                     d_f, sm_fShiftLoc, pbcAiuc);
                        break;
                    case F_LJ14:
                        pairs_gpu<calcVir, calcEner>(fTypeTid, numBonds, iatoms, d_forceParams,
                                                     d_xq, d_f, sm_fShiftLoc,
                                                     pbcAiuc, scaleFactor,
                                                     &vtotVdw_loc, &vtotElec_loc);
                        break;
                }
                break;
            }
        }

        if (calcEner)
        {
            #pragma unroll
            for (int j = 0; j < numFTypesOnGpu; j++)
            {
                if (__any(j == fType_shared_index))
                {
                    float vtot_shuffle = j == fType_shared_index ? vtot_loc : 0.0f;
                    #pragma unroll
                    for (unsigned int offset = (warp_size >> 1); offset > 0; offset >>= 1)
                    {
                        vtot_shuffle += __shfl_down(vtot_shuffle, offset);
                    }
                    if(threadIdx.x % warp_size == 0)
                    {
                        fType = fTypesOnGpu[j];
                        hipGlobalAtomicAdd((d_vTot + fType), vtot_shuffle);
                    }
                }
            }

            float vtotVdw_shuffle = vtotVdw_loc;
            float vtotElec_shuffle = vtotElec_loc;
            #pragma unroll
            for (unsigned int offset = (warp_size >> 1); offset > 0; offset >>= 1)
            {
                vtotVdw_shuffle += __shfl_down(vtotVdw_shuffle, offset);
                vtotElec_shuffle += __shfl_down(vtotElec_shuffle, offset);
            }
            if(threadIdx.x % warp_size == 0)
            {
                hipGlobalAtomicAdd(d_vTot + F_LJ14, vtotVdw_shuffle);
                hipGlobalAtomicAdd(d_vTot + F_COUL14, vtotElec_shuffle);
            }

        }

        /* Accumulate shift vectors from shared memory to global memory on the first SHIFTS threads of the block. */
        if (calcVir)
        {
            __syncthreads();
            if (threadIdx.x < SHIFTS)
            {
                fvec_inc_atomic(d_fShift[threadIdx.x], sm_fShiftLoc[threadIdx.x]);
            }
        }
    }

/*-------------------------------- End CUDA kernels-----------------------------*/


template<bool calcVir, bool calcEner>
void GpuBonded::Impl::launchKernel(const t_forcerec* fr, const matrix box)
{
    GMX_ASSERT(haveInteractions_,
               "Cannot launch bonded GPU kernels unless bonded GPU work was scheduled");
    static_assert(TPB_BONDED >= SHIFTS,
                  "TPB_BONDED must be >= SHIFTS for the virial kernel (calcVir=true)");

    PbcAiuc pbcAiuc;
    setPbcAiuc(fr->bMolPBC ? ePBC2npbcdim(fr->ePBC) : 0, box, &pbcAiuc);

    int fTypeRangeEnd = kernelParams_.fTypeRangeEnd[numFTypesOnGpu - 1];

    if (fTypeRangeEnd < 0)
    {
        return;
    }

    KernelLaunchConfig config;
    config.blockSize[0] = TPB_BONDED;
    config.blockSize[1] = 1;
    config.blockSize[2] = 1;
    config.gridSize[0]  = (fTypeRangeEnd + TPB_BONDED) / TPB_BONDED;
    config.gridSize[1]  = 1;
    config.gridSize[2]  = 1;
    config.stream       = stream_;

    auto kernelPtr            = exec_kernel_gpu<calcVir, calcEner>;
    kernelParams_.scaleFactor = fr->ic->epsfac * fr->fudgeQQ;
    kernelParams_.pbcAiuc     = pbcAiuc;

    //const auto kernelArgs = prepareGpuKernelArguments(kernelPtr, config, &kernelParams_);

    //launchGpuKernel(kernelPtr, config, nullptr, "exec_kernel_gpu<calcVir, calcEner>", kernelArgs);
    //launchGpuKernel(kernelPtr, config, nullptr, , kernelParams_);

    dim3 blockSize(config.blockSize[0], config.blockSize[1], config.blockSize[2]);
    dim3 gridSize(config.gridSize[0], config.gridSize[1], config.gridSize[2]);
 //   hipLaunchKernel((void*)kernel, gridSize, blockSize, const_cast<void**>(kernelArgs.data()),
 //                    config.sharedMemorySize, config.stream);
    hipLaunchKernelGGL(kernelPtr,
        gridSize,
        blockSize,
        config.sharedMemorySize,
        config.stream,
        kernelParams_.pbcAiuc,
        kernelParams_.scaleFactor,
        fixed_array<int>(kernelParams_.fTypesOnGpu),
        fixed_array<int>(kernelParams_.numFTypeIAtoms),
        fixed_array<int>(kernelParams_.numFTypeBonds),
        fixed_array<int>(kernelParams_.fTypeRangeStart),
        fixed_array<int>(kernelParams_.fTypeRangeEnd),
        kernelParams_.d_forceParams,
        kernelParams_.d_xq,
        kernelParams_.d_f,
        kernelParams_.d_fShift,
        kernelParams_.d_vTot,
        fixed_array<t_iatom*>(kernelParams_.d_iatoms)
    );

    hipError_t status = hipGetLastError();
    if (hipSuccess != status)
    {
        const std::string errorMessage =
                "GPU kernel (" + std::string("exec_kernel_gpu<calcVir, calcEner>")
                + ") failed to launch: " + std::string(hipGetErrorString(status));
        GMX_THROW(gmx::InternalError(errorMessage));
    }

}

void GpuBonded::launchKernel(const t_forcerec* fr, const gmx::StepWorkload& stepWork, const matrix box)
{
    if (stepWork.computeEnergy)
    {
        // When we need the energy, we also need the virial
        impl_->launchKernel<true, true>(fr, box);
    }
    else if (stepWork.computeVirial)
    {
        impl_->launchKernel<true, false>(fr, box);
    }
    else
    {
        impl_->launchKernel<false, false>(fr, box);
    }
}

} // namespace gmx
