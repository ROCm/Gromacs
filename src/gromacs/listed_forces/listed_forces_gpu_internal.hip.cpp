#include "hip/hip_runtime.h"
/*
 * This file is part of the GROMACS molecular simulation package.
 *
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
 *
 * \brief Implements HIP bonded functionality
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

#include <cassert>

#include "gromacs/math/math_constants.h"
#include "gromacs/gpu_utils/hiputils.hpp"
#include "gromacs/gpu_utils/typecasts.hpp"
#include "gromacs/gpu_utils/vectype_ops.hpp"
#include "gromacs/listed_forces/listed_forces_gpu.h"
#include "gromacs/math/units.h"
#include "gromacs/mdlib/force_flags.h"
#include "gromacs/mdtypes/interaction_const.h"
#include "gromacs/mdtypes/simulation_workload.h"
#include "gromacs/pbcutil/pbc_aiuc_hip.hpp"
#include "gromacs/timing/wallcycle.h"
#include "gromacs/utility/gmxassert.h"

#include "listed_forces_gpu_impl.h"

#if defined(_MSVC)
#    include <limits>
#endif

/*-------------------------------- HIP kernels-------------------------------- */
/*------------------------------------------------------------------------------*/

#define HIP_DEG2RAD_F (HIPRT_PI_F / 180.0F)

/*---------------- BONDED HIP kernels--------------*/

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
    warp_flags >>= (lane_id / warpSize) * warpSize;
    warp_flags |= uint64_t(1) << (warpSize - 1U);
    uint32_t valid_items = __lastbit_u32_u64(warp_flags) + 1U;

    float output = input;
    float value = 0.0f;
    #pragma unroll
    for(unsigned int offset = 1; offset < warpSize; offset *= 2)
    {
        value = __shfl_down(output, offset, warpSize);
        lane_id = __lane_id() & (warpSize - 1);
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

/* Harmonic */
__device__ __forceinline__ static void
harmonic_gpu(const float kA, const float xA, const float x, float* V, float* F)
{
    constexpr float half = 0.5F;
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
                          float3          gm_f[],
                          float3          sm_fShiftLoc[],
                          const PbcAiuc   pbcAiuc)
{
    if (i < numBonds)
    {
        const int3 bondData = *(reinterpret_cast<const int3*>(d_forceatoms + 3 * i));
        int        type     = bondData.x;
        int        ai       = bondData.y;
        int        aj       = bondData.z;

        /* dx = xi - xj, corrected for periodic boundary conditions. */
        float3 dx;
        int    ki = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[aj], dx);

        float dr2 = norm2(dx);
        float dr  = sqrt(dr2);

        float vbond;
        float fbond;
        harmonic_gpu(d_forceparams[type].harmonic.krA, d_forceparams[type].harmonic.rA, dr, &vbond, &fbond);

        if (calcEner)
        {
            *vtot_loc += vbond;
        }

        if (dr2 != 0.0F)
        {
            fbond *= __frsqrt_rn(dr2);

            float3 fij = fbond * dx;
            atomicAdd(&gm_f[ai], fij);
            atomicAdd(&gm_f[aj], -fij);
            if (calcVir && ki != gmx::c_centralShiftIndex)
            {
                atomicAdd(&sm_fShiftLoc[ki], fij);
                atomicAdd(&sm_fShiftLoc[gmx::c_centralShiftIndex], -fij);
            }
        }
    }
}

template<bool returnShift>
__device__ __forceinline__ static float bond_angle_gpu(const float4   xi,
                                                       const float4   xj,
                                                       const float4   xk,
                                                       const PbcAiuc& pbcAiuc,
                                                       float3*        r_ij,
                                                       float3*        r_kj,
                                                       float*         costh,
                                                       int*           t1,
                                                       int*           t2)
/* Return value is the angle between the bonds i-j and j-k */
{
    *t1 = pbcDxAiuc<returnShift>(pbcAiuc, xi, xj, *r_ij);
    *t2 = pbcDxAiuc<returnShift>(pbcAiuc, xk, xj, *r_kj);

    *costh   = cos_angle(*r_ij, *r_kj);
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
                           float3          gm_f[],
                           float3          sm_fShiftLoc[],
                           const PbcAiuc   pbcAiuc)
{
    if (i < numBonds)
    {
        const int4 angleData = *(reinterpret_cast<const int4*>(d_forceatoms + 4 * i));
        int        type      = angleData.x;
        int        ai        = angleData.y;
        int        aj        = angleData.z;
        int        ak        = angleData.w;

        float3 r_ij;
        float3 r_kj;
        float  cos_theta;
        int    t1;
        int    t2;
        float  theta = bond_angle_gpu<calcVir>(
                gm_xq[ai], gm_xq[aj], gm_xq[ak], pbcAiuc, &r_ij, &r_kj, &cos_theta, &t1, &t2);

        float va;
        float dVdt;
        harmonic_gpu(d_forceparams[type].harmonic.krA,
                     d_forceparams[type].harmonic.rA * HIP_DEG2RAD_F,
                     theta,
                     &va,
                     &dVdt);

        if (calcEner)
        {
            *vtot_loc += va;
        }

        float cos_theta2 = cos_theta * cos_theta;
        if (cos_theta2 < 1.0F)
        {
            float st    = dVdt * __frsqrt_rn(1.0F - cos_theta2);
            float sth   = st * cos_theta;
            float nrij2 = norm2(r_ij);
            float nrkj2 = norm2(r_kj);

            float nrij_1 = __frsqrt_rn(nrij2);
            float nrkj_1 = __frsqrt_rn(nrkj2);

            float cik = st * nrij_1 * nrkj_1;
            float cii = sth * nrij_1 * nrij_1;
            float ckk = sth * nrkj_1 * nrkj_1;

            float3 f_i = cii * r_ij - cik * r_kj;
            float3 f_k = ckk * r_kj - cik * r_ij;
            float3 f_j = -f_i - f_k;

            atomicAdd(&gm_f[ai], f_i);
            atomicAdd(&gm_f[aj], f_j);
            atomicAdd(&gm_f[ak], f_k);

            if (calcVir)
            {
                atomicAdd(&sm_fShiftLoc[t1], f_i);
                atomicAdd(&sm_fShiftLoc[gmx::c_centralShiftIndex], f_j);
                atomicAdd(&sm_fShiftLoc[t2], f_k);
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
                                 float3          gm_f[],
                                 float3          sm_fShiftLoc[],
                                 const PbcAiuc   pbcAiuc)
{
    if (i < numBonds)
    {
        const int4 ubData = *(reinterpret_cast<const int4*>(d_forceatoms + 4 * i));
        int        type   = ubData.x;
        int        ai     = ubData.y;
        int        aj     = ubData.z;
        int        ak     = ubData.w;

        float th0A = d_forceparams[type].u_b.thetaA * HIP_DEG2RAD_F;
        float kthA = d_forceparams[type].u_b.kthetaA;
        float r13A = d_forceparams[type].u_b.r13A;
        float kUBA = d_forceparams[type].u_b.kUBA;

        float3 r_ij;
        float3 r_kj;
        float  cos_theta;
        int    t1;
        int    t2;
        float  theta = bond_angle_gpu<calcVir>(
                gm_xq[ai], gm_xq[aj], gm_xq[ak], pbcAiuc, &r_ij, &r_kj, &cos_theta, &t1, &t2);

        float va;
        float dVdt;
        harmonic_gpu(kthA, th0A, theta, &va, &dVdt);

        if (calcEner)
        {
            *vtot_loc += va;
        }

        float3 r_ik;
        int    ki = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[ak], r_ik);

        float dr2 = norm2(r_ik);
        float dr  = dr2 * __frsqrt_rn(dr2);

        float vbond;
        float fbond;
        harmonic_gpu(kUBA, r13A, dr, &vbond, &fbond);

        float cos_theta2 = cos_theta * cos_theta;
        if (cos_theta2 < 1.0F)
        {
            float st  = dVdt * __frsqrt_rn(1.0F - cos_theta2);
            float sth = st * cos_theta;

            float nrkj2 = norm2(r_kj);
            float nrij2 = norm2(r_ij);

            float cik = st * __frsqrt_rn(nrkj2 * nrij2);
            float cii = sth / nrij2;
            float ckk = sth / nrkj2;

            float3 f_i = cii * r_ij - cik * r_kj;
            float3 f_k = ckk * r_kj - cik * r_ij;
            float3 f_j = -f_i - f_k;

            atomicAdd(&gm_f[ai], f_i);
            atomicAdd(&gm_f[aj], f_j);
            atomicAdd(&gm_f[ak], f_k);

            if (calcVir)
            {
                atomicAdd(&sm_fShiftLoc[t1], f_i);
                atomicAdd(&sm_fShiftLoc[gmx::c_centralShiftIndex], f_j);
                atomicAdd(&sm_fShiftLoc[t2], f_k);
            }
        }

        /* Time for the bond calculations */
        if (dr2 != 0.0F)
        {
            if (calcEner)
            {
                *vtot_loc += vbond;
            }

            fbond *= __frsqrt_rn(dr2);

            float3 fik = fbond * r_ik;
            atomicAdd(&gm_f[ai], fik);
            atomicAdd(&gm_f[ak], -fik);

            if (calcVir && ki != gmx::c_centralShiftIndex)
            {
                atomicAdd(&sm_fShiftLoc[ki], fik);
                atomicAdd(&sm_fShiftLoc[gmx::c_centralShiftIndex], -fik);
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
                                                      float3*        r_ij,
                                                      float3*        r_kj,
                                                      float3*        r_kl,
                                                      float3*        m,
                                                      float3*        n,
                                                      int*           t1,
                                                      int*           t2,
                                                      int*           t3)
{
    *t1 = pbcDxAiuc<returnShift>(pbcAiuc, xi, xj, *r_ij);
    *t2 = pbcDxAiuc<returnShift>(pbcAiuc, xk, xj, *r_kj);
    *t3 = pbcDxAiuc<returnShift>(pbcAiuc, xk, xl, *r_kl);

    *m         = cprod(*r_ij, *r_kj);
    *n         = cprod(*r_kj, *r_kl);
    float phi  = gmx_angle(*m, *n);
    float ipr  = iprod(*r_ij, *n);
    float sign = (ipr < 0.0F) ? -1.0F : 1.0F;
    phi        = sign * phi;

    return phi;
}


__device__ __forceinline__ static void
dopdihs_gpu(const float cpA, const float phiA, const int mult, const float phi, float* v, float* f)
{
    float mdphi, sdphi;

    mdphi = mult * phi - phiA * HIP_DEG2RAD_F;
    sdphi = __sinf(mdphi);
    *v    = cpA * (1.0F + __cosf(mdphi));
    *f    = -cpA * mult * sdphi;
}

template<bool calcVir>
__device__ static void do_dih_fup_gpu(const int            i,
                                      const int            j,
                                      const int            k,
                                      const int            l,
                                      const float          ddphi,
                                      const float3         r_ij,
                                      const float3         r_kj,
                                      const float3         r_kl,
                                      const float3         m,
                                      const float3         n,
                                      float3               gm_f[],
                                      float3               sm_fShiftLoc[],
                                      const PbcAiuc&       pbcAiuc,
                                      const float4         gm_xq[],
                                      const int            t1,
                                      const int            t2,
                                      const int gmx_unused t3)
{
    float iprm  = norm2(m);
    float iprn  = norm2(n);
    float nrkj2 = norm2(r_kj);
    float toler = nrkj2 * GMX_REAL_EPS;
    if ((iprm > toler) && (iprn > toler))
    {
        float  nrkj_1 = __frsqrt_rn(nrkj2); // replacing std::invsqrt call
        float  nrkj_2 = nrkj_1 * nrkj_1;
        float  nrkj   = nrkj2 * nrkj_1;
        float  a      = -ddphi * nrkj / iprm;
        float3 f_i    = a * m;
        float  b      = ddphi * nrkj / iprn;
        float3 f_l    = b * n;
        float  p      = iprod(r_ij, r_kj);
        p *= nrkj_2;
        float q = iprod(r_kl, r_kj);
        q *= nrkj_2;
        float3 uvec = p * f_i;
        float3 vvec = q * f_l;
        float3 svec = uvec - vvec;
        float3 f_j  = f_i - svec;
        float3 f_k  = f_l + svec;

        unsigned long long int b_ = __ballot(1);
        const int prev_lane_i = __shfl_up(i, 1);
        const int prev_lane_j = __shfl_up(j, 1);
        const int prev_lane_k = __shfl_up(k, 1);
        const int prev_lane_l = __shfl_up(l, 1);
        const bool headi = (threadIdx.x & (warpSize - 1)) == 0 || i != prev_lane_i;
        const bool headj = (threadIdx.x & (warpSize - 1)) == 0 || j != prev_lane_j;
        const bool headk = (threadIdx.x & (warpSize - 1)) == 0 || k != prev_lane_k;
        const bool headl = (threadIdx.x & (warpSize - 1)) == 0 || l != prev_lane_l;

        if (b_ == ~(unsigned long long int)0)
        {
            const float3 sumfi = {hipHeadSegmentedSum(f_i.x, headi), hipHeadSegmentedSum(f_i.y, headi), hipHeadSegmentedSum(f_i.z, headi)};
            const float3 sumfj = {hipHeadSegmentedSum(f_j.x, headj), hipHeadSegmentedSum(f_j.y, headj), hipHeadSegmentedSum(f_j.z, headj)};
            const float3 sumfk = {hipHeadSegmentedSum(f_k.x, headk), hipHeadSegmentedSum(f_k.y, headk), hipHeadSegmentedSum(f_k.z, headk)};
            const float3 sumfl = {hipHeadSegmentedSum(f_l.x, headl), hipHeadSegmentedSum(f_l.y, headl), hipHeadSegmentedSum(f_l.z, headl)};

            if (headi)
            {
                hipGlobalAtomicAdd(&gm_f[i].x, sumfi.x);
                hipGlobalAtomicAdd(&gm_f[i].y, sumfi.y);
                hipGlobalAtomicAdd(&gm_f[i].z, sumfi.z);
            }

            if (headj)
            {
                hipGlobalAtomicAdd(&gm_f[j].x, -sumfj.x);
                hipGlobalAtomicAdd(&gm_f[j].y, -sumfj.y);
                hipGlobalAtomicAdd(&gm_f[j].z, -sumfj.z);
            }

            if (headk)
            {
                hipGlobalAtomicAdd(&gm_f[k].x, -sumfk.x);
                hipGlobalAtomicAdd(&gm_f[k].y, -sumfk.y);
                hipGlobalAtomicAdd(&gm_f[k].z, -sumfk.z);
            }

            if (headl)
            {
                hipGlobalAtomicAdd(&gm_f[l].x, sumfl.x);
                hipGlobalAtomicAdd(&gm_f[l].y, sumfl.y);
                hipGlobalAtomicAdd(&gm_f[l].z, sumfl.z);
            }
        }
        else
        {
            hipGlobalAtomicAdd(&gm_f[i].x, f_i.x);
            hipGlobalAtomicAdd(&gm_f[i].y, f_i.y);
            hipGlobalAtomicAdd(&gm_f[i].z, f_i.z);
            hipGlobalAtomicAdd(&gm_f[j].x, -f_j.x);
            hipGlobalAtomicAdd(&gm_f[j].y, -f_j.y);
            hipGlobalAtomicAdd(&gm_f[j].z, -f_j.z);
            hipGlobalAtomicAdd(&gm_f[k].x, -f_k.x);
            hipGlobalAtomicAdd(&gm_f[k].y, -f_k.y);
            hipGlobalAtomicAdd(&gm_f[k].z, -f_k.z);
            hipGlobalAtomicAdd(&gm_f[l].x, f_l.x);
            hipGlobalAtomicAdd(&gm_f[l].y, f_l.y);
            hipGlobalAtomicAdd(&gm_f[l].z, f_l.z);
        }

        if (calcVir)
        {
            float3 dx_jl;
            int    t3 = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[l], gm_xq[j], dx_jl);

            hipLocalAtomicAdd(&sm_fShiftLoc[t1].x, f_i.x);
            hipLocalAtomicAdd(&sm_fShiftLoc[t1].y, f_i.y);
            hipLocalAtomicAdd(&sm_fShiftLoc[t1].z, f_i.z);
            hipLocalAtomicAdd(&sm_fShiftLoc[gmx::c_centralShiftIndex].x, -f_j.x);
            hipLocalAtomicAdd(&sm_fShiftLoc[gmx::c_centralShiftIndex].y, -f_j.y);
            hipLocalAtomicAdd(&sm_fShiftLoc[gmx::c_centralShiftIndex].z, -f_j.z);
            hipLocalAtomicAdd(&sm_fShiftLoc[t2].x, -f_k.x);
            hipLocalAtomicAdd(&sm_fShiftLoc[t2].y, -f_k.y);
            hipLocalAtomicAdd(&sm_fShiftLoc[t2].z, -f_k.z);
            hipLocalAtomicAdd(&sm_fShiftLoc[t3].x, f_l.x);
            hipLocalAtomicAdd(&sm_fShiftLoc[t3].y, f_l.y);
            hipLocalAtomicAdd(&sm_fShiftLoc[t3].z, f_l.z);
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
                          float3          gm_f[],
                          float3          sm_fShiftLoc[],
                          const PbcAiuc   pbcAiuc)
{
    if (i < numBonds)
    {
        int type = d_forceatoms[5 * i];
        int ai   = d_forceatoms[5 * i + 1];
        int aj   = d_forceatoms[5 * i + 2];
        int ak   = d_forceatoms[5 * i + 3];
        int al   = d_forceatoms[5 * i + 4];

        float3 r_ij;
        float3 r_kj;
        float3 r_kl;
        float3 m;
        float3 n;
        int    t1;
        int    t2;
        int    t3;
        float  phi = dih_angle_gpu<calcVir>(
                gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc, &r_ij, &r_kj, &r_kl, &m, &n, &t1, &t2, &t3);

        float vpd;
        float ddphi;
        dopdihs_gpu(d_forceparams[type].pdihs.cpA,
                    d_forceparams[type].pdihs.phiA,
                    d_forceparams[type].pdihs.mult,
                    phi,
                    &vpd,
                    &ddphi);

        if (calcEner)
        {
            *vtot_loc += vpd;
        }

        do_dih_fup_gpu<calcVir>(
                ai, aj, ak, al, ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc, pbcAiuc, gm_xq, t1, t2, t3);
    }
}

template<bool calcVir, bool calcEner>
__device__ void rbdihs_gpu(const int       i,
                           float*          vtot_loc,
                           const int       numBonds,
                           const t_iatom   d_forceatoms[],
                           const t_iparams d_forceparams[],
                           const float4    gm_xq[],
                           float3          gm_f[],
                           float3          sm_fShiftLoc[],
                           const PbcAiuc   pbcAiuc)
{
    constexpr float c0 = 0.0F, c1 = 1.0F, c2 = 2.0F, c3 = 3.0F, c4 = 4.0F, c5 = 5.0F;

    if (i < numBonds)
    {
        int type = d_forceatoms[5 * i];
        int ai   = d_forceatoms[5 * i + 1];
        int aj   = d_forceatoms[5 * i + 2];
        int ak   = d_forceatoms[5 * i + 3];
        int al   = d_forceatoms[5 * i + 4];

        float3 r_ij;
        float3 r_kj;
        float3 r_kl;
        float3 m;
        float3 n;
        int    t1;
        int    t2;
        int    t3;
        float  phi = dih_angle_gpu<calcVir>(
                gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc, &r_ij, &r_kj, &r_kl, &m, &n, &t1, &t2, &t3);

        /* Change to polymer convention */
        if (phi < c0)
        {
            phi += HIPRT_PI_F;
        }
        else
        {
            phi -= HIPRT_PI_F;
        }
        float cos_phi = __cosf(phi);
        /* Beware of accuracy loss, cannot use 1-sqrt(cos^2) ! */
        float sin_phi = __sinf(phi);

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

        do_dih_fup_gpu<calcVir>(
                ai, aj, ak, al, ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc, pbcAiuc, gm_xq, t1, t2, t3);
        if (calcEner)
        {
            *vtot_loc += v;
        }
    }
}

__device__ __forceinline__ static void make_dp_periodic_gpu(float* dp)
{
    /* dp cannot be outside (-pi,pi) */
    if (*dp >= HIPRT_PI_F)
    {
        *dp -= 2.0F * HIPRT_PI_F;
    }
    else if (*dp < -HIPRT_PI_F)
    {
        *dp += 2.0F * HIPRT_PI_F;
    }
}

template<bool calcVir, bool calcEner>
__device__ void idihs_gpu(const int       i,
                          float*          vtot_loc,
                          const int       numBonds,
                          const t_iatom   d_forceatoms[],
                          const t_iparams d_forceparams[],
                          const float4    gm_xq[],
                          float3          gm_f[],
                          float3          sm_fShiftLoc[],
                          const PbcAiuc   pbcAiuc)
{
    if (i < numBonds)
    {
        int type = d_forceatoms[5 * i];
        int ai   = d_forceatoms[5 * i + 1];
        int aj   = d_forceatoms[5 * i + 2];
        int ak   = d_forceatoms[5 * i + 3];
        int al   = d_forceatoms[5 * i + 4];

        float3 r_ij;
        float3 r_kj;
        float3 r_kl;
        float3 m;
        float3 n;
        int    t1;
        int    t2;
        int    t3;
        float  phi = dih_angle_gpu<calcVir>(
                gm_xq[ai], gm_xq[aj], gm_xq[ak], gm_xq[al], pbcAiuc, &r_ij, &r_kj, &r_kl, &m, &n, &t1, &t2, &t3);

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

        do_dih_fup_gpu<calcVir>(
                ai, aj, ak, al, -ddphi, r_ij, r_kj, r_kl, m, n, gm_f, sm_fShiftLoc, pbcAiuc, gm_xq, t1, t2, t3);

        if (calcEner)
        {
            *vtot_loc += -0.5F * ddphi * dp;
        }
    }
}

template<bool calcVir, bool calcEner>
__device__ void pairs_gpu(const int       i,
                          const int       numBonds,
                          const t_iatom   d_forceatoms[],
                          const t_iparams iparams[],
                          const float4    gm_xq[],
                          float3          gm_f[],
                          float3          sm_fShiftLoc[],
                          const PbcAiuc   pbcAiuc,
                          const float     scale_factor,
                          float*          vtotVdw_loc,
                          float*          vtotElec_loc)
{
    if (i < numBonds)
    {
        // TODO this should be made into a separate type, the GPU and CPU sizes should be compared
        const int3 pairData = *(reinterpret_cast<const int3*>(d_forceatoms + 3 * i));
        int        type     = pairData.x;
        int        ai       = pairData.y;
        int        aj       = pairData.z;

        float qq  = gm_xq[ai].w * gm_xq[aj].w;
        float c6  = iparams[type].lj14.c6A;
        float c12 = iparams[type].lj14.c12A;

        /* Do we need to apply full periodic boundary conditions? */
        float3 dr;
        int    fshift_index = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[ai], gm_xq[aj], dr);

        float r2    = norm2(dr);
        float rinv  = __frsqrt_rn(r2);
        float rinv2 = rinv * rinv;
        float rinv6 = rinv2 * rinv2 * rinv2;

        /* Calculate the Coulomb force * r */
        float velec = scale_factor * qq * rinv;

        /* Calculate the LJ force * r and add it to the Coulomb part */
        float fr = (12.0F * c12 * rinv6 - 6.0F * c6) * rinv6 + velec;

        float  finvr = fr * rinv2;
        float3 f     = finvr * dr;

        /* Add the forces */
        atomicAdd(&gm_f[ai], f);
        atomicAdd(&gm_f[aj], -f);
        if (calcVir && fshift_index != gmx::c_centralShiftIndex)
        {
            atomicAdd(&sm_fShiftLoc[fshift_index], f);
            atomicAdd(&sm_fShiftLoc[gmx::c_centralShiftIndex], -f);
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
                 float electrostaticsScaleFactor,
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
                 const float4* gm_xq,
                 //! Forces on atoms (on GPU)
                 float3* gm_f,
                 //! Force shifts on atoms (on GPU)
                 float3* gm_fShift,
                 //! Total Energy (on GPU)
                 float* d_vTot,
                 //! Interaction list atoms (on GPU)
                 const fixed_array<t_iatom*> d_iatoms)
{
    assert(blockDim.y == 1 && blockDim.z == 1);
    const int tid          = blockIdx.x * blockDim.x + threadIdx.x;
    float     vtot_loc     = 0.0F;
    float     vtotVdw_loc  = 0.0F;
    float     vtotElec_loc = 0.0F;

    extern __shared__ float3 sm_dynamicShmem[];
    float3* sm_fShiftLoc = sm_dynamicShmem;

    if (calcVir)
    {
        if (threadIdx.x < c_numShiftVectors)
        {
            sm_fShiftLoc[threadIdx.x] = make_float3(0.0F, 0.0F, 0.0F);
        }
        __syncthreads();
    }

    int  fType_shared_index = -1;
#pragma unroll
    for (int j = 0; j < numFTypesOnGpu; j++)
    {
        const int      numBonds = numFTypeBonds[j];
        const int      fTypeTid = tid - fTypeRangeStart[j];
        const t_iatom* iatoms   = d_iatoms[j];
        const int      fType    = fTypesOnGpu[j];
        const int      start    = fTypeRangeStart[j];
        const int      end      = fTypeRangeEnd[j];
        if (tid >= start && tid <= end)
        {
            fType_shared_index      = j;

            switch (fType)
            {
                case F_BONDS:
                    bonds_gpu<calcVir, calcEner>(fTypeTid,
                                                 &vtot_loc,
                                                 numBonds,
                                                 iatoms,
                                                 d_forceParams,
                                                 gm_xq,
                                                 gm_f,
                                                 sm_fShiftLoc,
                                                 pbcAiuc);
                    break;
                case F_ANGLES:
                    angles_gpu<calcVir, calcEner>(fTypeTid,
                                                  &vtot_loc,
                                                  numBonds,
                                                  iatoms,
                                                  d_forceParams,
                                                  gm_xq,
                                                  gm_f,
                                                  sm_fShiftLoc,
                                                  pbcAiuc);
                    break;
                case F_UREY_BRADLEY:
                    urey_bradley_gpu<calcVir, calcEner>(fTypeTid,
                                                        &vtot_loc,
                                                        numBonds,
                                                        iatoms,
                                                        d_forceParams,
                                                        gm_xq,
                                                        gm_f,
                                                        sm_fShiftLoc,
                                                        pbcAiuc);
                    break;
                case F_PDIHS:
                case F_PIDIHS:
                    pdihs_gpu<calcVir, calcEner>(fTypeTid,
                                                 &vtot_loc,
                                                 numBonds,
                                                 iatoms,
                                                 d_forceParams,
                                                 gm_xq,
                                                 gm_f,
                                                 sm_fShiftLoc,
                                                 pbcAiuc);
                    break;
                case F_RBDIHS:
                    rbdihs_gpu<calcVir, calcEner>(fTypeTid,
                                                  &vtot_loc,
                                                  numBonds,
                                                  iatoms,
                                                  d_forceParams,
                                                  gm_xq,
                                                  gm_f,
                                                  sm_fShiftLoc,
                                                  pbcAiuc);
                    break;
                case F_IDIHS:
                    idihs_gpu<calcVir, calcEner>(fTypeTid,
                                                 &vtot_loc,
                                                 numBonds,
                                                 iatoms,
                                                 d_forceParams,
                                                 gm_xq,
                                                 gm_f,
                                                 sm_fShiftLoc,
                                                 pbcAiuc);
                    break;
                case F_LJ14:
                    pairs_gpu<calcVir, calcEner>(fTypeTid,
                                                 numBonds,
                                                 iatoms,
                                                 d_forceParams,
                                                 gm_xq,
                                                 gm_f,
                                                 sm_fShiftLoc,
                                                 pbcAiuc,
                                                 electrostaticsScaleFactor,
                                                 &vtotVdw_loc,
                                                 &vtotElec_loc);
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
            int fType = fTypesOnGpu[j];
            if (__any(j == fType_shared_index))
            {
                float vtot_shuffle = j == fType_shared_index ? vtot_loc : 0.0f;
                #pragma unroll
                for (unsigned int offset = (warpSize >> 1); offset > 0; offset >>= 1)
                {
                    vtot_shuffle += __shfl_down(vtot_shuffle, offset);
                }
                if((threadIdx.x & (warpSize - 1)) == 0)
                {
                    hipGlobalAtomicAdd((d_vTot + fType), vtot_shuffle);
                }
            }
        }

        float vtotVdw_shuffle = vtotVdw_loc;
        float vtotElec_shuffle = vtotElec_loc;
        #pragma unroll
        for (unsigned int offset = (warpSize >> 1); offset > 0; offset >>= 1)
        {
            vtotVdw_shuffle += __shfl_down(vtotVdw_shuffle, offset);
            vtotElec_shuffle += __shfl_down(vtotElec_shuffle, offset);
        }

        if((threadIdx.x & (warpSize - 1)) == 0)
        { // One thread per warp accumulates partial sum into global sum
            hipGlobalAtomicAdd(d_vTot + F_LJ14, vtotVdw_shuffle);
            hipGlobalAtomicAdd(d_vTot + F_COUL14, vtotElec_shuffle);
        }
    }
    /* Accumulate shift vectors from shared memory to global memory on the first c_numShiftVectors threads of the block. */
    if (calcVir)
    {
        __syncthreads();
        if (threadIdx.x < c_numShiftVectors)
        {
            hipGlobalAtomicAdd(&gm_fShift[threadIdx.x].x, sm_fShiftLoc[threadIdx.x].x);
            hipGlobalAtomicAdd(&gm_fShift[threadIdx.x].y, sm_fShiftLoc[threadIdx.x].y);
            hipGlobalAtomicAdd(&gm_fShift[threadIdx.x].z, sm_fShiftLoc[threadIdx.x].z);
        }
    }
}


/*-------------------------------- End HIP kernels-----------------------------*/


template<bool calcVir, bool calcEner>
void ListedForcesGpu::Impl::launchKernel()
{
    GMX_ASSERT(haveInteractions_,
               "Cannot launch bonded GPU kernels unless bonded GPU work was scheduled");

    wallcycle_start_nocount(wcycle_, WallCycleCounter::LaunchGpu);
    wallcycle_sub_start(wcycle_, WallCycleSubCounter::LaunchGpuBonded);

    int fTypeRangeEnd = kernelParams_.fTypeRangeEnd[numFTypesOnGpu - 1];

    if (fTypeRangeEnd < 0)
    {
        return;
    }

    auto kernelPtr = exec_kernel_gpu<calcVir, calcEner>;

    //const auto kernelArgs = prepareGpuKernelArguments(
    //        kernelPtr, kernelLaunchConfig_, &kernelParams_, &d_xq_, &d_f_, &d_fShift_);

    //launchGpuKernel(kernelPtr,
    //                kernelLaunchConfig_,
    //                deviceStream_,
    //                nullptr,
    //                "exec_kernel_gpu<calcVir, calcEner>",
    //                kernelArgs);

    dim3 blockSize(kernelLaunchConfig_.blockSize[0], kernelLaunchConfig_.blockSize[1], kernelLaunchConfig_.blockSize[2]);
    dim3 gridSize(kernelLaunchConfig_.gridSize[0], kernelLaunchConfig_.gridSize[1], kernelLaunchConfig_.gridSize[2]);

    hipLaunchKernelGGL(kernelPtr,
        gridSize,
        blockSize,
        kernelLaunchConfig_.sharedMemorySize,
        deviceStream_.stream(),
        kernelParams_.pbcAiuc,
        kernelParams_.electrostaticsScaleFactor,
        fixed_array<int>(kernelParams_.fTypesOnGpu),
        fixed_array<int>(kernelParams_.numFTypeIAtoms),
        fixed_array<int>(kernelParams_.numFTypeBonds),
        fixed_array<int>(kernelParams_.fTypeRangeStart),
        fixed_array<int>(kernelParams_.fTypeRangeEnd),
        kernelParams_.d_forceParams,
        d_xq_,
        d_f_,
        d_fShift_,
        kernelParams_.d_vTot,
        fixed_array<t_iatom*>(kernelParams_.d_iatoms)
    );

    wallcycle_sub_stop(wcycle_, WallCycleSubCounter::LaunchGpuBonded);
    wallcycle_stop(wcycle_, WallCycleCounter::LaunchGpu);
}

void ListedForcesGpu::launchKernel(const gmx::StepWorkload& stepWork)
{
    if (stepWork.computeEnergy)
    {
        // When we need the energy, we also need the virial
        impl_->launchKernel<true, true>();
    }
    else if (stepWork.computeVirial)
    {
        impl_->launchKernel<true, false>();
    }
    else
    {
        impl_->launchKernel<false, false>();
    }
}

} // namespace gmx
