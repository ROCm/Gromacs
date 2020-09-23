#include <hip/hip_runtime.h>

#include <fstream>
#include <sstream>

#include "bondedCheck.h"
#include "gpu_vec_ops.h"
#include "myHipUtil.h"

/*-------------------------------- CUDA kernels-------------------------------- */
/*------------------------------------------------------------------------------*/

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
                atomicAdd(&gm_f[ai][m], fij);
                atomicAdd(&gm_f[aj][m], -fij);
                if (calcVir && ki != CENTRAL)
                {
                    atomicAdd(&sm_fShiftLoc[ki][m], fij);
                    atomicAdd(&sm_fShiftLoc[CENTRAL][m], -fij);
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
                atomicAdd(&gm_f[ai][m], f_i[m]);
                atomicAdd(&gm_f[aj][m], f_j[m]);
                atomicAdd(&gm_f[ak][m], f_k[m]);
                if (calcVir)
                {
                    atomicAdd(&sm_fShiftLoc[t1][m], f_i[m]);
                    atomicAdd(&sm_fShiftLoc[CENTRAL][m], f_j[m]);
                    atomicAdd(&sm_fShiftLoc[t2][m], f_k[m]);
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
                atomicAdd(&gm_f[ai][m], f_i[m]);
                atomicAdd(&gm_f[aj][m], f_j[m]);
                atomicAdd(&gm_f[ak][m], f_k[m]);
                if (calcVir)
                {
                    atomicAdd(&sm_fShiftLoc[t1][m], f_i[m]);
                    atomicAdd(&sm_fShiftLoc[CENTRAL][m], f_j[m]);
                    atomicAdd(&sm_fShiftLoc[t2][m], f_k[m]);
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
                atomicAdd(&gm_f[ai][m], fik);
                atomicAdd(&gm_f[ak][m], -fik);

                if (calcVir && ki != CENTRAL)
                {
                    atomicAdd(&sm_fShiftLoc[ki][m], fik);
                    atomicAdd(&sm_fShiftLoc[CENTRAL][m], -fik);
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
                                      const int  t3)
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
#pragma unroll
        for (int m = 0; (m < DIM); m++)
        {
            atomicAdd(&gm_f[i][m], f_i[m]);
            atomicAdd(&gm_f[j][m], -f_j[m]);
            atomicAdd(&gm_f[k][m], -f_k[m]);
            atomicAdd(&gm_f[l][m], f_l[m]);
        }

        if (calcVir)
        {
            fvec dx_jl;
            int  t3 = pbcDxAiuc<calcVir>(pbcAiuc, gm_xq[l], gm_xq[j], dx_jl);

#pragma unroll
            for (int m = 0; (m < DIM); m++)
            {
                atomicAdd(&sm_fShiftLoc[t1][m], f_i[m]);
                atomicAdd(&sm_fShiftLoc[CENTRAL][m], -f_j[m]);
                atomicAdd(&sm_fShiftLoc[t2][m], -f_k[m]);
                atomicAdd(&sm_fShiftLoc[t3][m], f_l[m]);
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
        float rinv2 = rinv * rinv;
        float rinv6 = rinv2 * rinv2 * rinv2;

        /* Calculate the Coulomb force * r */
        float velec = scale_factor * qq * rinv;

        /* Calculate the LJ force * r and add it to the Coulomb part */
        float fr = (12.0f * c12 * rinv6 - 6.0f * c6) * rinv6 + velec;

        float finvr = fr * rinv2;
        fvec  f;
        svmul_gpu(finvr, dr, f);

        /* Add the forces */
#pragma unroll
        for (int m = 0; m < DIM; m++)
        {
            atomicAdd(&gm_f[ai][m], f[m]);
            atomicAdd(&gm_f[aj][m], -f[m]);
            if (calcVir && fshift_index != CENTRAL)
            {
                atomicAdd(&sm_fShiftLoc[fshift_index][m], f[m]);
                atomicAdd(&sm_fShiftLoc[CENTRAL][m], -f[m]);
            }
        }

        if (calcEner)
        {
            *vtotVdw_loc += (c12 * rinv6 - c6) * rinv6;
            *vtotElec_loc += velec;
        }
    }
}


template<bool calcVir, bool calcEner>
__global__ void exec_kernel_gpu(BondedCudaKernelParameters kernelParams)
{
    // assert(blockDim.y == 1 && blockDim.z == 1);
    const int  tid          = blockIdx.x * blockDim.x + threadIdx.x;
    float      vtot_loc     = 0;
    float      vtotVdw_loc  = 0;
    float      vtotElec_loc = 0;
    __shared__ fvec sm_fShiftLoc[SHIFTS];

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
    bool threadComputedPotential = false;
#pragma unroll
    for (int j = 0; j < numFTypesOnGpu; j++)
    {
        if (tid >= kernelParams.fTypeRangeStart[j] && tid <= kernelParams.fTypeRangeEnd[j])
        {
            const int      numBonds = kernelParams.numFTypeBonds[j];
            int            fTypeTid = tid - kernelParams.fTypeRangeStart[j];
            const t_iatom* iatoms   = kernelParams.d_iatoms[j];
            fType                   = kernelParams.fTypesOnGpu[j];
            if (calcEner)
            {
                threadComputedPotential = true;
            }

            switch (fType)
            {
                case F_BONDS:
                    bonds_gpu<calcVir, calcEner>(fTypeTid, &vtot_loc, numBonds, iatoms,
                                                 kernelParams.d_forceParams, kernelParams.d_xq,
                                                 kernelParams.d_f, sm_fShiftLoc, kernelParams.pbcAiuc);
                    break;
                case F_ANGLES:
                    angles_gpu<calcVir, calcEner>(
                            fTypeTid, &vtot_loc, numBonds, iatoms, kernelParams.d_forceParams,
                            kernelParams.d_xq, kernelParams.d_f, sm_fShiftLoc, kernelParams.pbcAiuc);
                    break;
                case F_UREY_BRADLEY:
                    urey_bradley_gpu<calcVir, calcEner>(
                            fTypeTid, &vtot_loc, numBonds, iatoms, kernelParams.d_forceParams,
                            kernelParams.d_xq, kernelParams.d_f, sm_fShiftLoc, kernelParams.pbcAiuc);
                    break;
                case F_PDIHS:
                case F_PIDIHS:
                    pdihs_gpu<calcVir, calcEner>(fTypeTid, &vtot_loc, numBonds, iatoms,
                                                 kernelParams.d_forceParams, kernelParams.d_xq,
                                                 kernelParams.d_f, sm_fShiftLoc, kernelParams.pbcAiuc);
                    break;
                case F_RBDIHS:
                    rbdihs_gpu<calcVir, calcEner>(
                            fTypeTid, &vtot_loc, numBonds, iatoms, kernelParams.d_forceParams,
                            kernelParams.d_xq, kernelParams.d_f, sm_fShiftLoc, kernelParams.pbcAiuc);
                    break;
                case F_IDIHS:
                    idihs_gpu<calcVir, calcEner>(fTypeTid, &vtot_loc, numBonds, iatoms,
                                                 kernelParams.d_forceParams, kernelParams.d_xq,
                                                 kernelParams.d_f, sm_fShiftLoc, kernelParams.pbcAiuc);
                    break;
                case F_LJ14:
                    pairs_gpu<calcVir, calcEner>(fTypeTid, numBonds, iatoms, kernelParams.d_forceParams,
                                                 kernelParams.d_xq, kernelParams.d_f, sm_fShiftLoc,
                                                 kernelParams.pbcAiuc, kernelParams.scaleFactor,
                                                 &vtotVdw_loc, &vtotElec_loc);
                    break;
            }
            break;
        }
    }

    if (threadComputedPotential)
    {
        float* vtotVdw  = kernelParams.d_vTot + F_LJ14;
        float* vtotElec = kernelParams.d_vTot + F_COUL14;
        atomicAdd(kernelParams.d_vTot + fType, vtot_loc);
        atomicAdd(vtotVdw, vtotVdw_loc);
        atomicAdd(vtotElec, vtotElec_loc);
    }
    /* Accumulate shift vectors from shared memory to global memory on the first SHIFTS threads of the block. */
    if (calcVir)
    {
        __syncthreads();
        if (threadIdx.x < SHIFTS)
        {
            fvec_inc_atomic(kernelParams.d_fShift[threadIdx.x], sm_fShiftLoc[threadIdx.x]);
        }
    }
}

struct kernelTest {
     int natoms;
     int nshifts; 
     int f_nre; 
     size_t blockSize[3]; 
     size_t gridSize[3]; 
     int numFFParamTypes; 
     BondedCudaKernelParameters kernParam; 
}; 

static void setup_kernel_test(std::string dataDir, kernelTest &test, bool calcVir, bool calcEner)
{
    std::string fPath(dataDir); 
    std::string key1, key2; 

    fPath += "/exec_kernel_gpu_indata_"; 

    fPath += calcVir? "V1_" : "V0_" ; 
    fPath += calcEner? "E1.txt" : "E0.txt" ; 

    std::ifstream istr; 

    istr.open(fPath.c_str(), std::ofstream::in); 

    //  block and grid sizes used for scheduling the kernel
    istr >> key1 >> test.blockSize[0] >> test.blockSize[1] >> test.blockSize[2]; 
    istr >> key2 >> test.gridSize[0] >> test.gridSize[1] >> test.gridSize[2]; 

    // contents of pbcAiuc 
    PbcAiuc &pbcAiuc = test.kernParam.pbcAiuc; 

    istr >> key1  >> pbcAiuc.invBoxDiagZ  >>  pbcAiuc.boxZX >> pbcAiuc.boxZY >> pbcAiuc.boxZZ; 
    istr >> pbcAiuc.invBoxDiagY >> pbcAiuc.boxYX >> pbcAiuc.boxYY >> pbcAiuc.invBoxDiagX >> pbcAiuc.boxXX; 

    // scaleFactor
    istr >> key1 >> test.kernParam.scaleFactor; 

    // fTypesOnGpu
    istr >> key1; 
    for (int i=0; i < numFTypesOnGpu; i++) 
         istr >> test.kernParam.fTypesOnGpu[i]; 

    // numFTypeIAtoms
    istr >> key1;
    for (int i=0; i < numFTypesOnGpu; i++) 
         istr >> test.kernParam.numFTypeIAtoms[i]; 

    // numFTypeBonds
    istr >> key1; 
    for (int i=0; i < numFTypesOnGpu; i++) 
         istr >> test.kernParam.numFTypeBonds[i]; 

    // fTypeRangeStart
    istr >> key1; 
    for (int i=0; i < numFTypesOnGpu; i++) 
         istr >> test.kernParam.fTypeRangeStart[i]; 
      
    // fTypeRangeStart
    istr >> key1; 
    for (int i=0; i < numFTypesOnGpu; i++) 
         istr >> test.kernParam.fTypeRangeEnd[i]; 

    // d_forceParams 
    int  typeSize;  

    istr >> key1  >> typeSize >> test.numFFParamTypes; 

    unsigned char *h_bytes = new unsigned char[sizeof(t_iparams) * test.numFFParamTypes]; 
    unsigned char *hPtr = h_bytes;  
    
    for (int i=0; i < test.numFFParamTypes; i++) {
         for (int k=0; k < sizeof(t_iparams); k++) {
              int ival; 
              istr >> ival; 
              hPtr[k] = (unsigned char)ival;  
         }; 
         hPtr += sizeof(t_iparams); 
    }; 

    MY_HIP_CHECK( hipMalloc((void**)&test.kernParam.d_forceParams, sizeof(t_iparams) * test.numFFParamTypes) ); 
    MY_HIP_CHECK( hipMemcpyHtoD((void*)test.kernParam.d_forceParams, (void*)h_bytes, sizeof(t_iparams) * test.numFFParamTypes) ); 

    delete [] h_bytes; 

    // d_iatoms 
    istr >> key1; 
    for (int i=0; i < numFTypesOnGpu; i++) {
         int index; 

         istr >> key1  >> index >>  key2  >>  test.kernParam.numFTypeIAtoms[i]; 
 
         t_iatom *h_iatoms = new t_iatom[ test.kernParam.numFTypeIAtoms[i] ]; 
         
         for (int k=0; k < test.kernParam.numFTypeIAtoms[i]; k++) 
              istr >> h_iatoms[k];  

         MY_HIP_CHECK( hipMalloc((void**)&test.kernParam.d_iatoms[i], sizeof(t_iatom) * test.kernParam.numFTypeIAtoms[i]) ); 
         MY_HIP_CHECK( hipMemcpyHtoD((void*)test.kernParam.d_iatoms[i], (void*)h_iatoms, sizeof(t_iatom) * test.kernParam.numFTypeIAtoms[i]) ); 

         delete [] h_iatoms; 
    }; 

    // d_vTot
    istr >> key1 >> test.f_nre; 
    float *h_vTot = new float[test.f_nre]; 

    for (int i=0; i < F_NRE; i++) 
         istr >> h_vTot[i]; 

    MY_HIP_CHECK( hipMalloc((void**)&test.kernParam.d_vTot, sizeof(float) * test.f_nre) ); 
    MY_HIP_CHECK( hipMemcpyHtoD((void*)test.kernParam.d_vTot, (void*)h_vTot, sizeof(float) * test.f_nre) ); 

    delete [] h_vTot; 
 
    // d_xq
    istr >> key1 >> test.natoms; 
    float4 *h_xq = new float4[test.natoms]; 

    for (int i=0; i < test.natoms; i++)  
         istr >> h_xq[i].x >> h_xq[i].y >> h_xq[i].z >> h_xq[i].w; 

    MY_HIP_CHECK( hipMalloc((void**)&test.kernParam.d_xq, sizeof(float4) * test.natoms) ); 
    MY_HIP_CHECK( hipMemcpyHtoD((void*)test.kernParam.d_xq, h_xq, sizeof(float4) * test.natoms) ); 

    delete [] h_xq; 

    // d_f
    int natoms; 
 
    istr >> key1 >> natoms; 
    fvec *h_f = new fvec[test.natoms]; 

    for (int i=0; i < natoms; i++) 
         istr >> h_f[i][XX] >> h_f[i][YY] >> h_f[i][ZZ]; 

    MY_HIP_CHECK( hipMalloc((void**)&test.kernParam.d_f, sizeof(fvec) * test.natoms) ); 
    MY_HIP_CHECK( hipMemcpyHtoD((void*)test.kernParam.d_f, h_f, sizeof(fvec) * test.natoms) ); 

    delete [] h_f; 
      
    // d_fShift     
    istr >> key1 >> test.nshifts; 
    fvec *h_fShift = new fvec[test.nshifts]; 

    for (int i=0; i < test.nshifts; i++)   
         istr >> h_fShift[i][XX] >>  h_fShift[i][YY] >> h_fShift[i][ZZ]; 

    MY_HIP_CHECK( hipMalloc((void**)&test.kernParam.d_fShift, sizeof(fvec) * test.nshifts) ); 
    MY_HIP_CHECK( hipMemcpyHtoD((void*)test.kernParam.d_fShift, h_fShift, sizeof(fvec) * test.nshifts) ); 

    delete [] h_fShift; 

    istr.close(); 
}; 

static void save_kernel_test_output(std::string dataDir, kernelTest &test, bool calcVir, bool calcEner)
{
    std::string fPath(dataDir); 

    fPath += "/kernel_test_outdata_";
    
    fPath += calcVir? "V1_" : "V0_" ;
    fPath += calcEner? "E1.txt" : "E0.txt" ;
    
    std::ofstream ostr;
    
    ostr.open(fPath.c_str(), std::ofstream::out | std::ofstream::trunc);
    
    // d_vTot
    ostr << "d_vTot: " << test.f_nre << std::endl; 
    float *h_vTot = new float[test.f_nre]; 

    MY_HIP_CHECK( hipMemcpyDtoH((void*)h_vTot, (void*)test.kernParam.d_vTot, sizeof(float) * test.f_nre) ); 
    for (int i=0; i < F_NRE; i++) 
         ostr << h_vTot[i] << " "; 
    ostr << std::endl; 

    delete [] h_vTot; 
 
    // d_f
    ostr << "d_f: " << test.natoms << std::endl; 
    fvec *h_f = new fvec[test.natoms]; 

    MY_HIP_CHECK( hipMemcpyDtoH((void*)h_f, (void*)test.kernParam.d_f, sizeof(fvec) * test.natoms) ); 
    for (int i=0; i < test.natoms; i++) 
         ostr << h_f[i][XX] << " " << h_f[i][YY] << " " << h_f[i][ZZ]  << std::endl; 

    delete [] h_f; 
      
    // d_fShift     
    ostr << "d_fShift: " << test.nshifts << std::endl; 
    fvec *h_fShift = new fvec[test.nshifts]; 

    MY_HIP_CHECK( hipMemcpyDtoH((void*)h_fShift, (void*)test.kernParam.d_fShift, sizeof(fvec) * test.nshifts) ); 
    for (int i=0; i < test.nshifts; i++)   
         ostr << h_fShift[i][XX] << " " << h_fShift[i][YY] << " " << h_fShift[i][ZZ] << std::endl; 

    delete [] h_fShift; 

    ostr.close(); 
}; 

static void destroy_kernel_test(kernelTest &test)
{
    MY_HIP_CHECK( hipFree(test.kernParam.d_f) ); 
    MY_HIP_CHECK( hipFree(test.kernParam.d_vTot) ); 
    MY_HIP_CHECK( hipFree(test.kernParam.d_fShift) ); 
    MY_HIP_CHECK( hipFree((void*)test.kernParam.d_xq) ); 
    MY_HIP_CHECK( hipFree(test.kernParam.d_forceParams) ); 

    for (int i=0; i < numFTypesOnGpu; i++) {
         MY_HIP_CHECK( hipFree(test.kernParam.d_iatoms[i]) ); 
    }; 
}; 

int main(int argc, char *argv[])
{
    kernelTest  test;  
    hipStream_t stream; 

    if ( argc != 4 ) {
         std::cerr << "Invalid commandline parameters!" << std::endl; 
         throw std::runtime_error(""); 
    }
    
    std::string dataDir(argv[1]); 

    bool calcVir, calcEner; 

    calcVir = atoi(argv[2]) !=0 ? true : false; 
    calcEner = atoi(argv[3]) !=0 ? true : false; 

    MY_HIP_CHECK( hipStreamCreate(&stream) ); 

    void (*kernelFunc)(BondedCudaKernelParameters kernelParams); 

    if ( calcVir ) { 
         if ( calcEner ) 
              kernelFunc = exec_kernel_gpu<true, true>; 
         else 
              kernelFunc = exec_kernel_gpu<true, false>; 
    }
    else 
        if ( calcEner ) 
             kernelFunc = exec_kernel_gpu<false, false>; 
        else 
             kernelFunc = exec_kernel_gpu<false, false>; 

    setup_kernel_test(dataDir, test, calcVir, calcEner); 

    for (int iter=0; iter<1000; iter++) {
        hipLaunchKernelGGL(kernelFunc, dim3(test.gridSize[0],test.gridSize[1],test.gridSize[2]), dim3(test.blockSize[0],test.blockSize[1],test.blockSize[2]), 0, stream, test.kernParam);  
    }

    MY_HIP_CHECK( hipStreamSynchronize(stream) ); 

    save_kernel_test_output(dataDir, test, calcVir, calcEner); 

    destroy_kernel_test(test); 
}; 

