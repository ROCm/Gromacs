#ifndef _BONDED_CHECK_H_
#define _BONDED_CHECK_H_

#ifdef _HIP_KERNEL_TEST_
#include <hip/hip_runtime.h>
#else
#endif 

#include <array>

#include "types_def.h"
#include "ishift.h"

struct PbcAiuc
{
    //! 1/box[ZZ][ZZ]
    float invBoxDiagZ;
    //! box[ZZ][XX]
    float boxZX;
    //! box[ZZ][YY]
    float boxZY;
    //! box[ZZ][ZZ]
    float boxZZ;
    //! 1/box[YY][YY]
    float invBoxDiagY;
    //! box[YY][XX]
    float boxYX;
    //! box[YY][YY]
    float boxYY;
    //! 1/box[XX][XX]
    float invBoxDiagX;
    //! box[XX][XX]
    float boxXX;
};


template<bool returnShift>
static __forceinline__ __device__ int pbcDxAiuc(const PbcAiuc& pbcAiuc, const float4& r1, const float4& r2, fvec dr)
{
    dr[XX] = r1.x - r2.x;
    dr[YY] = r1.y - r2.y;
    dr[ZZ] = r1.z - r2.z;

    float shz = rintf(dr[ZZ] * pbcAiuc.invBoxDiagZ);
    dr[XX] -= shz * pbcAiuc.boxZX;
    dr[YY] -= shz * pbcAiuc.boxZY;
    dr[ZZ] -= shz * pbcAiuc.boxZZ;

    float shy = rintf(dr[YY] * pbcAiuc.invBoxDiagY);
    dr[XX] -= shy * pbcAiuc.boxYX;
    dr[YY] -= shy * pbcAiuc.boxYY;

    float shx = rintf(dr[XX] * pbcAiuc.invBoxDiagX);
    dr[XX] -= shx * pbcAiuc.boxXX;

    if (returnShift)
    {
        ivec ishift;

        ishift[XX] = -__float2int_rn(shx);
        ishift[YY] = -__float2int_rn(shy);
        ishift[ZZ] = -__float2int_rn(shz);

        return IVEC2IS(ishift);
    }
    else
    {
        return 0;
    }
};


static constexpr int numFTypesOnGpu = 8;

constexpr std::array<int, numFTypesOnGpu> fTypesOnGpu = { F_BONDS,  F_ANGLES, F_UREY_BRADLEY, F_PDIHS,  F_RBDIHS, F_IDIHS, F_PIDIHS, F_LJ14 };
constexpr int MAXFORCEPARAM = 12;
constexpr int NR_RBDIHS     = 6;
constexpr int NR_CBTDIHS    = 6;
constexpr int NR_FOURDIHS   = 4;

typedef union t_iparams {
    /* Some parameters have A and B values for free energy calculations.
     * The B values are not used for regular simulations of course.
     * Free Energy for nonbondeds can be computed by changing the atom type.
     * The harmonic type is used for all harmonic potentials:
     * bonds, angles and improper dihedrals
     */
    struct
    {
        real a, b, c;
    } bham;
    struct
    {
        real rA, krA, rB, krB;
    } harmonic;
    struct
    {
        real klinA, aA, klinB, aB;
    } linangle;
    struct
    {
        real lowA, up1A, up2A, kA, lowB, up1B, up2B, kB;
    } restraint;
    /* No free energy supported for cubic bonds, FENE, WPOL or cross terms */
    struct
    {
        real b0, kb, kcub;
    } cubic;
    struct
    {
        real bm, kb;
    } fene;
    struct
    {
        real r1e, r2e, krr;
    } cross_bb;
    struct
    {
        real r1e, r2e, r3e, krt;
    } cross_ba;
    struct
    {
        real thetaA, kthetaA, r13A, kUBA, thetaB, kthetaB, r13B, kUBB;
    } u_b;
    struct
    {
        real theta, c[5];
    } qangle;
    struct
    {
        real alpha;
    } polarize;
    struct
    {
        real alpha, drcut, khyp;
    } anharm_polarize;
    struct
    {
        real al_x, al_y, al_z, rOH, rHH, rOD;
    } wpol;
    struct
    {
        real a, alpha1, alpha2, rfac;
    } thole;
    struct
    {
        real c6, c12;
    } lj;
    struct
    {
        real c6A, c12A, c6B, c12B;
    } lj14;
    struct
    {
        real fqq, qi, qj, c6, c12;
    } ljc14;
    struct
    {
        real qi, qj, c6, c12;
    } ljcnb;
    /* Proper dihedrals can not have different multiplicity when
     * doing free energy calculations, because the potential would not
     * be periodic anymore.
     */
    struct
    {
        real phiA, cpA;
        int  mult;
        real phiB, cpB;
    } pdihs;
    struct
    {
        real dA, dB;
    } constr;
    /* Settle can not be used for Free energy calculations of water bond geometry.
     * Use shake (or lincs) instead if you have to change the water bonds.
     */
    struct
    {
        real doh, dhh;
    } settle;
    struct
    {
        real b0A, cbA, betaA, b0B, cbB, betaB;
    } morse;
    struct
    {
        real pos0A[DIM], fcA[DIM], pos0B[DIM], fcB[DIM];
    } posres;
    struct
    {
        real pos0[DIM], r, k;
        int  geom;
    } fbposres;
    struct
    {
        real rbcA[NR_RBDIHS], rbcB[NR_RBDIHS];
    } rbdihs;
    struct
    {
        real cbtcA[NR_CBTDIHS], cbtcB[NR_CBTDIHS];
    } cbtdihs;
    struct
    {
        real a, b, c, d, e, f;
    } vsite;
    struct
    {
        int  n;
        real a;
    } vsiten;
    /* NOTE: npair is only set after reading the tpx file */
    struct
    {
        real low, up1, up2, kfac;
        int  type, label, npair;
    } disres;
    struct
    {
        real phiA, dphiA, kfacA, phiB, dphiB, kfacB;
    } dihres;
    struct
    {
        int  ex, power, label;
        real c, obs, kfac;
    } orires;
    struct
    {
        int  table;
        real kA;
        real kB;
    } tab;
    struct
    {
        int cmapA, cmapB;
    } cmap;
    struct
    {
        real buf[MAXFORCEPARAM];
    } generic; /* Conversion */
} t_iparams;


struct BondedCudaKernelParameters
{
    //! Periodic boundary data
    PbcAiuc pbcAiuc;
    //! Scale factor
    float scaleFactor;
    //! The bonded types on GPU
    int fTypesOnGpu[numFTypesOnGpu];
    //! The number of interaction atom (iatom) elements for every function type
    int numFTypeIAtoms[numFTypesOnGpu];
    //! The number of bonds for every function type
    int numFTypeBonds[numFTypesOnGpu];
    //! The start index in the range of each interaction type
    int fTypeRangeStart[numFTypesOnGpu];
    //! The end index in the range of each interaction type
    int fTypeRangeEnd[numFTypesOnGpu];

    //! Force parameters (on GPU)
    t_iparams* d_forceParams;
    //! Coordinates before the timestep (on GPU)
    const float4* d_xq;
    //! Forces on atoms (on GPU)
    fvec* d_f;
    //! Force shifts on atoms (on GPU)
    fvec* d_fShift;
    //! Total Energy (on GPU)
    float* d_vTot;
    //! Interaction list atoms (on GPU)
    t_iatom* d_iatoms[numFTypesOnGpu];
}; 


#endif  // end of _BONDEDD_CHECK_H_
