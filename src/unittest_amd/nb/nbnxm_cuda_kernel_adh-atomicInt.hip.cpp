#include "hip/hip_runtime.h"
#include <iostream>
#include <fstream>
//#include "nbnxm_cuda_kernel_utils.hip.h"

//#include "gromacs/gpu_utils/hip_arch_utils.h"
//#include "gromacs/gpu_utils/cuda_kernel_utils.hip.h"
//#include "gromacs/math/utilities.h"
//#include "gromacs/pbcutil/ishift.h"
//
#define c_nbnxnGpuNumClusterPerSupercluster 8
#define c_nbnxnGpuJgroupSize (32 / c_nbnxnGpuNumClusterPerSupercluster)

#define NBNXN_MIN_RSQ 3.82e-07f

#define D_BOX_Z 1
#define D_BOX_Y 1
#define D_BOX_X 2
#define N_BOX_Z (2 * D_BOX_Z + 1)
#define N_BOX_Y (2 * D_BOX_Y + 1)
#define N_BOX_X (2 * D_BOX_X + 1)
#define N_IVEC (N_BOX_Z * N_BOX_Y * N_BOX_X)
#define CENTRAL (N_IVEC / 2)

#ifndef M_FLOAT_1_SQRTPI /* used in GPU kernels */
/* 1.0 / sqrt(M_PI) */
#    define M_FLOAT_1_SQRTPI 0.564189583547756f
#endif

static constexpr int c_nbnxnGpuClusterSize = 8;
static constexpr int c_nbnxnGpuClusterpairSplit = 1;
static const int c_clSize = c_nbnxnGpuClusterSize;
static const int c_numClPerSupercl = c_nbnxnGpuNumClusterPerSupercluster;
static constexpr int c_nbnxnGpuExclSize =
        c_nbnxnGpuClusterSize * c_nbnxnGpuClusterSize / c_nbnxnGpuClusterpairSplit;
static const float __device__ c_oneSixth    = 0.16666667f;
static const float __device__ c_oneTwelveth = 0.08333333f;
static const unsigned long c_fullWarpMask = 0xffffffffffffffff;
static const int warp_size_log2 = 6;

struct shift_consts_t
{
    float c2;
    float c3;
    float cpot;
};

typedef struct
{
    /* Returns the number of j-cluster groups in this entry */
    int numJClusterGroups() const { return cj4_ind_end - cj4_ind_start; }

    int sci;           /* i-super-cluster       */
    int shift;         /* Shift vector index plus possible flags */
    int cj4_ind_start; /* Start index into cj4  */
    int cj4_ind_end;   /* End index into cj4    */
} nbnxn_sci_t;

struct nbnxn_im_ei_t
{
    // The i-cluster interactions mask for 1 warp
    unsigned int imask = 0U;
    // Index into the exclusion array for 1 warp, default index 0 which means no exclusions
    int excl_ind = 0;
};

typedef struct
{
    int           cj[c_nbnxnGpuJgroupSize];         /* The 4 j-clusters */
    nbnxn_im_ei_t imei[c_nbnxnGpuClusterpairSplit]; /* The i-cluster mask data       for 2 warps */
} nbnxn_cj4_t;

struct nbnxn_excl_t
{
    /* Topology exclusion interaction bits per warp */
    unsigned int pair[c_nbnxnGpuExclSize];
};

typedef struct cu_atomdata
{
    int natoms;       /**< number of atoms                              */
    int natoms_local; /**< number of local atoms                        */
    int nalloc;       /**< allocation size for the atom data (xq, f)    */

    float4* xq; /**< atom coordinates + charges, size natoms      */
    float3* f;  /**< force output array, size natoms              */

    float* e_lj; /**< LJ energy output, size 1                     */
    float* e_el; /**< Electrostatics energy input, size 1          */

    float3* fshift; /**< shift forces                                 */

    int     ntypes;     /**< number of atom types                         */
    int*    atom_types; /**< atom type indices, size natoms               */
    float2* lj_comb;    /**< sqrt(c6),sqrt(c12) size natoms               */

    float3* shift_vec;         /**< shifts                                       */
    bool    bShiftVecUploaded; /**< true if the shift vector has been uploaded   */
}cu_atomdata_t;

typedef struct cu_nbparam
{

    int eeltype; /**< type of electrostatics, takes values from #eelCu */
    int vdwtype; /**< type of VdW impl., takes values from #evdwCu     */

    float epsfac;      /**< charge multiplication factor                      */
    float c_rf;        /**< Reaction-field/plain cutoff electrostatics const. */
    float two_k_rf;    /**< Reaction-field electrostatics constant            */
    float ewald_beta;  /**< Ewald/PME parameter                               */
    float sh_ewald;    /**< Ewald/PME correction term substracted from the direct-space potential */
    float sh_lj_ewald; /**< LJ-Ewald/PME correction term added to the correction potential        */
    float ewaldcoeff_lj; /**< LJ-Ewald/PME coefficient                          */

    float rcoulomb_sq; /**< Coulomb cut-off squared                           */

    float rvdw_sq;           /**< VdW cut-off squared                               */
    float rvdw_switch;       /**< VdW switched cut-off                              */
    float rlistOuter_sq;     /**< Full, outer pair-list cut-off squared             */
    float rlistInner_sq;     /**< Inner, dynamic pruned pair-list cut-off squared   */
    bool  useDynamicPruning; /**< True if we use dynamic pair-list pruning          */

    shift_consts_t  dispersion_shift; /**< VdW shift dispersion constants           */
    shift_consts_t  repulsion_shift;  /**< VdW shift repulsion constants            */

    /* LJ non-bonded parameters - accessed through texture memory */
    float*              nbfp; /**< nonbonded parameter table with C6/C12 pairs per atom type-pair, 2*ntype^2 elements */
    hipTextureObject_t nbfp_texobj; /**< texture object bound to nbfp */
    float*              nbfp_comb; /**< nonbonded parameter table per atom type, 2*ntype elements */
    hipTextureObject_t nbfp_comb_texobj; /**< texture object bound to nbfp_texobj */

    /* Ewald Coulomb force table data - accessed through texture memory */
    float               coulomb_tab_scale;  /**< table scale/spacing                        */
    float*              coulomb_tab;        /**< pointer to the table in the device memory  */
    hipTextureObject_t coulomb_tab_texobj; /**< texture object bound to coulomb_tab        */
}cu_nbparam_t;

typedef struct gpu_plist
{
    int na_c; /**< number of atoms per cluster                  */

    int                       nsci;       /**< size of sci, # of i clusters in the list     */
    int                       sci_nalloc; /**< allocation size of sci                       */
    nbnxn_sci_t*              sci;        /**< list of i-cluster ("super-clusters")         */

    int                       ncj4;          /**< total # of 4*j clusters                      */
    int                       cj4_nalloc;    /**< allocation size of cj4                       */
    nbnxn_cj4_t*              cj4;           /**< 4*j cluster list, contains j cluster number
                                                and index into the i cluster list            */
    int                       nimask;       /**< # of 4*j clusters * # of warps               */
    int                       imask_nalloc; /**< allocation size of imask                     */
    unsigned int*             imask;        /**< imask for 2 warps for each 4*j cluster group */
    nbnxn_excl_t*             excl;         /**< atom interaction bits                        */
    int                       nexcl;        /**< count for excl                               */
    int                       excl_nalloc;  /**< allocation size of excl                      */

    /* parameter+variables for normal and rolling pruning */
    bool haveFreshList; /**< true after search, indictes that initial pruning with outer prunning is needed */
    int  rollingPruningNumParts; /**< the number of parts/steps over which one cyle of roling pruning takes places */
    int  rollingPruningPart; /**< the next part to which the roling pruning needs to be applied */
}cu_plist_t;

__forceinline__ __host__ __device__ float norm2(float3 a)
{
    return (a.x * a.x + a.y * a.y + a.z * a.z);
}

__forceinline__ __host__ __device__ float3 make_float3(float s)
{
    return make_float3(s, s, s);
}

template<typename T>
__device__ __forceinline__ T LDG(const T* ptr)
{
    return *ptr;
}

static __forceinline__ __device__ void
                       convert_sigma_epsilon_to_c6_c12(const float sigma, const float epsilon, float* c6, float* c12)
{
    float sigma2, sigma6;

    sigma2 = sigma * sigma;
    sigma6 = sigma2 * sigma2 * sigma2;
    *c6    = epsilon * sigma6;
    *c12   = *c6 * sigma6;
}

static __forceinline__ __device__ void fetch_nbfp_c6_c12(float& c6, float& c12, const cu_nbparam_t nbparam, int baseIndex)
{
    float2* nbfp = (float2*)nbparam.nbfp;
    float2  c6c12;
    c6c12 = LDG(&nbfp[baseIndex]);
    c6    = c6c12.x;
    c12   = c6c12.y;
}

static __forceinline__ __device__ void
                       reduce_force_j_warp_shfl(float3 f, float3* fout, int tidxi, int aidx)
{
    f.x += __shfl_down(f.x, 1);
    f.y += __shfl_up(f.y, 1);
    f.z += __shfl_down(f.z, 1);

    if (tidxi & 1)
    {
        f.x = f.y;
    }

    f.x += __shfl_down(f.x, 2);
    f.z += __shfl_up(f.z, 2);

    if (tidxi & 2)
    {
        f.x = f.z;
    }

    f.x += __shfl_down(f.x, 4);

    if (tidxi < 3)
    {
        atomicAdd((int*)((&fout[aidx].x) + tidxi), f.x);
    }
}

static __forceinline__ __device__ void reduce_force_i_warp_shfl(float3             fin,
                                                                float3*            fout,
                                                                float*             fshift_buf,
                                                                bool               bCalcFshift,
                                                                int                tidxj,
                                                                int                aidx)
{
    fin.x += __shfl_down(fin.x, c_clSize);
    fin.y += __shfl_up(fin.y, c_clSize);
    fin.z += __shfl_down(fin.z, c_clSize);

    if (tidxj & 1)
    {
        fin.x = fin.y;
    }

    fin.x += __shfl_down(fin.x, 2 * c_clSize);
    fin.z += __shfl_up(fin.z, 2 * c_clSize);

    if (tidxj & 2)
    {
        fin.x = fin.z;
    }

    fin.x += __shfl_down(fin.x, 4*c_clSize);

    /* Threads 0,1,2 and 4,5,6 increment x,y,z for their warp */
    if (tidxj < 3)
    {
        atomicAdd((int*)(&fout[aidx].x + tidxj), fin.x);

        if (bCalcFshift)
        {
            *fshift_buf += fin.x;
        }
    }
}

static __forceinline__ __device__ void
                       reduce_energy_warp_shfl(float E_lj, float E_el, float* e_lj, float* e_el, int tidx, const unsigned long activemask)
{
    int i, sh;

    sh = 1;
#pragma unroll warp_size_log2
    for (i = 0; i < warp_size_log2; i++)
    {
        E_lj += __shfl_down(E_lj, sh);
        E_el += __shfl_down(E_el, sh);
        sh += sh;
    }

    /* The first thread in the warp writes the reduced energies */
    if (tidx == 0)
    {
        atomicAdd((int*)e_lj, E_lj);
        atomicAdd((int*)e_el, E_el);
    }
}

static __forceinline__ __device__ float calculate_lj_ewald_c6grid(const cu_nbparam_t nbparam, int typei, int typej)
{
    return LDG(&nbparam.nbfp_comb[2 * typei]) * LDG(&nbparam.nbfp_comb[2 * typej]);
}

static __forceinline__ __device__ void calculate_lj_ewald_comb_geom_F_E(const cu_nbparam_t nbparam,
                                                                        int                typei,
                                                                        int                typej,
                                                                        float              r2,
                                                                        float              inv_r2,
                                                                        float  lje_coeff2,
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
    expmcr2   = expf(-cr2);
    poly      = 1.0f + cr2 + 0.5f * cr2 * cr2;

    /* Subtract the grid force from the total LJ force */
    *F_invr += c6grid * (inv_r6_nm - expmcr2 * (inv_r6_nm * poly + lje_coeff6_6)) * inv_r2;

    /* Shift should be applied only to real LJ pairs */
    sh_mask = nbparam.sh_lj_ewald * int_bit;
    *E_lj += c_oneSixth * c6grid * (inv_r6_nm * (1.0f - expmcr2 * poly) + sh_mask);
}

static __forceinline__ __device__ float pmecorrF(float z2)
{
    const float FN6 = -1.7357322914161492954e-8f;
    const float FN5 = 1.4703624142580877519e-6f;
    const float FN4 = -0.000053401640219807709149f;
    const float FN3 = 0.0010054721316683106153f;
    const float FN2 = -0.019278317264888380590f;
    const float FN1 = 0.069670166153766424023f;
    const float FN0 = -0.75225204789749321333f;

    const float FD4 = 0.0011193462567257629232f;
    const float FD3 = 0.014866955030185295499f;
    const float FD2 = 0.11583842382862377919f;
    const float FD1 = 0.50736591960530292870f;
    const float FD0 = 1.0f;

    float z4;
    float polyFN0, polyFN1, polyFD0, polyFD1;

    z4 = z2 * z2;

    polyFD0 = FD4 * z4 + FD2;
    polyFD1 = FD3 * z4 + FD1;
    polyFD0 = polyFD0 * z4 + FD0;
    polyFD0 = polyFD1 * z2 + polyFD0;

    polyFD0 = 1.0f / polyFD0;

    polyFN0 = FN6 * z4 + FN4;
    polyFN1 = FN5 * z4 + FN3;
    polyFN0 = polyFN0 * z4 + FN2;
    polyFN1 = polyFN1 * z4 + FN1;
    polyFN0 = polyFN0 * z4 + FN0;
    polyFN0 = polyFN1 * z2 + polyFN0;

    return polyFN0 * polyFD0;
}

static __forceinline__ __device__ void calculate_force_switch_F_E(const cu_nbparam_t nbparam,
                                                                  float              c6,
                                                                  float              c12,
                                                                  float              inv_r,
                                                                  float              r2,
                                                                  float*             F_invr,
                                                                  float*             E_lj)
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
    r_switch = r_switch >= 0.0f ? r_switch : 0.0f;

    *F_invr += -c6 * (disp_shift_V2 + disp_shift_V3 * r_switch) * r_switch * r_switch * inv_r
               + c12 * (repu_shift_V2 + repu_shift_V3 * r_switch) * r_switch * r_switch * inv_r;
    *E_lj += c6 * (disp_shift_F2 + disp_shift_F3 * r_switch) * r_switch * r_switch * r_switch
             - c12 * (repu_shift_F2 + repu_shift_F3 * r_switch) * r_switch * r_switch * r_switch;
}
//[ADH]
#define EL_EWALD_ANA
#define LJ_COMB_LB

//[EAG]
//#define EL_EWALD_ANA
//#define VDW_CUTOFF_CHECK
//#define LJ_FORCE_SWITCH

//[AQP]
//#define EL_EWALD_ANA
//#define LJ_FORCE_SWITCH

//#define EL_RF
//#define LJ_COMB_LB
//#define LJ_EWALD_COMB_GEOM
//#define CALC_ENERGIES

#if defined EL_EWALD_ANA || defined EL_EWALD_TAB
/* Note: convenience macro, needs to be undef-ed at the end of the file. */
#    define EL_EWALD_ANY
#endif

#if defined EL_EWALD_ANY || defined EL_RF || defined LJ_EWALD \
        || (defined EL_CUTOFF && defined CALC_ENERGIES)

#    define EXCLUSION_FORCES
#endif

#if defined LJ_EWALD_COMB_GEOM || defined LJ_EWALD_COMB_LB
#    define LJ_EWALD
#endif

#if defined LJ_COMB_GEOM || defined LJ_COMB_LB
#    define LJ_COMB
#endif

#define NTHREAD_Z (1)
#define MIN_BLOCKS_PER_MP (16)
#define THREADS_PER_BLOCK (c_clSize * c_clSize * NTHREAD_Z)

//#ifdef PRUNE_NBL
//#    ifdef CALC_ENERGIES
//        __global__ void NB_KERNEL_FUNC_NAME(nbnxn_kernel, _VF_prune_cuda)
//#    else
//        __global__ void NB_KERNEL_FUNC_NAME(nbnxn_kernel, _F_prune_cuda)
//#    endif /* CALC_ENERGIES */
//#else
//#    ifdef CALC_ENERGIES
//        __global__ void NB_KERNEL_FUNC_NAME(nbnxn_kernel, _VF_cuda)
//#    else
//        __global__ void NB_KERNEL_FUNC_NAME(nbnxn_kernel, _F_cuda)
//#    endif /* CALC_ENERGIES */
//#endif     /* PRUNE_NBL */

//__launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
__launch_bounds__(THREADS_PER_BLOCK)
__global__ void nbnxn_kernel(const cu_atomdata_t atdat, const cu_nbparam_t nbparam, const cu_plist_t plist, bool bCalcFshift)
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

#    ifdef CALC_ENERGIES
#        ifdef EL_EWALD_ANY
    float                beta        = nbparam.ewald_beta;
    float                ewald_shift = nbparam.sh_ewald;
#        else
    float c_rf = nbparam.c_rf;
#        endif /* EL_EWALD_ANY */
    float*               e_lj        = atdat.e_lj;
    float*               e_el        = atdat.e_el;
#    endif     /* CALC_ENERGIES */

    /* thread/block/warp id-s */
    unsigned int tidxi  = hipThreadIdx_x;
    unsigned int tidxj  = hipThreadIdx_y;
    unsigned int tidx   = hipThreadIdx_y * hipBlockDim_x + hipThreadIdx_x;
#if NTHREAD_Z == 1
    unsigned int tidxz  = 0;
#else
    unsigned int tidxz  = hipThreadIdx_z;
#endif
    unsigned int bidx   = hipBlockIdx_x;

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
    unsigned int wexcl, imask, mask_ji;
    float4       xqbuf;
    float3       xi, xj, rv, f_ij, fcj_buf;
    float3       fci_buf[c_numClPerSupercl]; /* i force buffer */
    nbnxn_sci_t  nb_sci;

    /*! i-cluster interaction mask for a super-cluster with all c_numClPerSupercl=8 bits set */
    const unsigned superClInteractionMask = ((1U << c_numClPerSupercl) - 1U);

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
    sm_nextSlotPtr += (c_numClPerSupercl * c_clSize * sizeof(*xqib));

    /* shmem buffer for cj, for each warp separately */
    int* cjs = (int*)(sm_nextSlotPtr);
    /* the cjs buffer's use expects a base pointer offset for pairs of warps in the j-concurrent execution */
    cjs += tidxz * c_nbnxnGpuClusterpairSplit * c_nbnxnGpuJgroupSize;
    sm_nextSlotPtr += (NTHREAD_Z * c_nbnxnGpuClusterpairSplit * c_nbnxnGpuJgroupSize * sizeof(*cjs));

#    ifndef LJ_COMB
    /* shmem buffer for i atom-type pre-loading */
    int* atib = (int*)sm_nextSlotPtr;
    sm_nextSlotPtr += (c_numClPerSupercl * c_clSize * sizeof(*atib));
#    else
    /* shmem buffer for i-atom LJ combination rule parameters */
    float2* ljcpib = (float2*)sm_nextSlotPtr;
    sm_nextSlotPtr += (c_numClPerSupercl * c_clSize * sizeof(*ljcpib));
#    endif
    /*********************************************************************/

    nb_sci     = pl_sci[bidx];         /* my i super-cluster's index = current bidx */
    sci        = nb_sci.sci;           /* super-cluster */
    cij4_start = nb_sci.cj4_ind_start; /* first ...*/
    cij4_end   = nb_sci.cj4_ind_end;   /* and last index of j clusters */

    if (tidxz == 0)
    {
        /* Pre-load i-atom x and q into shared memory */
        ci = sci * c_numClPerSupercl + tidxj;
        ai = ci * c_clSize + tidxi;

        float* shiftptr = (float*)&shift_vec[nb_sci.shift];
        xqbuf = xq[ai] + make_float4(LDG(shiftptr), LDG(shiftptr + 1), LDG(shiftptr + 2), 0.0f);
        xqbuf.w *= nbparam.epsfac;
        xqib[tidxj * c_clSize + tidxi] = xqbuf;

#    ifndef LJ_COMB
        /* Pre-load the i-atom types into shared memory */
        atib[tidxj * c_clSize + tidxi] = atom_types[ai];
#    else
        /* Pre-load the LJ combination parameters into shared memory */
        ljcpib[tidxj * c_clSize + tidxi] = lj_comb[ai];
#    endif
    }
    __syncthreads();

    for (i = 0; i < c_numClPerSupercl; i++)
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
    if (nb_sci.shift == CENTRAL && pl_cj4[cij4_start].cj[0] == sci * c_numClPerSupercl)
    {
        /* we have the diagonal: add the charge and LJ self interaction energy term */
        for (i = 0; i < c_numClPerSupercl; i++)
        {
#            if defined EL_EWALD_ANY || defined EL_RF || defined EL_CUTOFF
            qi = xqib[i * c_clSize + tidxi].w;
            E_el += qi * qi;
#            endif

#            ifdef LJ_EWALD
#                if DISABLE_CUDA_TEXTURES
            E_lj += LDG(
                    &nbparam.nbfp[atom_types[(sci * c_numClPerSupercl + i) * c_clSize + tidxi] * (ntypes + 1) * 2]);
#                else
            E_lj += tex1Dfetch<float>(
                    nbparam.nbfp_texobj,
                    atom_types[(sci * c_numClPerSupercl + i) * c_clSize + tidxi] * (ntypes + 1) * 2);
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
    const int nonSelfInteraction = !(nb_sci.shift == CENTRAL & tidxj <= tidxi);
#    endif

    /* loop over the j clusters = seen by any of the atoms in the current super-cluster;
     * The loop stride NTHREAD_Z ensures that consecutive warps-pairs are assigned
     * consecutive j4's entries.
     */
    for (j4 = cij4_start + tidxz; j4 < cij4_end; j4 += NTHREAD_Z)
    {
        wexcl_idx = pl_cj4[j4].imei[0].excl_ind;
        imask     = pl_cj4[j4].imei[0].imask;
        wexcl     = excl[wexcl_idx].pair[(tidx) & (warpSize - 1)];

#    ifndef PRUNE_NBL
        if (imask)
#    endif
        {
            /* Pre-load cj into shared memory on both warps separately */
            if (((tidxj * hipBlockDim_x) % warpSize == 0) & (tidxi < c_nbnxnGpuJgroupSize))
            {
                cjs[tidxi] = pl_cj4[j4].cj[tidxi];
            }
//            __syncwarp(c_fullWarpMask); //cm todo
              __all(1);

            /* Unrolling this loop
               - with pruning leads to register spilling;
               - on Kepler and later it is much slower;
               Tested with up to nvcc 7.5 */
#    if !defined PRUNE_NBL
#        pragma unroll 4
#    endif
            for (jm = 0; jm < c_nbnxnGpuJgroupSize; jm++)
            {
                if (imask & (superClInteractionMask << (jm * c_numClPerSupercl)))
                {
                    mask_ji = (1U << (jm * c_numClPerSupercl));

                    cj = cjs[jm];
                    aj = cj * c_clSize + tidxj;

                    /* load j atom data */
                    xqbuf = xq[aj];
                    xj    = make_float3(xqbuf.x, xqbuf.y, xqbuf.z);
                    qj_f  = xqbuf.w;
#    ifndef LJ_COMB
                    typej = atom_types[aj];
#    else
                    ljcp_j = lj_comb[aj];
#    endif

                    fcj_buf = make_float3(0.0f);

#    if !defined PRUNE_NBL
#        pragma unroll 8
#    endif
                    for (i = 0; i < c_numClPerSupercl; i++)
                    {
                        if (imask & mask_ji)
                        {
                            ci = sci * c_numClPerSupercl + i; /* i cluster index */

                            /* all threads load an atom from i cluster ci into shmem! */
                            xqbuf = xqib[i * c_clSize + tidxi];
                            xi    = make_float3(xqbuf.x, xqbuf.y, xqbuf.z);

                            /* distance between i and j atoms */
                            rv = xi - xj;
                            r2 = norm2(rv);

#    ifdef PRUNE_NBL
                            /* If _none_ of the atoms pairs are in cutoff range,
                               the bit corresponding to the current
                               cluster-pair in imask gets set to 0. */
                            //if (!__any_sync(c_fullWarpMask, r2 < rlist_sq)) //cm todo
                            if (!__any(r2 < rlist_sq))
                            {
                                imask &= ~mask_ji;
                            }
#    endif

                            int_bit = (wexcl & mask_ji) ? 1.0f : 0.0f;

                            /* cutoff & exclusion check */
#    ifdef EXCLUSION_FORCES
                            if ((r2 < rcoulomb_sq) * (nonSelfInteraction | (ci != cj)))
#    else
                            if ((r2 < rcoulomb_sq) * int_bit)
#    endif
                            {
                                /* load the rest of the i-atom parameters */
                                qi = xqbuf.w;

#    ifndef LJ_COMB
                                /* LJ 6*C6 and 12*C12 */
                                typei = atib[i * c_clSize + tidxi];
                                fetch_nbfp_c6_c12(c6, c12, nbparam, ntypes * typei + typej);
#    else
                                ljcp_i       = ljcpib[i * c_clSize + tidxi];
#        ifdef LJ_COMB_GEOM
                                c6           = ljcp_i.x * ljcp_j.x;
                                c12          = ljcp_i.y * ljcp_j.y;
#        else
                                /* LJ 2^(1/6)*sigma and 12*epsilon */
                                sigma   = ljcp_i.x + ljcp_j.x;
                                epsilon = ljcp_i.y * ljcp_j.y;
#            if defined CALC_ENERGIES || defined LJ_FORCE_SWITCH || defined LJ_POT_SWITCH
                                convert_sigma_epsilon_to_c6_c12(sigma, epsilon, &c6, &c12);
#            endif
#        endif /* LJ_COMB_GEOM */
#    endif     /* LJ_COMB */

                                // Ensure distance do not become so small that r^-12 overflows
                                r2      = fmaxf(r2, NBNXN_MIN_RSQ);

                                inv_r  = rsqrtf(r2);
                                inv_r2 = inv_r * inv_r;
#    if !defined LJ_COMB_LB || defined CALC_ENERGIES
                                inv_r6 = inv_r2 * inv_r2 * inv_r2;
#        ifdef EXCLUSION_FORCES
                                /* We could mask inv_r2, but with Ewald
                                 * masking both inv_r6 and F_invr is faster */
                                inv_r6 *= int_bit;
#        endif /* EXCLUSION_FORCES */

                                F_invr = inv_r6 * (c12 * inv_r6 - c6) * inv_r2;
#        if defined CALC_ENERGIES || defined LJ_POT_SWITCH
                                E_lj_p = int_bit
                                         * (c12 * (inv_r6 * inv_r6 + nbparam.repulsion_shift.cpot) * c_oneTwelveth
                                            - c6 * (inv_r6 + nbparam.dispersion_shift.cpot) * c_oneSixth);
#        endif
#    else /* !LJ_COMB_LB || CALC_ENERGIES */
                                float sig_r  = sigma * inv_r;
                                float sig_r2 = sig_r * sig_r;
                                float sig_r6 = sig_r2 * sig_r2 * sig_r2;
#        ifdef EXCLUSION_FORCES
                                sig_r6 *= int_bit;
#        endif /* EXCLUSION_FORCES */

                                F_invr = epsilon * sig_r6 * (sig_r6 - 1.0f) * inv_r2;
#    endif     /* !LJ_COMB_LB || CALC_ENERGIES */

#    ifdef LJ_FORCE_SWITCH
#        ifdef CALC_ENERGIES
                                calculate_force_switch_F_E(nbparam, c6, c12, inv_r, r2, &F_invr, &E_lj_p);
#        else
                                calculate_force_switch_F(nbparam, c6, c12, inv_r, r2, &F_invr);
#        endif /* CALC_ENERGIES */
#    endif     /* LJ_FORCE_SWITCH */


#    ifdef LJ_EWALD
#        ifdef LJ_EWALD_COMB_GEOM
#            ifdef CALC_ENERGIES
                                calculate_lj_ewald_comb_geom_F_E(nbparam, typei, typej, r2, inv_r2,
                                                                 lje_coeff2, lje_coeff6_6, int_bit,
                                                                 &F_invr, &E_lj_p);
#            else
                                calculate_lj_ewald_comb_geom_F(nbparam, typei, typej, r2, inv_r2,
                                                               lje_coeff2, lje_coeff6_6, &F_invr);
#            endif /* CALC_ENERGIES */
#        elif defined LJ_EWALD_COMB_LB
                                calculate_lj_ewald_comb_LB_F_E(nbparam, typei, typej, r2, inv_r2,
                                                               lje_coeff2, lje_coeff6_6,
#            ifdef CALC_ENERGIES
                                                               int_bit, &F_invr, &E_lj_p
#            else
                                                               0, &F_invr, nullptr
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
                                vdw_in_range = (r2 < rvdw_sq) ? 1.0f : 0.0f;
                                F_invr *= vdw_in_range;
#        ifdef CALC_ENERGIES
                                E_lj_p *= vdw_in_range;
#        endif
#    endif /* VDW_CUTOFF_CHECK */

#    ifdef CALC_ENERGIES
                                E_lj += E_lj_p;
#    endif


#    ifdef EL_CUTOFF
#        ifdef EXCLUSION_FORCES
                                F_invr += qi * qj_f * int_bit * inv_r2 * inv_r;
#        else
                                F_invr += qi * qj_f * inv_r2 * inv_r;
#        endif
#    endif
#    ifdef EL_RF
                                F_invr += qi * qj_f * (int_bit * inv_r2 * inv_r - two_k_rf);
#    endif
#    if defined   EL_EWALD_ANA
                                F_invr += qi * qj_f
                                          * (int_bit * inv_r2 * inv_r + pmecorrF(beta2 * r2) * beta3);
#    elif defined EL_EWALD_TAB
                                F_invr += qi * qj_f
                                          * (int_bit * inv_r2
                                             - interpolate_coulomb_force_r(nbparam, r2 * inv_r))
                                          * inv_r;
#    endif /* EL_EWALD_ANA/TAB */

#    ifdef CALC_ENERGIES
#        ifdef EL_CUTOFF
                                E_el += qi * qj_f * (int_bit * inv_r - c_rf);
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
                                f_ij = rv * F_invr;

                                /* accumulate j forces in registers */
                                //fcj_buf -= f_ij; //cm todo
                                fcj_buf = fcj_buf - f_ij;

                                /* accumulate i forces in registers */
                                //fci_buf[i] += f_ij; //cm todo
                                fci_buf[i] = fci_buf[i] + f_ij;
                            }
                        }

                        /* shift the mask bit by 1 */
                        mask_ji += mask_ji;
                    }

                    /* reduce j forces */
                    reduce_force_j_warp_shfl(fcj_buf, f, tidxi, aj);
                }
            }
#    ifdef PRUNE_NBL
            /* Update the imask with the new one which does not contain the
               out of range clusters anymore. */
            pl_cj4[j4].imei[0].imask = imask;
#    endif
        }
        // avoid shared memory WAR hazards between loop iterations
//        __syncwarp(c_fullWarpMask); //cm todo
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
    for (i = 0; i < c_numClPerSupercl; i++)
    {
        ai = (sci * c_numClPerSupercl + i) * c_clSize + tidxi;
        reduce_force_i_warp_shfl(fci_buf[i], f, &fshift_buf, bCalcFshift, tidxj, ai);
    }

    /* aadd up local shift forces into global mem, tidxj indexes x,y,z */
    if (bCalcFshift && tidxj < 3)
    {
        atomicAdd((int*)(&(atdat.fshift[nb_sci.shift].x) + tidxj), fshift_buf);
    }

#    ifdef CALC_ENERGIES
    /* reduce the energies over warps and store into global memory */
    reduce_energy_warp_shfl(E_lj, E_el, e_lj, e_el, tidx, c_fullWarpMask);
#    endif
}

struct basicInfo {
    int natoms;
    int natoms_local;
    int nalloc;
    int ntypes;
    bool bShiftVecUploaded;
    int na_c;
    int nsci;
    int sci_nalloc;
    int ncj4;
    int cj4_nalloc;
    int nimask;
    int imask_nalloc;
    int nexcl;
    int excl_nalloc;
};

void initBasicInfo(const char* fileName, basicInfo *info) {
    std::ifstream myfile;
    myfile.open(fileName);

    myfile >> info->natoms;
    myfile >> info->natoms_local;
    myfile >> info->nalloc;
    myfile >> info->ntypes;
    myfile >> info->bShiftVecUploaded;
    myfile >> info->na_c;
    myfile >> info->nsci;
    myfile >> info->sci_nalloc;
    myfile >> info->ncj4;
    myfile >> info->cj4_nalloc;
    myfile >> info->nimask;
    myfile >> info->imask_nalloc;
    myfile >> info->nexcl;
    myfile >> info->excl_nalloc;

    myfile.close();
}

void dumpBasicInfo(basicInfo *info) {
    std::cout << info->natoms << std::endl;
    std::cout << info->natoms_local << std::endl;
    std::cout << info->nalloc << std::endl;
    std::cout << info->ntypes << std::endl;
    std::cout << info->bShiftVecUploaded << std::endl;
    std::cout << info->na_c << std::endl;
    std::cout << info->nsci << std::endl;
    std::cout << info->sci_nalloc << std::endl;
    std::cout << info->ncj4 << std::endl;
    std::cout << info->cj4_nalloc << std::endl;
    std::cout << info->nimask << std::endl;
    std::cout << info->imask_nalloc << std::endl;
    std::cout << info->nexcl << std::endl;
    std::cout << info->excl_nalloc << std::endl; 
}

void copy_xq_to_gpu(cu_atomdata_t *adat) {
    int natoms = adat->natoms;
    float *xq = new float[natoms*4];
    
    std::ifstream myfile;
    myfile.open("xq.txt");

    for (int i = 0; i < natoms*4; i++) {
        myfile >> xq[i];
    }

    hipMalloc((void**)&adat->xq, natoms * sizeof(float4));
    hipMemcpyHtoD(adat->xq, xq, natoms * sizeof(*adat->xq));

    myfile.close();

    delete []xq;
}

void dump_xq(cu_atomdata_t *adat) {
    std::ofstream myfile("xq_verify.txt", std::ofstream::out);
    int natoms = adat->natoms;

    float *xq   = new float[natoms*4];
    hipMemcpyDtoH(xq, adat->xq, natoms * sizeof(*adat->xq));

    for (int i = 0; i < natoms*4; i++) {
        myfile << xq[i] << std::endl;
    }

    myfile.close();

    delete []xq;
}	

void copy_energy_to_gpu(cu_atomdata_t *adat) {
    float *e_lj = new float[1];
    float *e_el = new float[1];
    
    std::ifstream myfile;
    myfile.open("energy.txt");
    myfile >> e_lj[0];
    myfile >> e_el[0];

    hipMalloc((void**)&adat->e_lj, sizeof(float));
    hipMalloc((void**)&adat->e_el, sizeof(float));
    hipMemcpyHtoD(adat->e_lj, e_lj, sizeof(float));
    hipMemcpyHtoD(adat->e_el, e_el, sizeof(float));

    myfile.close();

    delete []e_lj;
    delete []e_el;
}

void copy_atom_types_to_gpu(cu_atomdata_t *adat) {
    int natoms = adat->natoms;

    int *atom_types = new int[natoms];

    std::ifstream myfile;
    myfile.open("atom_types.txt");

    for (int i = 0; i < natoms; i++) {
        myfile >> atom_types[i];
    }

    hipMalloc((void**)&adat->atom_types, natoms * sizeof(int));
    hipMemcpyHtoD(adat->atom_types, atom_types, natoms * sizeof(int));

    myfile.close();

    delete []atom_types;
}

void copy_lj_comb_to_gpu(cu_atomdata_t *adat) {
    int natoms = adat->natoms;

    float *lj_comb = new float[natoms*2];

    std::ifstream myfile;
    myfile.open("lj_comb.txt");

    for (int i = 0; i < natoms*2; i++) {
        myfile >> lj_comb[i];
    }

    hipMalloc((void**)&adat->lj_comb, natoms * sizeof(float2));
    hipMemcpyHtoD(adat->lj_comb, lj_comb, natoms * sizeof(float2));

    myfile.close();

    delete []lj_comb;
}

void copy_shift_vec_to_gpu(cu_atomdata_t *adat) {
    int natoms = adat->natoms;

    float *shift_vec = new float[natoms*3];

    std::ifstream myfile;
    myfile.open("shift_vec.txt");

    for (int i = 0; i < natoms*3; i++) {
        myfile >> shift_vec[i];
    }

    hipMalloc((void**)&adat->shift_vec, natoms * sizeof(float3));
    hipMemcpyHtoD(adat->shift_vec, shift_vec, natoms * sizeof(float3));

    myfile.close();

    delete []shift_vec;
}

//========================================
void copy_sci_to_gpu(cu_plist_t *plist) {
    int nsci = plist->nsci;

    nbnxn_sci_t *sci = new nbnxn_sci_t[nsci];

    std::ifstream myfile;
    myfile.open("nbnxn_sci_t.txt");

    for (int i = 0; i < nsci; i++) {
        myfile >> sci[i].sci;
        myfile >> sci[i].shift;
        myfile >> sci[i].cj4_ind_start;
        myfile >> sci[i].cj4_ind_end;
    }

    hipMalloc((void**)&plist->sci, nsci * sizeof(nbnxn_sci_t));
    hipMemcpyHtoD(plist->sci, sci, nsci * sizeof(*plist->sci));

    myfile.close();

    delete []sci;
}

void dump_sci(cu_plist_t *plist) {
    std::ofstream myfile("nbnxn_sci_t_verify.txt", std::ofstream::out);
    int nsci = plist->nsci;

    nbnxn_sci_t *sci = new nbnxn_sci_t[nsci];

    hipMemcpyDtoH(sci, plist->sci, nsci * sizeof(*plist->sci));

    for (int i = 0; i < nsci; i++) {
	myfile << sci[i].sci << std::endl;
        myfile << sci[i].shift << std::endl;
        myfile << sci[i].cj4_ind_start << std::endl;
        myfile << sci[i].cj4_ind_end << std::endl;
    }

    myfile.close();

    delete []sci;
}

void copy_cj4_to_gpu(cu_plist_t *plist) {
    int ncj4 = plist->ncj4;

    nbnxn_cj4_t *cj4 = new nbnxn_cj4_t[ncj4];

    std::ifstream myfile;
    myfile.open("nbnxn_cj4_t.txt");

    for (int i = 0; i < ncj4; i++) {
        for (int j = 0; j < c_nbnxnGpuJgroupSize; j++) {
            myfile >> cj4[i].cj[j];
	}
	for (int k = 0; k < c_nbnxnGpuClusterpairSplit; k++) {
            myfile >> cj4[i].imei[k].imask;
            myfile >> cj4[i].imei[k].excl_ind;
        }
    }

    hipMalloc((void**)&plist->cj4, ncj4 * sizeof(nbnxn_cj4_t));
    hipMemcpyHtoD(plist->cj4, cj4, ncj4 * sizeof(*plist->cj4));

    myfile.close();

    delete []cj4;
}

void copy_excl_to_gpu(cu_plist_t *plist) {
    int nexcl = plist->nexcl;

    nbnxn_excl_t *excl = new nbnxn_excl_t[nexcl];

    std::ifstream myfile;
    myfile.open("nbnxn_excl_t.txt");

    for (int i = 0; i < nexcl; i++) {
        for (int j = 0; j < c_nbnxnGpuExclSize; j++) {
            myfile >> excl[i].pair[j];
	}	
    }

    hipMalloc((void**)&plist->excl, nexcl * sizeof(nbnxn_excl_t));
    hipMemcpyHtoD(plist->excl, excl, nexcl * sizeof(*plist->excl));

    myfile.close();

    delete []excl;
}

void init_nbparam(cu_nbparam_t *nbp, int nbfpSize) {

    std::ifstream myfile;
    myfile.open("nbp.txt");

    myfile >> nbp->eeltype;
    myfile >> nbp->vdwtype;

    myfile >> nbp->epsfac;
    myfile >> nbp->c_rf;
    myfile >> nbp->two_k_rf;
    myfile >> nbp->ewald_beta;
    myfile >> nbp->sh_ewald;
    myfile >> nbp->sh_lj_ewald;
    myfile >> nbp->ewaldcoeff_lj;

    myfile >> nbp->rcoulomb_sq;

    myfile >> nbp->rvdw_sq;
    myfile >> nbp->rvdw_switch;
    myfile >> nbp->rlistOuter_sq;
    myfile >> nbp->rlistInner_sq;
    myfile >> nbp->rlistInner_sq;

    myfile >> nbp->dispersion_shift.c2;
    myfile >> nbp->dispersion_shift.c3;
    myfile >> nbp->dispersion_shift.cpot;
    myfile >> nbp->repulsion_shift.c2;
    myfile >> nbp->repulsion_shift.c3;
    myfile >> nbp->repulsion_shift.cpot;

    float *nbfp  = new float[nbfpSize];

    for (int i = 0; i < nbfpSize; i++) {
        myfile >> nbfp[i];
    }

    hipMalloc((void**)&nbp->nbfp,  nbfpSize* sizeof(float));
    hipMemcpyHtoD(nbp->nbfp, nbfp, nbfpSize * sizeof(float));

    myfile.close();
    delete []nbfp;
}

void dump_nbparam(cu_nbparam_t *nbp, int nbfpSize) {

    std::ofstream myfile("nbp_verify.txt", std::ofstream::out);

    myfile << nbp->eeltype  << std::endl;
    myfile << nbp->vdwtype  << std::endl;

    myfile << nbp->epsfac        << std::endl;
    myfile << nbp->c_rf          << std::endl;
    myfile << nbp->two_k_rf      << std::endl;
    myfile << nbp->ewald_beta    << std::endl;
    myfile << nbp->sh_ewald      << std::endl;
    myfile << nbp->sh_lj_ewald   << std::endl;
    myfile << nbp->ewaldcoeff_lj << std::endl;

    myfile << nbp->rcoulomb_sq << std::endl;

    myfile << nbp->rvdw_sq << std::endl;
    myfile << nbp->rvdw_switch << std::endl;
    myfile << nbp->rlistOuter_sq << std::endl;
    myfile << nbp->rlistInner_sq << std::endl;
    myfile << nbp->rlistInner_sq << std::endl;

    myfile << nbp->dispersion_shift.c2 << std::endl;
    myfile << nbp->dispersion_shift.c3 << std::endl;
    myfile << nbp->dispersion_shift.cpot << std::endl;
    myfile << nbp->repulsion_shift.c2 << std::endl;
    myfile << nbp->repulsion_shift.c3 << std::endl;
    myfile << nbp->repulsion_shift.cpot << std::endl; 

    float *nbfp  = new float[nbfpSize];
    hipMemcpyDtoH(nbfp, nbp->nbfp, nbfpSize * sizeof(*nbp->nbfp));

    for (int i = 0; i < nbfpSize; i++) {
        myfile << nbfp[i] << std::endl;
    }

    myfile.close();
    delete []nbfp;
}

//#undef NTHREAD_Z
//#undef MIN_BLOCKS_PER_MP
//#undef THREADS_PER_BLOCK
//
//#undef EL_EWALD_ANY
//#undef EXCLUSION_FORCES
//#undef LJ_EWALD
//
//#undef LJ_COMB
//
enum eelCu
{
    eelCuCUT,
    eelCuRF,
    eelCuEWALD_TAB,
    eelCuEWALD_TAB_TWIN,
    eelCuEWALD_ANA,
    eelCuEWALD_ANA_TWIN,
    eelCuNR
};

/*! \brief VdW CUDA kernel flavors.
 *
 * The enumerates values correspond to the LJ implementations in the CUDA non-bonded
 * kernels.
 *
 * The column-order of pointers to different electrostatic kernels defined in
 * nbnxn_cuda.cu by the nb_*_kfunc_ptr function pointer table
 * should match the order of enumerated types below.
 */
enum evdwCu
{
    evdwCuCUT,
    evdwCuCUTCOMBGEOM,
    evdwCuCUTCOMBLB,
    evdwCuFSWITCH,
    evdwCuPSWITCH,
    evdwCuEWALDGEOM,
    evdwCuEWALDLB,
    evdwCuNR
};


static inline int calc_shmem_required_nonbonded(const int               num_threads_z,
                                                const cu_nbparam_t*     nbp)
{
    int shmem;


    /* size of shmem (force-buffers/xq/atom type preloading) */
    /* NOTE: with the default kernel on sm3.0 we need shmem only for pre-loading */
    /* i-atom x+q in shared memory */
    shmem = c_numClPerSupercl * c_clSize * sizeof(float4);
    /* cj in shared memory, for each warp separately */
    shmem += num_threads_z * c_nbnxnGpuClusterpairSplit * c_nbnxnGpuJgroupSize * sizeof(int);

    if (nbp->vdwtype == evdwCuCUTCOMBGEOM || nbp->vdwtype == evdwCuCUTCOMBLB)
    {
        /* i-atom LJ combination parameters in shared memory */
        shmem += c_numClPerSupercl * c_clSize * sizeof(float2);
    }
    else
    {
        /* i-atom types in shared memory */
        shmem += c_numClPerSupercl * c_clSize * sizeof(int);
    }

    return shmem;
}


int main (int argc, char* argv[]) {
    bool bCalcFshift = atoi(argv[1])!=0?true:false;

    basicInfo info;
    initBasicInfo("basic.txt", &info);
    //dumpBasicInfo(&info);

    // atomdata_t
    cu_atomdata_t *adat;
    adat = (cu_atomdata_t*)calloc(info.natoms, sizeof(cu_atomdata_t));
    adat->natoms            = info.natoms;
    adat->natoms_local      = info.natoms_local;
    adat->nalloc            = info.nalloc;
    adat->ntypes            = info.ntypes;
    adat->bShiftVecUploaded = info.bShiftVecUploaded;

    copy_xq_to_gpu(adat);
    copy_energy_to_gpu(adat);
    copy_atom_types_to_gpu(adat);
    copy_lj_comb_to_gpu(adat);
    copy_shift_vec_to_gpu(adat);

    //create output buffer
    hipMalloc((void**)&adat->f, adat->natoms * sizeof(float3));
    hipMalloc((void**)&adat->fshift, adat->natoms * sizeof(float3));

    // cu_plist_t
    cu_plist_t *plist;
    plist = (cu_plist_t*)calloc(1, sizeof(cu_plist_t));
    plist->na_c         = info.na_c;
    plist->nsci         = info.nsci;
    plist->sci_nalloc   = info.sci_nalloc;
    plist->ncj4         = info.ncj4;
    plist->cj4_nalloc   = info.cj4_nalloc;
    plist->nimask       = info.ncj4;
    plist->imask_nalloc = info.imask_nalloc;
    plist->nexcl        = info.nexcl;
    plist->excl_nalloc  = info.excl_nalloc;

    copy_sci_to_gpu(plist);
    copy_cj4_to_gpu(plist);
    copy_excl_to_gpu(plist);

    //dump_sci(plist);

    // cu_nbparam_t
    cu_nbparam_t *nbparam = new cu_nbparam_t();
    init_nbparam(nbparam, 2*adat->ntypes*adat->ntypes);
    //dump_nbparam(nbparam, 2*adat->ntypes*adat->ntypes);
    //
    int num_threads_z = 1;
    size_t sharedMemorySize = calc_shmem_required_nonbonded(num_threads_z, nbparam);
    
    for (int iter=0; iter<1000; iter++) {
        hipLaunchKernelGGL(nbnxn_kernel, dim3(plist->nsci, 1, 1), 
        	    dim3(c_clSize, c_clSize, num_threads_z), sharedMemorySize, 0
                    , *adat, *nbparam, *plist, bCalcFshift);
    }
    hipStreamSynchronize(0);

    
    return 0;
}
