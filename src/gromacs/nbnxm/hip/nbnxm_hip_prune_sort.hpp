#include "hip/hip_runtime.h"
#include "gmxpre.h"

#include "gromacs/gpu_utils/hip_arch_utils.hpp"
#include "gromacs/gpu_utils/typecasts.hpp"
#include "gromacs/math/utilities.h"
#include "gromacs/pbcutil/ishift.h"

#include "nbnxm_hip_kernel_utils.hpp"
#include "nbnxm_hip_types.h"

#define NTHREAD_Z 1

/*static __device__ unsigned char array_type_bit[] = {
    3u,    12u,  48u, 192u,
    15u,   51u, 195u,  60u,
    204u, 240u, 252u, 243u,
    207u,  63u, 255u};

static __forceinline__ __device__ unsigned char getTypeIndex(unsigned char imask_i)
{
    for( unsigned char index_array_type = 0; index_array_type < BITTYPES; index_array_type++ )
    {
        if( imask_i & array_type_bit[index_array_type] && ( imask_i & ~array_type_bit[index_array_type] ) == 0 )
            return index_array_type;
    }

    return 14u;
}*/

/*! \brief Nonbonded list pruning kernel.
 *
 *  true a new list from immediately after pair-list generation is pruned using rlistOuter,
 *  the pruned masks are stored in a separate buffer and the outer-list is pruned
 *  using the rlistInner distance; when false only the pruning with rlistInner is performed.
 *
 *  Kernel launch parameters:
 *   - #blocks   = #pair lists, blockId = pair list Id
 *   - #threads  = c_clSize^2
 *   - shmem     = see nbnxn_hip.cu:calc_shmem_required_prune()
 *
 *   Each thread calculates an i-j atom distance..
 */
 template<bool haveFreshList>
__launch_bounds__(c_clSize * c_clSize * NTHREAD_Z, 6) __global__
void nbnxn_kernel_sort_j_hip(const NBAtomDataGpu    atdat,
                             const NBParamGpu       nbparam,
                             const Nbnxm::gpu_plist plist,
                             int              numParts,
                             int              part)
#ifdef FUNCTION_DECLARATION_ONLY
; /* Only do function declaration, omit the function body. */

// Add extern declarations so each translation unit understands that
// there will be a definition provided.
extern template __launch_bounds__(c_clSize * c_clSize * NTHREAD_Z, 6) __global__ void
nbnxn_kernel_sort_j_hip<true>(const NBAtomDataGpu, const NBParamGpu, const Nbnxm::gpu_plist, int, int);
extern template __launch_bounds__(c_clSize * c_clSize * NTHREAD_Z, 6) __global__ void
nbnxn_kernel_sort_j_hip<false>(const NBAtomDataGpu, const NBParamGpu, const Nbnxm::gpu_plist, int, int);
#else
{

    /* convenience variables */
    const nbnxn_sci_t*     pl_sci     = haveFreshList ? plist.sci : plist.sci_sorted;
    nbnxn_cj4_t*       pl_cj4         = plist.cj4;
    nbnxn_cj4_ext_t*   pl_cj4_sorted  = plist.cj4_sorted;
    nbnxn_cj_sort_t *  pl_cj_sorted   = plist.cj_sorted;
    const float4*      xq             = atdat.xq;
    const float3*      shift_vec      = asFloat3(atdat.shiftVec);
    FastBuffer<nbnxn_excl_t> excl     = FastBuffer<nbnxn_excl_t>(plist.excl);

    float rlistOuter_sq = nbparam.rlistOuter_sq;
    float rlistInner_sq = nbparam.rlistInner_sq;
    float   rcoulomb_sq = nbparam.rcoulomb_sq;
    /* thread/block/warp id-s */
    unsigned int tidxi = threadIdx.x;
    unsigned int tidxj = threadIdx.y;
    unsigned int gidxj = tidxj / c_subGroup;

    unsigned int tidx  = tidxi + c_clSize * tidxj;

#    if NTHREAD_Z == 1
    unsigned int tidxz = 0;
#    else
    unsigned int tidxz = threadIdx.z;
#    endif

    unsigned int lane_id  = tidx + tidxz * c_clSize * c_clSize;
    unsigned int bidx  = blockIdx.x;
    unsigned int widx  = (threadIdx.y * c_clSize) / c_subWarp; /* warp index */

    /*********************************************************************
     * Set up shared memory pointers.
     * sm_nextSlotPtr should always be updated to point to the "next slot",
     * that is past the last point where data has been stored.
     */
    extern __shared__ char sm_dynamicShmem[];
    char*                  sm_nextSlotPtr = sm_dynamicShmem;
    static_assert(sizeof(char) == 1,
                  "The shared memory offset calculation assumes that char is 1 byte");

    /* shmem buffer for i x+q pre-loading */
    float4* xib = reinterpret_cast<float4*>(sm_nextSlotPtr);
    sm_nextSlotPtr += (c_nbnxnGpuNumClusterPerSupercluster * c_clSize * sizeof(*xib));

    /* shmem buffer for i x+q pre-loading */
    int* jm_count_sm = reinterpret_cast<int*>(sm_nextSlotPtr);
    sm_nextSlotPtr += (BITTYPES + 2) * sizeof(int);

    int* jm_offset_sm = reinterpret_cast<int*>(sm_nextSlotPtr);
    sm_nextSlotPtr += (BITTYPES + 2) * sizeof(int);

    int* current_types = reinterpret_cast<int*>(sm_nextSlotPtr);
    sm_nextSlotPtr += (NTHREAD_Z * 64 * sizeof(int));

    constexpr int items_per_thread = 1 + ((BITTYPES + 2) / (c_clSize * c_clSize * NTHREAD_Z));

    /*********************************************************************/

    nbnxn_sci_t nb_sci =
            pl_sci[bidx * numParts + part]; /* my i super-cluster's index = sciOffset + current bidx * numParts + part */
    int sci        = nb_sci.sci;           /* super-cluster */
    int cij4_start = nb_sci.cj4_ind_start; /* first ...*/
    int cij4_end   = nb_sci.cj4_ind_start + nb_sci.cj4_length;   /* and last index of j clusters */

    constexpr bool c_loadUsingAllXYThreads = (c_clSize == c_nbnxnGpuNumClusterPerSupercluster);
    if (tidxz == 0 && (c_loadUsingAllXYThreads || tidxj < c_nbnxnGpuNumClusterPerSupercluster))
    {
        /* Pre-load i-atom x and q into shared memory */
        int ci = sci * c_nbnxnGpuNumClusterPerSupercluster + tidxj;
        int ai = ci * c_clSize + tidxi;

        /* We don't need q, but using float4 in shmem avoids bank conflicts.
           (but it also wastes L2 bandwidth). */
        float4 tmp                    = xq[ai];
        float4 xi                     = tmp + shift_vec[nb_sci.shift];
        xib[tidxj * c_clSize + tidxi] = xi;
    }

    #pragma unroll
    for(int i = 0; i < items_per_thread; i++)
    {
        int type_index = lane_id * items_per_thread + i;
        if(type_index < BITTYPES + 2)
        {
            jm_count_sm[type_index] = 0;
            jm_offset_sm[type_index] = 0;
        }
    }
    __syncthreads();



    for (int j4 = cij4_start + tidxz; j4 < cij4_end; j4 += NTHREAD_Z)
    {
        pl_cj4_sorted[j4].imei[widx].imask = 0u;

        int wexcl_idx = pl_cj4[j4].imei[widx].excl_ind;
        unsigned int wexcl = excl[wexcl_idx].pair[tidx & (c_subWarp - 1)];

        unsigned int imask = pl_cj4[j4].imei[widx].imask;
        unsigned int imaskFull = imask;
#    pragma unroll c_nbnxnGpuJgroupSize
        for (int jm = 0; jm < c_nbnxnGpuJgroupSize; jm++)
        {
            int cj = pl_cj4[j4].cj[jm];
            int aj = cj * c_clSize + tidxj;

            unsigned int mask_ji = (1U << (jm * c_nbnxnGpuNumClusterPerSupercluster));
            unsigned char mask_i = 1U;
            unsigned char new_mask_i = 0U;

            /* load j atom data */
            float4 tmp = xq[aj];
            float3 xj  = make_float3(tmp.x, tmp.y, tmp.z);

            bool interaction = false, interactionFull = false;
#    pragma unroll c_nbnxnGpuNumClusterPerSupercluster
            for (int i = 0; i < c_nbnxnGpuNumClusterPerSupercluster; i++)
            {
                if (imask & mask_ji /*& wexcl*/)
                {
                    /* load i-cluster coordinates from shmem */
                    float4 xi = xib[i * c_clSize + tidxi];

                    /* distance between i and j atoms */
                    float3 rv = make_float3(xi.x, xi.y, xi.z) - xj;
                    float  r2 = norm2(rv);

                    /*if (!__nb_any((r2 < rlistOuter_sq) && (wexcl & mask_ji), widx))
                    {
                        imaskFull &= ~mask_ji;
                    }*/

                    /* If any atom pair is within range, set the bit
                       corresponding to the current cluster-pair. */
                    if (__nb_any_group((r2 < rlistOuter_sq) && (wexcl & mask_ji), gidxj))
                    {
                        interactionFull = true;
                    }

                    /* If any atom pair is within range, set the bit
                       corresponding to the current cluster-pair. */
                    if (__nb_any_group((r2 < rlistInner_sq) && (wexcl & mask_ji), gidxj))
                    {
                        interaction = true;
                        new_mask_i |= mask_i;
                    }
                }

                mask_ji += mask_ji;
                mask_i  += mask_i;
            }

            // Save the results
            if(tidxi == 0)
            {
                pl_cj_sorted[j4 * c_subGroupJ4Size + jm * c_subGroupN + gidxj].j     = (j4 - cij4_start) * c_subGroupJ4Size + jm * c_subGroupN + gidxj;
                pl_cj_sorted[j4 * c_subGroupJ4Size + jm * c_subGroupN + gidxj].mask  = new_mask_i;
                pl_cj_sorted[j4 * c_subGroupJ4Size + jm * c_subGroupN + gidxj].imask = imaskFull;

                unsigned char type = interactionFull ? (interaction ? 2u : 1u) : 0u;
                pl_cj_sorted[j4 * c_subGroupJ4Size + jm * c_subGroupN + gidxj].type = type;

                //printf("type: %d %d %d\n", static_cast<int>(type), static_cast<int>(new_mask_i), (int)type + new_mask_i);

                atomicAdd(&jm_count_sm[(int)type + new_mask_i], 1);
                //__syncthreads();
            }
        }
    }

    #pragma unroll
    for( int i = 0; i < items_per_thread; i++)
    {
        int type_index = lane_id * items_per_thread + i;
        if( 0 < type_index && type_index < BITTYPES + 2)
        {
            for( unsigned int j = 0; j < type_index; j++)
            {
                jm_offset_sm[BITTYPES + 1 - type_index] += jm_count_sm[BITTYPES + 1 - j];
            }
        }
    }
    __syncthreads();

    int j4_step = 0;
    int half_block_idx  = (threadIdx.y) / (c_clSize / 2);
    for (int j4 = cij4_start + tidxz * 2; j4 < cij4_end; j4 += (2 * NTHREAD_Z))
    {
        if( (j4 + half_block_idx) < cij4_end)
        {
            int j_old      = pl_cj_sorted[(j4 + half_block_idx) * c_subGroupJ4Size + (tidx & 31)].j;
            int j4_orig    = cij4_start + (j_old / (c_subGroupJ4Size));
            int jm_orig    = (j_old & (c_subGroupJ4Size - 1)) / c_subGroupN;
            int tidxj_orig = j_old & (c_subGroupN - 1);

            unsigned int imask = pl_cj_sorted[(j4 + half_block_idx) * c_subGroupJ4Size + (tidx & 31)].imask;
            unsigned char type_char = pl_cj_sorted[(j4 + half_block_idx) * c_subGroupJ4Size + (tidx & 31)].type;
            unsigned char mask = pl_cj_sorted[(j4 + half_block_idx) * c_subGroupJ4Size + (tidx & 31)].mask;

            int type = (int)type_char + mask;

            /*printf("%d %d %d %d x: %d y: %d z: %d\n",
                j_old, j4_orig, j4, static_cast<int>(type),
                tidxi, tidxj, tidxz);*/

            current_types[lane_id] = type;

            __syncthreads();

            int offset = 0;
            for (int current_types_index = 0; current_types_index < (tidxz * 64 + tidx); current_types_index++)
            {
                if(current_types[current_types_index] == type)
                    offset++;
            }

            /*printf("offset - %d %d %d x: %d y: %d z: %d\n",
                offset, static_cast<int>(type), jm_offset_sm[type],
                tidxi, tidxj, tidxz);*/

            int j_new     = jm_offset_sm[type] + offset;
            int j4_new    = cij4_start + (j_new / (c_subGroupJ4Size));
            int jm_new    = (j_new & (c_subGroupJ4Size - 1)) / c_subGroupN;
            int gidxj_new = j_new & (c_subGroupN - 1);


            pl_cj4_sorted[j4_new].j[jm_new * c_subGroupN + gidxj_new]        = j_old;
            pl_cj4_sorted[j4_new].excl_ind[jm_new * c_subGroupN + gidxj_new] = pl_cj4[j4_orig].imei[tidxj_orig/(c_clSize/c_nbnxnGpuClusterpairSplit)].excl_ind;
            pl_cj4_sorted[j4_new].cj[jm_new * c_subGroupN + gidxj_new]       = pl_cj4[j4_orig].cj[jm_orig];

            unsigned int partial_mask = imask & (superClInteractionMask << (jm_orig * c_nbnxnGpuNumClusterPerSupercluster));
            if( jm_new > jm_orig )
                partial_mask = partial_mask << ((jm_new - jm_orig) * c_nbnxnGpuNumClusterPerSupercluster);
            else
                partial_mask = partial_mask >> ((jm_orig - jm_new) * c_nbnxnGpuNumClusterPerSupercluster);

            /*printf("new - %d %d %d %d %d  %u bidx: %d x: %d y: %d z: %d\n",
                    j_new, j4_new, gidxj_new, gidxj_new/(c_clSize/c_nbnxnGpuClusterpairSplit), jm_new, partial_mask,
                    bidx, tidxi, tidxj, tidxz);*/

            atomicOr(&(pl_cj4_sorted[j4_new].imei[gidxj_new/(c_clSize/c_nbnxnGpuClusterpairSplit)].imask), partial_mask);
            pl_cj4_sorted[j4_new].imask[jm_new * c_subGroupN + gidxj_new] = partial_mask;

            if( (cij4_start + j4_step * (2 * NTHREAD_Z) + (lane_id / c_subGroupJ4Size) ) < cij4_end)
                atomicAdd(&jm_offset_sm[current_types[lane_id]], 1);
            __syncthreads();
        }
        j4_step++;
    }
}
#endif /* FUNCTION_DECLARATION_ONLY */

#undef NTHREAD_Z
