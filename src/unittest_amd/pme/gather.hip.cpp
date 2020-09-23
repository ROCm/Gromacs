#include "hip/hip_runtime.h"
#include <iostream>
#include <fstream>

#if ! __HIP_DEVICE_COMPILE__
#include <cassert>
#endif

constexpr int XX                         = 0;
constexpr int YY                         = 1;
constexpr int ZZ                         = 2;
constexpr int DIM                        = 3;
constexpr int c_pmeGpuOrder              = 4;
constexpr int warp_size                  = 64;
constexpr int c_pmeAtomDataAlignment     = 128;
constexpr int c_gatherMaxWarpsPerBlock   = 4;
constexpr int c_gatherMaxThreadsPerBlock = (c_gatherMaxWarpsPerBlock * warp_size);

constexpr int c_pmeSpreadGatherThreadsPerAtom           = c_pmeGpuOrder * c_pmeGpuOrder;
constexpr int c_pmeSpreadGatherThreadsPerAtom4ThPerAtom = c_pmeGpuOrder;

constexpr int c_pmeSpreadGatherAtomsPerWarp =
        (warp_size / c_pmeSpreadGatherThreadsPerAtom);
constexpr int c_pmeSpreadGatherAtomsPerWarp4ThPerAtom =
        (warp_size / c_pmeSpreadGatherThreadsPerAtom4ThPerAtom);

constexpr int c_pmeMaxUnitcellShift = 2;
constexpr int c_pmeNeighborUnitcellCount = 2 * c_pmeMaxUnitcellShift + 1;

constexpr bool c_usePadding = true;
constexpr bool c_skipNeutralAtoms = false;
static const bool c_useAtomDataPrefetch = true;

#define INLINE_EVERYWHERE __host__ __device__ __forceinline__
#define GMX_CUDA_MAX_BLOCKS_PER_MP 0
#define GMX_CUDA_MAX_THREADS_PER_MP 0

constexpr int c_gatherMinBlocksPerMP = GMX_CUDA_MAX_THREADS_PER_MP / c_gatherMaxThreadsPerBlock;

int __device__ __forceinline__ pme_gpu_check_atom_data_index(const int atomDataIndex, const int nAtomData)
{
    return c_usePadding ? 1 : (atomDataIndex < nAtomData);
}

int __device__ __forceinline__ pme_gpu_check_atom_charge(const float coefficient)
{
    assert(isfinite(coefficient));
    return c_skipNeutralAtoms ? (coefficient != 0.0f) : 1;
}

template<int order, int atomsPerWarp>
int INLINE_EVERYWHERE getSplineParamIndexBase(int warpIndex, int atomWarpIndex)
{
    assert((atomWarpIndex >= 0) && (atomWarpIndex < atomsPerWarp));
    const int dimIndex    = 0;
    const int splineIndex = 0;
    // The zeroes are here to preserve the full index formula for reference
    return (((splineIndex + order * warpIndex) * DIM + dimIndex) * atomsPerWarp + atomWarpIndex);
}

template<int order, int atomsPerWarp>
int INLINE_EVERYWHERE getSplineParamIndex(int paramIndexBase, int dimIndex, int splineIndex)
{
    assert((dimIndex >= XX) && (dimIndex < DIM));
    assert((splineIndex >= 0) && (splineIndex < order));
    return (paramIndexBase + (splineIndex * DIM + dimIndex) * atomsPerWarp);
}

template<typename T>
static __forceinline__ __device__ T fetchFromParamLookupTable(const T* d_ptr,
                                                              int      index)
{
    assert(index >= 0);
    T result;

    result = *(d_ptr + index);

    return result;
}

template<typename T, const int atomsPerBlock, const int dataCountPerAtom>
__device__ __forceinline__ void pme_gpu_stage_atom_data(const int nAtoms,
                                                        T* __restrict__ sm_destination,
                                                        const T* __restrict__ gm_source)
{
    static_assert(c_usePadding,
                  "With padding disabled, index checking should be fixed to account for spline "
                  "theta/dtheta pr-warp alignment");
    const int blockIndex       = blockIdx.y * gridDim.x + blockIdx.x;
    const int threadLocalIndex = ((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x) + threadIdx.x;
    const int localIndex       = threadLocalIndex;
    const int globalIndexBase = blockIndex * atomsPerBlock * dataCountPerAtom;
    const int globalIndex     = globalIndexBase + localIndex;
    const int globalCheck =
            pme_gpu_check_atom_data_index(globalIndex, nAtoms * dataCountPerAtom);
    if ((localIndex < atomsPerBlock * dataCountPerAtom) & globalCheck)
    {
        assert(isfinite(float(gm_source[globalIndex])));
        sm_destination[localIndex] = gm_source[globalIndex];
    }
}

template<const int order, const int atomsPerBlock, const int atomsPerWarp, const bool writeSmDtheta, const bool writeGlobal>
__device__ __forceinline__ void calculate_splines(float*                       d_theta,
                                                  float*                       d_dtheta,
                                                  const float*                 realGridSizeFP,
                                                  const float*                 recipBox,
                                                  const float*                 d_fractShiftsTable,
                                                  const int*                   d_gridlineIndicesTable,
                                                  int*                         d_gridlineIndices,
                                                  const int*                   tablesOffsets,
                                                  const int                    nAtoms,
                                                  const int                    atomIndexOffset,
                                                  const float3                 atomX,
                                                  const float                  atomCharge,
                                                  float* __restrict__ sm_theta,
                                                  float* __restrict__ sm_dtheta,
                                                  int* __restrict__ sm_gridlineIndices)
{
    /* Global memory pointers for output */
    float* __restrict__ gm_theta         = d_theta;
    float* __restrict__ gm_dtheta        = d_dtheta;
    int* __restrict__ gm_gridlineIndices = d_gridlineIndices;

    /* Fractional coordinates */
    __shared__ float sm_fractCoords[atomsPerBlock * DIM];

    /* Thread index w.r.t. block */
    const int threadLocalId =
            (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    /* Warp index w.r.t. block - could probably be obtained easier? */
    const int warpIndex = threadLocalId / warp_size;
    /* Atom index w.r.t. warp - alternating 0 1 0 1 .. */
    const int atomWarpIndex = threadIdx.z % atomsPerWarp;
    /* Atom index w.r.t. block/shared memory */
    const int atomIndexLocal = warpIndex * atomsPerWarp + atomWarpIndex;

    /* Atom index w.r.t. global memory */
    const int atomIndexGlobal = atomIndexOffset + atomIndexLocal;
    /* Spline contribution index in one dimension */
    const int threadLocalIdXY = (threadIdx.y * blockDim.x) + threadIdx.x;
    const int orderIndex      = threadLocalIdXY / DIM;
    /* Dimension index */
    const int dimIndex = threadLocalIdXY % DIM;

    /* Multi-purpose index of rvec/ivec atom data */
    const int sharedMemoryIndex = atomIndexLocal * DIM + dimIndex;

    float splineData[order];

    const int localCheck = (dimIndex < DIM) && (orderIndex < 1);
    const int globalCheck = pme_gpu_check_atom_data_index(atomIndexGlobal, nAtoms);

    /* we have 4 threads per atom, but can only use 3 here for the dimensions */
    if (localCheck && globalCheck)
    {
        /* Indices interpolation */

        if (orderIndex == 0)
        {
            int   tableIndex, tInt;
            float n, t;
            assert(atomIndexLocal < DIM * atomsPerBlock);

            switch (dimIndex)
            {
                case XX:
                    tableIndex = tablesOffsets[XX];
                    n          = realGridSizeFP[XX];
                    t          = atomX.x * recipBox[dimIndex*DIM+XX]
                        + atomX.y * recipBox[dimIndex*DIM+YY]
                        + atomX.z * recipBox[dimIndex*DIM+ZZ];
                    break;
                case YY:
                    tableIndex = tablesOffsets[YY];
                    n          = realGridSizeFP[YY];
                    t = atomX.y * recipBox[dimIndex*DIM+YY]
                                + atomX.z * recipBox[dimIndex*DIM+ZZ];
                    break;
                case ZZ:
                    tableIndex = tablesOffsets[ZZ];
                    n          = realGridSizeFP[ZZ];
                    t          = atomX.z * recipBox[dimIndex*DIM+ZZ];
                    break;
            }
            const float shift = c_pmeMaxUnitcellShift;
            /* Fractional coordinates along box vectors, adding a positive shift to ensure t is positive for triclinic boxes */
            t    = (t + shift) * n;
            tInt = (int)t;
            assert(sharedMemoryIndex < atomsPerBlock * DIM);
            sm_fractCoords[sharedMemoryIndex] = t - tInt;
            tableIndex += tInt;
            assert(tInt >= 0);
            assert(tInt < c_pmeNeighborUnitcellCount * n);

            sm_fractCoords[sharedMemoryIndex] +=
                    fetchFromParamLookupTable(d_fractShiftsTable,
                                              tableIndex);
            sm_gridlineIndices[sharedMemoryIndex] =
                    fetchFromParamLookupTable(d_gridlineIndicesTable,
                                              tableIndex);
            if (writeGlobal)
            {
                gm_gridlineIndices[atomIndexOffset * DIM + sharedMemoryIndex] =
                        sm_gridlineIndices[sharedMemoryIndex];
            }
        }
        const int chargeCheck = pme_gpu_check_atom_charge(atomCharge);
        if (chargeCheck)
        {
            float div;
            int o = orderIndex; // This is an index that is set once for PME_GPU_PARALLEL_SPLINE == 1

            const float dr = sm_fractCoords[sharedMemoryIndex];
            assert(isfinite(dr));

            /* dr is relative offset from lower cell limit */
            splineData[order - 1] = 0.0f;
            splineData[1]         = dr;
            splineData[0]         = 1.0f - dr;

#pragma unroll
            for (int k = 3; k < order; k++)
            {
                div               = 1.0f / (k - 1.0f);
                splineData[k - 1] = div * dr * splineData[k - 2];
#pragma unroll
                for (int l = 1; l < (k - 1); l++)
                {
                    splineData[k - l - 1] =
                            div * ((dr + l) * splineData[k - l - 2] + (k - l - dr) * splineData[k - l - 1]);
                }
                splineData[0] = div * (1.0f - dr) * splineData[0];
            }

            const int thetaIndexBase =
                    getSplineParamIndexBase<order, atomsPerWarp>(warpIndex, atomWarpIndex);
            const int thetaGlobalOffsetBase = atomIndexOffset * DIM * order;
            /* only calculate dtheta if we are saving it to shared or global memory */
            if (writeSmDtheta || writeGlobal)
            {
                /* Differentiation and storing the spline derivatives (dtheta) */
#pragma unroll
                for (o = 0; o < order; o++)
                {
                    const int thetaIndex =
                            getSplineParamIndex<order, atomsPerWarp>(thetaIndexBase, dimIndex, o);

                    const float dtheta = ((o > 0) ? splineData[o - 1] : 0.0f) - splineData[o];
                    assert(isfinite(dtheta));
                    assert(thetaIndex < order * DIM * atomsPerBlock);
                    if (writeSmDtheta)
                    {
                        sm_dtheta[thetaIndex] = dtheta;
                    }
                    if (writeGlobal)
                    {
                        const int thetaGlobalIndex  = thetaGlobalOffsetBase + thetaIndex;
                        gm_dtheta[thetaGlobalIndex] = dtheta;
                    }
                }
            }

            div                   = 1.0f / (order - 1.0f);
            splineData[order - 1] = div * dr * splineData[order - 2];
#pragma unroll
            for (int k = 1; k < (order - 1); k++)
            {
                splineData[order - k - 1] = div
                                            * ((dr + k) * splineData[order - k - 2]
                                            + (order - k - dr) * splineData[order - k - 1]);
            }
            splineData[0] = div * (1.0f - dr) * splineData[0];
            /* Storing the spline values (theta) */
#pragma unroll
            for (o = 0; o < order; o++)
            {
                const int thetaIndex =
                        getSplineParamIndex<order, atomsPerWarp>(thetaIndexBase, dimIndex, o);
                assert(thetaIndex < order * DIM * atomsPerBlock);
                sm_theta[thetaIndex] = splineData[o];
                assert(isfinite(sm_theta[thetaIndex]));
                if (writeGlobal)
                {
                    const int thetaGlobalIndex = thetaGlobalOffsetBase + thetaIndex;
                    gm_theta[thetaGlobalIndex] = splineData[o];
                }
            }
        }
    }
}

__device__ __forceinline__ float read_grid_size(const float* realGridSizeFP, const int dimIndex)
{
    switch (dimIndex)
    {
        case XX: return realGridSizeFP[XX];
        case YY: return realGridSizeFP[YY];
        case ZZ: return realGridSizeFP[ZZ];
    }
    assert(false);
    return 0.0f;
}

template<const int order, const int atomDataSize, const int blockSize>
__device__ __forceinline__ void reduce_atom_forces(float3* __restrict__ sm_forces,
                                                   const int    atomIndexLocal,
                                                   const int    splineIndex,
                                                   const int    lineIndex,
                                                   const float* realGridSizeFP,
                                                   float&       fx,
                                                   float&       fy,
                                                   float&       fz)
{
    if (!(order & (order - 1))) // Only for orders of power of 2
    {
        static_assert(order == 4, "Only order of 4 is implemented");
        static_assert(atomDataSize <= warp_size,
                      "TODO: rework for atomDataSize > warp_size (order 8 or larger)");
        const int width = atomDataSize;

        fx += __shfl_down(fx, 1, width);
        fy += __shfl_up(fy, 1, width);
        fz += __shfl_down(fz, 1, width);

        if (splineIndex & 1)
        {
            fx = fy;
        }

        fx += __shfl_down(fx, 2, width);
        fz += __shfl_up(fz, 2, width);

        if (splineIndex & 2)
        {
            fx = fz;
        }

        // We have to just further reduce those groups of 4
        for (int delta = 4; delta < atomDataSize; delta <<= 1)
        {
              fx += __shfl_down(fx, delta, width);
        }

        const int dimIndex = splineIndex;
        if (dimIndex < DIM)
        {
            const float n = read_grid_size(realGridSizeFP, dimIndex);
            *((float*)(&sm_forces[atomIndexLocal]) + dimIndex) = fx * n;
        }
    }
    else
    {
        // We use blockSize shared memory elements to read fx, or fy, or fz, and then reduce them to
        // fit into smemPerDim elements which are stored separately (first 2 dimensions only)
        const int         smemPerDim   = warp_size;
        const int         smemReserved = (DIM)*smemPerDim;
        __shared__ float  sm_forceReduction[smemReserved + blockSize];
        __shared__ float* sm_forceTemp[DIM];

        const int numWarps = blockSize / smemPerDim;
        const int minStride =
                max(1, atomDataSize / numWarps); // order 4: 128 threads => 4, 256 threads => 2, etc

#pragma unroll
        for (int dimIndex = 0; dimIndex < DIM; dimIndex++)
        {
            int elementIndex = smemReserved + lineIndex;
            // Store input force contributions
            sm_forceReduction[elementIndex] = (dimIndex == XX) ? fx : (dimIndex == YY) ? fy : fz;
            // sync here because two warps write data that the first one consumes below
            __syncthreads();
            // Reduce to fit into smemPerDim (warp size)
#pragma unroll
            for (int redStride = atomDataSize / 2; redStride > minStride; redStride >>= 1)
            {
                if (splineIndex < redStride)
                {
                    sm_forceReduction[elementIndex] += sm_forceReduction[elementIndex + redStride];
                }
            }
            __syncthreads();
            // Last iteration - packing everything to be nearby, storing convenience pointer
            sm_forceTemp[dimIndex] = sm_forceReduction + dimIndex * smemPerDim;
            int redStride          = minStride;
            if (splineIndex < redStride)
            {
                const int packedIndex = atomIndexLocal * redStride + splineIndex;
                sm_forceTemp[dimIndex][packedIndex] =
                        sm_forceReduction[elementIndex] + sm_forceReduction[elementIndex + redStride];
            }
            __syncthreads();
        }

        assert((blockSize / warp_size) >= DIM);
        // assert (atomsPerBlock <= warp_size);

        const int warpIndex = lineIndex / warp_size;
        const int dimIndex  = warpIndex;

        // First 3 warps can now process 1 dimension each
        if (dimIndex < DIM)
        {
            int sourceIndex = lineIndex % warp_size;
#pragma unroll
            for (int redStride = minStride / 2; redStride > 1; redStride >>= 1)
            {
                if (!(splineIndex & redStride))
                {
                    sm_forceTemp[dimIndex][sourceIndex] += sm_forceTemp[dimIndex][sourceIndex + redStride];
                }
            }

            //__syncwarp();
            __all(1);

            const float n         = read_grid_size(realGridSizeFP, dimIndex);
            const int   atomIndex = sourceIndex / minStride;

            if (sourceIndex == minStride * atomIndex)
            {
                *((float*)(&sm_forces[atomIndex]) + dimIndex) =
                        (sm_forceTemp[dimIndex][sourceIndex] + sm_forceTemp[dimIndex][sourceIndex + 1]) * n;
            }
        }
    }
}

template<const int order, const bool overwriteForces, const bool wrapX, const bool wrapY, const bool readGlobal, const bool useOrderThreads>
__launch_bounds__(c_gatherMaxThreadsPerBlock, c_gatherMinBlocksPerMP) __global__
        void pme_gather_kernel(const int    nAtoms,
			       const float* d_coefficients,
			       const float* d_coordinates,
			       const float* d_realGrid,
			       float*       d_theta,
			       float*       d_dtheta,
			       const float* d_fractShiftsTable,
			       const float* recipBox,
			       const float* realGridSizeFP,
			       const int*   d_gridlineIndicesTable,
			       int*         d_gridlineIndices,
			       const int*   tablesOffsets,
			       const int*   realGridSize,
			       const int*   realGridSizePadded,
			       float*       d_forces)
{
    /* Global memory pointers */
    const float* __restrict__ gm_coefficients = d_coefficients;
    const float* __restrict__ gm_grid         = d_realGrid;
    float* __restrict__ gm_forces             = d_forces;

    /* Global memory pointers for readGlobal */
    const float* __restrict__ gm_theta         = d_theta;
    const float* __restrict__ gm_dtheta        = d_dtheta;
    const int* __restrict__ gm_gridlineIndices = d_gridlineIndices;

    float3 atomX;
    float  atomCharge;

    /* Some sizes */
    const int atomsPerBlock =
            useOrderThreads ? (c_gatherMaxThreadsPerBlock / c_pmeSpreadGatherThreadsPerAtom4ThPerAtom)
                            : (c_gatherMaxThreadsPerBlock / c_pmeSpreadGatherThreadsPerAtom);
    const int blockIndex = blockIdx.y * gridDim.x + blockIdx.x;

    /* Number of data components and threads for a single atom */
    const int atomDataSize = useOrderThreads ? c_pmeSpreadGatherThreadsPerAtom4ThPerAtom
                                             : c_pmeSpreadGatherThreadsPerAtom;
    const int atomsPerWarp = useOrderThreads ? c_pmeSpreadGatherAtomsPerWarp4ThPerAtom
                                             : c_pmeSpreadGatherAtomsPerWarp;

    const int blockSize = atomsPerBlock * atomDataSize;
    assert(blockSize == blockDim.x * blockDim.y * blockDim.z);

    /* These are the atom indices - for the shared and global memory */
    const int atomIndexLocal  = threadIdx.z;
    const int atomIndexOffset = blockIndex * atomsPerBlock;
    const int atomIndexGlobal = atomIndexOffset + atomIndexLocal;

    /* Early return for fully empty blocks at the end
     * (should only happen for billions of input atoms)
     */
    if (atomIndexOffset >= nAtoms)
    {
        return;
    }
    // 4 warps per block, 8 atoms per warp *3 *4
    const int        splineParamsSize    = atomsPerBlock * DIM * order;
    const int        gridlineIndicesSize = atomsPerBlock * DIM;
    __shared__ int   sm_gridlineIndices[gridlineIndicesSize];
    __shared__ float sm_theta[splineParamsSize];
    __shared__ float sm_dtheta[splineParamsSize];

    /* Spline Z coordinates */
    const int ithz = threadIdx.x;

    /* These are the spline contribution indices in shared memory */
    const int splineIndex = threadIdx.y * blockDim.x + threadIdx.x;
    const int lineIndex   = (threadIdx.z * (blockDim.x * blockDim.y))
                          + splineIndex; /* And to all the block's particles */

    const int threadLocalId =
            (threadIdx.z * (blockDim.x * blockDim.y)) + blockDim.x * threadIdx.y + threadIdx.x;
    const int threadLocalIdMax = blockDim.x * blockDim.y * blockDim.z;

    if (readGlobal)
    {
        /* Read splines */
        const int localGridlineIndicesIndex = threadLocalId;
        const int globalGridlineIndicesIndex = blockIndex * gridlineIndicesSize + localGridlineIndicesIndex;
        const int globalCheckIndices         = pme_gpu_check_atom_data_index(
                globalGridlineIndicesIndex, nAtoms * DIM);
        if ((localGridlineIndicesIndex < gridlineIndicesSize) & globalCheckIndices)
        {
            sm_gridlineIndices[localGridlineIndicesIndex] = gm_gridlineIndices[globalGridlineIndicesIndex];
            assert(sm_gridlineIndices[localGridlineIndicesIndex] >= 0);
        }
        /* The loop needed for order threads per atom to make sure we load all data values, as each thread must load multiple values
           with order*order threads per atom, it is only required for each thread to load one data value */

        const int iMin = 0;
        const int iMax = useOrderThreads ? 3 : 1;

        for (int i = iMin; i < iMax; i++)
        {
            int localSplineParamsIndex =
                    threadLocalId
                    + i * threadLocalIdMax; /* i will always be zero for order*order threads per atom */
            int globalSplineParamsIndex = blockIndex * splineParamsSize + localSplineParamsIndex;
            int globalCheckSplineParams = pme_gpu_check_atom_data_index(
                    globalSplineParamsIndex, nAtoms * DIM * order);
            if ((localSplineParamsIndex < splineParamsSize) && globalCheckSplineParams)
            {
                sm_theta[localSplineParamsIndex]  = gm_theta[globalSplineParamsIndex];
                sm_dtheta[localSplineParamsIndex] = gm_dtheta[globalSplineParamsIndex];
                assert(isfinite(sm_theta[localSplineParamsIndex]));
                assert(isfinite(sm_dtheta[localSplineParamsIndex]));
            }
        }
        __syncthreads();
    }
    else
    {
        /* Recaclulate  Splines  */
        if (c_useAtomDataPrefetch)
        {
            // charges
            __shared__ float sm_coefficients[atomsPerBlock];
            // Coordinates
            __shared__ float sm_coordinates[DIM * atomsPerBlock];
            /* Staging coefficients/charges */
            pme_gpu_stage_atom_data<float, atomsPerBlock, 1>(nAtoms, sm_coefficients,
                                                             d_coefficients);

            /* Staging coordinates */
            pme_gpu_stage_atom_data<float, atomsPerBlock, DIM>(nAtoms, sm_coordinates,
                                                               d_coordinates);
            __syncthreads();
            atomX.x    = sm_coordinates[atomIndexLocal * DIM + XX];
            atomX.y    = sm_coordinates[atomIndexLocal * DIM + YY];
            atomX.z    = sm_coordinates[atomIndexLocal * DIM + ZZ];
            atomCharge = sm_coefficients[atomIndexLocal];
        }
        else
        {
            atomCharge = gm_coefficients[atomIndexGlobal];
            atomX.x    = d_coordinates[atomIndexGlobal * DIM + XX];
            atomX.y    = d_coordinates[atomIndexGlobal * DIM + YY];
            atomX.z    = d_coordinates[atomIndexGlobal * DIM + ZZ];
        }
        calculate_splines<order, atomsPerBlock, atomsPerWarp, true, false>(
                d_theta, d_dtheta, realGridSizeFP, recipBox, d_fractShiftsTable,
		d_gridlineIndicesTable, d_gridlineIndices, tablesOffsets, nAtoms,
		atomIndexOffset, atomX, atomCharge, sm_theta, sm_dtheta, sm_gridlineIndices);
        __all(1);
    }
    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;

    const int globalCheck = pme_gpu_check_atom_data_index(atomIndexGlobal, nAtoms);
    const int chargeCheck = pme_gpu_check_atom_charge(gm_coefficients[atomIndexGlobal]);

    if (chargeCheck & globalCheck)
    {
        const int nx  = realGridSize[XX];
        const int ny  = realGridSize[YY];
        const int nz  = realGridSize[ZZ];
        const int pny = realGridSizePadded[YY];
        const int pnz = realGridSizePadded[ZZ];

        const int atomWarpIndex = atomIndexLocal % atomsPerWarp;
        const int warpIndex     = atomIndexLocal / atomsPerWarp;

        const int splineIndexBase = getSplineParamIndexBase<order, atomsPerWarp>(warpIndex, atomWarpIndex);
        const int splineIndexZ = getSplineParamIndex<order, atomsPerWarp>(splineIndexBase, ZZ, ithz);
        const float2 tdz       = make_float2(sm_theta[splineIndexZ], sm_dtheta[splineIndexZ]);

        int       iz     = sm_gridlineIndices[atomIndexLocal * DIM + ZZ] + ithz;
        const int ixBase = sm_gridlineIndices[atomIndexLocal * DIM + XX];

        if (iz >= nz)
        {
            iz -= nz;
        }
        int constOffset, iy;

        const int ithyMin = useOrderThreads ? 0 : threadIdx.y;
        const int ithyMax = useOrderThreads ? order : threadIdx.y + 1;
        for (int ithy = ithyMin; ithy < ithyMax; ithy++)
        {
            const int splineIndexY = getSplineParamIndex<order, atomsPerWarp>(splineIndexBase, YY, ithy);
            const float2 tdy       = make_float2(sm_theta[splineIndexY], sm_dtheta[splineIndexY]);

            iy = sm_gridlineIndices[atomIndexLocal * DIM + YY] + ithy;
            if (wrapY & (iy >= ny))
            {
                iy -= ny;
            }
            constOffset = iy * pnz + iz;

#pragma unroll
            for (int ithx = 0; (ithx < order); ithx++)
            {
                int ix = ixBase + ithx;
                if (wrapX & (ix >= nx))
                {
                    ix -= nx;
                }
                const int gridIndexGlobal = (ix * pny * pnz + constOffset);
                assert(gridIndexGlobal >= 0);
                const float gridValue = gm_grid[gridIndexGlobal];
                assert(isfinite(gridValue));
                const int splineIndexX =
                        getSplineParamIndex<order, atomsPerWarp>(splineIndexBase, XX, ithx);
                const float2 tdx  = make_float2(sm_theta[splineIndexX], sm_dtheta[splineIndexX]);
                const float  fxy1 = tdz.x * gridValue;
                const float  fz1  = tdz.y * gridValue;
                fx += tdx.y * tdy.x * fxy1;
                fy += tdx.x * tdy.y * fxy1;
                fz += tdx.x * tdy.x * fz1;
            }
        }
    }

    // Reduction of partial force contributions
    __shared__ float3 sm_forces[atomsPerBlock];
    reduce_atom_forces<order, atomDataSize, blockSize>(sm_forces, atomIndexLocal, splineIndex, lineIndex,
                                                       realGridSizeFP, fx, fy, fz);
    __syncthreads();

    /* Calculating the final forces with no component branching, atomsPerBlock threads */
    const int forceIndexLocal  = threadLocalId;
    const int forceIndexGlobal = atomIndexOffset + forceIndexLocal;
    const int calcIndexCheck = pme_gpu_check_atom_data_index(forceIndexGlobal, nAtoms);
    if ((forceIndexLocal < atomsPerBlock) & calcIndexCheck)
    {
        const float3 atomForces     = sm_forces[forceIndexLocal];
        const float  negCoefficient = -gm_coefficients[forceIndexGlobal];
        float3       result;
        result.x = negCoefficient * recipBox[XX*DIM+XX] * atomForces.x;
        result.y = negCoefficient
                   * (recipBox[XX*DIM+YY] * atomForces.x
                      + recipBox[YY*DIM+YY] * atomForces.y);
        result.z = negCoefficient
                   * (recipBox[XX*DIM+ZZ] * atomForces.x
                      + recipBox[YY*DIM+ZZ] * atomForces.y
                      + recipBox[ZZ*DIM+ZZ] * atomForces.z);
        sm_forces[forceIndexLocal] = result;
    }

    __all(1);
    assert(atomsPerBlock <= warp_size);

    /* Writing or adding the final forces component-wise, single warp */
    const int blockForcesSize = atomsPerBlock * DIM;
    const int numIter         = (blockForcesSize + warp_size - 1) / warp_size;
    const int iterThreads     = blockForcesSize / numIter;
    if (threadLocalId < iterThreads)
    {
#pragma unroll
        for (int i = 0; i < numIter; i++)
        {
            int       outputIndexLocal  = i * iterThreads + threadLocalId;
            int       outputIndexGlobal = blockIndex * blockForcesSize + outputIndexLocal;
            const int globalOutputCheck =
                    pme_gpu_check_atom_data_index(outputIndexGlobal, nAtoms * DIM);
            if (globalOutputCheck)
            {
                const float outputForceComponent = ((float*)sm_forces)[outputIndexLocal];
                if (overwriteForces)
                {
                    gm_forces[outputIndexGlobal] = outputForceComponent;
                }
                else
                {
                    gm_forces[outputIndexGlobal] += outputForceComponent;
                }
            }
        }
    }
}

template <typename T>
void initValueFromFile(const char* fileName, int dataSize, int totalSize, T* out) {
    std::ifstream myfile;
    myfile.open(fileName);

    for (int i = 0; i < dataSize; i++) {
        myfile >> out[i];
    }
    myfile.close();

    for (int i = dataSize; i < totalSize; i++) {
        out[i] = 0;
    }
}

int main(int argc, char* argv[]) {
    int nAtoms                 = atoi(argv[1]);
    const bool useOrderThreads = atoi(argv[2]); 

    const int order  = 4;

    int nAtomsPadded = ((nAtoms + c_pmeAtomDataAlignment - 1)/c_pmeAtomDataAlignment) * c_pmeAtomDataAlignment;
    int atomPerBlock = useOrderThreads?
            c_gatherMaxThreadsPerBlock/c_pmeSpreadGatherThreadsPerAtom4ThPerAtom:
            c_gatherMaxThreadsPerBlock/c_pmeSpreadGatherThreadsPerAtom;

    int grid_dim_x = nAtomsPadded/atomPerBlock;

    int *h_realGridSize, *h_realGridSizePadded, *h_tableOffsets;
    int *d_realGridSize, *d_realGridSizePadded, *d_tableOffsets;
    h_realGridSize       = new int[DIM];
    h_realGridSizePadded = new int[DIM];
    h_tableOffsets       = new int[DIM];
    initValueFromFile("realGridSize.txt", DIM, DIM, h_realGridSize);
    initValueFromFile("realGridSizePadded.txt", DIM, DIM, h_realGridSizePadded);
    initValueFromFile("tableOffsets.txt", DIM, DIM, h_tableOffsets);

    float *h_recipBox, *h_realGridSizeFP;
    float *d_recipBox, *d_realGridSizeFP;
    h_recipBox           = new float[DIM*DIM];
    h_realGridSizeFP     = new float[DIM];
    initValueFromFile("recipBox.txt", DIM*DIM, DIM*DIM, h_recipBox);
    initValueFromFile("realGridSizeFP.txt", DIM, DIM, h_realGridSizeFP);

    float *h_theta, *h_dtheta;
    float *d_theta, *d_dtheta;
    int   *h_gridlineIndices;
    int   *d_gridlineIndices;
    h_theta           = new float[nAtomsPadded*DIM*order];
    h_dtheta          = new float[nAtomsPadded*DIM*order];
    h_gridlineIndices = new int[nAtomsPadded*DIM];
    initValueFromFile("d_theta.txt", nAtomsPadded*DIM*order, nAtomsPadded*DIM*order, h_theta);
    initValueFromFile("d_dtheta.txt", nAtomsPadded*DIM*order, nAtomsPadded*DIM*order, h_dtheta);
    initValueFromFile("d_gridlineIndices.txt", nAtoms*DIM, nAtomsPadded*DIM, h_gridlineIndices);

    float *h_coordinates, *h_coefficients;
    float *d_coordinates, *d_coefficients;
    h_coordinates  = new float[nAtomsPadded*DIM];
    h_coefficients = new float[nAtomsPadded];
    initValueFromFile("d_coordinates.txt", nAtomsPadded * DIM, nAtomsPadded * DIM, h_coordinates);
    initValueFromFile("d_coefficients.txt", nAtomsPadded, nAtomsPadded, h_coefficients);

    const int    nx                  = h_realGridSize[XX];
    const int    ny                  = h_realGridSize[YY];
    const int    nz                  = h_realGridSize[ZZ];
    const int    cellCount           = c_pmeNeighborUnitcellCount;
    const int    newFractShiftsSize  = cellCount * (nx + ny + nz);
    const int    nRealGridSize = nx * ny * nz;

    int *h_gridlineIndicesTable;
    int *d_gridlineIndicesTable;
    h_gridlineIndicesTable = new int[newFractShiftsSize];
    initValueFromFile("d_gridlineIndicesTable.txt", newFractShiftsSize, newFractShiftsSize, h_gridlineIndicesTable);

    float *h_fractShiftsTable;
    float *d_fractShiftsTable;
    h_fractShiftsTable = new float[newFractShiftsSize];
    initValueFromFile("d_fractShiftsTable.txt", newFractShiftsSize, newFractShiftsSize, h_fractShiftsTable);

    float *h_realGrid;
    float *d_realGrid;
    h_realGrid = new float[nRealGridSize];
    std::cout << nRealGridSize;
    initValueFromFile("d_realGrid.txt", nRealGridSize, nRealGridSize, h_realGrid);

    hipMalloc((void**)&d_realGridSize, DIM * sizeof(int));
    hipMalloc((void**)&d_realGridSizePadded, DIM * sizeof(int));
    hipMalloc((void**)&d_tableOffsets, DIM * sizeof(int));

    hipMalloc((void**)&d_recipBox, DIM * DIM * sizeof(float));
    hipMalloc((void**)&d_realGridSizeFP, DIM * sizeof(float));

    hipMalloc((void**)&d_theta, nAtomsPadded * DIM * order * sizeof(float));
    hipMalloc((void**)&d_dtheta, nAtomsPadded * DIM * order * sizeof(float));
    hipMalloc((void**)&d_gridlineIndices, nAtomsPadded * DIM * sizeof(int));

    hipMalloc((void**)&d_coordinates, nAtomsPadded * DIM *sizeof(float));
    hipMalloc((void**)&d_coefficients, nAtomsPadded * sizeof(float));
    hipMalloc((void**)&d_fractShiftsTable, newFractShiftsSize * sizeof(float));

    hipMalloc((void**)&d_gridlineIndicesTable, newFractShiftsSize * sizeof(int));
    hipMalloc((void**)&d_realGrid, nRealGridSize * sizeof(float));

    float *d_forces;
    hipMalloc((void**)&d_forces, nAtomsPadded * DIM * sizeof(float));

    hipMemcpyHtoD(d_realGridSize, h_realGridSize, DIM * sizeof(int));
    hipMemcpyHtoD(d_realGridSizePadded, h_realGridSizePadded, DIM * sizeof(int));
    hipMemcpyHtoD(d_tableOffsets, h_tableOffsets, DIM * sizeof(int));

    hipMemcpyHtoD(d_recipBox, h_recipBox, DIM * DIM * sizeof(float));
    hipMemcpyHtoD(d_realGridSizeFP, h_realGridSizeFP, DIM * sizeof(float));

    hipMemcpyHtoD(d_theta, h_theta, nAtomsPadded * DIM * order * sizeof(float));
    hipMemcpyHtoD(d_dtheta, h_dtheta, nAtomsPadded * DIM * order * sizeof(float));
    hipMemcpyHtoD(d_gridlineIndices, h_gridlineIndices, nAtomsPadded * DIM * sizeof(int));

    hipMemcpyHtoD(d_coordinates, h_coordinates, nAtomsPadded * DIM * sizeof(float));
    hipMemcpyHtoD(d_coefficients, h_coefficients, nAtomsPadded * sizeof(float));

    hipMemcpyHtoD(d_fractShiftsTable, h_fractShiftsTable, newFractShiftsSize * sizeof(int));
    hipMemcpyHtoD(d_gridlineIndicesTable, h_gridlineIndicesTable, newFractShiftsSize * sizeof(int));

    hipMemcpyHtoD(d_realGrid, h_realGrid, nRealGridSize * sizeof(float));

    void (*gatherKernel)(const int nAtoms,
             const float* d_coefficients,
             const float* d_coordinates,
             const float* d_realGrid,
             float*       d_theta,
             float*       d_dtheta,
             const float* d_fractShiftsTable,
             const float* recipBox,
             const float* realGridSizeFP,
             const int*   d_gridlineIndicesTable,
             int*         d_gridlineIndices,
             const int*   tablesOffsets,
             const int*   realGridSize,
             const int*   realGridSizePadded,
             float*       d_forces);

    if (useOrderThreads) {
        gatherKernel = pme_gather_kernel<4,true, true, true, false, true>;
    } else {
        gatherKernel = pme_gather_kernel<4,true, true, true, true, false>;
    }

    for (int iter=0; iter<1000; iter++) {
        hipLaunchKernelGGL(gatherKernel, dim3(grid_dim_x, 1, 1), dim3(order, useOrderThreads?1:order, atomPerBlock), 0, 0
                    , nAtoms, d_coefficients, d_coordinates
		    , d_realGrid, d_theta, d_dtheta
		    , d_fractShiftsTable, d_recipBox
                    , d_realGridSizeFP, d_gridlineIndicesTable
		    , d_gridlineIndices, d_tableOffsets
		    , d_realGridSize, d_realGridSizePadded
                    , d_forces);
    }
    hipStreamSynchronize(0);
    //float h_forces[nAtoms * DIM]; 
    //hipMemcpyDtoH(h_forces, d_forces, nAtoms * DIM * sizeof(float));
    //for (int i = 0; i < nAtoms * DIM; i++)
    //    std::cout << h_forces[i] << std::endl;

    return 0;
}
