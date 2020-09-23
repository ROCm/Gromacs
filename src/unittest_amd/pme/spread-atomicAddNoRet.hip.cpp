#include "hip/hip_runtime.h"
#include <iostream>
#include <fstream>

#if ! __HIP_DEVICE_COMPILE__
#include <cassert>
#endif

//#include "gromacs/gpu_utils/cuda_kernel_utils.hip.h"

//#include "pme.hip.h"
//#include "pme_calculate_splines.hip.h"
//#include "pme_gpu_utils.h"
//#include "pme_grid.h"
//
constexpr int XX                         = 0;
constexpr int YY                         = 1;
constexpr int ZZ                         = 2;
constexpr int DIM                        = 3;
constexpr int c_pmeGpuOrder              = 4;
constexpr int warp_size                  = 64;
constexpr int c_pmeAtomDataAlignment     = 128;
constexpr int c_spreadMaxWarpsPerBlock   = 8;
constexpr int c_spreadMaxThreadsPerBlock = (c_spreadMaxWarpsPerBlock * warp_size);

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

#define HIP_PI_F 3.141592654f
#define CLANG_DISABLE_OPTIMIZATION_ATTRIBUTE

#define INLINE_EVERYWHERE __host__ __device__ __forceinline__

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
static __forceinline__ __device__ T fetchFromParamLookupTable(const T*                  d_ptr,
                                                              int                       index)
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

template<const int order, const bool useOrderThreads>
__device__ __forceinline__ void spread_charges(const bool wrapX, const bool wrapY,
                                               float*                       d_realGrid,
                                               int                          atomIndexOffset,
                                               const int                    nAtoms,
                                               const int*                   realGridSize,
                                               const int*                   realGridSizePadded,
                                               const float*                 atomCharge,
                                               const int* __restrict__ sm_gridlineIndices,
                                               const float* __restrict__ sm_theta)
{
    /* Global memory pointer to the output grid */
    float* __restrict__ gm_grid = d_realGrid;


    const int atomsPerWarp = useOrderThreads ? c_pmeSpreadGatherAtomsPerWarp4ThPerAtom
                                             : c_pmeSpreadGatherAtomsPerWarp;

    const int nx  = realGridSize[XX];
    const int ny  = realGridSize[YY];
    const int nz  = realGridSize[ZZ];
    const int pny = realGridSizePadded[YY];
    const int pnz = realGridSizePadded[ZZ];

    const int offx = 0, offy = 0, offz = 0; // unused for now

    const int atomIndexLocal  = threadIdx.z;
    const int atomIndexGlobal = atomIndexOffset + atomIndexLocal;

    const int globalCheck = pme_gpu_check_atom_data_index(atomIndexGlobal, nAtoms);
    const int chargeCheck = pme_gpu_check_atom_charge(*atomCharge);
    if (chargeCheck & globalCheck)
    {
        // Spline Z coordinates
        const int ithz = threadIdx.x;

        const int ixBase = sm_gridlineIndices[atomIndexLocal * DIM + XX] - offx;
        const int iyBase = sm_gridlineIndices[atomIndexLocal * DIM + YY] - offy;
        int       iz     = sm_gridlineIndices[atomIndexLocal * DIM + ZZ] - offz + ithz;
        if (iz >= nz)
        {
            iz -= nz;
        }
        /* Atom index w.r.t. warp - alternating 0 1 0 1 .. */
        const int atomWarpIndex = atomIndexLocal % atomsPerWarp;
        /* Warp index w.r.t. block - could probably be obtained easier? */
        const int warpIndex = atomIndexLocal / atomsPerWarp;

        const int splineIndexBase = getSplineParamIndexBase<order, atomsPerWarp>(warpIndex, atomWarpIndex);
        const int splineIndexZ = getSplineParamIndex<order, atomsPerWarp>(splineIndexBase, ZZ, ithz);
        const float thetaZ     = sm_theta[splineIndexZ];

        /* loop not used if order*order threads per atom */
        const int ithyMin = useOrderThreads ? 0 : threadIdx.y;
        const int ithyMax = useOrderThreads ? order : threadIdx.y + 1;
        for (int ithy = ithyMin; ithy < ithyMax; ithy++)
        {
            int iy = iyBase + ithy;
            if (wrapY & (iy >= ny))
            {
                iy -= ny;
            }

            const int splineIndexY = getSplineParamIndex<order, atomsPerWarp>(splineIndexBase, YY, ithy);
            float       thetaY = sm_theta[splineIndexY];
            const float Val    = thetaZ * thetaY * (*atomCharge);
            assert(isfinite(Val));
            const int offset = iy * pnz + iz;

#pragma unroll
            for (int ithx = 0; (ithx < order); ithx++)
            {
                int ix = ixBase + ithx;
                if (wrapX & (ix >= nx))
                {
                    ix -= nx;
                }
                const int gridIndexGlobal = ix * pny * pnz + offset;
                const int splineIndexX =
                        getSplineParamIndex<order, atomsPerWarp>(splineIndexBase, XX, ithx);
                const float thetaX = sm_theta[splineIndexX];
                assert(isfinite(thetaX));
                assert(isfinite(gm_grid[gridIndexGlobal]));
                atomicAddNoRet(gm_grid + gridIndexGlobal, thetaX * Val);
            }
        }
    }
}

template<const int order, const bool useOrderThreads, const bool writeGlobal>
__launch_bounds__(c_spreadMaxThreadsPerBlock) CLANG_DISABLE_OPTIMIZATION_ATTRIBUTE __global__
        void pme_spline_and_spread_kernel(const bool computeSplines,
             const bool spreadCharges, const bool wrapX, const bool wrapY, 
             const int nAtoms,
             const float *d_coefficients,
             const float *d_coordinates,
	     const float *realGridSizeFP,
	     const float *recipBox,
             const float *d_fractShiftsTable,
             const int   *d_gridlineIndicesTable,
	     const int   *tablesOffsets,
             const int   *realGridSize,
             const int   *realGridSizePadded,
             float       *d_theta,
             float       *d_dtheta,
             int         *d_gridlineIndices,
             float       *d_realGrid)
{
    const int atomsPerBlock =
            useOrderThreads ? c_spreadMaxThreadsPerBlock / c_pmeSpreadGatherThreadsPerAtom4ThPerAtom
                            : c_spreadMaxThreadsPerBlock / c_pmeSpreadGatherThreadsPerAtom;
    // Gridline indices, ivec
    __shared__ int sm_gridlineIndices[atomsPerBlock * DIM];
    // Spline values
    __shared__ float sm_theta[atomsPerBlock * DIM * order];
    float            dtheta;

    const int atomsPerWarp = useOrderThreads ? c_pmeSpreadGatherAtomsPerWarp4ThPerAtom
                                             : c_pmeSpreadGatherAtomsPerWarp;

    float3 atomX;
    float  atomCharge;

    const int blockIndex      = blockIdx.y * gridDim.x + blockIdx.x;
    const int atomIndexOffset = blockIndex * atomsPerBlock;

    /* Thread index w.r.t. block */
    const int threadLocalId =
            (threadIdx.z * (blockDim.x * blockDim.y)) + (threadIdx.y * blockDim.x) + threadIdx.x;
    /* Warp index w.r.t. block - could probably be obtained easier? */
    const int warpIndex = threadLocalId / warp_size;

    /* Atom index w.r.t. warp */
    const int atomWarpIndex = threadIdx.z % atomsPerWarp;
    /* Atom index w.r.t. block/shared memory */
    const int atomIndexLocal = warpIndex * atomsPerWarp + atomWarpIndex;
    /* Atom index w.r.t. global memory */
    const int atomIndexGlobal = atomIndexOffset + atomIndexLocal;

    /* Early return for fully empty blocks at the end
     * (should only happen for billions of input atoms)
     */
    if (atomIndexOffset >= nAtoms)
    {
        return;
    }
  /* Charges, required for both spline and spread */
    if (c_useAtomDataPrefetch)
    {
        __shared__ float sm_coefficients[atomsPerBlock];
        pme_gpu_stage_atom_data<float, atomsPerBlock, 1>(nAtoms, sm_coefficients,
                                                         d_coefficients);
        __syncthreads();
        atomCharge = sm_coefficients[atomIndexLocal];
    }
    else
    {
        atomCharge = d_coefficients[atomIndexGlobal];
    }

    if (computeSplines)
    {
        if (c_useAtomDataPrefetch)
        {
            // Coordinates
            __shared__ float sm_coordinates[DIM * atomsPerBlock];

            /* Staging coordinates */
            pme_gpu_stage_atom_data<float, atomsPerBlock, DIM>(nAtoms, sm_coordinates,
                                                               d_coordinates);
            __syncthreads();
            atomX.x = sm_coordinates[atomIndexLocal * DIM + XX];
            atomX.y = sm_coordinates[atomIndexLocal * DIM + YY];
            atomX.z = sm_coordinates[atomIndexLocal * DIM + ZZ];
        }
        else
        {
            atomX.x = d_coordinates[atomIndexGlobal * DIM + XX];
            atomX.y = d_coordinates[atomIndexGlobal * DIM + YY];
            atomX.z = d_coordinates[atomIndexGlobal * DIM + ZZ];
        }
        calculate_splines<order, atomsPerBlock, atomsPerWarp, false, writeGlobal>(
                d_theta, d_dtheta, realGridSizeFP, recipBox, d_fractShiftsTable,
	        d_gridlineIndicesTable, d_gridlineIndices, tablesOffsets, nAtoms,
		atomIndexOffset, atomX, atomCharge, sm_theta, &dtheta, sm_gridlineIndices);
        __all(1);
    }
    else
    {
        /* Staging the data for spread
         * (the data is assumed to be in GPU global memory with proper layout already,
         * as in after running the spline kernel)
         */
        /* Spline data - only thetas (dthetas will only be needed in gather) */
        pme_gpu_stage_atom_data<float, atomsPerBlock, DIM * order>(nAtoms, sm_theta,
                                                                   d_theta);
        /* Gridline indices */
        pme_gpu_stage_atom_data<int, atomsPerBlock, DIM>(nAtoms, sm_gridlineIndices,
                                                         d_gridlineIndices);

        __syncthreads();
    }

    /* Spreading */
    if (spreadCharges)
    {
        spread_charges<order, useOrderThreads>(wrapX, wrapY, d_realGrid,
                atomIndexOffset, nAtoms, realGridSize, realGridSizePadded,
                &atomCharge, sm_gridlineIndices, sm_theta);
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

int main(int argc, char *argv[]) {

    int nAtoms                 = atoi(argv[1]);
    const bool useOrderThreads = atoi(argv[2]);

    std::cout << "useOrderThreads:" << useOrderThreads << std::endl;

    const int order  = 4;

    int nAtomsPadded = ((nAtoms + c_pmeAtomDataAlignment - 1)/c_pmeAtomDataAlignment) * c_pmeAtomDataAlignment;
    int atomPerBlock = useOrderThreads?
	    c_spreadMaxThreadsPerBlock/c_pmeSpreadGatherThreadsPerAtom4ThPerAtom:
            c_spreadMaxThreadsPerBlock/c_pmeSpreadGatherThreadsPerAtom;

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

    float *d_realGrid;
    hipMalloc((void**)&d_realGrid, nRealGridSize * sizeof(float));

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

    void (*spreadKernel)(const bool computeSplines,
             const bool spreadCharges, const bool wrapX, const bool wrapY,
             const int nAtoms,
             const float *d_coefficients,
             const float *d_coordinates,
             const float *realGridSizeFP,
             const float *recipBox,
             const float *d_fractShiftsTable,
             const int   *d_gridlineIndicesTable,
             const int   *tablesOffsets,
             const int   *realGridSize,
             const int   *realGridSizePadded,
             float       *d_theta,
             float       *d_dtheta,
             int         *d_gridlineIndices,
             float       *d_realGrid);
    
    if (useOrderThreads) {
        spreadKernel = pme_spline_and_spread_kernel<4,true, false>;
    } else {
        spreadKernel = pme_spline_and_spread_kernel<4,false, true>;
    }
    for (int iter=0; iter<1000; iter++) {
        hipLaunchKernelGGL(spreadKernel, dim3(grid_dim_x, 1, 1), dim3(order, useOrderThreads?1:order, atomPerBlock), 0, 0
    		    , true, true, true, true
    		    , nAtoms, d_coefficients, d_coordinates
    		    , d_realGridSizeFP, d_recipBox, d_fractShiftsTable
    		    , d_gridlineIndicesTable,d_tableOffsets, d_realGridSize
    		    , d_realGridSizePadded, d_theta, d_dtheta, d_gridlineIndices
    		    , d_realGrid);
    }
    hipStreamSynchronize(0);
    //float *h_realGrid = new float[nRealGridSize];

    //hipMemcpyDtoH(h_realGrid, d_realGrid, nRealGridSize * sizeof(float));

    //for (int i = 0; i < nRealGridSize; i++)
    //    std::cout << h_realGrid[i] << std::endl;

    return 0;
}
