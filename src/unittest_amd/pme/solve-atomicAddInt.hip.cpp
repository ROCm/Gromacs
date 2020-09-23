#include "hip/hip_runtime.h"
#include <iostream>
#include <fstream>

#if ! __HIP_DEVICE_COMPILE__
#include <cassert>
#endif

#include <cmath>

constexpr int XX                       = 0;
constexpr int YY                       = 1;
constexpr int ZZ                       = 2;
constexpr int DIM                      = 3;
constexpr int warp_size                = 64;
constexpr int c_virialAndEnergyCount   = 7;
constexpr int c_solveMaxWarpsPerBlock  = 4;
constexpr int c_solveMaxThreadsPerBlock = (c_solveMaxWarpsPerBlock * warp_size);

#define HIP_PI_F 3.141592654f
#define CLANG_DISABLE_OPTIMIZATION_ATTRIBUTE

enum class GridOrdering
{
    YZX,
    XYZ
};

/*! \brief
 * PME complex grid solver kernel function.
 *
 * \tparam[in] gridOrdering             Specifies the dimension ordering of the complex grid.
 * \tparam[in] computeEnergyAndVirial   Tells if the reciprocal energy and virial should be
 * computed. \param[in]  kernelParams             Input PME CUDA data in constant memory.
 */
template<GridOrdering gridOrdering, bool computeEnergyAndVirial>
__launch_bounds__(c_solveMaxThreadsPerBlock) CLANG_DISABLE_OPTIMIZATION_ATTRIBUTE __global__
        void pme_solve_kernel(const float *d_splineModuli,
            const float *recipBox,
            const int   *complexGridSizePadded,
            const int   *complexGridSize,
            const int   *realGridSize,
            const int   *splineValuesOffset,
            float       *d_fourierGrid,
            float       *d_virialAndEnergy,
            float       boxVolume,
            float       elFactor,
            float       ewaldFactor)
{
    /* This kernel supports 2 different grid dimension orderings: YZX and XYZ */
    int majorDim, middleDim, minorDim;
    switch (gridOrdering)
    {
        case GridOrdering::YZX:
            majorDim  = YY;
            middleDim = ZZ;
            minorDim  = XX;
            break;

        case GridOrdering::XYZ:
            majorDim  = XX;
            middleDim = YY;
            minorDim  = ZZ;
            break;

        default: assert(false);
    }

    /* Global memory pointers */
    const float* __restrict__ gm_splineValueMajor =
            d_splineModuli + splineValuesOffset[majorDim];
    const float* __restrict__ gm_splineValueMiddle =
            d_splineModuli + splineValuesOffset[middleDim];
    const float* __restrict__ gm_splineValueMinor =
            d_splineModuli + splineValuesOffset[minorDim];
    float* __restrict__ gm_virialAndEnergy = d_virialAndEnergy;
    float2* __restrict__ gm_grid           = (float2*)d_fourierGrid;

    /* Various grid sizes and indices */
    const int localOffsetMinor = 0, localOffsetMajor = 0, localOffsetMiddle = 0; // unused
    const int localSizeMinor   = complexGridSizePadded[minorDim];
    const int localSizeMiddle  = complexGridSizePadded[middleDim];
    const int localCountMiddle = complexGridSize[middleDim];
    const int localCountMinor  = complexGridSize[minorDim];
    const int nMajor           = realGridSize[majorDim];
    const int nMiddle          = realGridSize[middleDim];
    const int nMinor           = realGridSize[minorDim];
    const int maxkMajor        = (nMajor + 1) / 2;  // X or Y
    const int maxkMiddle       = (nMiddle + 1) / 2; // Y OR Z => only check for !YZX
    const int maxkMinor        = (nMinor + 1) / 2;  // Z or X => only check for YZX

    /* Each thread works on one cell of the Fourier space complex 3D grid (gm_grid).
     * Each block handles up to c_solveMaxThreadsPerBlock cells -
     * depending on the grid contiguous dimension size,
     * that can range from a part of a single gridline to several complete gridlines.
     */
    const int threadLocalId     = threadIdx.x;
    const int gridLineSize      = localCountMinor;
    const int gridLineIndex     = threadLocalId / gridLineSize;
    const int gridLineCellIndex = threadLocalId - gridLineSize * gridLineIndex;
    const int gridLinesPerBlock = max(blockDim.x / gridLineSize, 1);
    const int activeWarps       = (blockDim.x / warp_size);
    const int indexMinor        = blockIdx.x * blockDim.x + gridLineCellIndex;
    const int indexMiddle       = blockIdx.y * gridLinesPerBlock + gridLineIndex;
    const int indexMajor        = blockIdx.z;

    /* Optional outputs */
    float energy = 0.0f;
    float virxx  = 0.0f;
    float virxy  = 0.0f;
    float virxz  = 0.0f;
    float viryy  = 0.0f;
    float viryz  = 0.0f;
    float virzz  = 0.0f;

    assert(indexMajor < complexGridSize[majorDim]);
    if ((indexMiddle < localCountMiddle) & (indexMinor < localCountMinor)
        & (gridLineIndex < gridLinesPerBlock))
    {
        /* The offset should be equal to the global thread index for coalesced access */
        const int gridIndex = (indexMajor * localSizeMiddle + indexMiddle) * localSizeMinor + indexMinor;
        float2* __restrict__ gm_gridCell = gm_grid + gridIndex;

        const int kMajor = indexMajor + localOffsetMajor;
        /* Checking either X in XYZ, or Y in YZX cases */
        const float mMajor = (kMajor < maxkMajor) ? kMajor : (kMajor - nMajor);

        const int kMiddle = indexMiddle + localOffsetMiddle;
        float     mMiddle = kMiddle;
        /* Checking Y in XYZ case */
        if (gridOrdering == GridOrdering::XYZ)
        {
            mMiddle = (kMiddle < maxkMiddle) ? kMiddle : (kMiddle - nMiddle);
        }
        const int kMinor = localOffsetMinor + indexMinor;
        float     mMinor = kMinor;
        /* Checking X in YZX case */
        if (gridOrdering == GridOrdering::YZX)
        {
            mMinor = (kMinor < maxkMinor) ? kMinor : (kMinor - nMinor);
        }
        /* We should skip the k-space point (0,0,0) */
        const bool notZeroPoint = (kMinor > 0) | (kMajor > 0) | (kMiddle > 0);

        float mX, mY, mZ;
        switch (gridOrdering)
        {
            case GridOrdering::YZX:
                mX = mMinor;
                mY = mMajor;
                mZ = mMiddle;
                break;

            case GridOrdering::XYZ:
                mX = mMajor;
                mY = mMiddle;
                mZ = mMinor;
                break;

            default: assert(false);
        }

        /* 0.5 correction factor for the first and last components of a Z dimension */
        float corner_fac = 1.0f;
        switch (gridOrdering)
        {
            case GridOrdering::YZX:
                if ((kMiddle == 0) | (kMiddle == maxkMiddle))
                {
                    corner_fac = 0.5f;
                }
                break;

            case GridOrdering::XYZ:
                if ((kMinor == 0) | (kMinor == maxkMinor))
                {
                    corner_fac = 0.5f;
                }
                break;

            default: assert(false);
        }

        if (notZeroPoint)
        {
            const float mhxk = mX * recipBox[XX*DIM+XX];
            const float mhyk = mX * recipBox[XX*DIM+YY]
                               + mY * recipBox[YY*DIM+YY];
            const float mhzk = mX * recipBox[XX*DIM+ZZ]
                               + mY * recipBox[YY*DIM+ZZ]
                               + mZ * recipBox[ZZ*DIM+ZZ];

            const float m2k = mhxk * mhxk + mhyk * mhyk + mhzk * mhzk;
            assert(m2k != 0.0f);
            // TODO: use LDG/textures for gm_splineValue
            float denom = m2k * float(HIP_PI_F) * boxVolume
                          * gm_splineValueMajor[kMajor] * gm_splineValueMiddle[kMiddle]
                          * gm_splineValueMinor[kMinor];
            assert(isfinite(denom));
            assert(denom != 0.0f);

            const float tmp1   = expf(-ewaldFactor * m2k);
            const float etermk = elFactor * tmp1 / denom;

            float2       gridValue    = *gm_gridCell;
            const float2 oldGridValue = gridValue;
            gridValue.x *= etermk;
            gridValue.y *= etermk;
            *gm_gridCell = gridValue;

            if (computeEnergyAndVirial)
            {
                const float tmp1k =
                        2.0f * (gridValue.x * oldGridValue.x + gridValue.y * oldGridValue.y);

                float vfactor = (ewaldFactor + 1.0f / m2k) * 2.0f;
                float ets2    = corner_fac * tmp1k;
                energy        = ets2;

                float ets2vf = ets2 * vfactor;

                virxx = ets2vf * mhxk * mhxk - ets2;
                virxy = ets2vf * mhxk * mhyk;
                virxz = ets2vf * mhxk * mhzk;
                viryy = ets2vf * mhyk * mhyk - ets2;
                viryz = ets2vf * mhyk * mhzk;
                virzz = ets2vf * mhzk * mhzk - ets2;
            }
        }
    }

    /* Optional energy/virial reduction */
    if (computeEnergyAndVirial)
    {
        /* A tricky shuffle reduction inspired by reduce_force_j_warp_shfl.
         * The idea is to reduce 7 energy/virial components into a single variable (aligned by 8).
         * We will reduce everything into virxx.
         */

        /* We can only reduce warp-wise */
        const int          width      = warp_size;

        virxx += __shfl_down(virxx, 1, width);
        viryy += __shfl_up(viryy, 1, width);
        virzz += __shfl_down(virzz, 1, width);
        virxy += __shfl_up(virxy, 1, width);
        virxz += __shfl_down(virxz, 1, width);
        viryz += __shfl_up(viryz, 1, width);
        energy += __shfl_down(energy, 1, width);
        if (threadLocalId & 1)
        {
            virxx = viryy; // virxx now holds virxx and viryy pair sums
            virzz = virxy; // virzz now holds virzz and virxy pair sums
            virxz = viryz; // virxz now holds virxz and viryz pair sums
        }

        virxx += __shfl_down(virxx, 2, width);
        virzz += __shfl_up(virzz, 2, width);
        virxz += __shfl_down(virxz, 2, width);
        energy += __shfl_up(energy, 2, width);
        if (threadLocalId & 2)
        {
            virxx = virzz;  // virxx now holds quad sums of virxx, virxy, virzz and virxy
            virxz = energy; // virxz now holds quad sums of virxz, viryz, energy and unused paddings
        }

        virxx += __shfl_down(virxx, 4, width);
        virxz += __shfl_up(virxz, 4, width);
        if (threadLocalId & 4)
        {
            virxx = virxz; // virxx now holds all 7 components' octet sums + unused paddings
        }

        /* We only need to reduce virxx now */
#pragma unroll
        for (int delta = 8; delta < width; delta <<= 1)
        {
              virxx += __shfl_down(virxx, delta, width);
        }
        /* Now first 7 threads of each warp have the full output contributions in virxx */

        const int  componentIndex      = threadLocalId & (warp_size - 1);
        const bool validComponentIndex = (componentIndex < c_virialAndEnergyCount);
        /* Reduce 7 outputs per warp in the shared memory */
        const int stride =
                8; // this is c_virialAndEnergyCount==7 rounded up to power of 2 for convenience, hence the assert
        assert(c_virialAndEnergyCount == 7);
        const int        reductionBufferSize = (c_solveMaxThreadsPerBlock / warp_size) * stride;
        __shared__ float sm_virialAndEnergy[reductionBufferSize];

        if (validComponentIndex)
        {
            const int warpIndex                                     = threadLocalId / warp_size;
            sm_virialAndEnergy[warpIndex * stride + componentIndex] = virxx;
        }
        __syncthreads();

        /* Reduce to the single warp size */
        const int targetIndex = threadLocalId;
#pragma unroll
        for (int reductionStride = reductionBufferSize >> 1; reductionStride >= warp_size;
             reductionStride >>= 1)
        {
            const int sourceIndex = targetIndex + reductionStride;
            if ((targetIndex < reductionStride) & (sourceIndex < activeWarps * stride))
            {
                // TODO: the second conditional is only needed on first iteration, actually - see if compiler eliminates it!
                sm_virialAndEnergy[targetIndex] += sm_virialAndEnergy[sourceIndex];
            }
            __syncthreads();
        }

        /* Now use shuffle again */
        /* NOTE: This reduction assumes there are at least 4 warps (asserted).
         *       To use fewer warps, add to the conditional:
         *       && threadLocalId < activeWarps * stride
         */
        //assert(activeWarps * stride >= warp_size);
        if (threadLocalId < warp_size)
        {
            float output = sm_virialAndEnergy[threadLocalId];
#pragma unroll
            for (int delta = stride; delta < warp_size; delta <<= 1)
            {
                  output += __shfl_down(output, delta, warp_size);
            }
            /* Final output */
            if (validComponentIndex)
            {
                //assert(isfinite(output));
                atomicAdd((int*)(gm_virialAndEnergy + componentIndex), (int)(output));
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

    bool computeEnergyAndVirial = atoi(argv[1]);

    GridOrdering gridOrdering = GridOrdering::XYZ;
    int majorDim = -1, middleDim = -1, minorDim = -1;
    switch (gridOrdering)
    {
        case GridOrdering::YZX:
            majorDim  = YY;
            middleDim = ZZ;
            minorDim  = XX;
            break;

        case GridOrdering::XYZ:
            majorDim  = XX;
            middleDim = YY;
            minorDim  = ZZ;
            break;

	default:
	    return -1;
    }

    float *h_recipBox;
    float *d_recipBox;
    h_recipBox = new float[DIM*DIM];
    initValueFromFile("recipBox.txt", DIM*DIM, DIM*DIM, h_recipBox);

    int *h_splineValuesOffset;
    int *d_splineValuesOffset;
    h_splineValuesOffset = new int[DIM];
    initValueFromFile("splineValuesOffset.txt", DIM, DIM, h_splineValuesOffset);

    int *h_complexGridSize, *h_complexGridSizePadded, *h_realGridSize;
    int *d_complexGridSize, *d_complexGridSizePadded, *d_realGridSize;

    h_realGridSize          = new int[DIM];
    h_complexGridSize       = new int[DIM];
    h_complexGridSizePadded = new int[DIM];
    initValueFromFile("realGridSize.txt", DIM, DIM, h_realGridSize);
    initValueFromFile("complexGridSize.txt", DIM, DIM, h_complexGridSize);
    initValueFromFile("complexGridSizePadded.txt", DIM, DIM, h_complexGridSizePadded);

    float *h_splineModuli;
    float *d_splineModuli;

    const int newSplineValuesSize = h_realGridSize[XX] +
        h_realGridSize[YY] +
        h_realGridSize[ZZ];

    h_splineModuli = new float[newSplineValuesSize];
    initValueFromFile("d_splineModuli.txt", newSplineValuesSize, newSplineValuesSize, h_splineModuli);

    float *h_fourierGrid;
    float *d_fourierGrid;

    const int complexGridSize = h_complexGridSize[0] * h_complexGridSize[1] * h_complexGridSize[2] * 2;

    h_fourierGrid = new float[complexGridSize];
    initValueFromFile("d_fourierGrid.txt", complexGridSize, complexGridSize, h_fourierGrid);

    const int maxBlockSize      = c_solveMaxThreadsPerBlock;
    const int gridLineSize      = h_complexGridSize[minorDim];
    const int gridLinesPerBlock = std::max(maxBlockSize / gridLineSize, 1);
    const int blocksPerGridLine = (gridLineSize + maxBlockSize - 1) / maxBlockSize;
    int       cellsPerBlock;
    if (blocksPerGridLine == 1)
    {
        cellsPerBlock = gridLineSize * gridLinesPerBlock;
    }
    else
    {
        cellsPerBlock = (gridLineSize + blocksPerGridLine - 1) / blocksPerGridLine;
    }

    int blockSize[3] = {1,1,1};
    int gridSize[3]  = {1,1,1};

    blockSize[0] = (cellsPerBlock + warpSize - 1) / warpSize * warpSize;
    gridSize[0]  = blocksPerGridLine;
    gridSize[1]  = (h_complexGridSize[middleDim] + gridLinesPerBlock - 1)/gridLinesPerBlock;
    gridSize[2]  = h_complexGridSize[majorDim];

    hipMalloc((void**)&d_recipBox, DIM * DIM * sizeof(float));

    hipMalloc((void**)&d_realGridSize, DIM * sizeof(int));
    hipMalloc((void**)&d_complexGridSize, DIM * sizeof(int));
    hipMalloc((void**)&d_complexGridSizePadded, DIM * sizeof(int));

    hipMalloc((void**)&d_splineValuesOffset, DIM * sizeof(int));

    hipMalloc((void**)&d_splineModuli, newSplineValuesSize * sizeof(float));
    hipMalloc((void**)&d_fourierGrid, complexGridSize * sizeof(float));

    float *d_virialAndEnergy;
    hipMalloc((void**)&d_virialAndEnergy, c_virialAndEnergyCount * sizeof(float));
    hipMemset(d_virialAndEnergy, 0, c_virialAndEnergyCount * sizeof(float));

    hipMemcpyHtoD(d_recipBox, h_recipBox, DIM * DIM * sizeof(float));

    hipMemcpyHtoD(d_realGridSize, h_realGridSize, DIM * sizeof(int));
    hipMemcpyHtoD(d_complexGridSize, h_complexGridSize, DIM * sizeof(int));
    hipMemcpyHtoD(d_complexGridSizePadded, h_complexGridSizePadded, DIM * sizeof(int));
    hipMemcpyHtoD(d_splineValuesOffset, h_splineValuesOffset, DIM * sizeof(int));

    hipMemcpyHtoD(d_splineModuli, h_splineModuli, newSplineValuesSize * sizeof(float));
    hipMemcpyHtoD(d_fourierGrid, h_fourierGrid, complexGridSize * sizeof(float));

    void (*solveKernel)(const float *d_splineModuli,
            const float *recipBox,
            const int   *complexGridSizePadded,
            const int   *complexGridSize,
            const int   *realGridSize,
            const int   *splineValuesOffset,
            float       *d_fourierGrid,
            float       *d_virialAndEnergy,
            float       boxVolume,
            float       elFactor,
            float       ewaldFactor);

    if (computeEnergyAndVirial) {
        solveKernel = pme_solve_kernel<GridOrdering::XYZ, true>;
    } else {
        solveKernel = pme_solve_kernel<GridOrdering::XYZ, false>;
    }

    // boxVolume:949.076 elFactor:138.935 ewaldFactor:0.819457
    for (int iter=0; iter<1000; iter++) {
        hipLaunchKernelGGL(solveKernel, dim3(gridSize[0], gridSize[1], gridSize[2])
		    , dim3(blockSize[0], blockSize[1], blockSize[2]), 0, 0
		    , d_splineModuli, d_recipBox, d_complexGridSizePadded
		    , d_complexGridSize, d_realGridSize
		    , d_splineValuesOffset, d_fourierGrid, d_virialAndEnergy
                    , 1.0f, 1.0f, 1.0f);
    }
    hipStreamSynchronize(0);
//#if DEBUG
//    float *h_virialAndEnergy = new float[c_virialAndEnergyCount];
//    hipMemcpyDtoH(h_virialAndEnergy, d_virialAndEnergy, c_virialAndEnergyCount * sizeof(float));
//    for (int i = 0; i < c_virialAndEnergyCount; i++)
//        std::cout << h_virialAndEnergy[i] << std::endl;
//#endif


    return 0;
}
