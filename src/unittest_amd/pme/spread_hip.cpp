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
constexpr int c_pmeOrder              = 4;
constexpr int warp_size                  = 64;
constexpr int c_pmeAtomDataBlockSize     = 128;
constexpr int c_spreadMaxWarpsPerBlock   = 8;
constexpr int c_spreadMaxThreadsPerBlock = (c_spreadMaxWarpsPerBlock * warp_size);

constexpr int c_pmeMaxUnitcellShift = 2;
constexpr int c_pmeNeighborUnitcellCount = 2 * c_pmeMaxUnitcellShift + 1;

constexpr bool c_usePadding = true;
constexpr bool c_skipNeutralAtoms = false;
static const bool c_useAtomDataPrefetch = true;

template<typename ValueType>
using DeviceBuffer = ValueType*;
typedef float real;

#    define HIDE_FROM_OPENCL_COMPILER(x) x
#    define NUMFEPSTATES 2

enum class ThreadsPerAtom : int
{
    Order,
    OrderSquared,
    Count
};

namespace gmx
{

template<typename ValueType>
class BasicVector
{
public:
    //! Underlying raw C array type (rvec/dvec/ivec).
    using RawArray = ValueType[DIM];

    // The code here assumes ValueType has been deduced as a data type like int
    // and not a pointer like int*. If there is a use case for a 3-element array
    // of pointers, the implementation will be different enough that the whole
    // template class should have a separate partial specialization. We try to avoid
    // accidental matching to pointers, but this assertion is a no-cost extra check.
    //
    // TODO: Use std::is_pointer_v when CUDA 11 is a requirement.
    //static_assert(!std::is_pointer<std::remove_cv_t<ValueType>>::value,
    //              "BasicVector value type must not be a pointer.");

    //! Constructs default (uninitialized) vector.
    BasicVector() {}
    //! Constructs a vector from given values.
    BasicVector(ValueType x, ValueType y, ValueType z) : x_{ x, y, z } {}
    /*! \brief
     * Constructs a vector from given values.
     *
     * This constructor is not explicit to support implicit conversions
     * that allow, e.g., calling `std::vector<RVec>:``:push_back()` directly
     * with an `rvec` parameter.
     */
    BasicVector(const RawArray x) : x_{ x[XX], x[YY], x[ZZ] } {}
    //! Default copy constructor.
    BasicVector(const BasicVector& src) = default;
    //! Default copy assignment operator.
    BasicVector& operator=(const BasicVector& v) = default;
    //! Default move constructor.
    BasicVector(BasicVector&& src) noexcept = default;
    //! Default move assignment operator.
    BasicVector& operator=(BasicVector&& v) noexcept = default;
    //! Indexing operator to make the class work as the raw array.
    ValueType& operator[](int i) { return x_[i]; }
    //! Indexing operator to make the class work as the raw array.
    ValueType operator[](int i) const { return x_[i]; }
    //! Allow inplace addition for BasicVector
    BasicVector<ValueType>& operator+=(const BasicVector<ValueType>& right)
    {
        return *this = *this + right;
    }
    //! Allow inplace subtraction for BasicVector
    BasicVector<ValueType>& operator-=(const BasicVector<ValueType>& right)
    {
        return *this = *this - right;
    }
    //! Allow vector addition
    BasicVector<ValueType> operator+(const BasicVector<ValueType>& right) const
    {
        return { x_[0] + right[0], x_[1] + right[1], x_[2] + right[2] };
    }
    //! Allow vector subtraction
    BasicVector<ValueType> operator-(const BasicVector<ValueType>& right) const
    {
        return { x_[0] - right[0], x_[1] - right[1], x_[2] - right[2] };
    }
    //! Allow vector scalar division
    BasicVector<ValueType> operator/(const ValueType& right) const
    {
        assert((right != 0 && "Cannot divide by zero"));

        return *this * (1 / right);
    }
    //! Scale vector by a scalar
    BasicVector<ValueType>& operator*=(const ValueType& right)
    {
        x_[0] *= right;
        x_[1] *= right;
        x_[2] *= right;

        return *this;
    }
    //! Divide vector by a scalar
    BasicVector<ValueType>& operator/=(const ValueType& right)
    {
        assert((right != 0 && "Cannot divide by zero"));

        return *this *= 1 / right;
    }
    //! Return dot product
    ValueType dot(const BasicVector<ValueType>& right) const
    {
        return x_[0] * right[0] + x_[1] * right[1] + x_[2] * right[2];
    }

    //! Allow vector vector multiplication (cross product)
    BasicVector<ValueType> cross(const BasicVector<ValueType>& right) const
    {
        return { x_[YY] * right.x_[ZZ] - x_[ZZ] * right.x_[YY],
                 x_[ZZ] * right.x_[XX] - x_[XX] * right.x_[ZZ],
                 x_[XX] * right.x_[YY] - x_[YY] * right.x_[XX] };
    }

    //! Return normalized to unit vector
    BasicVector<ValueType> unitVector() const
    {
        const ValueType vectorNorm = norm();
        assert((vectorNorm != 0 && "unitVector() should not be called with a zero vector"));

        return *this / vectorNorm;
    }

    //! Length^2 of vector
    ValueType norm2() const { return dot(*this); }

    //! Norm or length of vector
    ValueType norm() const { return std::sqrt(norm2()); }

    //! cast to RVec
    BasicVector<real> toRVec() const { return { real(x_[0]), real(x_[1]), real(x_[2]) }; }

    //! cast to IVec
    BasicVector<int> toIVec() const
    {
        return { static_cast<int>(x_[0]), static_cast<int>(x_[1]), static_cast<int>(x_[2]) };
    }

    //! cast to DVec
    BasicVector<double> toDVec() const { return { double(x_[0]), double(x_[1]), double(x_[2]) }; }

    //! Converts to a raw C array where implicit conversion does not work.
    RawArray& as_vec() { return x_; }

private:
    RawArray x_;
};
//typedef BasicVector<real> RVec;
} //namespace gmx

#define GMX_UNUSED_VALUE(value) (void)value
#define HIP_PI_F 3.141592654f
#define CLANG_DISABLE_OPTIMIZATION_ATTRIBUTE

#define INLINE_EVERYWHERE __host__ __device__ __forceinline__
#        define gmx_unused __attribute__((unused))

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

template<typename T>
__device__ inline void assertIsFinite(T arg);

template<>
__device__ inline void assertIsFinite(float3 gmx_unused arg)
{
    assert(isfinite(float(arg.x)));
    assert(isfinite(float(arg.y)));
    assert(isfinite(float(arg.z)));
}

template<typename T>
__device__ inline void assertIsFinite(T gmx_unused arg)
{
    assert(isfinite(float(arg)));
}

template<typename T, const int atomsPerBlock, const int dataCountPerAtom>
__device__ __forceinline__ void pme_gpu_stage_atom_data(T* __restrict__ sm_destination,
                                                        const T* __restrict__ gm_source)
{
    const int blockIndex       = blockIdx.y * gridDim.x + blockIdx.x;
    const int threadLocalIndex = ((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x) + threadIdx.x;
    const int localIndex       = threadLocalIndex;
    const int globalIndexBase = blockIndex * atomsPerBlock * dataCountPerAtom;
    const int globalIndex     = globalIndexBase + localIndex;
    if (localIndex < atomsPerBlock * dataCountPerAtom)
    {
        assertIsFinite(gm_source[globalIndex]);
        sm_destination[localIndex] = gm_source[globalIndex];
    }
}

template<const int order, const int atomsPerBlock, const int atomsPerWarp, const bool writeSmDtheta, const bool writeGlobal, ThreadsPerAtom threadsPerAtom>
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

    const int orderIndex      = threadIdx.y;
    /* Dimension index */
    const int dimIndex = threadIdx.x;

    /* Multi-purpose index of rvec/ivec atom data */
    const int sharedMemoryIndex = atomIndexLocal * DIM + dimIndex;

    float splineData[order];

    const int localCheck = (dimIndex < DIM);

    /* we have 4 threads per atom, but can only use 3 here for the dimensions */
    if (localCheck)
    {
        /* Indices interpolation */

        if (orderIndex == 0)
        {
            int   tableIndex, tInt;
            float n, t;
            assert(atomIndexLocal < DIM * atomsPerBlock);
            /* Accessing fields in fshOffset/nXYZ/recipbox/... with dimIndex offset
             * puts them into local memory(!) instead of accessing the constant memory directly.
             * That's the reason for the switch, to unroll explicitly.
             * The commented parts correspond to the 0 components of the recipbox.
             */
	    switch (dimIndex)
            {
                case XX:
                    tableIndex = tablesOffsets[XX];
                    n          = realGridSizeFP[XX];
                    t          = atomX.x * recipBox[dimIndex*DIM*XX]
                        + atomX.y * recipBox[dimIndex*DIM*YY]
                        + atomX.z * recipBox[dimIndex*DIM*ZZ];
			//printf("XX, %d, %f, %f\n", tableIndex, n, t);
                    break;

                case YY:
                    tableIndex = tablesOffsets[YY];
                    n          = realGridSizeFP[YY];
                    t = /*atomX.x * kernelParams.current.recipBox[dimIndex][XX] + */ atomX.y
                                * recipBox[dimIndex*DIM*YY]
                        + atomX.z * recipBox[dimIndex*DIM*ZZ];
			//printf("YY, %d, %f, %f\n", tableIndex, n, t);
                    break;

                case ZZ:
                    tableIndex = tablesOffsets[ZZ];
                    n          = realGridSizeFP[ZZ];
                    t          = /*atomX.x * kernelParams.current.recipBox[dimIndex][XX] + atomX.y * kernelParams.current.recipBox[dimIndex][YY] + */ atomX
                                .z
                        * recipBox[dimIndex*DIM*ZZ];
			//printf("ZZ, %d, %f, %f\n", tableIndex, n, t);
                    break;
            }
            const float shift = c_pmeMaxUnitcellShift;
            /* Fractional coordinates along box vectors, adding a positive shift to ensure t is positive for triclinic boxes */
	    //printf("%f, %f, %f\n", t, shift, n);
            t    = (t + shift) * n;
            tInt = (int)t;
            assert(sharedMemoryIndex < atomsPerBlock * DIM);
            sm_fractCoords[sharedMemoryIndex] = t - tInt;
            tableIndex += tInt;
            assert(tInt >= 0);
            assert(tInt < c_pmeNeighborUnitcellCount * n);

	    // TODO have shared table for both parameters to share the fetch, as index is always same?
            // TODO compare texture/LDG performance
            sm_fractCoords[sharedMemoryIndex] +=
                    fetchFromParamLookupTable(d_fractShiftsTable,
                                              tableIndex);
            sm_gridlineIndices[sharedMemoryIndex] =
                    fetchFromParamLookupTable(d_gridlineIndicesTable,
                                              tableIndex);
	    //printf("sm_fractCoords, %d, %f, %f\n", sharedMemoryIndex, sm_fractCoords[sharedMemoryIndex], sm_gridlineIndices[sharedMemoryIndex]);
            if (writeGlobal)
            {
                gm_gridlineIndices[atomIndexOffset * DIM + sharedMemoryIndex] =
                        sm_gridlineIndices[sharedMemoryIndex];
            }
        }
	__syncthreads();

        /* B-spline calculation */

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
		const int ithyMin = (threadsPerAtom == ThreadsPerAtom::Order) ? 0 : threadIdx.y;
                const int ithyMax = (threadsPerAtom == ThreadsPerAtom::Order) ? order : threadIdx.y + 1;
#pragma unroll
	       	for (int ithy = ithyMin; ithy < ithyMax; ithy++)
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
	    const int ithyMin = (threadsPerAtom == ThreadsPerAtom::Order) ? 0 : threadIdx.y;
            const int ithyMax = (threadsPerAtom == ThreadsPerAtom::Order) ? order : threadIdx.y + 1;
#pragma unroll
	    for (int ithy = ithyMin; ithy < ithyMax; ithy++)
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

template<int order, int gridIndex, ThreadsPerAtom threadsPerAtom>
__device__ __forceinline__ void spread_charges(const bool wrapX, const bool wrapY,
                                               float*                       d_realGrid0,
                                               float*                       d_realGrid1,
                                               int                          atomIndexOffset,
                                               const int                    nAtoms,
                                               const int*                   realGridSize,
                                               const int*                   realGridSizePadded,
                                               const float*                 atomCharge,
                                               const int* __restrict__ sm_gridlineIndices,
                                               const float* __restrict__ sm_theta)
{
    /* Global memory pointer to the output grid */
    float* __restrict__ gm_grid = gridIndex? d_realGrid1 : d_realGrid0;

    // Number of atoms processed by a single warp in spread and gather
    const int threadsPerAtomValue = (threadsPerAtom == ThreadsPerAtom::Order) ? order : order * order;
    const int atomsPerWarp        = warp_size / threadsPerAtomValue;

    const int nx  = realGridSize[XX];
    const int ny  = realGridSize[YY];
    const int nz  = realGridSize[ZZ];
    const int pny = realGridSizePadded[YY];
    const int pnz = realGridSizePadded[ZZ];

    const int offx = 0, offy = 0, offz = 0; // unused for now

    const int atomIndexLocal  = threadIdx.z;

    const int chargeCheck = pme_gpu_check_atom_charge(*atomCharge);
    if (chargeCheck)
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
	const int ithyMin = (threadsPerAtom == ThreadsPerAtom::Order) ? 0 : threadIdx.y;
        const int ithyMax = (threadsPerAtom == ThreadsPerAtom::Order) ? order : threadIdx.y + 1;
        for (int ithy = ithyMin; ithy < ithyMax; ithy++)
        {
            int iy = iyBase + ithy;
            if (wrapY & (iy >= ny))
            {
                iy -= ny;
            }

            const int splineIndexY = getSplineParamIndex<order, atomsPerWarp>(splineIndexBase, YY, ithy);
            float       thetaY = sm_theta[splineIndexY];
	    //printf("%f, %f, %f\n", thetaZ, thetaY, *atomCharge);
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
                atomicAdd(gm_grid + gridIndexGlobal, thetaX * Val);
            }
        }
    }
}

template<int order, int numGrids, bool writeGlobal, ThreadsPerAtom threadsPerAtom>
__launch_bounds__(c_spreadMaxThreadsPerBlock) CLANG_DISABLE_OPTIMIZATION_ATTRIBUTE __global__
        void pme_spline_and_spread_kernel(const bool computeSplines,
             const bool spreadCharges, const bool wrapX, const bool wrapY,
             const int nAtoms,
             const float3 *d_coordinates,
             const float *d_coefficients0,
             const float *d_coefficients1,
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
             float       *d_realGrid0,
	     float       *d_realGrid1)
{
    const int threadsPerAtomValue = (threadsPerAtom == ThreadsPerAtom::Order) ? order : order * order;
    const int atomsPerBlock       = c_spreadMaxThreadsPerBlock / threadsPerAtomValue;
    // Number of atoms processed by a single warp in spread and gather
    const int atomsPerWarp = warp_size / threadsPerAtomValue;
    // Gridline indices, ivec
    __shared__ int sm_gridlineIndices[atomsPerBlock * DIM];
    // Charges
    __shared__ float sm_coefficients[atomsPerBlock];
    // Spline values
    __shared__ float sm_theta[atomsPerBlock * DIM * order];
    float            dtheta;

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
        pme_gpu_stage_atom_data<float, atomsPerBlock, 1>(sm_coefficients,
                                                         d_coefficients0);
        __syncthreads();
        atomCharge = sm_coefficients[atomIndexLocal];
    }
    else
    {
        atomCharge = d_coefficients0[atomIndexGlobal];
    }

    if (computeSplines)
    {
	const float3* __restrict__ gm_coordinates = d_coordinates;
      //  for(int i=0;i<100;i++)

        if (c_useAtomDataPrefetch)
        {
            // Coordinates
            __shared__ float3 sm_coordinates[atomsPerBlock];

            /* Staging coordinates */
            pme_gpu_stage_atom_data<float3, atomsPerBlock, 1>(sm_coordinates,
                                                               gm_coordinates);
            __syncthreads();
	    atomX = sm_coordinates[atomIndexLocal];
        }
        else
        {
	    atomX = gm_coordinates[atomIndexGlobal];
        }
    		printf("(%f,%f,%f) %d\n", atomX.x, atomX.y, atomX.z, atomIndexLocal);
	calculate_splines<order, atomsPerBlock, atomsPerWarp, false, writeGlobal, ThreadsPerAtom::Order>(
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
        pme_gpu_stage_atom_data<float, atomsPerBlock, DIM * order>(sm_theta,
                                                                   d_theta);
        /* Gridline indices */
        pme_gpu_stage_atom_data<int, atomsPerBlock, DIM>(sm_gridlineIndices,
                                                         d_gridlineIndices);

        __syncthreads();
    }

    /* Spreading */
    if (spreadCharges)
    {
	spread_charges<order, 0, threadsPerAtom>(wrapX, wrapY, d_realGrid0,
			d_realGrid1, atomIndexOffset, nAtoms, realGridSize, realGridSizePadded,
                	&atomCharge, sm_gridlineIndices, sm_theta);
    }
    if (numGrids == 2)
    {
        __syncthreads();
        if (c_useAtomDataPrefetch)
        {
            pme_gpu_stage_atom_data<float, atomsPerBlock, 1>(sm_coefficients,
                                                             d_coefficients1);
            __syncthreads();
            atomCharge = sm_coefficients[atomIndexLocal];
        }
        else
        {
            atomCharge = d_coefficients1[atomIndexGlobal];
        }
        if (spreadCharges)
        {
            spread_charges<order, 1, threadsPerAtom>(wrapX, wrapY, d_realGrid0,
                        d_realGrid1, atomIndexOffset, nAtoms, realGridSize, realGridSizePadded,
                        &atomCharge, sm_gridlineIndices, sm_theta);
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

template <typename T>
void initVector3ValueFromFile(const char* fileName, int dataSize, int totalSize, T* out) {
    std::ifstream myfile;
    myfile.open(fileName);
    
    float3 value;
    for (int i = 0; i < dataSize; i++) {
        myfile >> value.x;//out[i];
        myfile >> value.y;//out[i];
        myfile >> value.z;//out[i];
	out[i] = value;
    }
    myfile.close();

    for (int i = dataSize; i < totalSize; i++) {
        out[i] = float3 {0.0, 0.0, 0.0};
    }
}


int main(int argc, char *argv[]) {

    int nAtoms                 = atoi(argv[1]);
    const bool useOrderThreads = atoi(argv[2]);

    std::cout << "useOrderThreads:" << useOrderThreads << std::endl;

    const int order  = 4;

    int nAtomsPadded = ((nAtoms + c_pmeAtomDataBlockSize - 1)/c_pmeAtomDataBlockSize) * c_pmeAtomDataBlockSize;
    int threadsPerAtomValue = order;
    int atomsPerBlock       = c_spreadMaxThreadsPerBlock / threadsPerAtomValue;

    int grid_dim_x = nAtomsPadded/atomsPerBlock;

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

    float3 *h_coordinates;
    float *h_coefficients[NUMFEPSTATES];
    float3 *d_coordinates;
    float *d_coefficients[NUMFEPSTATES];
    h_coordinates  = new float3[nAtomsPadded*DIM];
    h_coefficients[0] = new float[nAtomsPadded];
    h_coefficients[1] = new float[nAtomsPadded];
    initVector3ValueFromFile("d_coordinates.txt", nAtomsPadded * DIM, nAtomsPadded * DIM, h_coordinates);
    initValueFromFile("d_coefficients0.txt", nAtomsPadded, nAtomsPadded, h_coefficients[0]);
    initValueFromFile("d_coefficients1.txt", nAtomsPadded, nAtomsPadded, h_coefficients[1]);

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

    hipMalloc((void**)&d_coordinates, nAtomsPadded * DIM *sizeof(float3));
    hipMalloc((void**)&d_coefficients[0], nAtomsPadded * sizeof(float));
    hipMalloc((void**)&d_coefficients[1], nAtomsPadded * sizeof(float));
    hipMalloc((void**)&d_fractShiftsTable, newFractShiftsSize * sizeof(float));

    hipMalloc((void**)&d_gridlineIndicesTable, newFractShiftsSize * sizeof(int));

    float *d_realGrid[NUMFEPSTATES];
    hipMalloc((void**)&d_realGrid[0], nRealGridSize * sizeof(float));
    hipMalloc((void**)&d_realGrid[1], nRealGridSize * sizeof(float));

    hipMemcpyHtoD(d_realGridSize, h_realGridSize, DIM * sizeof(int));
    hipMemcpyHtoD(d_realGridSizePadded, h_realGridSizePadded, DIM * sizeof(int));
    hipMemcpyHtoD(d_tableOffsets, h_tableOffsets, DIM * sizeof(int));

    hipMemcpyHtoD(d_recipBox, h_recipBox, DIM * DIM * sizeof(float));
    hipMemcpyHtoD(d_realGridSizeFP, h_realGridSizeFP, DIM * sizeof(float));

    hipMemcpyHtoD(d_theta, h_theta, nAtomsPadded * DIM * order * sizeof(float));
    hipMemcpyHtoD(d_dtheta, h_dtheta, nAtomsPadded * DIM * order * sizeof(float));
    hipMemcpyHtoD(d_gridlineIndices, h_gridlineIndices, nAtomsPadded * DIM * sizeof(int));

    hipMemcpyHtoD(d_coordinates, h_coordinates, nAtomsPadded * DIM * sizeof(float3));
    printf("%d %d*********\n", DIM, nAtomsPadded);
   // for(int i=0;i<DIM*nAtomsPadded;i++)
    //printf("(%f,%f,%f)\n", d_coordinates[i].x,d_coordinates[i].y,d_coordinates[i].z);
    hipMemcpyHtoD(d_coefficients[0], h_coefficients[0], nAtomsPadded * sizeof(float));
    hipMemcpyHtoD(d_coefficients[1], h_coefficients[1], nAtomsPadded * sizeof(float));

    hipMemcpyHtoD(d_fractShiftsTable, h_fractShiftsTable, newFractShiftsSize * sizeof(int));
    hipMemcpyHtoD(d_gridlineIndicesTable, h_gridlineIndicesTable, newFractShiftsSize * sizeof(int));

    void (*spreadKernel)(const bool computeSplines,
             const bool spreadCharges, const bool wrapX, const bool wrapY,
             const int nAtoms,
             const float3 *d_coordinates,
             const float *d_coefficients0,
             const float *d_coefficients1,
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
             float       *d_realGrid0,
	     float       *d_realGrid1);
    
        //if (numGrids == 2)
        {
            spreadKernel = pme_spline_and_spread_kernel<c_pmeOrder, 2, false, ThreadsPerAtom::Order>;
        }
       // else
       // {
       //     kernelPtr = pme_spline_and_spread_kernel<c_pmeOrder, true, true, c_wrapX, c_wrapY, 1, false, ThreadsPerAtom::Order>;
       // }
  //  for (int iter=0; iter<1000; iter++) {
        hipLaunchKernelGGL(spreadKernel, 
			dim3(grid_dim_x, 1, 1), 
			dim3(order, 1, atomsPerBlock), 0, 0
    		    , true, true, true, true
    		    , nAtoms, d_coordinates, d_coefficients[0], d_coefficients[1]
    		    , d_realGridSizeFP, d_recipBox, d_fractShiftsTable
    		    , d_gridlineIndicesTable,d_tableOffsets, d_realGridSize
    		    , d_realGridSizePadded, d_theta, d_dtheta, d_gridlineIndices
    		    , d_realGrid[0], d_realGrid[1]);
   // }
    hipStreamSynchronize(0);
    /*
printf("********************hello\n");
    float *h_realGrid = new float[nRealGridSize];

    hipMemcpyDtoH(h_realGrid, d_realGrid[0], nRealGridSize * sizeof(float));
    for (int i = 0; i < nRealGridSize; i++)
        std::cout << h_realGrid[i] << std::endl;
*/
    return 0;
}
