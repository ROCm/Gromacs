#ifndef _GPU_VEC_OPS_H_ 
#define _GPU_VEC_OPS_H_

#include "types_def.h"

/* maths operations */
__forceinline__ __device__ void svmul_gpu(float a, const fvec v1, fvec v2)
{
    v2[XX] = a * v1[XX];
    v2[YY] = a * v1[YY];
    v2[ZZ] = a * v1[ZZ];
}


__forceinline__ __device__ void fvec_add_gpu(const fvec a, const fvec b, fvec c)
{
    float x, y, z;

    x = a[XX] + b[XX];
    y = a[YY] + b[YY];
    z = a[ZZ] + b[ZZ];

    c[XX] = x;
    c[YY] = y;
    c[ZZ] = z;
}

__forceinline__ __device__ void ivec_add_gpu(const ivec a, const ivec b, ivec c)
{
    int x, y, z;

    x = a[XX] + b[XX];
    y = a[YY] + b[YY];
    z = a[ZZ] + b[ZZ];

    c[XX] = x;
    c[YY] = y;
    c[ZZ] = z;
}

__forceinline__ __device__ void fvec_inc_atomic(fvec a, const fvec b)
{
    atomicAddNoRet(&a[XX], b[XX]);
    atomicAddNoRet(&a[YY], b[YY]);
    atomicAddNoRet(&a[ZZ], b[ZZ]);
}

__forceinline__ __device__ void fvec_inc_gpu(fvec a, const fvec b)
{
    float x, y, z;

    x = a[XX] + b[XX];
    y = a[YY] + b[YY];
    z = a[ZZ] + b[ZZ];

    a[XX] = x;
    a[YY] = y;
    a[ZZ] = z;
}

__forceinline__ __device__ void fvec_dec_atomic(fvec a, const fvec b)
{
    atomicAddNoRet(&a[XX], -1.0f * b[XX]);
    atomicAddNoRet(&a[YY], -1.0f * b[YY]);
    atomicAddNoRet(&a[ZZ], -1.0f * b[ZZ]);
}

__forceinline__ __device__ void fvec_dec_gpu(fvec a, const fvec b)
{
    float x, y, z;

    x = a[XX] - b[XX];
    y = a[YY] - b[YY];
    z = a[ZZ] - b[ZZ];

    a[XX] = x;
    a[YY] = y;
    a[ZZ] = z;
}

__forceinline__ __device__ void cprod_gpu(const fvec a, const fvec b, fvec c)
{
    c[XX] = a[YY] * b[ZZ] - a[ZZ] * b[YY];
    c[YY] = a[ZZ] * b[XX] - a[XX] * b[ZZ];
    c[ZZ] = a[XX] * b[YY] - a[YY] * b[XX];
}

__forceinline__ __device__ float iprod_gpu(const fvec a, const fvec b)
{
    return (a[XX] * b[XX] + a[YY] * b[YY] + a[ZZ] * b[ZZ]);
}

__forceinline__ __device__ float norm_gpu(const fvec a)
{
    return sqrt(iprod_gpu(a, a));
}

__forceinline__ __device__ float gmx_angle_gpu(const fvec a, const fvec b)
{
    fvec  w;
    float wlen, s;

    cprod_gpu(a, b, w);

    wlen = norm_gpu(w);
    s    = iprod_gpu(a, b);

    return atan2f(wlen, s); // requires float
}

__forceinline__ __device__ void clear_ivec_gpu(ivec a)
{
    a[XX] = 0;
    a[YY] = 0;
    a[ZZ] = 0;
}
__forceinline__ __device__ void fvec_sub_gpu(const fvec a, const fvec b, fvec c)
{
    float x, y, z;

    x = a[XX] - b[XX];
    y = a[YY] - b[YY];
    z = a[ZZ] - b[ZZ];

    c[XX] = x;
    c[YY] = y;
    c[ZZ] = z;
}

__forceinline__ __device__ float norm2_gpu(const fvec a)
{
    return a[XX] * a[XX] + a[YY] * a[YY] + a[ZZ] * a[ZZ];
}

__forceinline__ __device__ void copy_fvec_gpu(const fvec a, fvec b)
{
    b[XX] = a[XX];
    b[YY] = a[YY];
    b[ZZ] = a[ZZ];
}

__forceinline__ __device__ void copy_ivec_gpu(const ivec a, ivec b)
{
    b[XX] = a[XX];
    b[YY] = a[YY];
    b[ZZ] = a[ZZ];
}

__forceinline__ __device__ float cos_angle_gpu(const fvec a, const fvec b)
{
    /*
     *                  ax*bx + ay*by + az*bz
     * cos-vec (a,b) =  ---------------------
     *                      ||a|| * ||b||
     */
    float cosval;
    int   m;
    float aa, bb, ip, ipa, ipb, ipab;

    ip = ipa = ipb = 0.0f;
    for (m = 0; (m < DIM); m++)
    {
        aa = a[m];
        bb = b[m];
        ip += aa * bb;
        ipa += aa * aa;
        ipb += bb * bb;
    }
    ipab = ipa * ipb;
    if (ipab > 0.0f)
    {
        cosval = ip * rsqrt(ipab);
    }
    else
    {
        cosval = 1.0f;
    }
    if (cosval > 1.0f)
    {
        return 1.0f;
    }
    if (cosval < -1.0f)
    {
        return -1.0f;
    }

    return cosval;
}


__device__ static inline void unitv_gpu(const fvec src, fvec dest)
{
    float linv;

    linv     = rsqrt(norm2_gpu(src));
    dest[XX] = linv * src[XX];
    dest[YY] = linv * src[YY];
    dest[ZZ] = linv * src[ZZ];
}

#endif
