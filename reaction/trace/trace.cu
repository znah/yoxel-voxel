//
#include "utils.h"

__constant__ float4x4 c_proj2wldMtx;
__constant__ float3   c_eyePos;
__constant__ int2     c_viewSize;

texture<uchar,  3, cudaReadModeNormalizedFloat> volumeTex;   // 3D texture
texture<float4, 1, cudaReadModeElementType>     transferTex; // 1D transfer function texture


__device__ float3 calcRayDir(int x, int y)
{
    const float eps = 1e-5f;
    float px = 2.0f * (float)x / c_viewSize.x - 1.0f;
    float py = 2.0f * (float)y / c_viewSize.y - 1.0f;
    float4 v = make_float4( px, py, 0, 1 );
    v = mul(c_proj2wldMtx, v);
    float invW = 1.0f / v.w;
    v.x *= invW;
    v.y *= invW;
    v.z *= invW;
    float3 dir = normalize(make_float3(v) - c_eyePos);
    if (fabsf(dir.x) < eps) dir.x = copysignf(eps, dir.x);
    if (fabsf(dir.y) < eps) dir.y = copysignf(eps, dir.y);
    if (fabsf(dir.z) < eps) dir.z = copysignf(eps, dir.z);
    return dir;
}

__device__ float4 castRay(float3 orig, float3 dir)
{
    float tx_coef = 1.0f / fabs(dir.x);
    float ty_coef = 1.0f / fabs(dir.y);
    float tz_coef = 1.0f / fabs(dir.z);

    float tx_bias = -tx_coef * orig.x;
    float ty_bias = -ty_coef * orig.y;
    float tz_bias = -tz_coef * orig.z;

    int octant_mask = 0;
    if (dir.x < 0.0f) octant_mask ^= 1, tx_bias = -tx_coef - tx_bias;
    if (dir.y < 0.0f) octant_mask ^= 2, ty_bias = -ty_coef - ty_bias;
    if (dir.z < 0.0f) octant_mask ^= 4, tz_bias = -tz_coef - tz_bias;

    float t_enter = fmaxf(tx_bias, ty_bias, tz_bias);
    float t_exit  = fminf(tx_coef + tx_bias, ty_coef + ty_bias, tz_coef + tz_bias);
    
    t_enter = fmaxf(0.0f, t_enter);
    if (t_exit < 0.0f || t_enter > t_exit)
        return make_float4(0, 0, 0, 0);







    return make_float4(t_enter, t_exit, 0, 0);

}
 
extern "C"
__global__ void Trace(uint * d_img)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    float3 dir = calcRayDir(x, y);
    float4 res = castRay(c_eyePos, dir);

    int ofs = x + y * c_viewSize.x;
    d_img[ofs] = float2uint(res);
}
