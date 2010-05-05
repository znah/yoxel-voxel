//
#include "utils.h"

__constant__ float4x4 c_proj2wldMtx;
__constant__ float3   c_eyePos;
__constant__ int2     c_viewSize;

texture<uchar,  3, cudaReadModeNormalizedFloat> volumeTex;   // 3D texture
texture<float4, 1, cudaReadModeElementType>     transferTex; // 1D transfer function texture
texture<uchar,  3, cudaReadModeElementType>     markTex;

struct RayData
{
    float3 orig;
    float3 dir;
};

__device__ RayData PrepareRay(int x, int y)
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
    
    RayData ray;
    ray.orig = c_eyePos;
    ray.dir  = dir;
    return ray;
}


__device__ bool walkBrick(RayData ray, float t_enter, float t_exit, float4 & accum)
{
    const float dt = 1.0f / 1024.0f;
    for (float t = t_enter; t < t_exit; t += dt)
    {
        float3 p = ray.orig + t * ray.dir;
        float v = tex3D(volumeTex, p.x, p.y, p.z);
        float4 col = make_float4(saturate( (v-0.5f)*4.0 ));
        col.w *= 0.5;
        col.x *= col.w;
        col.y *= col.w;
        col.z *= col.w;

        accum += col * (1.0 - accum.w);
        if (accum.w > 0.99f)
            return true;
    }
    return false;
}


__device__ float3 intersect(float3 orig, float3 dt, float3 lo, float3 hi, float & t_enter, float & t_exit)
{
    float3 tlo = dt*(lo - orig);
    float3 thi = dt*(hi - orig);

    float3 t1 = fminf(tlo, thi);
    float3 t2 = fmaxf(tlo, thi);

    t_enter = max(t1.x, max(t1.y, t1.z));
    t_exit  = min(t2.x, min(t2.y, t2.z));

    if (t_exit == t2.x) return make_float3(1, 0, 0);
    if (t_exit == t2.y) return make_float3(0, 1, 0);
    return make_float3(0, 0, 1);
}


__device__ float4 castRay(RayData ray)
{
    /*float3 t_coef, t_bias;
    t_coef.x = 1.0f / fabs(ray.dir.x);
    t_coef.y = 1.0f / fabs(ray.dir.y);
    t_coef.z = 1.0f / fabs(ray.dir.z);

    t_bias.x = -t_coef.x * ray.orig.x;
    t_bias.y = -t_coef.y * ray.orig.y;
    t_bias.z = -t_coef.z * ray.orig.z;

    int octant_mask = 0;
    if (ray.dir.x < 0.0f) octant_mask ^= 1, t_bias.x = -t_coef.x - t_bias.x;
    if (ray.dir.y < 0.0f) octant_mask ^= 2, t_bias.y = -t_coef.y - t_bias.y;
    if (ray.dir.z < 0.0f) octant_mask ^= 4, t_bias.z = -t_coef.z - t_bias.z;*/

    float3 dt = make_float3(1.0f) / ray.dir;

    float t_enter, t_exit;
    intersect(ray.orig, dt, make_float3(0.0), make_float3(1.0), t_enter, t_exit);

    if (t_exit < 0.0f || t_enter > t_exit)
        return make_float4(0, 0, 0, 0);

    const float gridSize  = 64.0f;
    const float brickSize = 1.0f / gridSize;

    float3 p = ray.orig + t_enter * ray.dir;
    p = make_float3(make_int3(p * gridSize)) * brickSize;

    float3 dp;
    dp.x = copysignf(1.0f, ray.dir.x);
    dp.y = copysignf(1.0f, ray.dir.y);
    dp.z = copysignf(1.0f, ray.dir.z);

    float t_stop = t_exit;

    float4 accum = make_float4(0);
    while (true)
    {
      float3 step = intersect(ray.orig, dt, p, p+brickSize, t_enter, t_exit) * dp;
      if (t_enter > t_stop)
        break;

      float3 p2 = p + brickSize*0.5f;
      //if (tex3D(markTex, p2.x, p2.y, p2.z) > 0)
        //walkBrick(ray, t_enter, t_exit, accum);
      accum.x += 0.1;
      p += step;
    }

    walkBrick(ray, t_enter, t_exit, accum);
    return accum;
}
 
extern "C"
__global__ void Trace(uint * d_img)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;

    RayData ray = PrepareRay(x, y);
    float4 res = castRay(ray);

    int ofs = x + y * c_viewSize.x;
    d_img[ofs] = float2uint(res);
}
