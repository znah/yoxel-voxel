#pragma once

#include "cutil_math.h"

typedef unsigned int uint;
typedef unsigned char uchar;

struct float4x4
{
    float4 m[4];
};

// transform vector by matrix (no translation)
__device__
float3 mul(const float4x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float4x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = dot(v, M.m[3]);
    return r;
}


__device__ uint float2uint(float4 v)
{
    uint res = (uint)(saturate(v.x) * 255);
    res |= (uint)(saturate(v.y) * 255) <<  8;
    res |= (uint)(saturate(v.z) * 255) << 16;
    res |= (uint)(saturate(v.w) * 255) << 24;
    return res;
}


__device__ float fmaxf(float a, float b, float c)
{
  return fmaxf( fmaxf(a, b), c );
}

__device__ float fminf(float a, float b, float c)
{
  return fminf( fminf(a, b), c );
}