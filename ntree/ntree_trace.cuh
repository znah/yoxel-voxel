#pragma once

typedef texture<uchar4, 3, cudaReadModeNormalizedFloat> VoxDataTex;
typedef texture<uchar4, 3, cudaReadModeElementType> VoxChildTex;
typedef texture<uint4, 1, cudaReadModeElementType> VoxNodeTex;

extern "C"
{
  
}