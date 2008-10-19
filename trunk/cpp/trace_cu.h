#pragma once

#include <cuda_runtime.h>

#pragma pack(push, 4)

typedef unsigned int uint;
typedef unsigned char uchar;
typedef unsigned short ushort;

  // octant - 3bit, qx - 7 bit, qy - 6 bit
typedef ushort VoxNormal;

inline VoxNormal PackNormal(float x, float y, float z)
{
  int octant = 0;
  if (x < 0) { octant |= 1; x = -x; }
  if (y < 0) { octant |= 2; y = -y; }
  if (z < 0) { octant |= 4; z = -z; }

  float t = 1.0f / (x + y + z);
  float px = t*x;
  float py = t*y;

  int qx = (int)(127.0f*px);
  int qy = (int)(63.0f*py);

  VoxNormal res = 0;
  res |= octant;
  res |= qx << 3;
  res |= qy << (3+7);
  return res;
}

inline __device__ __host__ void UnpackNormal(ushort packed, float & x, float & y, float & z)
{
  int octant = packed & 7;
  int qx = (packed >> 3) & 127;
  int qy = (packed >> (3+7)) & 63;
  
  x = qx / 127.0f;
  y = qy / 63.0f;
  z = 1.0f - x - y;
  float invLen = 1.0f / sqrtf(x*x + y*y + z*z);
  x *= invLen;
  y *= invLen;
  z *= invLen;

  if (octant & 1) x = -x;
  if (octant & 2) y = -y;
  if (octant & 4) z = -z;
}

// color - 16bit, normal - 16bit
typedef uint VoxData;

inline VoxData PackVoxData(uchar4 c, VoxNormal n)
{
  ushort c16 = 0;
  c16 |= (c.x>>3);          // red
  c16 |= (c.y>>2) << 5;     // green
  c16 |= (c.z>>3) << (5+6); // blue
  return c16 | (n<<16);
}

inline __device__ __host__  void UnpackVoxData(VoxData vd, uchar4 & c, VoxNormal & n)
{
  n = vd>>16;
  c.x = (vd << 3) & 0xf8;
  c.y = (vd >> (5-2)) & 0xfc;
  c.z = (vd >> (5+6-3)) & 0xf8;
}

typedef int VoxNodeId;

const VoxNodeId EmptyNode = -1;
const VoxNodeId FullNode  = -2;

union VoxNodeInfo
{
  uint pad;
  struct
  {
  uchar selfChildId:3;
  bool  emptyFlag:1;
  uchar leafFlags:8;
  };
};

typedef int VoxChild; // VoxData or VoxNodeId

struct VoxNode
{
  VoxNodeInfo flags;
  VoxNodeId   parent;
  VoxData     data;
  VoxChild   child[8];
};

inline __device__ __host__ bool GetLeafFlag(const VoxNode & node, int i)
{
  return (node.flags.leafFlags & (1<<i)) != 0;
}

inline __device__ __host__ void SetLeafFlag(VoxNode & node, int i, bool leaf)
{
  uchar mask = 1 << i;
  node.flags.leafFlags &= ~mask;
  if (leaf)
    node.flags.leafFlags |= mask;
}


struct VoxStructTree
{
  VoxNodeId     root;
  VoxNode     * nodes;
};

struct RenderParams
{
  float3 eye;
  float3 dir;
  float3 right;
  float3 up;

  float3 lightPos;
};

struct RayData
{
  float3 dir;
  float t;
  VoxNodeId endNode;
  int endNodeChild;
  float endNodeSize;
};

struct TraceParams
{
  float3 start;
  
  float detailCoef;
  VoxNodeId startNode;
  float3 startNodePos;
  float  startNodeSize;
  int startNodeLevel;
};

#pragma pack(pop)

