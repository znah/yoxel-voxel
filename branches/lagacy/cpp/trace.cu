#include "stdafx.h"

#include "trace_cu.h"
#include "trace_utils.h"

#define USE_TEXLOOKUP

#define INIT_THREAD \
  int xi = blockIdx.x * blockDim.x + threadIdx.x; \
  int yi = blockIdx.y * blockDim.y + threadIdx.y; \
  int sx = rp.viewWidth;                          \
  int sy = rp.viewHeight;                         \
  if (xi >= sx || yi >= sy ) return; \
  int tid = yi*sx + xi;        \

__constant__ VoxStructTree tree;
__constant__ RenderParams rp;

texture<uint, 1, cudaReadModeElementType> nodes_tex;


#define GET_FIELD( id, fld ) ( tree.nodes[id].fld )

#ifdef USE_TEXLOOKUP
  #define NODE_SZ (sizeof(VoxNode)/4)
  #define GET_TEXNODE_FIELD( p, fld ) ( tex1Dfetch(nodes_tex, (p)+(fld)) )

  typedef uint NodePtr;
  __constant__ NodePtr InvalidPtr = 0xffffffff;

  __device__ NodePtr GetNodePtr(VoxNodeId id) { return id*NODE_SZ;  }
  __device__ VoxNodeId Ptr2Id(NodePtr p) { return p/NODE_SZ; }

  __device__ VoxNodeInfo GetNodeInfo(NodePtr p) { return GET_TEXNODE_FIELD(p, 0); }
  __device__ VoxNodeId   GetParent  (NodePtr p) { return GET_TEXNODE_FIELD(p, 1); }
  __device__ VoxChild    GetChild   (NodePtr p, int chId) { return GET_TEXNODE_FIELD(p, 3 + chId); }
#else
  typedef const VoxNode * NodePtr;
  __constant__ NodePtr InvalidPtr = NULL;

  __device__ NodePtr GetNodePtr(VoxNodeId id) { return tree.nodes + id;  }
  __device__ VoxNodeId Ptr2Id(NodePtr p) { return p - tree.nodes; }

  __device__ const VoxNodeInfo & GetNodeInfo(NodePtr p) { return p->flags; }
  __device__ const VoxNodeId   & GetParent  (NodePtr p) { return p->parent; }
  __device__ const VoxChild    & GetChild   (NodePtr p, int chId) { return p->child[chId]; }
#endif

__device__ VoxData GetVoxData  (VoxNodeId id) { return tree.nodes[id].data; }


extern "C" {


__global__ void InitEyeRays(RayData * rays, float * noiseBuf)
{
  INIT_THREAD

  float3 dir = rp.dir + 2*(xi-sx/2)*rp.right/sx + 2*(yi-sy/2)*rp.up/sy;
  dir = normalize(dir);

  int noiseBase = (tid*3 + rp.rndSeed) % (3*sx*sy-3);
  point_3f noiseShift = point_3f(noiseBuf[noiseBase], noiseBuf[noiseBase+1], noiseBuf[noiseBase+2]) * rp.ditherCoef;
  rays[tid].pos = rp.eyePos + noiseShift;
  rays[tid].dir = dir;

  rays[tid].endNode = 0;
  rays[tid].endNodeChild = EmptyNode;
}


__global__ void InitFishEyeRays(RayData * rays)
{
  INIT_THREAD

  
  float2 p = make_float2(xi-sx/2, yi-sy/2);
  p /= 0.5f*sx;

  const float pi = 3.141593f;
  p *= 0.8;

  float r = length(p);
  float phi = atan2(p.y, p.x);
  float theta = pi/2-asin(r);
  
  float ct = __cosf(theta);
  float3 v = make_float3(__cosf(phi)*ct, __sinf(phi)*ct, __sinf(theta));

  float3 dir = v.x*rp.right + v.y*rp.up + v.z*rp.dir;
  dir = normalize(dir);        

  rays[tid].dir.x = dir.x;
  rays[tid].dir.y = dir.y;
  rays[tid].dir.z = dir.z;

  rays[tid].endNode = 0;
  rays[tid].endNodeChild = EmptyNode;
}

__global__ void Trace(RayData * rays)
{
  INIT_THREAD

  if (IsNull(rays[tid].endNode))
    return;

  point_3f dir = rays[tid].dir;
  AdjustDir(dir);

  point_3f t1, t2;
  uint dirFlags = 0;
  if (!SetupTrace(rays[tid].pos, dir, t1, t2, dirFlags)) //rp.eyePos
  {
    rays[tid].endNode = EmptyNode;
    return;
  }

  NodePtr nodePtr = GetNodePtr(tree.root);
  int childId = 0;
  int level = 0;
  float nodeSize = pow(0.5f, level);
  int count = 0;

  enum States { ST_EXIT, ST_ANALYSE, ST_SAVE, ST_GOUP, ST_GODOWN, ST_GONEXT };
  int state = ST_ANALYSE;
  while (state != ST_EXIT)
  {
    ++count;
    switch (state)
    {
      case ST_ANALYSE:
      {
        childId = -1;
        if (maxCoord(t1) * rp.detailCoef > nodeSize/2)  { state = GetEmptyFlag(GetNodeInfo(nodePtr)) ? ST_GOUP : ST_SAVE; break; }
        
        childId = FindFirstChild(t1, t2);
        state = ST_GODOWN;
        break;
      }
      
      case ST_GODOWN:
      {
        if (minCoord(t2) < 0) { state = ST_GONEXT; break; }

        VoxNodeInfo nodeInfo = GetNodeInfo(nodePtr);
        int realChildId = childId^dirFlags;
        if (GetLeafFlag(nodeInfo, realChildId)) { state = ST_SAVE; break; }
        
        // no performance gain for some reason
        // if (GetNullFlag(nodeInfo, realChildId)) { state = ST_GONEXT; break; } 
        
        VoxNodeId ch = GetChild(nodePtr, realChildId);
        if (IsNull(ch)) { state = ST_GONEXT; break; }
        nodePtr = GetNodePtr(ch);
        ++level;
        nodeSize /= 2;
        state = ST_ANALYSE;
        break;
      }
      
      case ST_GONEXT:
      {
        state = GoNext(childId, t1, t2) ? ST_GODOWN : ST_GOUP;
        break;
      }

      case ST_GOUP:
      {
        VoxNodeId p = GetParent(nodePtr);
        if (IsNull(p)) 
        { 
          rays[tid].endNode = EmptyNode;
          state = ST_EXIT; 
          break; 
        }

        for (int i = 0; i < 3; ++i)
        {
          int mask = 1<<i;
          float dt = t2[i] - t1[i];
          ((childId & mask) == 0) ? t2[i] += dt : t1[i] -= dt;
        }
        childId = GetSelfChildId(GetNodeInfo(nodePtr))^dirFlags;
        nodePtr = GetNodePtr(p);
        --level;
        nodeSize *= 2;
        state = ST_GONEXT;
        break;
      }

      case ST_SAVE:
      {
        rays[tid].endNode = Ptr2Id(nodePtr);
        rays[tid].endNodeChild = childId^dirFlags;
        rays[tid].t = maxCoord(t1);
        rays[tid].endNodeSize = nodeSize;
        state = ST_EXIT;
        break;
      }
    }
  }
  rays[tid].perfCount = count;
}

__device__ point_3f CalcLighting(point_3f pos, point_3f normal, point_3f color)
{
  point_3f accum = rp.ambient * color;
  for (int i = 0; i < MaxLightsNum; ++i)
  {
    if (!rp.lights[i].enabled)
      continue;

    point_3f lightDir = rp.lights[i].pos - pos;
    float lightDist2 = dot(lightDir, lightDir);
    float lightDist = sqrtf(lightDist2);
    float attenuation = 1.0f / dot(point_3f(1.0f, lightDist, lightDist2), rp.lights[i].attenuationCoefs);
    lightDir /= lightDist;

    point_3f diffuse = rp.lights[i].diffuse * color * max(dot(lightDir, normal), 0.0f);
    
    point_3f viewerDir = normalize(rp.eyePos - pos);
    point_3f hv = normalize(viewerDir + lightDir);
    point_3f specular = rp.lights[i].specular * pow(max(0.0f, dot(hv, normal)), rp.specularExp);

    accum += (diffuse + specular) * attenuation;
  }
  return accum;
}

__global__ void ShadeCounter(const RayData * eyeRays, uchar4 * img)
{
  INIT_THREAD;

  int count = eyeRays[tid].perfCount;
  count = min(count, 255);
  img[tid] = make_uchar4(count, count, count, 255);
  return;
}

__global__ void ShadeSimple(const RayData * eyeRays, uchar4 * img, ushort4 * accum)
{
  INIT_THREAD;

  VoxNodeId node = eyeRays[tid].endNode;
  
  point_3f res(0, 0, 0);
  if (IsNull(node))
  {
    float h = (float)yi / sy;
    res = point_3f(128, 128, 255) * (1.0 - h);
    if (node != EmptyNode)
      res = point_3f(1, 0, 0);
  }
  else
  {
    float3 p = rp.eyePos;                          
    float3 dir = eyeRays[tid].dir;
    float t = eyeRays[tid].t;

    VoxData vd;
    int childId = eyeRays[tid].endNodeChild;
    if (childId < 0)
      vd = GetVoxData(node);
    else
      vd = GetChild(GetNodePtr(node), childId);

    Color16  c16;
    Normal16 n16;
    UnpackVoxData(vd, c16, n16);
    uchar4 col;
    float3 norm;
    col = UnpackColorCU(c16);
    UnpackNormal(n16, norm.x, norm.y, norm.z);

    float3 pt = p + dir*t;
    point_3f materialColor = point_3f(col.x, col.y, col.z) / 256.0f;
    res = fminf(CalcLighting(pt, norm, materialColor) * 256.0f, point_3f(255, 255, 255));
  }

  ushort4 accRes = make_ushort4(res.x, res.y, res.z, 0);
  if (rp.accumIter > 0)
  {
    ushort4 prev = accum[tid];
    accRes.x += prev.x;
    accRes.y += prev.y;
    accRes.z += prev.z;
  }
  if (rp.accumIter < 255)
    accum[tid] = accRes;
  float accumCoef = 1.0f / (rp.accumIter+1);
  img[tid] = make_uchar4(accRes.x * accumCoef, accRes.y * accumCoef, accRes.z * accumCoef, 255);
}

void Run_InitEyeRays(GridShape grid, RayData * rays, float * noiseBuf)
{
  InitEyeRays<<<grid.grid, grid.block>>>(rays, noiseBuf);
}

void Run_Trace(GridShape grid, RayData * rays)
{
  Trace<<<grid.grid, grid.block>>>(rays);
}

void Run_ShadeSimple(GridShape grid, const RayData * eyeRays, uchar4 * img, ushort4 * accum)
{
  ShadeSimple<<<grid.grid, grid.block>>>(eyeRays, img, accum);
}

void Run_ShadeCounter(GridShape grid, const RayData * eyeRays, uchar4 * img)
{
  ShadeCounter<<<grid.grid, grid.block>>>(eyeRays, img);
}


}