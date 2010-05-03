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

  point_3f dir = rp.dir + 2*(xi-sx/2)*rp.right/sx + 2*(yi-sy/2)*rp.up/sy;
  dir = normalize(dir);
  AdjustDir(dir);

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

  point_3f dir = v.x*rp.right + v.y*rp.up + v.z*rp.dir;
  dir = normalize(dir);        
  AdjustDir(dir);

  rays[tid].dir.x = dir.x;
  rays[tid].dir.y = dir.y;
  rays[tid].dir.z = dir.z;

  rays[tid].endNode = 0;
  rays[tid].endNodeChild = EmptyNode;
}

__device__ int pos2ch(int3 pos)
{
  int childId;
  childId  = pos.x & 1;
  childId |= (pos.y & 1) << 1;
  childId |= (pos.z & 1) << 2;
  return childId;
}

__device__ float maxv(float3 p)
{
  return fmaxf( fmaxf(p.x, p.y), p.z );
}

__device__ float minv(float3 p)
{
  return fminf( fminf(p.x, p.y), p.z );
}

/*inline __device__ float3 operator+(const float3 & p, float a)
{
  return p + make_float3(a);
}*/
__device__ int3 operator+(const int3 & p, int a)
{
  return p + make_int3(a);
}

__device__ int FindFirstChild2(float t_enter, float3 t_coef, float3 t_bias, int3 & pos, float nodeSize)
{
  float3 corner = make_float3(pos) + 0.5;
  corner *= nodeSize;
  float3 t_center = corner * t_coef + t_bias;
  int ch = 0;
  pos.x <<= 1;
  pos.y <<= 1;
  pos.z <<= 1;
  if (t_center.x < t_enter) ch |= 1, ++pos.x;
  if (t_center.y < t_enter) ch |= 2, ++pos.y;
  if (t_center.z < t_enter) ch |= 4, ++pos.z;
  return ch;
}

__global__ void Trace(RayData * rays)
{
  INIT_THREAD;

  RayData & ray = rays[tid];

  if (IsNull(ray.endNode))
    return;
  ray.endNode = EmptyNode;
  ray.perfCount = 0;

  point_3f dir = ray.dir;

  float3 t_coef, t_bias;
  t_coef.x = 1.0f / fabs(dir.x);
  t_coef.y = 1.0f / fabs(dir.y);
  t_coef.z = 1.0f / fabs(dir.z);
  t_bias.x = -t_coef.x * ray.pos.x;
  t_bias.y = -t_coef.y * ray.pos.y;
  t_bias.z = -t_coef.z * ray.pos.z;
  uint octant_mask = 0;
  if (dir.x < 0.0f) octant_mask |= 1, t_bias.x = -t_coef.x - t_bias.x;
  if (dir.y < 0.0f) octant_mask |= 2, t_bias.y = -t_coef.y - t_bias.y;
  if (dir.z < 0.0f) octant_mask |= 4, t_bias.z = -t_coef.z - t_bias.z;

  float t_enter = maxv(t_bias);
  float t_exit  = minv(t_coef + t_bias);
  if (t_exit < 0.0f || t_enter > t_exit)
     return;

  NodePtr nodePtr = GetNodePtr(tree.root);
  VoxNodeInfo nodeInfo = GetNodeInfo(nodePtr);
  int childId = 0;
  int3 pos = make_int3(0);
  childId = FindFirstChild2(t_enter, t_coef, t_bias, pos, 1.0f);
  float nodeSize = 0.5f;

  enum Action { ACT_UNKNOWN, ACT_SAVE, ACT_DOWN, ACT_NEXT };
  enum Condition {
    CND_HIT_EMPTY = 1,
    CND_HIT_LEAF  = 2,
    CND_EARLY     = 4,
  };
  while (true)
  {
    float3 dt = nodeSize * t_coef;
    float3 t = make_float3(pos) * dt + t_bias;
    t_enter = maxv(t);
    t += dt;
    t_exit  = minv(t);
    int exitPlane = argmin(t);

    int realChildId = childId^octant_mask;
    uint cond = 0;
    if (GetNullFlag(nodeInfo, realChildId))
      cond |= CND_HIT_EMPTY;
    if (GetLeafFlag(nodeInfo, realChildId))
      cond |= CND_HIT_LEAF;
    if (t_exit < 0.0f) 
      cond |= CND_EARLY;

    int action = ACT_UNKNOWN;
    if (cond & (CND_HIT_EMPTY | CND_EARLY))
      action = ACT_NEXT;
    if (cond == CND_HIT_LEAF)
      action = ACT_SAVE;
    if (action == ACT_UNKNOWN)
      action = ACT_DOWN;

    if (action == ACT_SAVE)
    {
      ray.endNode = Ptr2Id(nodePtr);
      ray.endNodeChild = realChildId;
      ray.t = t_enter;
      ray.endNodeSize = nodeSize;
      break;
    }

    if (action == ACT_DOWN)
    {
      VoxNodeId ch = GetChild(nodePtr, realChildId);
      nodePtr = GetNodePtr(ch);
      childId = FindFirstChild2(t_enter, t_coef, t_bias, pos, nodeSize);
      nodeSize *= 0.5f;
      nodeInfo = GetNodeInfo(nodePtr);
      continue;
    }

    // GO NEXT
    uint exitMask = 1 << exitPlane;
    while (childId & exitMask)
    {
      // GO UP
      VoxNodeId p = GetParent(nodePtr);
      
      if (IsNull(p))
      {
        return;
      }
      pos.x >>= 1;
      pos.y >>= 1;
      pos.z >>= 1;
      childId = pos2ch(pos);
      nodeSize *= 2.0f;
      nodePtr = GetNodePtr(p);
      nodeInfo = GetNodeInfo(nodePtr);
    }
    childId |= exitMask;
    if (exitMask == 1) ++pos.x;
    if (exitMask == 2) ++pos.y;
    if (exitMask == 4) ++pos.z;
  }


  /*enum States { ST_EXIT, ST_ANALYSE, ST_SAVE, ST_GOUP, ST_GODOWN, ST_GONEXT };
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
        int realChildId = childId^octant_mask;
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
        state = GoNext(childId, t1, t2, pos) ? ST_GODOWN : ST_GOUP;
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
        childId = GetSelfChildId(GetNodeInfo(nodePtr))^octant_mask;
        nodePtr = GetNodePtr(p);
        --level;
        nodeSize *= 2;
        state = ST_GONEXT;
        break;
      }

      case ST_SAVE:
      {
        rays[tid].endNode = Ptr2Id(nodePtr);
        rays[tid].endNodeChild = childId^octant_mask;
        rays[tid].t = maxCoord(t1);
        rays[tid].endNodeSize = nodeSize;
        state = ST_EXIT;
        break;
      }
    }
  }*/
  //rays[tid].perfCount = count;
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
    res = point_3f(128, 128, 255) * (1.0f - h);
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