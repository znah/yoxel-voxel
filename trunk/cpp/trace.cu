#include "trace_cu.h"
#include "cutil_math.h"

//#define USE_TEXLOOKUP

#define INIT_THREAD \
  int xi = blockIdx.x * blockDim.x + threadIdx.x; \
  int yi = blockIdx.y * blockDim.y + threadIdx.y; \
  int sx = blockDim.x * gridDim.x;                \
  int sy = blockDim.y * gridDim.y;                \
  int tid = yi*sx + xi;        \

__constant__ VoxStructTree tree;

texture<uint, 1, cudaReadModeElementType> flags_tex;
texture<VoxNodeId, 1, cudaReadModeElementType> parents_tex;
texture<VoxNodeId, 1, cudaReadModeElementType> children_tex;
texture<uchar4, 1, cudaReadModeElementType> colors_tex;
texture<char4, 1, cudaReadModeElementType> normals_tex;

texture<uint, 1, cudaReadModeElementType> nodes_tex;
texture<uint, 1, cudaReadModeElementType> leafs_tex;


#define GET_FIELD( id, fld ) ( ((id)&VOX_LEAF) ? tree.leafs[IDX(id)].fld : tree.nodes[IDX(id)].fld )

#ifdef USE_TEXLOOKUP
  #define NODE_SZ (sizeof(VoxNode)/4)
  #define LEAF_SZ (sizeof(VoxLeaf)/4)
  #define GET_TEXNODE_FIELD( id, fld ) ( tex1Dfetch(nodes_tex, IDX(id)*NODE_SZ+(fld)) )

  __device__ uint GetSelfChildId(VoxNodeId id) { return GET_TEXNODE_FIELD( id, 0 ) & 7; }
  __device__ bool GetEmptyFlag(VoxNodeId id) { return ((GET_TEXNODE_FIELD( id, 0 )>>3) & 0x1) != 0; }

  __device__ VoxNodeId   GetParent (VoxNodeId id) { return GET_TEXNODE_FIELD(id, 1); }
  __device__ VoxNodeId   GetChild  (VoxNodeId id, int chId) { return GET_TEXNODE_FIELD(id, 4 + chId); }

#else

  __device__ uint GetSelfChildId(VoxNodeId id) { return GET_FIELD(id, flags.selfChildId); }
  __device__ bool GetEmptyFlag(VoxNodeId id) { return GET_FIELD(id, flags.emptyFlag); }

  __device__ VoxNodeId   & GetParent (VoxNodeId id) { return GET_FIELD(id, parent); }
  __device__ VoxNodeId   & GetChild  (VoxNodeId id, int chId) { return tree.nodes[IDX(id)].child[chId]; }
#endif

__device__ uchar4 GetColor  (VoxNodeId id) { return GET_FIELD(id, color); }
__device__ char4 GetNormal  (VoxNodeId id) { return GET_FIELD(id, normal); }


__device__ bool IsLeaf(VoxNodeId id) { return (id & VOX_LEAF) != 0; }


struct point_3f : public float3
{
  __host__ __device__ float & operator[](int i) {return *(&x+i); }
  __host__ __device__ point_3f(const float3 & p) { x = p.x; y = p.y; z = p.z; }
  __host__ __device__ point_3f() { x = 0; y = 0; z = 0; }

}; 

__host__ __device__ float max(const float3 & p) { return fmaxf(p.x, fmaxf(p.y, p.z)); }
__host__ __device__ float min(const float3 & p) { return fminf(p.x, fminf(p.y, p.z)); }

__host__ __device__ int argmin(const float3 & p) 
{
  if (p.x > p.y)
    return (p.y < p.z) ? 1 : 2;
  else
    return (p.x < p.z) ? 0 : 2;
}

__host__ __device__ int argmax(const float3 & p) 
{
  if (p.x < p.y)
    return (p.y > p.z) ? 1 : 2;
  else
    return (p.x > p.z) ? 0 : 2;
}


template<class T>
__host__ __device__ void swap(T & a, T & b) { T c = a; a = b; b = c; }

__device__ uint FindFrstChild(point_3f & t1, point_3f & t2)
{
  uint childId = 0;
  point_3f tm = 0.5f * (t1 + t2);
  float tEnter = max(t1);
  for (int i = 0; i < 3; ++i)
  {
    if (tm[i] > tEnter)
      t2[i] = tm[i];
    else
    {
      t1[i] = tm[i];
      childId |= 1<<i;
    }
  }
  return childId;
}

template<int ExitPlane>
__device__ bool GoNextTempl(uint & childId, point_3f & t1, point_3f & t2)
{
  int mask = 1<<ExitPlane;
  if ((childId & mask) != 0)
    return false;

  childId ^= mask;

  float dt = t2[ExitPlane] - t1[ExitPlane];
  t1[ExitPlane] = t2[ExitPlane];
  t2[ExitPlane] += dt;
  return true;
}

__device__ bool GoNext(uint & childId, point_3f & t1, point_3f & t2)
{
  // argmin
  if (t2.x > t2.y)
    return (t2.y < t2.z) ? GoNextTempl<1>(childId, t1, t2) : GoNextTempl<2>(childId, t1, t2);
  else
    return (t2.x < t2.z) ? GoNextTempl<0>(childId, t1, t2) : GoNextTempl<2>(childId, t1, t2);
}

extern "C" {

__global__ void InitEyeRays(RenderParams rp, RayData * rays)
{
  INIT_THREAD

  float3 dir = rp.dir + 2*(xi-sx/2)*rp.right/sx + 2*(yi-sy/2)*rp.up/sx;
  dir = normalize(dir);

  rays[tid].dir.x = dir.x;
  rays[tid].dir.y = dir.y;
  rays[tid].dir.z = dir.z;

  rays[tid].endNode = 0;
}


__global__ void InitFishEyeRays(RenderParams rp, RayData * rays)
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
}

__global__ void InitShadowRays(RenderParams rp, const RayData * eyeRays, RayData * shadowRays)
{
  INIT_THREAD

  if (eyeRays[tid].endNode < 0)
  {
    shadowRays[tid].endNode = -1;
    return;
  }

  float3 endPos = rp.eye + eyeRays[tid].dir*eyeRays[tid].t;
  float3 dir = normalize(endPos - rp.lightPos);
  
  shadowRays[tid].dir = dir;
  shadowRays[tid].endNode = 0;
}


__global__ void Trace(TraceParams tp, RayData * rays)
{
  INIT_THREAD

  if (rays[tid].endNode < 0)
    return;

  const float eps = 1e-8f;
  point_3f dir = rays[tid].dir;
  for (int i = 0; i < 3; ++i)
    if (abs(dir[i]) < eps)
      dir[i] = (dir[i] < 0) ? -eps : eps;

  float3 p = tp.start;
  point_3f t1 = (tp.startNodePos - p) / dir;
  point_3f t2 = (tp.startNodePos + make_float3(tp.startNodeSize) - p) / dir;
  uint dirFlags = 0;
  for (int i = 0; i < 3; ++i)
    if (dir[i] < 0)
    {
      dirFlags |= 1<<i;
      swap(t1[i], t2[i]);
    }

  if (max(t1) >= min(t2))
  {
    rays[tid].endNode = -1;
    return;
  }

  VoxNodeId node = tp.startNode;
  uint childId = 0;
  int level = tp.startNodeLevel;
  float nodeSize = pow(0.5f, level);

  enum States { ST_EXIT, ST_ANALYSE, ST_SAVE, ST_GOUP, ST_GODOWN, ST_GONEXT };
  int state = ST_ANALYSE;
  while (state != ST_EXIT)
  {
    switch (state)
    {
      case ST_ANALYSE:
      {
        if (IsLeaf(node)) { state = ST_SAVE; break; }
        if (max(t1) * tp.detailCoef > nodeSize/2)  { state = GetEmptyFlag(node) ? ST_GOUP : ST_SAVE; break; }
        
        childId = FindFrstChild(t1, t2);
        state = ST_GODOWN;
        break;
      }
      
      case ST_GODOWN:
      {
        if (min(t2) < 0) { state = ST_GONEXT; break; }
        
        VoxNodeId ch = GetChild(node, childId^dirFlags);
        if (ch == EmptyNode) {state = ST_GONEXT; break; }
        node = ch;
        ++level;
        nodeSize /= 2;
        state = (node != FullNode) ? ST_ANALYSE : ST_SAVE;
        break;
      }
      
      case ST_GONEXT:
      {
        state = GoNext(childId, t1, t2) ? ST_GODOWN : ST_GOUP;
        break;
      }

      case ST_GOUP:
      {
        VoxNodeId p = GetParent(node);
        if (p < 0) { 
          rays[tid].endNode = -1;
          state = ST_EXIT; 
          break; 
        }

        for (int i = 0; i < 3; ++i)
        {
          int mask = 1<<i;
          float dt = t2[i] - t1[i];
          ((childId & mask) == 0) ? t2[i] += dt : t1[i] -= dt;
        }
        childId = GetSelfChildId(node)^dirFlags;
        node = p;
        --level;
        nodeSize *= 2;
        state = ST_GONEXT;
        break;
      }

      case ST_SAVE:
      {
        rays[tid].endNode = node;
        rays[tid].t = max(t1);
        rays[tid].endNodeSize = nodeSize;
        state = ST_EXIT;
        break;
      }
    }
  }
}

__global__ void ShadeShadow(RenderParams rp, const RayData * eyeRays, const RayData * shadowRays, uchar4 * img)
{
  INIT_THREAD

  VoxNodeId node = eyeRays[tid].endNode;
  if (node < 0)
  {
    img[tid] = make_uchar4(0, 0, 0, 255);
    return;
  }
  float3 p = rp.eye;                          
  float3 dir = eyeRays[tid].dir;
  float t = eyeRays[tid].t;

  uchar4 col = GetColor(node);
  char4 np = GetNormal(node);
  float3 norm = make_float3(np.x, np.y, np.z);
  norm /= 127.0;
  float3 pt = p + dir*t;
  float3 lightDir = rp.lightPos - pt;
  float lightDist = length(lightDir);
  lightDir /= lightDist;
  float fade = 1.0 / (lightDist*lightDist);

  if (shadowRays[tid].t < lightDist-shadowRays[tid].endNodeSize*3)
    fade = 0;

  float diff = 0.7 * max(dot(lightDir, norm), 0.0f);
  float amb = 0.3;
  float l = diff * fade + amb;

  float3 viewerDir = normalize(make_float3(0)-dir);
  float3 hv = normalize(viewerDir + lightDir);
  float spec = pow(max(0.0f, dot(hv, norm)), 10) * 50;
  spec *= fade;


  float3 res = make_float3(col.x*l + spec, col.y*l + spec, col.z*l + spec);
  res = fminf(res, make_float3(255));

  img[tid] = make_uchar4(res.x, res.y, res.z, 255);
}


__global__ void ShadeSimple(RenderParams rp, const RayData * eyeRays, const RayData * shadowRays, uchar4 * img)
{
  INIT_THREAD

  VoxNodeId node = eyeRays[tid].endNode;
  if (node < 0)
  {
    img[tid] = make_uchar4(0, node == EmptyNode ? 0 : 64, 0, 255);
    return;
  }


  float3 p = rp.eye;                          
  float3 dir = eyeRays[tid].dir;
  float t = eyeRays[tid].t;

  uchar4 col = GetColor(node);
  char4 np = GetNormal(node);
  float3 norm = make_float3(np.x, np.y, np.z);
  norm /= 127.0;
  float3 pt = p + dir*t;
  float3 lightDir = rp.lightPos - pt;
  float lightDist = length(lightDir);
  lightDir /= lightDist;
  float fade = 1.0;//0.3 / (lightDist*lightDist);

  float diff = 0.7 * max(dot(lightDir, norm), 0.0f);
  float amb = 0.3;
  float l = diff * fade + amb;

  float3 viewerDir = normalize(make_float3(0)-dir);
  float3 hv = normalize(viewerDir + lightDir);
  float spec = pow(max(0.0f, dot(hv, norm)), 10) * 50;
  spec *= fade;


  float3 res = make_float3(col.x*l + spec, col.y*l + spec, col.z*l + spec);
  res = fminf(res, make_float3(255));

  img[tid] = make_uchar4(res.x, res.y, res.z, 255);
}


}
