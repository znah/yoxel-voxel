#include "stdafx.h"

#include "trace_cu.h"
#include "trace_utils.h"

#define INIT_THREAD \
  int xi = blockIdx.x * blockDim.x + threadIdx.x; \
  int yi = blockIdx.y * blockDim.y + threadIdx.y; \
  int sx = rp.viewWidth;                          \
  int sy = rp.viewHeight;                         \
  if (xi >= sx || yi >= sy ) return; \
  int tid = yi*sx + xi;        \

__constant__ VoxStructTree tree;
__constant__ RenderParams rp;

__constant__ int c_popc8LUT[256] =
{
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
};

texture<uint, 1, cudaReadModeElementType> nodes_tex;


#define GET_FIELD( id, fld ) ( tree.nodes[id].fld )

#if USE_TEXLOOKUP
  #define NODE_SZ (sizeof(VoxNode)/4)
  #define GET_TEXNODE_FIELD( p, fld ) ( tex1Dfetch(nodes_tex, (p)+(fld)) )

  typedef uint NodePtr;
  __constant__ NodePtr InvalidPtr = 0xffffffff;

  __device__ NodePtr GetNodePtr(VoxNodeId id) { return id*NODE_SZ;  }
  __device__ VoxNodeId Ptr2Id(NodePtr p) { return p/NODE_SZ; }

  __device__ VoxNodeInfo GetNodeInfo(NodePtr p) { return GET_TEXNODE_FIELD(p, 0); }
  __device__ VoxChild    GetChild   (NodePtr p, int chId) { return GET_TEXNODE_FIELD(p, 2 + chId); }
#else
  typedef const VoxNode * NodePtr;
  __constant__ NodePtr InvalidPtr = NULL;

  __device__ NodePtr GetNodePtr(VoxNodeId id) { return tree.nodes + id;  }
  __device__ VoxNodeId Ptr2Id(NodePtr p) { return p - tree.nodes; }

  __device__ const VoxNodeInfo & GetNodeInfo(NodePtr p) { return p->flags; }
  __device__ const VoxChild    & GetChild   (NodePtr p, int chId) { return p->child[chId]; }
#endif

__device__ VoxData GetVoxData  (VoxNodeId id) { return tree.nodes[id].data; }


extern "C" {


__global__ void InitEyeRays(RayData * rays, const float * noiseBuf, const int * shuffleBuf)
{
  INIT_THREAD

  point_3f dir = rp.dir + 2*(xi-sx/2)*rp.right/sx + 2*(yi-sy/2)*rp.up/sy;
  dir = normalize(dir);
  AdjustDir(dir);

  int noiseBase = (tid*3 + rp.rndSeed) % (3*sx*sy-3);
  point_3f noiseShift = point_3f(noiseBuf[noiseBase], noiseBuf[noiseBase+1], noiseBuf[noiseBase+2]) * rp.ditherCoef;

  int idx = tid;
  if (shuffleBuf != NULL)
    idx = shuffleBuf[tid];

  rays[idx].pos = rp.eyePos + noiseShift;
  rays[idx].dir = dir;

  rays[idx].endNode = 0;
  rays[idx].endNodeChild = EmptyNode;
  rays[idx].unshuffleIndex = tid;
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

__device__ int3 operator+(const int3 & p, int a)
{
  return p + make_int3(a);
}

__device__ int popc8(uint mask)
{
    return c_popc8LUT[mask];
}

__device__ int popc16(uint mask)
{
    return c_popc8LUT[mask & 0xffu] + c_popc8LUT[mask >> 8];
}

__device__ int popc32(uint mask)
{
    int result = c_popc8LUT[mask & 0xffu];
    result += c_popc8LUT[(mask >> 8) & 0xffu];
    result += c_popc8LUT[(mask >> 16) & 0xffu];
    result += c_popc8LUT[mask >> 24];
    return result;
}


__device__ void FindFirstChild2(float t_enter, float3 t_coef, float3 t_bias, int3 & pos, float nodeSize)
{
  float3 corner = make_float3(pos) + 0.5;
  corner *= nodeSize;
  float3 t_center = corner * t_coef + t_bias;
  pos.x <<= 1;
  pos.y <<= 1;
  pos.z <<= 1;
  if (t_center.x < t_enter) ++pos.x;
  if (t_center.y < t_enter) ++pos.y;
  if (t_center.z < t_enter) ++pos.z;
}

const int STACK_DEPTH = 16;
const int BLOCK_THREADNUM = 128;

__shared__ NodePtr s_stack[STACK_DEPTH * BLOCK_THREADNUM];

struct Stack
{
  int level;
  int sbase;
  NodePtr stack[STACK_DEPTH];

  __device__ Stack() : level(0), sbase(threadIdx.x + threadIdx.y * blockDim.x) {}

  __device__ void push(NodePtr p) 
  {
#if SHARED_STACK
    s_stack[sbase + (level++)*BLOCK_THREADNUM] = p;
#else
    stack[level++] = p;
#endif
  }

  __device__ bool trypop(int upcount) { return level >= upcount; }

  __device__ NodePtr pop(int upcount) 
  {
    level -= upcount;
#if SHARED_STACK
    return s_stack[sbase + level*BLOCK_THREADNUM];
#else
    return stack[level];
#endif
  }
};

__global__ void Trace(RayData * rays)
{
  INIT_THREAD;

  RayData & ray = rays[tid];

  if (IsNull(ray.endNode))
    return;
  ray.endNode = EmptyNode;
  ray.perfCount = 0;

  float3 t_coef, t_bias;
  uint octant_mask = 0;

  {
    float3 dir = ray.dir;
    t_coef.x = 1.0f / fabs(dir.x);
    t_coef.y = 1.0f / fabs(dir.y);
    t_coef.z = 1.0f / fabs(dir.z);
    t_bias.x = -t_coef.x * ray.pos.x;
    t_bias.y = -t_coef.y * ray.pos.y;
    t_bias.z = -t_coef.z * ray.pos.z;
    
    if (dir.x < 0.0f) octant_mask |= 1, t_bias.x = -t_coef.x - t_bias.x;
    if (dir.y < 0.0f) octant_mask |= 2, t_bias.y = -t_coef.y - t_bias.y;
    if (dir.z < 0.0f) octant_mask |= 4, t_bias.z = -t_coef.z - t_bias.z;
  }

  float t_enter = maxv(t_bias);
  float t_exit  = minv(t_coef + t_bias);
  if (t_exit < 0.0f || t_enter > t_exit)
     return;

  NodePtr nodePtr = GetNodePtr(tree.root);
  VoxNodeInfo nodeInfo = GetNodeInfo(nodePtr);
  int3 pos = make_int3(0);
  FindFirstChild2(t_enter, t_coef, t_bias, pos, 1.0f);
  float nodeSize = 0.5f;
  int count = 0;

  Stack stack;

  enum Action { ACT_UNKNOWN, ACT_SAVE, ACT_DOWN, ACT_NEXT };
  enum Condition {
    CND_HIT_EMPTY = 1,
    CND_HIT_LEAF  = 2,
    CND_EARLY     = 4,
    CND_LOD       = 8,
    CND_LOD_EMPTY = 16
  };
  while (true)
  {
    ++count;
    float3 t2 = make_float3(pos + 1) * nodeSize * t_coef + t_bias;
    t_exit = minv(t2);

    int realChildId = pos2ch(pos)^octant_mask;

    int action = ACT_DOWN;
    if (t_exit < 0.0f) 
      action = ACT_NEXT;
    else
    {
      if (/*stack.level > 8 ||*/ t_enter * rp.detailCoef > 2.0f * nodeSize) // LOD ?
      {
        if (GetEmptyFlag(nodeInfo))
          action = ACT_NEXT;
        else
        {
          realChildId = -1;
          action = ACT_SAVE;
        }
      }
      else
      {
        if (GetNullFlag(nodeInfo, realChildId))
          action = ACT_NEXT;
        else if (GetLeafFlag(nodeInfo, realChildId))
          action = ACT_SAVE;
      }
    }

    if (action == ACT_SAVE)
    {
      ray.endNode = Ptr2Id(nodePtr);
      ray.endNodeChild = realChildId;
      ray.t = t_enter;
      ray.endNodeSize = nodeSize;
      ray.perfCount = count;
      break;
    }

    if (action == ACT_DOWN)
    {
      stack.push(nodePtr);
      VoxNodeId ch = GetChild(nodePtr, realChildId);
      nodePtr = GetNodePtr(ch);
      FindFirstChild2(t_enter, t_coef, t_bias, pos, nodeSize);
      nodeSize *= 0.5f;
      nodeInfo = GetNodeInfo(nodePtr);
      continue;
    }

    // GO NEXT
    int diff;  
    if (t_exit == t2.x) {diff = pos.x; ++pos.x;}
    if (t_exit == t2.y) {diff = pos.y; ++pos.y;}
    if (t_exit == t2.z) {diff = pos.z; ++pos.z;}
    diff = (diff^(diff+1))>>1;
    if (diff != 0)
    {
      int upcount = popc16(diff);
      if (!stack.trypop(upcount))
      {
        ray.perfCount = count;
        return;
      }
      nodePtr = stack.pop(upcount);
      nodeInfo = GetNodeInfo(nodePtr);
      pos.x >>= upcount;
      pos.y >>= upcount;
      pos.z >>= upcount;
      nodeSize *= 1<<upcount;
    }
    t_enter = t_exit;
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
  int idx = eyeRays[tid].unshuffleIndex;
  img[idx] = make_uchar4(count, count, count, 255);
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

  int idx = eyeRays[tid].unshuffleIndex;

  ushort4 accRes = make_ushort4(res.x, res.y, res.z, 0);
  if (rp.accumIter > 0)
  {
    ushort4 prev = accum[idx];
    accRes.x += prev.x;
    accRes.y += prev.y;
    accRes.z += prev.z;
  }
  if (rp.accumIter < 255)
    accum[idx] = accRes;
  float accumCoef = 1.0f / (rp.accumIter+1);
  img[idx] = make_uchar4(accRes.x * accumCoef, accRes.y * accumCoef, accRes.z * accumCoef, 255);
}

void Run_InitEyeRays(GridShape grid, RayData * rays, float * noiseBuf, const int * shuffleBuf)
{
  InitEyeRays<<<grid.grid, grid.block>>>(rays, noiseBuf, shuffleBuf);
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