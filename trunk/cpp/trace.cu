#include "stdafx.h"

#include "trace_cu.h"
#include "trace_utils.h"

#define USE_TEXLOOKUP

#define INIT_THREAD \
  int xi = blockIdx.x * blockDim.x + threadIdx.x; \
  int yi = blockIdx.y * blockDim.y + threadIdx.y; \
  int sx = blockDim.x * gridDim.x;                \
  int sy = blockDim.y * gridDim.y;                \
  int tid = yi*sx + xi;        \

__constant__ VoxStructTree tree;

texture<uint, 1, cudaReadModeElementType> nodes_tex;


#define GET_FIELD( id, fld ) ( tree.nodes[id].fld )

#ifdef USE_TEXLOOKUP
  #define NODE_SZ (sizeof(VoxNode)/4)
  #define GET_TEXNODE_FIELD( p, fld ) ( tex1Dfetch(nodes_tex, (p)+(fld)) )

  typedef uint NodePtr;
  __device__ NodePtr GetNodePtr(VoxNodeId id) { return id*NODE_SZ;  }
  __device__ VoxNodeId Ptr2Id(NodePtr p) { return p/NODE_SZ; }

  __device__ VoxNodeInfo GetNodeInfo(NodePtr p) { return GET_TEXNODE_FIELD(p, 0); }
  __device__ VoxNodeId   GetParent  (NodePtr p) { return GET_TEXNODE_FIELD(p, 1); }
  __device__ VoxChild    GetChild   (NodePtr p, int chId) { return GET_TEXNODE_FIELD(p, 3 + chId); }
#else
  typedef const VoxNode * NodePtr;
  __device__ NodePtr GetNodePtr(VoxNodeId id) { return tree.nodes + id;  }
  __device__ VoxNodeId Ptr2Id(NodePtr p) { return p - tree.nodes; }

  __device__ const VoxNodeInfo & GetNodeInfo(NodePtr p) { return p->flags; }
  __device__ const VoxNodeId   & GetParent  (NodePtr p) { return p->parent; }
  __device__ const VoxChild    & GetChild   (NodePtr p, int chId) { return p->child[chId]; }
#endif

__device__ VoxData GetVoxData  (VoxNodeId id) { return tree.nodes[id].data; }


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
  rays[tid].endNodeChild = EmptyNode;
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
  rays[tid].endNodeChild = EmptyNode;
}

__global__ void Trace(RenderParams rp, RayData * rays)
{
  INIT_THREAD

  if (IsNull(rays[tid].endNode))
    return;

  point_3f dir = rays[tid].dir;
  AdjustDir(dir);

  point_3f t1, t2;
  uint dirFlags = 0;
  if (!SetupTrace(rp.eye, dir, t1, t2, dirFlags))
    return;

  NodePtr nodePtr = GetNodePtr(tree.root);
  int childId = 0;
  int level = 0;
  float nodeSize = pow(0.5f, level);

  enum States { ST_EXIT, ST_ANALYSE, ST_SAVE, ST_GOUP, ST_GODOWN, ST_GONEXT };
  int state = ST_ANALYSE;
  while (state != ST_EXIT)
  {
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

        if (GetLeafFlag(GetNodeInfo(nodePtr), childId^dirFlags)) { state = ST_SAVE; break; }
        
        VoxNodeId ch = GetChild(nodePtr, childId^dirFlags);
        if (IsNull(ch)) {state = ST_GONEXT; break; }
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
}

__global__ void ShadeSimple(RenderParams rp, const RayData * eyeRays, const RayData * shadowRays, uchar4 * img)
{
  INIT_THREAD

  VoxNodeId node = eyeRays[tid].endNode;
  if (IsNull(node))
  {
    img[tid] = make_uchar4(0, node == EmptyNode ? 0 : 64, 0, 255);
    return;
  }

  float3 p = rp.eye;                          
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
