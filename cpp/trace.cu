#include "stdafx.h"

#include "trace_cu.h"
#include "trace_utils.h"

#define USE_TEXLOOKUP

#define INIT_THREAD \
  const int xi = blockIdx.x * blockDim.x + threadIdx.x; \
  const int yi = blockIdx.y * blockDim.y + threadIdx.y; \
  const int sx = rp.viewWidth;                          \
  const int sy = rp.viewHeight;                         \
  if (xi >= sx || yi >= sy ) return; \
  const int tid = yi*sx + xi;        \

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


__device__ float3 CalcRayDirView(int xi, int yi)
{
  const int sx = rp.viewWidth;
  const int sy = rp.viewHeight;
  return point_3f(2*rp.fovCoef*(float)(xi-sx/2)/sx, 2*rp.fovCoef*(float)(yi-sy/2)/sx, -1);
}

__device__ float3 CalcRayDirWorld(int xi, int yi)
{
  point_3f dir = CalcRayDirView(xi, yi);
  dir = rp.viewToWldMtx * dir;
  dir = dir - rp.eyePos;
  return dir;
}

/*__global__ void InitFishEyeRays(RayData * rays)
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
}*/

__global__ void Trace()
{
  INIT_THREAD

  rp.rays[tid].endNode = 0;
  rp.rays[tid].endNodeChild = EmptyNode;


  if (IsNull(rp.rays[tid].endNode))
    return;

  point_3f dir = CalcRayDirWorld(xi, yi);

  point_3f t1, t2;
  uint dirFlags = 0;
  if (!SetupTrace(rp.eyePos, dir, t1, t2, dirFlags)) //rp.eyePos
  {
    rp.rays[tid].endNode = EmptyNode;
    return;
  }

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
          rp.rays[tid].endNode = EmptyNode;
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
        rp.rays[tid].endNode = Ptr2Id(nodePtr);
        rp.rays[tid].endNodeChild = childId^dirFlags;
        rp.zBuf[tid] = maxCoord(t1);
        rp.rays[tid].endNodeSize = nodeSize;
        state = ST_EXIT;
        break;
      }
    }
  }
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

/*__device__ point_3f SampleWldPos(int xi, int yi)
{
  int tid = yi * rp.viewWidth + xi;
  return rp.eyePos + rp.rays[tid].t*rp.rays[tid].dir;
}

__device__ point_3f SampleNormal(int xi, int yi)
{
  //int tid = yi * rp.viewWidth + xi;
  int step = 1;
  //int step = max((int)(rp.rays[tid].endNodeSize / (rp.rays[tid].t*rp.detailCoef)), 1);
  
  if (xi < step || xi > rp.viewWidth-step-1)
    return make_float3(0, 0, 1);
  if (yi < step || yi > rp.viewHeight-step-1)
    return make_float3(0, 0, 1);

  point_3f u1 = SampleWldPos(xi-step, yi);
  point_3f u2 = SampleWldPos(xi+step, yi);
  point_3f du = u2-u1;
  point_3f v1 = SampleWldPos(xi, yi-step);
  point_3f v2 = SampleWldPos(xi, yi+step);
  point_3f dv = v2-v1;
  point_3f n = normalize(cross(du, dv));

  return n;
}*/

__global__ void ShadeSimple(uchar4 * img)
{
  INIT_THREAD

  VoxNodeId node = rp.rays[tid].endNode;
  if (IsNull(node))
  {
    img[tid] = make_uchar4(0, node == EmptyNode ? 0 : 64, 0, 255);
    return;
  }

  float3 p = rp.eyePos;                          
  float3 dir = CalcRayDirWorld(xi, yi);
  float dl = length(dir);
  dir /= dl;
  float t = rp.zBuf[tid] / dl;

  VoxData vd;
  int childId = rp.rays[tid].endNodeChild;
  if (childId < 0)
    vd = GetVoxData(node);
  else
    vd = GetChild(GetNodePtr(node), childId);

  Color16  c16;
  Normal16 n16;
  UnpackVoxData(vd, c16, n16);
  uchar4 col;
  col = UnpackColorCU(c16);

  point_3f norm;
  //if (((xi/256 + yi/256) & 1) != 0)
    UnpackNormal(n16, norm.x, norm.y, norm.z);
  //else
  //  norm = SampleNormal(xi, yi);

  float3 pt = p + dir*t;
  point_3f materialColor = point_3f(col.x, col.y, col.z) / 256.0f;
  point_3f res = fminf(CalcLighting(pt, norm, materialColor) * 256.0f, point_3f(255, 255, 255));

  img[tid] = make_uchar4(res.x, res.y, res.z, 255);
}


extern "C" {

void Run_Trace(GridShape grid)
{
  Trace<<<grid.grid, grid.block>>>();
}

void Run_ShadeSimple(GridShape grid, uchar4 * img)
{
  ShadeSimple<<<grid.grid, grid.block>>>(img);
}
}