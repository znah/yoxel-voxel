#include "stdafx.h"

#include "trace_spu.h"
#include "trace_utils.h"

#include <libmisc.h>



int tag_id;
trace_spu_params params __attribute__ ((aligned (16)));

Color32 result[BlockSize*BlockSize] __attribute__ ((aligned (16)));

const int CacheSize = 2048;
VoxNodeId cacheIds[CacheSize];
VoxNode cacheNodes[CacheSize] __attribute__ ((aligned (16)));

int missCount = 0;
int fetchCount = 0;

const VoxNode FetchNode(VoxNodeId nodeId)
{
  int ofs = nodeId % CacheSize;
  if (__builtin_expect((cacheIds[ofs] != nodeId), 0))
  {
    const VoxNode * node_ptr = params.nodes + nodeId;
    spu_mfcdma32((void *)(cacheNodes + ofs), (unsigned int)node_ptr, sizeof(VoxNode), tag_id, MFC_GET_CMD);
    (void)spu_mfcstat(MFC_TAG_UPDATE_ALL);
    cacheIds[ofs] = nodeId;
    ++missCount;
  }
  ++fetchCount;
  return cacheNodes[ofs];
}


struct TraceResult
{
  VoxData data;
  float t;
};


typedef vector float float4;


inline GLOBAL_FUNC int FindFirstChildSPU(float4 & t1, float4 & t2)
{
  float4 tm = spu_splats(0.5f) * (t1 + t2);
  float tEnter = max_vec_float3(t1);
  
  vector unsigned int cmp = spu_cmpgt(spu_splats(tEnter), tm);
  t2 = spu_sel(tm, t2, cmp);
  t1 = spu_sel(t1, tm, cmp);

  
  //int childId = spu_gather(cmp)[0] >> 1;
  //static const int lookup[8] = {0, 4, 2, 6, 1, 5, 3, 7};
  //childId = lookup[childId];
  int childId = 0;
  if (cmp[0]) childId |= 1<<0;
  if (cmp[1]) childId |= 1<<1;
  if (cmp[2]) childId |= 1<<2;

  //printf("%d %d\n", childId, spu_gather(cmp)[0]);
  return childId;
}

inline GLOBAL_FUNC  bool GoNextSPU(int & childId, float4 & t1, float4 & t2)
{
  int exitPlane;
  
  // argmin
  if (t2[0] > t2[1])
    exitPlane = (t2[1] < t2[2]) ? 1 : 2;
  else
    exitPlane = (t2[0] < t2[2]) ? 0 : 2;

  int mask = 1<<exitPlane;
  if ((childId & mask) != 0)
    return false;

  childId ^= mask;

  float dt = t2[exitPlane]-t1[exitPlane];
//  t1[exitPlane] = t2[exitPlane];
  
  t1[exitPlane] = t2[exitPlane];
  t2[exitPlane] += dt;
  return true;

}



bool RecTrace(VoxNodeId nodeId, float4 t1, float4 t2, const uint dirFlags, TraceResult & res)
{
  if (IsNull(nodeId) || min_vec_float3(t2) <= 0)
    return false;

  const VoxNode node = FetchNode(nodeId);
  int ch = FindFirstChildSPU(t1, t2);
  while (!GetLeafFlag(node.flags, ch^dirFlags))
  {
    if (RecTrace(node.child[ch^dirFlags], t1, t2, dirFlags, res))
      return true;

    if (!GoNextSPU(ch, t1, t2))
      return false;
  } 

  res.data = node.child[ch^dirFlags];
  res.t = max_vec_float3(t1);
  return true;
}


SimpleShader shader;

void RenderBlock(const point_2i & base)
{
  for (int y = 0; y < BlockSize; ++y)
    for (int x = 0; x < BlockSize; ++x)
    {
      int ofs = y*BlockSize + x;
      result[ofs] = Color32(0, 0, 0, 255);

      point_3f dir = cg::normalized(params.rdd.dir0 + params.rdd.du*(base.x + x) + params.rdd.dv*(base.y + y));
      AdjustDir(dir);
      point_3f t1, t2;
      uint dirFlags;
      if (!SetupTrace(params.pos, dir, t1, t2, dirFlags))
        continue;

      
      float4 vt1 = {t1.x, t1.y, t1.z, 0};
      float4 vt2 = {t2.x, t2.y, t2.z, 0};
      TraceResult res;
      if (!RecTrace(params.root, vt1, vt2, dirFlags, res))
        continue;
      
      result[ofs] = shader.Shade(res.data, dir, res.t);
    }
}


int main(unsigned long long spu_id __attribute__ ((unused)), unsigned long long parm)
{
  tag_id = mfc_tag_reserve();
  spu_writech(MFC_WrTagMask, -1);
  
  spu_mfcdma32((void *)(&params), (unsigned int)parm, sizeof(trace_spu_params), tag_id, MFC_GET_CMD);
  (void)spu_mfcstat(MFC_TAG_UPDATE_ALL);

  for (int i = 0; i < CacheSize; ++i)
    cacheIds[i] = EmptyNode;

  shader.viewerPos = params.pos;
  shader.lightPos = params.pos;

  point_2i gridSize = params.viewSize / BlockSize;
  int blockNum = gridSize.x * gridSize.y;
  for (int block = params.blockStart; block < blockNum; block += params.blockStride)
  {
    int bx = block % gridSize.x;
    int by = block / gridSize.x;
    point_2i base = point_2i(bx, by) * BlockSize;
    RenderBlock(base);
    for (int row = 0; row < BlockSize; ++row)
    {
      spu_mfcdma32((void *)(result + BlockSize*row), 
        (unsigned int)(params.colorBuf + (base.y+row) * params.viewSize.x + base.x), 
        sizeof(Color32) * BlockSize, tag_id, MFC_PUT_CMD);
    }
    (void)spu_mfcstat(MFC_TAG_UPDATE_ALL);
  }

  printf("fetch: %d miss: %d\n", fetchCount, missCount);
  return 0;
}
