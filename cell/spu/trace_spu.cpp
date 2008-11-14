#include "stdafx.h"

#include "trace_spu.h"
#include "trace_utils.h"

//using namespace cg;

int tag_id;
trace_spu_params params __attribute__ ((aligned (16)));

Color32 result[BlockSize*BlockSize] __attribute__ ((aligned (16)));

const int CacheSize = 512;
VoxNodeId cacheIds[CacheSize];
VoxNode cacheNodes[CacheSize] __attribute__ ((aligned (16)));

int missCount = 0;
int fetchCount = 0;

const VoxNode FetchNode(VoxNodeId nodeId)
{
  int ofs = nodeId % CacheSize;
  if (cacheIds[ofs] != nodeId)
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
  VoxNode node;
  int child;
  float t;
};

bool RecTrace(VoxNodeId nodeId, point_3f t1, point_3f t2, const uint dirFlags, TraceResult & res)
{
  if (IsNull(nodeId) || minCoord(t2) <= 0)
    return false;

  const VoxNode node = FetchNode(nodeId);
  int ch = FindFirstChild(t1, t2);
  while (true)
  {
    if (GetLeafFlag(node.flags, ch^dirFlags))
    {
      res.node = node;
      res.child = ch^dirFlags;
      res.t = maxCoord(t1);
      return true;
    }

    if (RecTrace(node.child[ch^dirFlags], t1, t2, dirFlags, res))
      return true;

    if (!GoNext(ch, t1, t2))
      return false;
  }
}


SimpleShader shader;

void RenderBlock(const point_2i & base)
{
  for (int y = 0; y < BlockSize; ++y)
    for (int x = 0; x < BlockSize; ++x)
    {
      int ofs = y*BlockSize + x;
      result[ofs] = Color32(x*16, y*16, 0, 255);

      point_3f dir = cg::normalized(params.rdd.dir0 + params.rdd.du*(base.x + x) + params.rdd.dv*(base.y + y));
      AdjustDir(dir);
      point_3f t1, t2;
      uint dirFlags;
      if (!SetupTrace(params.pos, dir, t1, t2, dirFlags))
        continue;

      TraceResult res;
      if (!RecTrace(params.root, t1, t2, dirFlags, res))
        continue;
      
      result[ofs] = shader.Shade(res.node.child[res.child], dir, res.t);
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
      //printf("%u \n", result + BlockSize*row);
      //printf("%u \n", result + BlockSize*row);

      spu_mfcdma32((void *)(result + BlockSize*row), 
        (unsigned int)(params.colorBuf + (base.y+row) * params.viewSize.x + base.x), 
        sizeof(Color32) * BlockSize, tag_id, MFC_PUT_CMD);
      (void)spu_mfcstat(MFC_TAG_UPDATE_ALL);
    }
  }

  printf("fetch: %d miss: %d\n", fetchCount, missCount);
  return 0;
}
