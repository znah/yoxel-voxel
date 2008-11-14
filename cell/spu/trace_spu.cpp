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
  VoxData data;
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
      res.data = node.child[ch^dirFlags];
      res.t = maxCoord(t1);
      return true;
    }

    if (RecTrace(node.child[ch^dirFlags], t1, t2, dirFlags, res))
      return true;

    if (!GoNext(ch, t1, t2))
      return false;
  }
}

bool StacklessTrace(point_3f t1, point_3f t2, const uint dirFlags, TraceResult & res)
{
  VoxNode node = FetchNode(params.root);
  int childId = 0;
  //int level = 0;
  //float nodeSize = 1.0f; //pow(0.5f, level);

  enum States { ST_EXIT, ST_ANALYSE, ST_SAVE, ST_GOUP, ST_GODOWN, ST_GONEXT };
  int state = ST_ANALYSE;
  while (state != ST_EXIT)
  {
    switch (state)
    {
      case ST_ANALYSE:
      {
        //childId = -1;
        //if (maxCoord(t1) * rp.detailCoef > nodeSize/2)  { state = GetEmptyFlag(GetNodeInfo(node)) ? ST_GOUP : ST_SAVE; break; }
        
        childId = FindFirstChild(t1, t2);
        state = ST_GODOWN;
        break;
      }
      
      case ST_GODOWN:
      {
        if (minCoord(t2) < 0) { state = ST_GONEXT; break; }

        if (GetLeafFlag(GetNodeInfo(node), childId^dirFlags)) { state = ST_SAVE; break; }
        
        VoxNodeId ch = GetChild(node, childId^dirFlags);
        if (IsNull(ch)) {state = ST_GONEXT; break; }
        node = FetchNode(ch); //node = ch;
        //++level;
        //nodeSize /= 2;
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
        VoxNodeId p = node.parent;
        if (IsNull(p)) { 
          return false;
          //rays[tid].endNode = EmptyNode;
          //state = ST_EXIT; 
          //break; 
        }

        for (int i = 0; i < 3; ++i)
        {
          int mask = 1<<i;
          float dt = t2[i] - t1[i];
          ((childId & mask) == 0) ? t2[i] += dt : t1[i] -= dt;
        }
        childId = GetSelfChildId(node.flags)^dirFlags;
        node = FetchNode(p);
        --level;
        nodeSize *= 2;
        state = ST_GONEXT;
        break;
      }

      case ST_SAVE:
      {
        res.data = node.child[childId^dirFlags];
        res.t = maxCoord(t1);
        state = ST_EXIT;
        break;
      }
    }
  }
  return true;
}


SimpleShader shader;

void RenderBlock(const point_2i & base)
{
  for (int y = 0; y < BlockSize; ++y)
    for (int x = 0; x < BlockSize; ++x)
    {
      int ofs = y*BlockSize + x;
      result[ofs] = Color32(x*256/BlockSize, y*256/BlockSize, 0, 255);

      point_3f dir = cg::normalized(params.rdd.dir0 + params.rdd.du*(base.x + x) + params.rdd.dv*(base.y + y));
      AdjustDir(dir);
      point_3f t1, t2;
      uint dirFlags;
      if (!SetupTrace(params.pos, dir, t1, t2, dirFlags))
        continue;

      TraceResult res;
      if (!RecTrace(params.root, t1, t2, dirFlags, res))
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
      (void)spu_mfcstat(MFC_TAG_UPDATE_ALL);
    }
  }

  printf("fetch: %d miss: %d\n", fetchCount, missCount);
  return 0;
}
