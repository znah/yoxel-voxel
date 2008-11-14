#include "stdafx.h"

#include "trace_spu.h"
#include "trace_utils.h"

//using namespace cg;

int tag_id;
trace_spu_params params __attribute__ ((aligned (16)));

Color32 result[MaxRowSize] __attribute__ ((aligned (16)));

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

  //const VoxNode & node = (*m_svo)[nodeId];
  VoxNode node __attribute__ ((aligned (16)));
  const VoxNode * node_ptr = params.nodes + nodeId;
  spu_mfcdma32((void *)(&node), (unsigned int)node_ptr, sizeof(node), tag_id, MFC_GET_CMD);
  (void)spu_mfcstat(MFC_TAG_UPDATE_ALL);

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




int main(unsigned long long spu_id __attribute__ ((unused)), unsigned long long parm)
{
  tag_id = mfc_tag_reserve();
  spu_writech(MFC_WrTagMask, -1);
  
  spu_mfcdma32((void *)(&params), (unsigned int)parm, sizeof(trace_spu_params), tag_id, MFC_GET_CMD);
  (void)spu_mfcstat(MFC_TAG_UPDATE_ALL);

  SimpleShader shader;
  shader.viewerPos = params.pos;
  shader.lightPos = params.pos;

  for (int y = params.start.y; y < params.end.y; ++y)
  {
    for (int x = params.start.x; x < params.end.x; ++x)
    {
      result[x] = Color32(y*256/768, 0, 0, 255);

      point_3f dir = cg::normalized(params.rdd.dir0 + params.rdd.du*x + params.rdd.dv*y);
      AdjustDir(dir);
      point_3f t1, t2;
      uint dirFlags;
      if (!SetupTrace(params.pos, dir, t1, t2, dirFlags))
        continue;

      TraceResult res;
      if (!RecTrace(params.root, t1, t2, dirFlags, res))
        continue;
      
      result[x] = shader.Shade(res.node.child[res.child], dir, res.t);
    }
   spu_mfcdma32((void *)result, (unsigned int)(params.colorBuf + y * params.viewSize.x), sizeof(Color32) * params.viewSize.x, tag_id, MFC_PUT_CMD);
	 (void)spu_mfcstat(MFC_TAG_UPDATE_ALL);

  }

  return 0;
}
