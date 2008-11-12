#include "stdafx.h"

#include "stdafx.h"
#include "renderer_base.h"
#include "trace_utils.h"

class TreaderRenderer : public RendererBase
{
public:
  virtual const Color32 * RenderFrame();

private:
  bool RecTrace(VoxNodeId nodeId, point_3f t1, point_3f t2, const uint dirFlags, TraceResult & res);
};
