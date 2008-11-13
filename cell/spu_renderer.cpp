#include "stdafx.h"
#include "rendrer_base.h"

#include <boost/thread/thread.hpp>
#include <libspe2.h>

#include "spu/trace_spu.h"

extern spe_program_handle_t trace_spu;


class SPURenderer : public RendererBase
{
public:
  const Color32 * RenderFrame();
private:
  struct RenderThread;
};

shared_ptr<ISVORenderer> CreateSPURenderer()
{
  return shared_ptr<ISVORenderer>(new SPURenderer);
}



#define CHECK( cond ) if (!(cond)) printf("check at %d: " #cond "\n", __LINE__ )


struct RenderThread
{
  TreadedRenderer * renderer;
  RayDirData rdd;
  point_2i start, end;

  void SPURenderer::operator()()
  {
    spe_context_ptr_t ctx = spe_context_create (0, NULL);
    CHECK(ctx != NULL);

    int res = spe_program_load(ctx, &trace_sup);
    CHECK(res == 0);

    trace_spu_params params __attribute__ ((aligned (16)));
    params.rdd = rdd;
    params.start = start;
    params.end = end;
    params.viewSize = renderer->m_viewSize;
    params.colorBuf = &renderer->m_colorBuf[0];
    params.root = &renderer->m_svo->GetRoot();
    params.nodes = &(*renderer->m_svo)[0];

    unsigned int entry = SPE_DEFAULT_ENTRY;
    res = spe_context_run(ctx, &entry, 0, &params, NULL, NULL);
    CHECK(res >= 0);
    
    int res = spe_context_destroy(ctx);
    CHECK(res == 0);
  }
};


const Color32 * SPURenderer::RenderFrame()
{
  if (m_svo == NULL)
    return NULL;

  RayDirData rdd;
  InitRayDir(rdd);

  int spe_threads = spe_cpu_info_get(SPE_COUNT_USABLE_SPES, -1);

  int ystep = m_viewSize.y / spe_threads;

  boost::thread_group threads;
  for (int i = 0; i < ThreadNum; ++i)
  {
    RenderThreadData td;
    td.renderer = this;
    td.rdd = rdd;
    td.start = point_2i(0, ystep*i);
    td.end = point_2i(m_viewSize.x, ystep*(i+1));
    threads.create_thread(td);
  }
  threads.join_all();
}
