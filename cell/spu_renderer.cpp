#include "stdafx.h"
#include "rendrer_base.h"

#include <boost/thread/thread.hpp>
#include <libspe2.h>


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




struct RenderThread
{
  TreadedRenderer * renderer;
  RayDirData rdd;
  point_2i start, size;

  void SPURenderer::operator()()
  {
    


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
    td.size = point_2i(m_viewSize.x, ystep);
    threads.create_thread(td);
  }
  threads.join_all();





}
