#include "stdafx.h"
#include "svodata.h"
#include "svorenderer.h"

#include <Magick++.h>

using namespace Magick;
using std::cout;
using std::endl;

inline double mytime()
{
#ifdef WIN32
  return (double)clock() / CLOCKS_PER_SEC;
#else
  timeval tv;
  gettimeofday(&tv, NULL);
  return (double)tv.tv_sec + (double)tv.tv_usec/1000000.0f; 
#endif
}


void testRenderer(shared_ptr<ISVORenderer> renderer, SVOData & scene, const char * outfn)
{
  renderer->SetScene(&scene);
  renderer->SetResolution(1024, 768);
  renderer->SetViewPos(point_3f(0.5f, 0.5f, 0.3f));
  renderer->SetViewDir(point_3f(-1, -1, -1.5));

  double start = mytime();
  const Color32 * frameBuf = renderer->RenderFrame();
  double dt = (mytime() - start) * 1000.0f;
  printf("time: %f ms\n", (float)dt);

  if (outfn != NULL)
  {
    point_2i size = renderer->GetResolution();
    Image img(size.x, size.y, "RGBA", CharPixel, frameBuf);
    img.quality(50);
    img.write(outfn);
  }
}

int main()
{
  cout << "loading scene..." << endl;
  SVOData scene;
  scene.Load("data/scene.vox");

  cout << "\ntesting PPU renderer..." << endl;
  shared_ptr<ISVORenderer> renderer = CreateThreadedRenderer();
  testRenderer(renderer, scene, "test_ppu.jpg");
  
#ifdef TARGET_PPU
  cout << "\ntesting SPU renderer..." << endl;
  shared_ptr<ISVORenderer> spu_renderer = CreateSPURenderer();
  testRenderer(spu_renderer, scene, "test_spu.jpg");
#endif

  return 0;
}
