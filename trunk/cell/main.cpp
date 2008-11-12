#include "stdafx.h"
#include "svodata.h"
#include "svorenderer.h"

#include <Magick++.h>

using namespace Magick;

inline float mytime()
{
#ifdef WIN32
  return (float)clock() / CLOCKS_PER_SEC;
#else
  timeval tv;
  timezone tz;
  gettimeofday(&tv,&tz);
  return (float)tv.tv_sec + (float)tv1.tv_usec/1000000f.0; 
#endif
}


const Color32 * testRenderer(shared_ptr<ISVORenderer> renderer, SVOData & scene)
{
  renderer->SetScene(&scene);
  renderer->SetResolution(1024, 768);
  renderer->SetViewPos(point_3f(0.5f, 0.5f, 0.3f));
  renderer->SetViewDir(point_3f(-1, -1, -1.5));

  float start = mytime();
  const Color32 * res = renderer->RenderFrame();
  float dt = (mytime() - start) * 1000.0f;
  printf("time: %f ms\n", dt);

  return res;
}

int main()
{
  SVOData scene;
  scene.Load("../data/scene.vox");
  shared_ptr<ISVORenderer> renderer = CreateThreadedRenderer();
  shared_ptr<ISVORenderer> refRenderer = CreateSimpleRenderer();

  testRenderer(refRenderer, scene);
  const Color32 * frameBuf = testRenderer(renderer, scene);

  point_2i size = renderer->GetResolution();

  Image img(size.x, size.y, "RGBA", CharPixel, frameBuf);
  img.quality(75);
  img.write("test.jpg");

  return 0;
}
