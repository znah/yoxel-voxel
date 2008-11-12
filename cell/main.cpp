#include "stdafx.h"
#include "svodata.h"
#include "svorenderer.h"

#include <Magick++.h>


using namespace Magick;


const Color32 * testRenderer(shared_ptr<ISVORenderer> renderer, SVOData & scene)
{
  renderer->SetScene(&scene);
  renderer->SetResolution(1024, 768);
  renderer->SetViewPos(point_3f(0.5f, 0.5f, 0.3f));
  renderer->SetViewDir(point_3f(-1, -1, -1.5));
  
  clock_t start = clock();
  const Color32 * res = renderer->RenderFrame();
  float dt = 1000.0f * (float)(clock() - start) / CLOCKS_PER_SEC;
  printf("render time: %f ms\n", dt);

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
