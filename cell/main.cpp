#include "stdafx.h"
#include "svodata.h"
#include "svorenderer.h"

#include <Magick++.h>


using namespace Magick;


int main()
{
  std::cout << "test" << std::endl;

  SVOData scene;
  scene.Load("../data/scene.vox");

  shared_ptr<ISVORenderer> renderer = CreateSimpleRenderer();
  renderer->SetScene(&scene);

  renderer->SetViewPos(point_3f(0.5, 0.5, 0.5));
  renderer->SetViewDir(cprf(-135, -45, 0));

  const Color32 * frameBuf = renderer->RenderFrame();
  point_2i size = renderer->GetResolution();

  Image img(size.x, size.y, "RGBA", CharPixel, frameBuf);
  img.write("test.jpg");



  return 0;
}
