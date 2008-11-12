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
  
  renderer->SetResolution(1024, 768);

  renderer->SetViewPos(point_3f(0.5f, 0.5f, 0.3f));
  renderer->SetViewDir(point_3f(-1, -1, -1.5));

  //const Color32 * frameBuf = renderer->RenderFrame();
  point_2i size = renderer->GetResolution();

  //Image img(size.x, size.y, "RGBA", CharPixel, frameBuf);
  //img.write("test.jpg");

  return 0;
}
