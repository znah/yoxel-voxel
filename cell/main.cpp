#include "stdafx.h"
#include "svodata.h"
#include "svorenderer.h"

int main()
{
  std::cout << "test" << std::endl;

  SVOData scene;
  scene.Load("../data/scene.vox");

  shared_ptr<ISVORenderer> renderer = CreateSimpleRenderer();
  renderer->SetScene(&scene);


  return 0;
}
