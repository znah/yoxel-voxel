#include "stdafx.h"
#include "svodata.h"
#include "svorenderer.h"



int main()
{
  std::cout << "test" << std::endl;

  SVOData svo;
  svo.Load("../data/scene.vox");

  return 0;
}
