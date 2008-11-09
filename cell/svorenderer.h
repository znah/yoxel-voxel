#pragma once

#include "svodata.h"

class ISVORenderer
{
protected:
  ISVORenderer() {}
public:
  virtual ~ISVORenderer() {}
  
  virtual void SetScene(SVOData * svo) = 0;
  
  virtual void SetViewPos(const point_3f & pos) = 0;
  virtual void SetViewDir(const cprf & rotation) = 0;
  
  virtual void SetResolution(int width, int height) = 0;
  virtual point_2i GetResolution() const = 0;

  virtual const Color32 * RenderFrame() = 0;
};

shared_ptr<ISVORenderer> CreateSimpleRenderer();
