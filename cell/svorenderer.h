#pragma once

#include "svodata.h"
#include <tr1/memory>

class ISVORenderer
{
protected:
  ISVORenderer() {}
public:
  virtual ~ISVORenderer() {}
  
  virtual void SetSVO(SVOData * svo) = 0;
  
  virtual void SetViewPos(const point_3f & pos) = 0;
  virtual void SetViewDir(float crs, float pitch, float roll) = 0;
  
  virtual void SetResolution(int width, int height) = 0;
  virtual point_2i GetResolution() const = 0;

  virtual const Color32 * RenderFrame() = 0;
};

