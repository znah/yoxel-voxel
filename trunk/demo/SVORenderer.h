#pragma once

#include "DynamicSVO.h"
#include "CudaSVO.h"
#include "cu_cpp.h"
#include "trace_cu.h"

class SVORenderer
{
public:
  SVORenderer();
  ~SVORenderer();

  void SetScene(DynamicSVO * svo) { m_svo.SetSVO(svo); }

  void SetViewPos(const point_3f & pos) { m_pos = pos; }
  void SetViewDir(const point_3f & dir) { m_dir = dir; }
  void SetViewUp (const point_3f & up) { m_up = up; }

  void SetViewSize(int width, int height);

  void SetFOV(float fov) { m_fov = fov; }

  void Render(void * d_dstBuf);

private:
  CudaSVO m_svo;

  point_3f m_pos;
  point_3f m_dir;
  point_3f m_up;
  point_2i m_viewSize;
  float m_fov;

  CuVector<RayData> m_rayDataBuf;
};
