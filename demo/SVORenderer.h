#pragma once

#include "DynamicSVO.h"
#include "CudaSVO_rt.h"
#include "cu_cpp.h"
#include "trace_cu.h"

class SVORenderer
{
public:
  SVORenderer();
  ~SVORenderer();

  void SetScene(DynamicSVO * svo);

  void SetViewPos(const point_3f & pos) { m_pos = pos; }
  void SetViewDir(const point_3f & dir) { m_dir = dir; }
  void SetViewUp (const point_3f & up) { m_up = up; }

  void SetViewSize(int width, int height);

  void SetFOV(float fov) { m_fov = fov; }
  float GetFOV() const { return m_fov; }

  void SetDetailCoef(float coef) { m_detailCoef = coef; }
  float GetDetailCoef() const { return m_detailCoef; }

  void Render(void * d_dstBuf);

private:
  CudaSVO m_svo;

  point_3f m_pos;
  point_3f m_dir;
  point_3f m_up;
  point_2i m_viewSize;
  float m_fov;

  float m_detailCoef;

  CuVector<RayData> m_rayDataBuf;

  const textureReference * m_dataTexRef;
};
