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

  void SetSSNA(bool enable) { m_ssna = enable; }
  bool GetSSNA() const { return m_ssna; }

  void SetShowNormals(bool enable) { m_showNormals = enable; }
  bool GetShowNormals() const { return m_showNormals; }

  void SetLigth(int i, const LightParams & lp) { m_lights[i] = lp; }

  void Render(void * d_dstBuf);
  void UpdateSVO();

  void DumpTraceData(std::string);

private:
  CudaSVO m_svo;

  bool m_ssna;
  bool m_showNormals;

  point_3f m_pos;
  point_3f m_dir;
  point_3f m_up;
  point_2i m_viewSize;
  float m_fov;
  float m_ditherCoef;

  LightParams m_lights[MaxLightsNum];

  float m_detailCoef;

  CuVector<RayData> m_rayDataBuf;
  CuVector<float> m_zbuf[2];

  const textureReference * m_dataTexRef;

  void InitBlur();
};
