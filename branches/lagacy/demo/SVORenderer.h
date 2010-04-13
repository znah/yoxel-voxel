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

  void SetViewPos(const point_3f & pos) { CheckSet(m_pos, pos); }
  void SetViewDir(const point_3f & dir) { CheckSet(m_dir, dir); }
  void SetViewUp (const point_3f & up)  { CheckSet(m_up, up); }

  void SetViewSize(int width, int height);

  void SetFOV(float fov) { CheckSet(m_fov, fov); }
  float GetFOV() const { return m_fov; }

  void SetDetailCoef(float coef) { CheckSet(m_detailCoef, coef); }
  float GetDetailCoef() const { return m_detailCoef; }

  void SetDither(float coef) { CheckSet(m_ditherCoef, coef); }
  float GetDither() const { return m_ditherCoef; }

  void SetLigth(int i, const LightParams & lp) { m_lights[i] = lp; }

  void Render(void * d_dstBuf);
  void UpdateSVO();

  void ResetAccum();

private:
  template <class T>
  void CheckSet(T & val, const T & newVal)
  {
    if (val != newVal)
      ResetAccum();
    val = newVal;
  }

  CudaSVO m_svo;

  point_3f m_pos;
  point_3f m_dir;
  point_3f m_up;
  point_2i m_viewSize;
  float m_fov;
  float m_ditherCoef;

  LightParams m_lights[MaxLightsNum];

  float m_detailCoef;

  CuVector<RayData> m_rayDataBuf;
  CuVector<float> m_noiseBuf;
  CuVector<ushort4> m_accumBuf;

  const textureReference * m_dataTexRef;

  int m_accumIter;
};
