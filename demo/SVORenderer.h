#pragma once

#include "DynamicSVO.h"
#include "CudaSVO_rt.h"
#include "cu_cpp.h"
#include "trace_cu.h"

struct ProfileStats
{
  float traceTime;
  ProfileStats() : traceTime(0) {}
};

class SVORenderer
{
public:
  enum ShadeMode { 
    SM_SIMPLE = 0, 
    SM_COUNTER, 
    SM_MAX };


  SVORenderer();
  ~SVORenderer();

  void SetScene(DynamicSVO * svo);

  void SetViewPos(const point_3f & pos) { CheckSet(m_pos, pos); }
  void SetViewDir(const point_3f & dir) { CheckSet(m_dir, dir); }
  void SetViewUp (const point_3f & up)  { CheckSet(m_up, up); }

  void SetViewSize(int width, int height);

  void SetFOV(float fov) { CheckSet(m_fov, fov); }
  float GetFOV() const { return m_fov; }

  void SetDetailCoef(float coef) { CheckSet(m_detailCoef, coef); printf("%f\n", coef); }
  float GetDetailCoef() const { return m_detailCoef; }

  void SetDither(float coef) { CheckSet(m_ditherCoef, coef); }
  float GetDither() const { return m_ditherCoef; }

  void SetShuffle(bool enable) { CheckSet(m_shuffleEnabled, enable); }
  bool GetShuffle() const { return m_shuffleEnabled; }

  void SetLight(int i, const LightParams & lp) { m_lights[i] = lp; }

  void SetShadeMode(int mode) { CheckSet(m_shadeMode, mode); }
  int GetShadeMode() const { return m_shadeMode; }

  void Render(void * d_dstBuf);
  void UpdateSVO();

  void ResetAccum();

  const ProfileStats & GetProfile() const { return m_profStats; }

  std::string GetInfoString() const;

private:
  template <class T>
  void CheckSet(T & val, const T & newVal)
  {
    if (val != newVal)
    {
      ResetAccum();
      m_profStats = ProfileStats();
    }
    val = newVal;
  }

  CudaSVO m_svo;

  point_3f m_pos;
  point_3f m_dir;
  point_3f m_up;
  point_2i m_viewSize;
  float m_fov;
  
  float m_ditherCoef;
  int m_shadeMode;
  bool m_shuffleEnabled;

  LightParams m_lights[MaxLightsNum];

  float m_detailCoef;

  CuVector<RayData> m_rayDataBuf;
  CuVector<float> m_noiseBuf;
  CuVector<ushort4> m_accumBuf;
  CuVector<int> m_shuffleBuf;

  const textureReference * m_dataTexRef;

  int m_accumIter;

  ProfileStats m_profStats;
};
