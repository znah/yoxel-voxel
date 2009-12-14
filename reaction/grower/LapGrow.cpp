#include "stdafx.h"
#include "LapGrow.h"

point_2f LapGrow::neib(const point_2f & p, int i)
{
  static const point_2f ofs[NeibNum] = { point_2f( 1, 0),
                                         point_2f( 0, 1),
                                         point_2f(-1, 0),
                                         point_2f( 0,-0)};
  return p + ofs[i];
}

LapGrow::LapGrow() 
  : m_nu(2.0f)
  , m_chargeR(1.0f)
{
  point_2f p0 = point_2f(0, 0);
  m_black.push_back(p0);
  for (int i = 0; i < NeibNum; ++i)
    TryAddGrey(neib(p0, i));
}

bool LapGrow::TryMarkSpace(const point_2f & p)
{
  point_2i id = floori(p + 0.5f);
  return m_mark.insert(id).second;
}

bool LapGrow::TryAddGrey(const point_2f & p)
{
  if (!TryMarkSpace(p))
    return false;
  GraySite site = {p, calcPhi(p)};
  m_grey.push_back(site);
  return true;
}

float LapGrow::calcPhi(const point_2f & p)
{
  float res = 0;
  for (int i = 0; i < m_black.size(); ++i)
    res += calcPhi(m_black[i], p);
  return res;
}

float LapGrow::calcPhi(const point_2f & p0, const point_2f & p)
{
  float r = length(p0-p);
  return 1.0f - m_chargeR / r;
}

void LapGrow::GrowParticle()
{
  m_prob.resize(m_grey.size());
  float lo(m_grey[0].phi), hi(m_grey[0].phi);
  for (int i = 1; i < m_grey.size(); ++i)
  {
    float v = m_grey[i].phi;
    lo = min(lo, v);
    hi = max(hi, v);
  }
  if (hi - lo < epsf)
    std::fill(m_prob.begin(), m_prob.end(), 1.0f / m_prob.size());
  else
  {
    float invd = 1.0f / (hi - lo);
    for (int i = 0; i < m_prob.size(); ++i)
      m_prob[i] = (m_grey[i].phi - lo) * invd;
    float invsum = 1.0f / std::accumulate(m_prob.begin(), m_prob.end(), 0.0f);
    for (int i = 0; i < m_prob.size(); ++i)
      m_prob[i] *= invsum;
  }
  std::partial_sum( m_prob.begin(), m_prob.end(), m_prob.begin());
  float r = randf();
  std::lower_bound();
    


  

}