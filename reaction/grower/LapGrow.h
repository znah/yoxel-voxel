#pragma once

class LapGrow
{
public:
  LapGrow();
  void GrowParticle();

private:
  std::vector<point_2f> m_black;
  struct GraySite { point_2f pos; float phi; };
  std::vector<GraySite> m_grey;
  
  float m_nu;
  float m_chargeR;

  static const int NeibNum = 4;
  point_2f neib(const point_2f & p, int i);
  
  std::set<point_2i> m_mark;
  bool TryMarkSpace(const point_2f & p);
  bool TryAddGrey(const point_2f & p);
  float calcPhi(const point_2f & p);
  float calcPhi(const point_2f & p0, const point_2f & p);

  std::vector<float> m_prob;
};