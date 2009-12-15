#pragma once

class LapGrow
{
public:
  LapGrow();
  void GrowParticle();
  void SetExponent(float exponent) { m_exponent = exponent; }

  int size() const { return m_black.size(); }
  const point_2f * black() const { return &m_black[0]; }


private:
  std::vector<point_2f> m_black;
  struct GraySite { point_2f pos; float phi; };
  std::vector<GraySite> m_grey;
  
  float m_exponent;
  float m_chargeR;

  static const int NeibNum = 4;
  point_2f neib(const point_2f & p, int i);
  
  struct point_cmp
  {
    bool operator()(const point_2i & a, const point_2i & b)
    {
      if (a.x != b.x)
        return a.x < b.x;
      return a.y < b.y;
    }
  };
  std::set<point_2i, point_cmp> m_mark;
  bool TryMarkSpace(const point_2f & p);
  bool TryAddGrey(const point_2f & p);
  float calcPhi(const point_2f & p);
  float calcPhi(const point_2f & p0, const point_2f & p);

  std::vector<float> m_prob;
};