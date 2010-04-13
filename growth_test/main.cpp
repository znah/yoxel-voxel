#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <numeric>

using std::vector;

const int FieldSize = 512;
const float gridStep = 0.001;

inline int ofs(int x, int y) { return y*FieldSize + x; }

struct point_2i
{
  int x, y;
};

inline float calcPot(const point_2i & p1, const point_2i & p2)
{
  float dx = (p1.x - p2.x);
  float dy = (p1.y - p2.y);
  float r = sqrtf((float)(dx*dx+dy*dy));
  return  - 1.0 / sqrtf(r);
}

int main()
{
  vector<point_2i> cluster;
  vector<point_2i> candPos;
  vector<float> candPot;
  vector<int> field(FieldSize*FieldSize, 0);
  int f2 = FieldSize/2;
  point_2i center = {f2, f2};
  for (int dy = -1; dy < 2; ++dy)
    for (int dx = -1; dx < 2; ++dx)
    {
      field[ofs(f2+dx, f2+dy)] = (dx==0 && dy==0) ? 1 : 2;
      point_2i p = {f2+dx, f2+dy};
      if (!(dx==0 && dy==0))
      {
        
        candPos.push_back(p);
        candPot.push_back(calcPot(center, p));
      }
      else
        cluster.push_back(p);
    }


  vector<float> prob;
  for (int iter = 0; iter < 4000; ++iter)
  {
    prob.resize(candPot.size());
    float pmin = *std::min_element(candPot.begin(), candPot.end());
    float pmax = *std::max_element(candPot.begin(), candPot.end());
    float invdp = 1.0f / (pmax-pmin);
    float sum = 0;
    for (size_t i = 0; i < candPot.size(); ++i)
    {
      float v = powf((candPot[i] - pmin) * invdp, 10.0);
      prob[i] = v;
      sum += v;
    }
    for (size_t i = 0; i < prob.size(); ++i)
      prob[i] /= sum;

    std::partial_sum(prob.begin(), prob.end(), prob.begin());
    float r = (float)rand() / RAND_MAX + (float)rand() / RAND_MAX / RAND_MAX;
    int chosen = std::upper_bound(prob.begin(), prob.end(), r) - prob.begin();
    //int chosen = std::min_element(prob.begin(), prob.end()) - prob.begin();
    
    point_2i p = candPos[chosen];
    field[ofs(p.x, p.y)] = 1;

    if (candPos.size() > 1)
    {
      candPos[chosen] = candPos.back();
      candPot[chosen] = candPot.back();
    }
    candPos.pop_back();
    candPot.pop_back();
    
    for (int dy = -1; dy < 2; ++dy)
      for (int dx = -1; dx < 2; ++dx)
      {
        int x = p.x + dx;
        int y = p.y + dy;
        if (x < 0 || x >= FieldSize)
          continue;
        if (y < 0 || y >= FieldSize)
          continue;
        if (field[ofs(x, y)] != 0)
          continue;
        field[ofs(x, y)] = 2;
        point_2i newpt = {x, y};

        float newPot = 0;
        for (size_t i = 0; i < cluster.size(); ++i)
          newPot += calcPot(cluster[i], newpt);

        candPos.push_back(newpt);
        candPot.push_back(newPot);
      }

    for (size_t i = 0; i < candPot.size(); ++i)
      candPot[i] += calcPot(p, candPos[i]);

    if (iter % 100 == 0)
      std::cout << iter << std::endl;
  }

  std::ofstream file("out.dat", std::ios::binary);
  file.write((char*)&field[0], field.size()*sizeof(int));
  
  return 0;
}