#include "stdafx.h"
#include "scene.h"

#include "ntree/ntree.h"
#include "trace_utils.h"

using namespace ntree;

////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// Scene ////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

Scene::Scene() 
  : m_root(NULL)
  , m_treeDepth(5)
{}

Scene::~Scene()
{
  m_root = DelTree(m_root);
}


void Scene::Load(const char * filename)
{

}

void Scene::Save(const char * filename)
{

}


void Scene::AddVolume(cg::point_3i pos, cg::point_3i size, const ValueType * data)
{
  RangeBuilder builder;
  builder.dstRange = range_3i(pos, size);
  builder.data = data;
  m_root = builder.build(m_root, m_treeDepth);
}

std::string Scene::GetStats()
{
  StatsBuilder stat;
  stat.walk(m_root, 0);

  std::ostringstream ss;
  ss << "( ";
  for (size_t i = 0; i < stat.count.size(); ++i)
    ss << ((i > 0) ? ", " : "") << stat.count[i];
  ss << " )";
  return ss.str();
};


inline point_3i FindFirstChild2(point_3f & t1, point_3f & t2)
{
  point_3i pt;
  for (int i = 0; i < NodeSizePow; ++i)
  {
    int ch = FindFirstChild(t1, t2);
    pt.x = (pt.x << 1) | (ch&1);
    pt.y = (pt.y << 1) | ((ch>>1)&1);
    pt.z = (pt.z << 1) | ((ch>>2)&1);
  }
  return pt;
}

inline int argmin(const point_3f & p)
{
  if (p.x > p.y)
    return (p.y < p.z) ? 1 : 2;
  else
    return (p.x < p.z) ? 0 : 2;
}

inline bool GoNext2(point_3i & ch, point_3f & t1, point_3f & t2)
{
  int exitPlane = argmin(t2);
  if (ch[exitPlane] >= NodeSize-1)
    return false;

  ++ch[exitPlane];
  float dt = t2[exitPlane] - t1[exitPlane];
  t1[exitPlane] = t2[exitPlane];
  t2[exitPlane] += dt;
  return true;
}

bool accumLight(float dt, ValueType sample, point_4f & res)
{
  dt *= 256;
  point_4f v = point_4f(sample.x, sample.y, sample.z, sample.w) / 256;
  float opaq = 1.0f - powf((1.0f - v.w), dt);
  v &= point_4f(opaq, opaq, opaq, 1.0f) * (1.0f - res.w);
  res += v;
  return false;
}

bool walkNode(NodePtr node, point_3f t1, point_3f t2, const uint dirFlags, point_4f & res)
{
  if (minCoord(t2) <= 0)
    return false;

  point_3i ch = FindFirstChild2(t1, t2);
  bool done = false;
  do {
    int ci = pt2ci(ch);
    if (node->child != NULL && node->child[ci] != NULL)
      done = walkNode(node->child[ci], t1, t2, dirFlags, res);
    else
    {
      float dt = minCoord(t2) - maxCoord(t1);
      done = accumLight(dt, node->data[ci], res);
    }
  } while (!done && GoNext2(ch, t1, t2));
  return done;
}

ValueType Scene::TraceRay(const point_3f & p, point_3f dir)
{
  AdjustDir(dir);
  point_4f res;
  point_3f t1, t2;
  uint dirFlags;
  if (SetupTrace(p, dir, t1, t2, dirFlags))
    walkNode(m_root, t1, t2, dirFlags, res);
  else
    res = point_4f(0, 0, 0, 1);
  res *= 256;
  for (int i = 0; i < 4; ++i)
    res[i] = cg::bound(res[i], 0.0f, 255.0f);
  uchar4 c = {(uchar)res.x, (uchar)res.y, (uchar)res.z, (uchar)res.w};
  return c;
}
