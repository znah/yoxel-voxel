#include "stdafx.h"
#include "scene.h"

#include "ntree/ntree.h"

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

