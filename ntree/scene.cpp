#include "stdafx.h"
#include "scene.h"

#include "ntree/ntree.h"
#include "trace_utils.h"
#include "ntree_trace.cuh"

using namespace ntree;

////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// Scene ////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

Scene::Scene() 
  : m_treeDepth(5)
//  , m_dataPool(GetDataTex(), 5, point_3i(64, 64, 64))
//  , m_gridPool(GetNodeTex(), 4, point_3i(32, 16, 16))
//  , m_rootGrid(GPUNull)
//  , m_rootData(GPUNull)
{
  SetViewSize(point_2i(800, 600));
}

Scene::~Scene()
{
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
  builder.build(m_root, ntree::CalcSceneSize(m_treeDepth));
}

std::string Scene::GetStats()
{
  StatsBuilder stat;
  stat.walk(m_root);

  std::ostringstream ss;
  //ss << "( ";
  //for (size_t i = 0; i < stat.count.size(); ++i)
  //  ss << ((i > 0) ? ", " : "") << stat.count[i];
  //ss << " )";
  int vol = stat.grids * GridSize3 + stat.bricks * BrickSize3;
  vol *= 4;
  ss << "grids: " << stat.grids << "  bricks: " << stat.bricks << "  vol: " << (vol / 1024.0);
  
  return ss.str();
};

void Scene::SetViewSize(point_2i size) 
{ 
  m_viewSize = size;
}


/*void Scene::Render(uchar4 * img, float * debug)
{
  RenderParams rp;
  rp.viewSize = make_int2(m_viewSize.x, m_viewSize.y);
  const float fov = 70.0f;
  rp.fovCoef = tan(cg::grad2rad(0.5f * fov));

  point_3f target(0.5f, 0.5f, 0.5f);
  point_3f eye(2.0f, 1.5f, 1.5f);
  matrix_4f view2wld = MakeViewToWld(eye, target - eye, point_3f(0, 0, 1));
  matrix_4f wld2view;
  cg::inverse(view2wld, wld2view);
  rp.viewToWldMtx = make_float4x4(view2wld);
  rp.wldToViewMtx = make_float4x4(wld2view);
  rp.eyePos = make_float3(eye);
  rp.rootGrid = m_rootGrid;
  rp.rootData = m_rootData;

  RunTrace(rp, m_imgBuf.d_ptr(), m_debugBuf.d_ptr());
  m_imgBuf.read(0, m_imgBuf.size(), img);
  m_debugBuf.read(0, m_debugBuf.size(), debug);
}*/