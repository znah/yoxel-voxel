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
  builder.build(m_root, 1 << (m_treeDepth*2 + 2));
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

/*void Scene::UpdateGPU()
{
  NHood nhood;
  for (int i = 0; i < 8; ++i)
  {
    nhood.p[i] = (i == 0) ? m_root : NULL;
    nhood.data[i] = make_uchar4(0, 0, 0, 0);
  }
  
  UpdateGPURec(nhood, m_rootData, m_rootGrid);
}

inline uint ofsInNode(const point_3i & p)
{
  return (p.z * NodeSize + p.y) * NodeSize + p.x;
}

void Scene::GetNHood(const NHood & nhood, const point_3i & p, NHood & res)
{
  for (int i = 0; i < 8; ++i)
  {
    point_3i ofs(i & 1, (i>>1)&1, (i>>2)&1);
    point_3i np = p + ofs;
    point_3i fp = np / NodeSize;
    np %= NodeSize;
    int nid = fp.x + (fp.y<<1) + (fp.z<<2);
    NodePtr snode = nhood.p[nid];
    int sofs = ofsInNode(np);
    if (snode == NULL)
    {
      res.p[i] = NULL;
      res.data[i] = nhood.data[nid];
    }
    else if (snode->child == NULL)
    {
      res.p[i] = NULL;
      res.data[i] = snode->data[sofs];
    }
    else
      res.p[i] = snode->child[sofs];
  }
}

struct GPUChildData
{
  GPURef data;
  GPURef child;
  uint res1;
  uint res2;
};

void Scene::UpdateGPURec(const NHood & nhood, GPURef & dataRef, GPURef & childRef)
{
  STATIC_ASSERT( sizeof(GPUChildData) == sizeof(uint4), GPUChildData_size_error );

  if (nhood.p[0] == NULL)
  {
    dataRef = GPUNull;
    childRef = GPUNull;
    return;
  }

  UploadData(nhood, dataRef);

  if (nhood.p[0]->child == NULL)
  {
    childRef = GPUNull;
    return;
  }
    
  GPUChildData childBuf[NodeSize3];
  GPUChildData * dst = childBuf;

  for (int z = 0; z < NodeSize; ++z)
  for (int y = 0; y < NodeSize; ++y)
  for (int x = 0; x < NodeSize; ++x)
  {
    point_3i p(x, y, z);
    NHood sub;
    GetNHood(nhood, p, sub);
    UpdateGPURec(sub, dst->data, dst->child);
    dst->res1 = 0;
    dst->res2 = 0;
    ++dst;
  }

  childRef = m_gridPool.CreateBrick((uint4*)childBuf);
}*/

/*void Scene::UploadData(const NHood & nhood, GPURef & dataRef)
{
  const int size = NodeSize + 1;
  const int size3 = size * size * size;
  ValueType data[size3];
  std::fill(data, data + size3, make_uchar4(0, 0, 0, 0));
  ValueType * dst = data;
  for (int z = 0; z <= NodeSize; ++z)
  for (int y = 0; y <= NodeSize; ++y)
  for (int x = 0; x <= NodeSize; ++x)
  {
    int fx = (x == NodeSize ? 1 : 0);
    int fy = (y == NodeSize ? 1 : 0);
    int fz = (z == NodeSize ? 1 : 0);
    int nid = fx + (fy<<1) + (fz<<2);
    NodePtr snode = nhood.p[nid];
    int sofs = 0;
    if (fx) sofs += x;
    if (fy) sofs += y * NodeSize;
    if (fz) sofs += z * NodeSize * NodeSize;

    if (snode != NULL)
    {
      Assert(snode->data != NULL);
      *dst = snode->data[sofs];
    }
    else
      *dst = nhood.data[nid];
    
    ++dst;
  }
  dataRef = m_dataPool.CreateBrick(data);
}*/

void Scene::SetViewSize(point_2i size) 
{ 
  m_viewSize = size;
  m_imgBuf.resize(size.x * size.y);
  m_debugBuf.resize(size.x * size.y);
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