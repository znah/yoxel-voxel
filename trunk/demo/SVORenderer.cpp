#include "stdafx.h"
#include "SVORenderer.h"

#include "trace_cu.h"

SVORenderer::SVORenderer() 
: m_dir(1, 0, 0)
, m_up(0, 0, 1)
, m_viewSize(640, 480)
, m_fov(70.0f)
, m_detailCoef(1.0)
, m_ditherCoef(1.0f/2048.0f)
{
  cudaGetTextureReference(&m_dataTexRef, "nodes_tex");
}

SVORenderer::~SVORenderer() {}

void SVORenderer::SetScene(DynamicSVO * svo) 
{ 
  m_svo.SetSVO(svo);
  UpdateSVO();
}

void SVORenderer::UpdateSVO()
{ 
  m_svo.Update(); 

  int size(0);
  VoxNode * d_ptr = m_svo.GetNodes(size);
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint>();
  cudaBindTexture(0, m_dataTexRef, (const void *)d_ptr, &channelDesc, size);
  CUT_CHECK_ERROR("ttt");

  VoxStructTree tree;
  tree.root = m_svo.GetRoot();
  tree.nodes = (VoxNode*)d_ptr;
  CuSetSymbol(tree, "tree");
}


void SVORenderer::SetViewSize(int width, int height)
{
  m_viewSize = point_2i(width, height);
  m_rayDataBuf.resize(width * height);
  
  std::vector<float> noiseBuf(3*width * height);
  for (size_t i = 0; i < noiseBuf.size(); ++i)
    noiseBuf[i] = cg::rand(1.0f)-0.5f;
  
  m_noiseBuf.resize(noiseBuf.size());
  m_noiseBuf.write(0, noiseBuf.size(), &noiseBuf[0]);


}

inline dim3 MakeGrid(const point_2i & size, const dim3 & block)
{
  return dim3(iDivUp(size.x, block.x), iDivUp(size.y, block.y), 1);
}

void SVORenderer::Render(void * d_dstBuf)
{
  dim3 block(16, 28, 1);
  dim3 grid = MakeGrid(m_viewSize, block);


  RenderParams rp;
  rp.viewWidth = m_viewSize.x;
  rp.viewHeight = m_viewSize.y;
  rp.detailCoef = m_detailCoef * cg::grad2rad(m_fov / 2) / m_viewSize.x;

  rp.eye = m_pos;

  point_3f vfwd = cg::normalized(m_dir);
  point_3f vright = cg::normalized(vfwd ^ m_up);
  point_3f vup = vright ^ vfwd;
  
  float du = tan(cg::grad2rad(m_fov / 2));
  float dv = du * m_viewSize.y / max(m_viewSize.x, 1);

  rp.dir = vfwd;
  rp.right = vright * du;
  rp.up = vup * dv;
  
  rp.lightPos = m_pos;
  
  rp.ditherCoef = m_ditherCoef;
  rp.rndSeed = 0;//cg::rand((int)m_noiseBuf.size());

  Run_InitEyeRays(grid, block, rp, m_rayDataBuf.d_ptr(), m_noiseBuf.d_ptr());
  Run_Trace(grid, block, rp, m_rayDataBuf.d_ptr());
  Run_ShadeSimple(grid, block, rp, m_rayDataBuf.d_ptr(), (uchar4*)d_dstBuf);
}
