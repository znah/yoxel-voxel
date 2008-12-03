#include "stdafx.h"
#include "SVORenderer.h"

#include "trace_cu.h"

SVORenderer::SVORenderer() 
: m_dir(1, 0, 0)
, m_up(0, 0, 1)
, m_viewSize(640, 480)
, m_fov(70.0f)
{
  cudaGetTextureReference(&m_dataTexRef, "nodes_tex");
}

SVORenderer::~SVORenderer() {}

void SVORenderer::SetScene(DynamicSVO * svo) 
{ 
  m_svo.SetSVO(svo);
  
  int size(0);
  CUdeviceptr d_ptr(0);
  m_svo.GetNodes(d_ptr, size);
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<uint>();
  cudaBindTexture(0, m_dataTexRef, (const void *)d_ptr, &channelDesc, size);

  VoxStructTree tree;
  tree.root = m_svo.GetRoot();
  tree.nodes = (VoxNode*)d_ptr;
  CuSetSymbol(tree, "tree");
}

void SVORenderer::SetViewSize(int width, int height)
{
  m_viewSize = point_2i(width, height);
  m_rayDataBuf.resize(width * height);
}

inline dim3 MakeGrid(const point_2i & size, const dim3 & block)
{
  return dim3(iDivUp(size.x, block.x), iDivUp(size.y, block.y), 1);
}

void SVORenderer::Render(void * d_dstBuf)
{
  dim3 block(16, 16, 1);
  dim3 grid = MakeGrid(m_viewSize, block);

  RenderParams rp;
  rp.viewWidth = m_viewSize.x;
  rp.viewHeight = m_viewSize.x;
  rp.detailCoef;

  rp.eye = m_pos;
  rp.dir = cg::normalized(m_dir);
  rp.up = cg::normalized(m_up);
  rp.right = cg::normalized(rp.dir ^ rp.up);
  rp.lightPos = m_pos;

  CUT_CHECK_ERROR("ttt");

  Run_InitEyeRays(block, grid, rp, m_rayDataBuf.d_ptr());
  Run_Trace(block, grid, rp, m_rayDataBuf.d_ptr());
  Run_ShadeSimple(block, grid, rp, m_rayDataBuf.d_ptr(), (uchar4*)d_dstBuf);

  CUT_CHECK_ERROR("ttt");
}
