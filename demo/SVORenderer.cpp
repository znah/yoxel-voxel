#include "stdafx.h"
#include "SVORenderer.h"

#include "trace_cu.h"
#include "format.h"


SVORenderer::SVORenderer() 
: m_dir(1, 0, 0)
, m_up(0, 0, 1)
, m_viewSize(640, 480)
, m_fov(70.0f)
, m_detailCoef(1.0)
, m_ditherCoef(0*1.0f/2048.0f)
, m_ssna(true)
, m_showNormals(false)
{
  cudaGetTextureReference(&m_dataTexRef, "nodes_tex");

  for (int i = 0; i < MaxLightsNum; ++i)
    m_lights[i].enabled = false;
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
  m_zBuf.resize(width * height);
  m_zBuf2.resize(width * height);
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
  rp.fovCoef = tan( 0.5f * cg::grad2rad(m_fov));
  rp.detailCoef = m_detailCoef * cg::grad2rad(m_fov / 2) / m_viewSize.x;

  rp.eyePos = make_float3(m_pos);
  
  rp.viewToWldMtx = cg::MakeViewToWld(m_pos, m_dir, m_up);
  cg::inverse(rp.viewToWldMtx, rp.wldToViewMtx);
  
  rp.specularExp = 10.0;
  rp.ambient = make_float3(0.1f);

  for (int i = 0; i < MaxLightsNum; ++i)
    rp.lights[i] = m_lights[i];

  rp.rays = m_rayDataBuf.d_ptr();
  rp.zBuf = m_zBuf.d_ptr();
  rp.zBuf2 = m_zBuf2.d_ptr();

  rp.ssna = m_ssna;
  rp.showNormals = m_showNormals;

  CuSetSymbol(rp, "rp");

  Run_Trace(make_grid2d(m_viewSize, point_2i(16, 28)));
  CUT_CHECK_ERROR("ttt");

  Run_BlurZBuf(make_grid2d(m_viewSize, point_2i(16, 16)));
  CUT_CHECK_ERROR("ttt");

  Run_ShadeSimple(make_grid2d(m_viewSize, point_2i(16, 16)), (uchar4*)d_dstBuf);
  CUT_CHECK_ERROR("ttt");
}

template <class T>
inline void WriteVec(std::string fn, const std::vector<T> & buf)
{
  std::ofstream ss(fn.c_str(), std::ios::binary);
  ss.write((const char*)&buf[0], (int)(sizeof(T)*buf.size()));
}

void SVORenderer::DumpTraceData(std::string fnbase)
{
  std::vector<RayData> buf;
  m_rayDataBuf.read(buf);
  std::vector<float> distBuf(buf.size());
  std::vector<Color32> colorBuf(buf.size());
  std::vector<point_3f> normalBuf(buf.size());

  for (size_t i = 0; i < buf.size(); ++i)
  {
    const RayData & rd = buf[i];
    //distBuf[i] = rd.t;

    if (IsNull(rd.endNode))
      continue;

    const VoxNode & node = m_svo.GetSVO()->GetNodes()[rd.endNode];
    VoxData data;
    if (rd.endNodeChild < 0)
      data = node.data;
    else
      data = node.child[rd.endNodeChild];
    
    Color16 c16;
    Normal16 n16;
    UnpackVoxData(data, c16, n16);
    colorBuf[i] = UnpackColor(c16);
    normalBuf[i] = UnpackNormal(n16);
  }

  fnbase += formatStr("_{0}x{1}") % m_viewSize.x % m_viewSize.y;
  WriteVec(fnbase + ".dist",   distBuf);
  WriteVec(fnbase + ".color",  colorBuf);
  WriteVec(fnbase + ".normal", normalBuf);
}
