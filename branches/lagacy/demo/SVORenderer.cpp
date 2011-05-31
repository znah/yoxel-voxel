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
, m_accumIter(0)
, m_shadeMode(SM_SIMPLE)
, m_shuffleEnabled(false)
, m_lodLimit(16)
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
  
  ResetAccum();
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


  std::vector<int> shuffleBuf(width * height);
  for (int i = 0; i < (int)shuffleBuf.size(); ++i)
    shuffleBuf[i] = i;
  std::random_shuffle(shuffleBuf.begin(), shuffleBuf.end());
  m_shuffleBuf.resize(shuffleBuf.size());
  m_shuffleBuf.write(0, shuffleBuf.size(), &shuffleBuf[0]);
  
  m_accumBuf.resize(width * height);
  ResetAccum();
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
  rp.minNodeSize = 1.0 / (1<<m_lodLimit);

  rp.eyePos = make_float3(m_pos);

  point_3f vfwd = cg::normalized(m_dir);
  point_3f vright = cg::normalized(vfwd ^ m_up);
  point_3f vup = vright ^ vfwd;
  
  float du = tan(cg::grad2rad(m_fov / 2));
  float dv = du * m_viewSize.y / max(m_viewSize.x, 1);

  rp.dir = make_float3(vfwd);
  rp.right = make_float3(vright * du);
  rp.up = make_float3(vup * dv);
  
  rp.specularExp = 10.0;
  rp.ambient = make_float3(0.1f);

  for (int i = 0; i < MaxLightsNum; ++i)
    rp.lights[i] = m_lights[i];

  rp.ditherCoef = m_ditherCoef;
  rp.rndSeed = m_accumIter == 0 ? 0 : cg::rand((int)m_noiseBuf.size());

  rp.accumIter = m_accumIter;
  if (m_accumIter < 255)
    ++m_accumIter;

  CuSetSymbol(rp, "rp");

  Run_InitEyeRays(make_grid2d(m_viewSize, point_2i(16, 16)), m_rayDataBuf.d_ptr(), m_noiseBuf.d_ptr(),
    m_shuffleEnabled ? m_shuffleBuf.d_ptr() : NULL);
  CUT_CHECK_ERROR("ttt");

  CuTimer timer;
  timer.start();
  Run_Trace(make_grid2d(m_viewSize, point_2i(TRACE_BLOCK_X, TRACE_BLOCK_Y)), m_rayDataBuf.d_ptr());
  CUT_CHECK_ERROR("ttt");
  float traceTime = timer.stop();
  m_profStats.traceTime = (m_profStats.traceTime == 0) ? traceTime : (0.8f * m_profStats.traceTime  + 0.2f * traceTime);
  

  if (m_shadeMode == SM_SIMPLE)
    Run_ShadeSimple(make_grid2d(m_viewSize, point_2i(16, 16)), m_rayDataBuf.d_ptr(), (uchar4*)d_dstBuf, m_accumBuf.d_ptr());
  else
    Run_ShadeCounter(make_grid2d(m_viewSize, point_2i(16, 16)), m_rayDataBuf.d_ptr(), (uchar4*)d_dstBuf);
  CUT_CHECK_ERROR("ttt");
}

void SVORenderer::ResetAccum()
{
  m_accumIter = 0;
}

std::string SVORenderer::GetInfoString() const
{
  std::string res("--\n");
  res += format("resolution {0} {1}\n") % m_viewSize.x % m_viewSize.y;
  res += format("detailCoef {0}\n") % m_detailCoef;
  res += format("ditherCoef {0}\n") % m_ditherCoef;
  int svoSize = m_svo.Source()->GetNodes().getPageNum() * m_svo.Source()->GetNodes().getPageSize();
  res += format("svoSize {0}\n") % svoSize;
  res += format("shuffle {0}\n") % m_shuffleEnabled;
  res += format("prof.traceTime {0}\n") % m_profStats.traceTime;
  res += format("USE_TEXLOOKUP {0}\n") % USE_TEXLOOKUP;
  res += format("SHARED_STACK {0}\n") % SHARED_STACK;
  res += format("lodLimit {0}\n") % m_lodLimit;
  return res;
}

void SVORenderer::SaveCounters(std::string filename)
{
  std::vector<RayData> rayBuf;
  m_rayDataBuf.read(rayBuf);
  std::vector<int> counters(rayBuf.size());
  std::vector<int> enter(rayBuf.size());
  std::vector<int> exit(rayBuf.size());
  for (int i = 0; i < rayBuf.size(); ++i)
  {
    counters[i] = rayBuf[i].perfCount;
    enter[i] = rayBuf[i].enterTime;
    exit[i] = rayBuf[i].exitTime;
  }

  {
    std::ofstream file(filename.c_str(), std::ios::binary);
    file.write((char*)&counters[0], counters.size() * sizeof(counters[0]));
  }
  {
    std::ofstream file(("enter_" + filename).c_str(), std::ios::binary);
    file.write((char*)&enter[0], enter.size() * sizeof(enter[0]));
  }
  {
    std::ofstream file(("exit_" + filename).c_str(), std::ios::binary);
    file.write((char*)&exit[0], exit.size() * sizeof(exit[0]));
  }

}