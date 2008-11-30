#include "stdafx.h"
#include "SVORenderer.h"

#include "trace_cu.h"

SVORenderer::SVORenderer() 
: m_dir(1, 0, 0)
, m_up(0, 0, 1)
, m_viewSize(640, 480)
, m_fov(70.0f)
{}

SVORenderer::~SVORenderer() {}

void SVORenderer::SetViewSize(int width, int height)
{
  m_viewSize = point_2i(width, height);
  m_rayDataBuf.resize(width * height);
}


void SVORenderer::Render(void * d_dstBuf)
{

}
