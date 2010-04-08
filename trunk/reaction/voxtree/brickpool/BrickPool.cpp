#include "stdafx.h"
#include "BrickPool.h"

CuBrickPool::CuBrickPool(CuBrickPoolParams params)
: m_params(params)
, m_poolArray(NULL)
, d_mappedSlices(NULL)
{
  const int brickSize = m_params.brickSize;
  const int brickSize3 = brickSize * brickSize * brickSize;

  CUDA_ARRAY3D_DESCRIPTOR decs;
  decs.Width  = m_params.sizeX * brickSize;
  decs.Height = m_params.sizeY * brickSize;
  decs.Depth  = m_params.sizeZ * brickSize;
  decs.Format = CU_AD_FORMAT_UNSIGNED_INT8;
  decs.NumChannels = 1;
  decs.Flags = 0;
  cuArray3DCreate(&m_poolArray, &decs);
  assert(m_poolArray != NULL);

  cuMemAlloc(&d_mappedSlices, m_params.sizeX * m_params.sizeY * brickSize3 * m_params.maxMappedSliceNum);
  assert(d_mappedSlices != NULL);

}

CuBrickPool::~CuBrickPool()
{
  if (m_poolArray != NULL)
  { 
    cuArrayDestroy(m_poolArray);
    m_poolArray = NULL;
  }
  if (d_mappedSlices != NULL)
  {
    cuMemFree(d_mappedSlices);
    d_mappedSlices = NULL;
  }
}

CuBrickPoolMapping CuBrickPool::alloc_map(int count)
{
  CuBrickPoolMapping mapping;
  mapping.sliceNum = 0;

  return mapping;

}

void CuBrickPool::unmap()
{

}
