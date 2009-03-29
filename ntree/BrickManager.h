#pragma once

#include "ntree/nodes.h"

template<class T>
class BrickPool
{
public:
  BrickPool(const textureReference * tex, int brickSize, point_3i poolSize) 
    : m_tex(tex)
    , m_cuArray(NULL)
    , m_extent(poolSize)
    , m_brickSize(brickSize)
    , m_head(-1)
    , m_count(0)
  {
    m_mark.resize(poolSize.x * poolSize.y * poolSize.z, -1);

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<T>();
    point_3i texSize = poolSize * brickSize;
    cudaExtent extent = make_cudaExtent(texSize.x, texSize.y, texSize.z);
    CUDA_SAFE_CALL( cudaMalloc3DArray(&m_cuArray, &desc, extent) );
    CUDA_SAFE_CALL( cudaBindTextureToArray(m_tex, m_cuArray, &desc) );
  }

  ~BrickPool()
  {
    CUDA_SAFE_CALL( cudaFreeArray(m_cuArray) );
    m_cuArray = NULL;
  }

  GPURef CreateBrick(const T * data = NULL)
  {
    Assert(m_count < (int)m_mark.size());
    
    GPURef id = m_count;
    ++m_count;

    // TODO brick release

    if (data != NULL)
      SetBrick(id, data);
    return id;
  }


  void SetBrick(GPURef id, const T * data)
  {
    //Assert();
    point_3i pos = id2pos(id);
    cudaMemcpy3DParms params = {0};
    params.srcPtr = make_cudaPitchedPtr((void*)data, sizeof(T) * m_brickSize, m_brickSize, m_brickSize);
    params.dstArray = m_cuArray;
    params.dstPos = make_cudaPos(pos.x * m_brickSize, pos.y * m_brickSize, pos.z * m_brickSize);
    params.extent = make_cudaExtent(m_brickSize, m_brickSize, m_brickSize);
    params.kind = cudaMemcpyHostToDevice;
    CUDA_SAFE_CALL( cudaMemcpy3D(&params) );
  }

private:
  GPURef pos2id(const point_3i & pos) const { return pos.x + m_extent.x * (pos.y + pos.z * m_extent.y); }
  point_3i id2pos(GPURef id) const 
  { 
    point_3i res;
    res.x = id % m_extent.x;
    id /= m_extent.x;
    res.y = id % m_extent.y;
    id /= m_extent.y;
    res.z = id % m_extent.z;
    return res;
  }

  const textureReference * m_tex;
  cudaArray * m_cuArray;

  int m_brickSize;
  std::vector<int> m_mark;
  point_3i m_extent;
  int m_head;
  int m_count;
};