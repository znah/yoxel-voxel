#pragma once

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

  uchar4 CreateBrick(const T * data = NULL)
  {
    Assert(m_count < (int)m_mark.size());
    
    int ofs = m_count;
    ++m_count;

    // TODO brick release

    uchar4 id = ofs2id(ofs);
    if (data != NULL)
      SetBrick(id, data);
    return id;
  }


  void SetBrick(uchar4 id, const T * data)
  {
    //Assert();
    cudaMemcpy3DParms params = {0};
    params.srcPtr = make_cudaPitchedPtr((void*)data, sizeof(T) * m_brickSize, m_brickSize, m_brickSize);
    params.dstArray = m_cuArray;
    params.dstPos = make_cudaPos(id.x * m_brickSize, id.y * m_brickSize, id.z * m_brickSize);
    params.extent = make_cudaExtent(m_brickSize, m_brickSize, m_brickSize);
    params.kind = cudaMemcpyHostToDevice;
    CUDA_SAFE_CALL( cudaMemcpy3D(&params) );
  }

private:
  int id2ofs(uchar4 id) const { return id.x + m_extent.x * (id.y + id.z * m_extent.y); }
  uchar4 ofs2id(int ofs) const 
  { 
    uchar4 res;
    res.x = ofs % m_extent.x;
    ofs /= m_extent.x;
    res.y = ofs % m_extent.y;
    ofs /= m_extent.y;
    res.z = ofs % m_extent.z;
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