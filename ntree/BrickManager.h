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
    cudaMalloc3DArray(&m_cuArray, &desc, extent);
    cudaBindTextureToArray(tex, m_cuArray, &desc);
  }

  ~BrickPool()
  {
    cudaFreeArray(m_cuArray);
    m_cuArray = NULL;
  }

  uchar4 CreateBrick(const T * data = NULL)
  {
    uchar4 id;
    if (data != NULL)
      SetBrick(id, data);
    return id;
  }


  void SetBrick(uchar4 id, const T * data)
  {
    cudaMemcpy3DParms params = {0};
    params.srcPtr = make_cudaPitchedPtr(data, sizeof(T) * m_brickSize, m_brickSize, m_brickSize);
    params.dstArray = m_cuArray;
    params.dstPos = make_cudaPos(id.x * m_brickSize, id.y * m_brickSize, id.z * m_brickSize);
    params.extent = make_cudaExtent(m_brickSize, m_brickSize, m_brickSize);
    params.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&params);
  }

private:
  const textureReference * m_tex;
  cudaArray * m_cuArray;

  int m_brickSize;
  std::vector<int> m_mark;
  point_3i m_extent;
  int m_head;
  int m_count;
};