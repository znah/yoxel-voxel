#pragma once

template<class ValueType>
class BrickPool
{
public:
  BrickPool(const textureReference * tex, int brickSize, point_3i poolSize) 
    : m_tex(tex)
    , cuArray(NULL)
    , m_size(poolSize)
    , m_brickSize
    , m_head(-1)
    , m_count(0)
  {
    resize(point_3i(4, 4, 4));
  }

  uchar4 CreateBrick(const ValueType * data = NULL)
  {
    uchar4 id;
    if (data != NULL)
      SetBrick(id, data);
    return id;
  }


  void SetBrick(uchar4 id, const ValueType * data)
  {

  }


private:
  void resize(point_3i size)
  {
    if (m_size == size)
      return;

    cudaChannelFormatDesc desc = cudaCreateChannelDesc<ValueType>();
    cudaArray * newArray(NULL);
    cudaExtent extent = {size.x, size.y, size.z};
    cudaMalloc3DArray(&newArray, &desc, extent);

    
    
  }

  const textureReference * tex;
  cudaArray * cuArray;

  int m_brickSize;
  std::vector<int> m_mark;
  point_3i m_size;
  int m_head;
  int m_count;
};