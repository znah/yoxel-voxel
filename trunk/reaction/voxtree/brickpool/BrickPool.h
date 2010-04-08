#pragma once

typedef uint8 CellType;

struct CuBrickPoolParams
{
  int sizeX;
  int sizeY;
  int sizeZ;
  int brickSize;
  int maxMappedSliceNum;

  CuBrickPoolParams()
  : sizeX(96)
  , sizeY(96)
  , sizeZ(96)
  , brickSize(5)
  , maxMappedSliceNum(16)
  {}
};

struct CuBrickPoolMapping
{
  int sliceNum;
  int mappedItemNum;
  CUdeviceptr d_mappedSlices;
  CUdeviceptr d_mappedIdxs;
  CUdeviceptr d_sliceMap2Pool;
};


class CuBrickPool : public noncopyable
{
public:
  CuBrickPool(CuBrickPoolParams params);
  ~CuBrickPool();

  int capacity() const;
  int size() const;

  CuBrickPoolMapping alloc_map(int count);
  void unmap();

private:
  const CuBrickPoolParams m_params;

  CUarray     m_poolArray;
  CUdeviceptr d_mappedSlices;
  

};