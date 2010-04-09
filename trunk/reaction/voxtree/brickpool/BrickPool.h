#pragma once

class CuBrickPoolManager : public noncopyable
{
public:
  struct Params
  {
    int sizeX;
    int sizeY;
    int sizeZ;
    int mappingSlotNum;
    CUdeviceptr d_mapSlotsMarkEnum;

    Params()
      : sizeX(96)
      , sizeY(96)
      , sizeZ(96)
      , mappingSlotNum(16)
      , d_mapSlotsMarkEnum(NULL)
    {}
  };

  CuBrickPoolManager(const Params & params);
  ~CuBrickPoolManager();

  int capacity() const { return m_capacity; }
  int brickCount() const { return m_brickCount; }
  
  int allocMap(int count);
  const std::vector<int> slot2slice() const { return m_slot2pool; }
  int mappedBrickCount() const { return m_mappedBrickCount; };


private:
  const Params m_params;
  int m_capacity;

  std::vector<int> m_brickMark;
  typedef std::pair<int, int> CountSlicePair;
  std::vector<CountSlicePair> m_sliceBrickCounters;
  int m_brickCount;

  std::vector<int> m_hostEnumBuf;
  int m_mappedBrickCount;
  std::vector<int> m_slot2pool;
};
