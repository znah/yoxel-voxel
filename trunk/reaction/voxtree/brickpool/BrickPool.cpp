#include "stdafx.h"
#include "BrickPool.h"

CuBrickPoolManager::CuBrickPoolManager(const Params & params)
: m_params(params)
, m_mappedBrickCount(0)
{
  const int sliceItemNum = m_params.sizeX * m_params.sizeY;

  m_capacity = m_params.sizeX * m_params.sizeY * m_params.sizeZ;
  m_brickMark.assign( m_capacity, 0 );
  for (int i = 0; i < m_params.sizeZ; ++i)
    m_sliceBrickCounters.push_back(CountSlicePair(0, i));
  m_brickCount = 0;

  m_hostEnumBuf.assign(sliceItemNum * m_params.mappingSlotNum, 0);
}

CuBrickPoolManager::~CuBrickPoolManager() {}

int CuBrickPoolManager::allocMap(int count)
{
  const int sliceItemNum = m_params.sizeX * m_params.sizeY;
  std::sort(m_sliceBrickCounters.begin(), m_sliceBrickCounters.end());
  m_slot2pool.erase(m_slot2pool.begin(), m_slot2pool.end());
  
  int allocated = 0;
  int allocSlot = 0;
  for (allocSlot = 0; allocSlot < m_params.mappingSlotNum && allocated < count; ++allocSlot)
  {
    int sliceFreeNum = sliceItemNum - m_sliceBrickCounters[allocSlot].first;
    int slice = m_sliceBrickCounters[allocSlot].second;
    if (sliceFreeNum == 0)
      break;

    int prevAllocated = allocated;

    int poolBegin = slice * sliceItemNum;
    int slotBegin = allocSlot * sliceItemNum;
    for (int i = 0; i < sliceItemNum; ++i)
    {
      int poolIdx = poolBegin + i;
      int slotIdx = slotBegin + i;
      if (allocated < count && m_brickMark[poolIdx] == 0)
      {
        m_hostEnumBuf[slotIdx] = allocated;
        m_brickMark[poolIdx] = 1;
        ++allocated;
      }
      else
        m_hostEnumBuf[slotIdx] = -1;
    }
    m_slot2pool.push_back(slice);
    m_sliceBrickCounters[allocSlot].first += allocated - prevAllocated;
  }

  m_mappedBrickCount = allocated;
  m_brickCount += allocated;
  if (m_mappedBrickCount > 0)
    CU_SAFE_CALL( cuMemcpyHtoD(m_params.d_mapSlotsMarkEnum, &m_hostEnumBuf[0], sliceItemNum * m_slot2pool.size() * sizeof(int)) );
  return m_mappedBrickCount;
}
