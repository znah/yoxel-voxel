#pragma once

#include "DeviceBuf.h"
#include "DynamicSVO.h"

class CudaSVO : public boost::noncopyable
{
private:
  DynamicSVO * m_svo;
  int m_curVersion;

  DeviceBuf m_nodes;

  template <class T>
  void UpdatePages(DeviceBuf & buf, const HomoStorage<T> & storage);

public:
  CudaSVO();

  void SetSVO(DynamicSVO * svo);
  void Update();

  VoxNodeId GetRoot();
  void GetNodes(CUdeviceptr & ptr, int & size);
};