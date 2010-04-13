#pragma once

#include "DynamicSVO.h"
#include "cu_cpp.h"

class CudaSVO : public noncopyable
{
private:
  DynamicSVO * m_svo;
  int m_curVersion;

  CuVector<VoxNode> m_nodes;

  template <class T>
  void UpdatePages(CuVector<T> & buf, const HomoStorage<T> & storage);

public:
  CudaSVO();

  void SetSVO(DynamicSVO * svo);
  void Update();

  VoxNodeId GetRoot();
  VoxNode * GetNodes(int & size);
};
