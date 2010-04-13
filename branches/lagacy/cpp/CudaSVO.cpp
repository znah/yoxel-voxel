#include "stdafx.h"

#include "CudaSVO.h"

CudaSVO::CudaSVO() : m_svo(NULL), m_curVersion(0)
{
}

void CudaSVO::SetSVO(DynamicSVO * svo)
{
  m_svo = svo;
  m_curVersion = 0;
  Update();
}

template <class T>
void CudaSVO::UpdatePages(DeviceBuf & buf, const HomoStorage<T> & storage)
{
  int pageNum = storage.getPageNum();
  int neededBufSize = pageNum * storage.getPageSize();
  int srcSize = storage.size() * sizeof(T);
  const char * srcPtr = (const char *)&storage[0];
  if (buf.size() < neededBufSize)
  {
    buf.resize(neededBufSize * 3 / 2);
    cuMemcpyHtoD(buf.ptr(), srcPtr, srcSize);
  }
  else
  {
    for (int page = 0; page < pageNum; ++page)
    {
      if (storage.getPageVer(page) <= m_curVersion)
        continue;

      int start = page * storage.getPageSize();
      int stop = std::min(start + storage.getPageSize(), srcSize);
      cuMemcpyHtoD(buf.ptr() + start, srcPtr + start, stop - start);
    }
  }
}

void CudaSVO::Update()
{
  UpdatePages(m_nodes, m_svo->GetNodes());
  m_curVersion = m_svo->GetCurVersion();
}

VoxNodeId CudaSVO::GetRoot()
{
  Assert(m_svo != NULL);
  return m_svo->GetRoot();
}

void CudaSVO::GetNodes(CUdeviceptr & ptr, int & size)
{
  ptr = m_nodes.ptr();
  size = m_nodes.size();
}
