#include "stdafx.h"
#include "CudaSVO_rt.h"

CudaSVO::CudaSVO() : m_svo(NULL), m_curVersion(0)
{
}

void CudaSVO::SetSVO(DynamicSVO * svo)
{
  m_svo = svo;
  m_curVersion = 0;
  Update();
}

DynamicSVO * CudaSVO::GetSVO()
{
  return m_svo;
}

template <class T>
void CudaSVO::UpdatePages(CuVector<T> & buf, const HomoStorage<T> & storage)
{
  int pageNum = storage.getPageNum();
  size_t neededBufSize = pageNum * storage.getPageCapacity();
  int srcSize = storage.size();
  const T * srcPtr = &storage[0];
  if (buf.size() < neededBufSize)
  {
    buf.resize(neededBufSize * 3 / 2);
    //cuMemcpyHtoD(buf.ptr(), srcPtr, srcSize);
    buf.write(0, srcSize, srcPtr);
  }
  else
  {
    for (int page = 0; page < pageNum; ++page)
    {
      if (storage.getPageVer(page) <= m_curVersion)
        continue;

      int start = page * storage.getPageCapacity();
      int stop = std::min(start + storage.getPageCapacity(), srcSize);
      buf.write(start, stop-start, srcPtr + start);
      //cuMemcpyHtoD(buf.ptr() + start, srcPtr + start, stop - start);
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

VoxNode * CudaSVO::GetNodes(int & size)
{
  size = (int)m_nodes.size()*sizeof(VoxNode);
  return m_nodes.d_ptr();
}
