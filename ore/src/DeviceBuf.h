#pragma once

#include <boost/noncopyable.hpp>
#include <cuda.h>


class DeviceBuf : public boost::noncopyable
{
private:
  CUdeviceptr m_ptr;
  int m_size;

public:
  DeviceBuf(int size = 0) : m_ptr(0), m_size(0) { resize(size); }
  ~DeviceBuf() { resize(0); }

  void resize(int size) {
    if (m_size == size)
      return;

    if (m_ptr != 0) 
    {
      cuMemFree(m_ptr);
      m_ptr = 0;
    }

    if (size > 0)
    {
      cuMemAlloc(&m_ptr, size);
    }
    m_size = size;
  }

  CUdeviceptr ptr() { return m_ptr; }
  int size() const { return m_size; }
};