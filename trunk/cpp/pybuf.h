#pragma once

#include "Python.h"
#include <vector>

template <class T>
inline PyObject * ToPyBuffer(const T * ptr, int count)
{
  PyObject * pybuf = PyBuffer_New( count*sizeof(T) );
  int len;
  T * dst;
  PyObject_AsWriteBuffer(pybuf, (void**)&dst, &len);
  std::copy(ptr, ptr+count, dst);
  return pybuf;
}


template <class T>
inline PyObject * ToPyBuffer(const std::vector<T> & src)
{
  return ToPyBuffer(&src[0], src.size());
}

