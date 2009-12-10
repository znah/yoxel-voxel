#include "stdafx.h"

using namespace boost::python;

namespace numpy
{
template <class T> struct TypeTrairs {};

template <> struct TypeTrairs<float> 
{ 
  static const char * name() { return "float32"; } 
};

template <> struct TypeTrairs<int> 
{ 
  static const char * name() { return "int32"; } 
};

template <> struct TypeTrairs<unsigned int> 
{ 
  static const char * name() { return "uint32"; } 
};

}


template <class T>
T * get_array_ptr(object a)
{
  // TODO: type check
  return (float*)(int)extract<int>(a.attr("ctypes").attr("data"));
}

/*point_3i get_array_shape(object a)
{
  tuple shape = extract<tuple>(a.attr("shape"));
  int ndim = len(shape);
  if (ndim > 3)
    throw std::runtime_error("array ndim > 3");
  point_3i res(1, 1, 1);
  for (int i = 0; i < ndim; ++i)
    res[i] = extract<int>(shape[i]);
  return res;
}*/

void test(int i)
{
  printf("asdad %d\n", i);
}

BOOST_PYTHON_MODULE(_grower)
{
  def("test", test);
}