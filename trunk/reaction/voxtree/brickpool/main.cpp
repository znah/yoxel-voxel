#include "stdafx.h"
#include "BrickPool.h"

using namespace boost::python;

object numpy;

template <class T>
T * get_array_ptr(object a)
{
  // TODO: type check
  return (T*)(int)extract<int>(a.attr("ctypes").attr("data"));
}

int test(int a)
{
  return a*a;
}


object CuBrickPool_slot2slice(const CuBrickPoolManager & poolman)
{
  int n = poolman.slot2slice().size();
  object a = numpy.attr("zeros")(n, "int32");
  int * dst = get_array_ptr<int>(a);
  for (int i = 0; i < n; ++i)
    dst[i] = poolman.slot2slice()[i];
  return a;
}


BOOST_PYTHON_MODULE(_brickpool)
{
  def("test", test);

  {
    typedef CuBrickPoolManager::Params cl;
    class_<cl>("CuBrickPoolManager_Params")
      .def_readwrite("sizeX", &cl::sizeX)
      .def_readwrite("sizeY", &cl::sizeY)
      .def_readwrite("sizeZ", &cl::sizeZ)
      .def_readwrite("mappingSlotNum", &cl::mappingSlotNum)
      .def_readwrite("d_mapSlotsMarkEnum", &cl::d_mapSlotsMarkEnum);
  }

  {
    typedef CuBrickPoolManager cl;
    class_<cl, boost::noncopyable>("CuBrickPoolManager", init<cl::Params>())
      .add_property("capacity", &cl::capacity)
      .add_property("brickCount", &cl::brickCount)
      .add_property("mappedBrickCount", &cl::mappedBrickCount)
      .add_property("slot2slice", &CuBrickPool_slot2slice)
      .def("allocMap", &cl::allocMap);
  }

  numpy = import("numpy");
}