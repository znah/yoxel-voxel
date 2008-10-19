#include <boost/python.hpp>
#include <iostream>
#include <stdexcept>

#include "DynamicSVO.h"
#include "CudaSVO.h"
#include "builders.h"
#include "pybuf.h"


namespace py = boost::python;


struct rgba
{
  uchar r, g, b, a;

  rgba(uchar r_, uchar g_, uchar b_, uchar a_) 
    : r(r_), g(g_), b(b_), a(a_) 
  {}

  rgba(uchar r_, uchar g_, uchar b_) 
    : r(r_), g(g_), b(b_), a(255) 
  {}

  rgba(py::object obj)
  {
    r = py::extract<uchar>(obj[0]);
    g = py::extract<uchar>(obj[1]);
    b = py::extract<uchar>(obj[2]);
    a = py::extract<uchar>(obj[3]);
  }

  operator uchar4() { return make_uchar4(r, g, b, a); }
};


py::tuple CudaSVO_GetData(CudaSVO * svo)
{
  VoxNodeId root = svo->GetRoot();

  CUdeviceptr ptr(0);
  int size(0);

  svo->GetNodes(ptr, size);
  py::tuple nodes = py::make_tuple(ptr, size);

  py::tuple res = py::make_tuple(root, nodes);
  return res;
}

RawSource * MakeRawSource(const cg::point_3i & size, py::object colors, py::object normals)
{
  const void * colorsPtr;
  const void * normalsPtr;
  Py_ssize_t len;
  if (PyObject_AsReadBuffer(colors.ptr(), &colorsPtr, &len))
    throw py::error_already_set();
  if (PyObject_AsReadBuffer(normals.ptr(), &normalsPtr, &len))
    throw py::error_already_set();

  if (size.x*size.y*size.z*sizeof(uchar4) != len)
    throw std::logic_error("incorrect data buffer size");

  return new RawSource(size, (const uchar4 *)colorsPtr, (const char4 *)normalsPtr);
}

IsoSource * MakeIsoSource(const cg::point_3i & size, py::object data)
{
  const void * ptr;
  Py_ssize_t len;
  if (PyObject_AsReadBuffer(data.ptr(), &ptr, &len))
    throw py::error_already_set();

  if (size.x*size.y*size.z*sizeof(uchar) != len)
    throw std::logic_error("incorrect data buffer size");

  return new IsoSource(size, (const uchar *)ptr);
}



BOOST_PYTHON_MODULE(_ore)
{
    using namespace boost::python;
    using namespace cg;

    class_<point_3i>("point_3i", init<int, int, int>())
        .def_readwrite("x", &point_3i::x)
        .def_readwrite("y", &point_3i::y)
        .def_readwrite("z", &point_3i::z);

    class_<point_3f>("point_3f", init<float, float, float>())
      .def_readwrite("x", &point_3f::x)
      .def_readwrite("y", &point_3f::y)
      .def_readwrite("z", &point_3f::z);

    class_<uchar4>("uchar4", no_init);
    class_<rgba>("rgba", init<uchar, uchar, uchar, uchar>())
      .def(init<py::object>())
      .def(init<uchar, uchar, uchar>());
    implicitly_convertible<rgba, uchar4>();


    enum_<BuildMode>("BuildMode")
      .value("GROW", BUILD_MODE_GROW)
      .value("CLEAR", BUILD_MODE_CLEAR);

    class_<VoxelSource>("VoxelSource", no_init)
      .def("GetSize", &VoxelSource::GetSize)
      .def("GetPivot", &VoxelSource::GetPivot);

    class_<RawSource, bases<VoxelSource> >("RawSource", no_init);
    def("MakeRawSource", MakeRawSource, py::return_value_policy<py::manage_new_object>());

    class_<SphereSource, bases<VoxelSource> >("SphereSource", init<int, uchar4, bool>());

    class_<IsoSource, bases<VoxelSource> >("IsoSource", no_init)
      .def("SetIsoLevel", &IsoSource::SetIsoLevel)
      .def("SetInside", &IsoSource::SetInside)
      .def("SetColor", &IsoSource::SetColor);
    def("MakeIsoSource", MakeIsoSource, py::return_value_policy<py::manage_new_object>());

    class_<DynamicSVO>("DynamicSVO")
      .def("BuildRange", &DynamicSVO::BuildRange)
      .def("Save", &DynamicSVO::Save)
      .def("Load", &DynamicSVO::Load)
      .def("TraceRay", &DynamicSVO::TraceRay)
      .add_property("nodecount", &DynamicSVO::GetNodeCount)
      .def("CountChangedPages", &DynamicSVO::CountChangedPages)
      .def("CountTransfrerSize", &DynamicSVO::CountTransfrerSize);

    class_<CudaSVO, boost::noncopyable>("CudaSVO")
      .def("SetSVO", &CudaSVO::SetSVO)
      .def("Update", &CudaSVO::Update)
      .def("GetData", &CudaSVO_GetData);
}
