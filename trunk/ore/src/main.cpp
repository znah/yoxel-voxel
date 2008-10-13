#include <boost/python.hpp>
#include <iostream>
#include <stdexcept>

#include "DynamicSVO.h"
#include "CudaSVO.h"
#include "builders.h"
#include "pybuf.h"


namespace py = boost::python;

void p_test(py::object obj)
{
  std::cout << py::extract<int>(obj[2]) << std::endl;
}

void ex_test()
{
  throw std::logic_error("sdfs sdfs");
}

py::tuple DynamicSVO_ExportStructTree(DynamicSVO * bld)
{
  VoxNodeId root = bld->GetRoot();
  py::tuple res = py::make_tuple(root, 
    py::handle<>(ToPyBuffer( bld->GetNodes().GetBuf() )), 
    py::handle<>(ToPyBuffer( bld->GetLeafs().GetBuf() ))
    );
  return res;
}

py::tuple CudaSVO_GetData(CudaSVO * svo)
{
  VoxNodeId root = svo->GetRoot();

  CUdeviceptr ptr(0);
  int size(0);

  svo->GetNodes(ptr, size);
  py::tuple nodes = py::make_tuple(ptr, size);
  svo->GetLeafs(ptr, size);
  py::tuple leafs = py::make_tuple(ptr, size);

  py::tuple res = py::make_tuple(root, nodes, leafs);
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

SphereSource * MakeShpereSource(int radius, py::object color, bool inverted)
{
  uchar r = py::extract<uchar>(color[0]);
  uchar g = py::extract<uchar>(color[1]);
  uchar b = py::extract<uchar>(color[2]);

  return new SphereSource(radius, make_uchar4(r, g, b, 255), inverted);
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

    def("p_test", p_test);
    def("ex_test", ex_test);

    enum_<BuildMode>("BuildMode")
      .value("GROW", BUILD_MODE_GROW)
      .value("CLEAR", BUILD_MODE_CLEAR);

    class_<VoxelSource>("VoxelSource", no_init)
      .def("GetSize", &VoxelSource::GetSize)
      .def("GetPivot", &VoxelSource::GetPivot);

    class_<RawSource, bases<VoxelSource> >("RawSource", no_init);
    def("MakeRawSource", MakeRawSource, py::return_value_policy<py::manage_new_object>());
    def("MakeShpereSource", MakeShpereSource, py::return_value_policy<py::manage_new_object>());

    class_<SphereSource, bases<VoxelSource> >("SphereSource", no_init);

    class_<DynamicSVO>("DynamicSVO")
      .def("BuildRange", &DynamicSVO::BuildRange)
      .def("ExportStructTree", &DynamicSVO_ExportStructTree)
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
