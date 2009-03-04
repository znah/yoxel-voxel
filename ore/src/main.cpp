#include "stdafx.h"

#include <boost/python.hpp>

#include "DynamicSVO.h"
//#include "CudaSVO.h"
#include "builders.h"
#include "pybuf.h"


namespace py = boost::python;


/*py::tuple CudaSVO_GetData(CudaSVO * svo)
{
  VoxNodeId root = svo->GetRoot();

  CUdeviceptr ptr(0);
  int size(0);

  svo->GetNodes(ptr, size);
  py::tuple nodes = py::make_tuple(ptr, size);

  py::tuple res = py::make_tuple(root, nodes);
  return res;
}*/

inline Color32 extractColor32(py::object obj)
{
  Color32 res;
  for (int i = 0; i < 3; ++i)
    res[i] = py::extract<uchar>(obj[i]);
  res[3] = 255;
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

  if (size.x*size.y*size.z*4 != len)
    throw std::logic_error("incorrect data buffer size");

  return new RawSource(size, (const Color32 *)colorsPtr, (const Normal32 *)normalsPtr);
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

SphereSource * MakeSphereSource(int radius, py::object color, bool inverted)
{
  return new SphereSource(radius, extractColor32(color), inverted);
}

void IsoSource_SetColor(IsoSource * dst, py::object color)
{
  dst->SetColor(extractColor32(color));
}

py::list DynamicSVO_GetNodeCountByLevel(DynamicSVO * svo)
{
  std::vector<int> res = svo->GetNodeCountByLevel();
  py::list lst;
  for (int i = 0; i < res.size(); ++i)
    lst.append(res[i]);
  return lst;
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


    enum_<BuildMode>("BuildMode")
      .value("GROW", BUILD_MODE_GROW)
      .value("CLEAR", BUILD_MODE_CLEAR);

    class_<VoxelSource>("VoxelSource", no_init)
      .def("GetSize", &VoxelSource::GetSize)
      .def("GetPivot", &VoxelSource::GetPivot);

    class_<RawSource, bases<VoxelSource> >("RawSource", no_init);
    def("MakeRawSource", MakeRawSource, py::return_value_policy<py::manage_new_object>());

    class_<SphereSource, bases<VoxelSource> >("SphereSource", no_init);
    def("MakeSphereSource", MakeSphereSource, py::return_value_policy<py::manage_new_object>());

    class_<IsoSource, bases<VoxelSource> >("IsoSource", no_init)
      .def("SetIsoLevel", &IsoSource::SetIsoLevel)
      .def("SetInside", &IsoSource::SetInside)
      .def("SetColor", IsoSource_SetColor);
    def("MakeIsoSource", MakeIsoSource, py::return_value_policy<py::manage_new_object>());

    class_<DynamicSVO>("DynamicSVO")
      .def("BuildRange", &DynamicSVO::BuildRange)
      .def("Save", &DynamicSVO::Save)
      .def("Load", &DynamicSVO::Load)
      .def("TraceRay", &DynamicSVO::TraceRay)
      .add_property("nodecount", &DynamicSVO::GetNodeCount)
      .def("CountChangedPages", &DynamicSVO::CountChangedPages)
      .def("CountTransfrerSize", &DynamicSVO::CountTransfrerSize)
      .def("GetNodeCountByLevel1", &DynamicSVO_GetNodeCountByLevel);

    /*class_<CudaSVO, boost::noncopyable>("CudaSVO")
      .def("SetSVO", &CudaSVO::SetSVO)
      .def("Update", &CudaSVO::Update)
      .def("GetData", &CudaSVO_GetData);*/
}
