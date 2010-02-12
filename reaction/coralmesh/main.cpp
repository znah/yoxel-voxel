#include "stdafx.h"
#include "CoralMesh.h"

using namespace boost::python;

object numpy;

template <class T>
T * get_array_ptr(object a)
{
  // TODO: type check
  return (T*)(int)extract<int>(a.attr("ctypes").attr("data"));
}

void test(int i)
{
  printf("asdad %d\n", i);
}

int CoralMesh_AddVert(CoralMesh & mesh, float x, float y, float z)
{
  return mesh.AddVert(point_3f(x, y, z));
}

object CoralMesh_GetPositions(CoralMesh & mesh)
{
  int n = mesh.GetVertNum();
  object a = numpy.attr("zeros")(make_tuple(n, 3), "float32");
  point_3f * dst = get_array_ptr<point_3f>(a);
  const point_3f * src = mesh.GetPositions();
  std::copy(src, src + n, dst);
  return a;
}

object CoralMesh_GetNormals(CoralMesh & mesh)
{
  int n = mesh.GetVertNum();
  object a = numpy.attr("zeros")(make_tuple(n, 3), "float32");
  point_3f * dst = get_array_ptr<point_3f>(a);
  const point_3f * src = mesh.GetNormals();
  std::copy(src, src + n, dst);
  return a;
}

object CoralMesh_GetFaces(CoralMesh & mesh)
{
  int n = mesh.GetFaceNum();
  object a = numpy.attr("zeros")(make_tuple(n, 3), "int32");
  face_t * dst = get_array_ptr<face_t>(a);
  const face_t * src = mesh.GetFaces();
  std::copy(src, src + n, dst);
  return a;
}

void CoralMesh_Grow(CoralMesh & mesh, float mergeDist, float splitDist, object amounts)
{
  const float * amountsPtr = get_array_ptr<float>(amounts);
  mesh.Grow(mergeDist, splitDist, amountsPtr);
}


BOOST_PYTHON_MODULE(_coralmesh)
{
  class_<CoralMesh>("CoralMesh")
    .def("add_face", &CoralMesh::AddFace)
    .def("add_vert", CoralMesh_AddVert)
    .def("get_vert_num", &CoralMesh::GetVertNum)
    .def("get_positions", CoralMesh_GetPositions)
    .def("get_normals", CoralMesh_GetNormals)
    .def("get_faces", CoralMesh_GetFaces)
    .def("update_normals", &CoralMesh::UpdateNormals)
    .def("grow", CoralMesh_Grow);

  def("test", test);

  numpy = import("numpy");
}