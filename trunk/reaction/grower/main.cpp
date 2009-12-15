#include "stdafx.h"
#include "LapGrow.h"

using namespace boost::python;


template <class T>
T * get_array_ptr(object a)
{
  // TODO: type check
  return (T*)(int)extract<int>(a.attr("ctypes").attr("data"));
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

object LapGrow_FetchBlack(const LapGrow & grow)
{
  int sz = grow.size();
  object numpy = import("numpy");
  object a = numpy.attr("zeros")(make_tuple(sz, 2), "float32");
  point_2f * dst = get_array_ptr<point_2f>(a);
  const point_2f * src = grow.black();
  std::copy(src, src + sz, dst);
  return a;
};

BOOST_PYTHON_MODULE(_grower)
{
  class_<LapGrow>("LapGrow")
    .def("GrowParticle", &LapGrow::GrowParticle)
    .def("SetExponent", &LapGrow::SetExponent)
    .def("FetchBlack", LapGrow_FetchBlack);
}