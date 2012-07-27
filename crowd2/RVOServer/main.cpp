#include "stdafx.h"

//#include "path_field.h"
//#include "RVO.h"
//#include "KdTree.h"

#include <boost/python/suite/indexing/vector_indexing_suite.hpp>

namespace py = boost::python;

/*
py::object py_calc_distmap(py::object py_obst) 
{
    numpy_boost<float, 2> obst_array(py_obst.ptr());
    numpy_boost<float, 2> dist_array(obst_array.shape());
    int h = obst_array.shape()[0], w = obst_array.shape()[1];
    int path_shape[3] = {h, w, 2};
    numpy_boost<float, 3> path_array(path_shape);
    
    calc_distmap(obst_array, dist_array, path_array);
    return py::make_tuple(to_object(dist_array), to_object(path_array));
} */

//extern void export_sim();
//extern void export_graph();



template <class T>
void export_vector()
{
    typedef std::vector<T> V;
    std::string s = typeid(V).name();
    py::class_<V>(s.c_str())
        .def(py::vector_indexing_suite<V>() );
}

BOOST_PYTHON_MODULE(RVOServer)
{
    import_array();

    export_vector<int>();
    //export_vector<float2>();

    //py::def("calc_distmap", py_calc_distmap);

    //export_graph();
    //export_sim();
}
