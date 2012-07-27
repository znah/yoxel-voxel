#pragma once

namespace py = boost::python;

using boost::scoped_ptr;

//typedef glm::ivec2 int2;
//typedef glm::vec2 float2;


template<class T, int D>
py::object to_object(numpy_boost<T, D> & arr)
{
    return py::object(py::handle<>(py::borrowed(arr.py_ptr())));
}

template <class T>
PyObject * pyptr(T obj) { return py::object(obj).ptr(); }

