#include "stdafx.h"

using namespace boost::python;

object numpy;


int test(int a)
{
  return a*a;
}


BOOST_PYTHON_MODULE(_brickpool)
{
  def("test", test);

  numpy = import("numpy");
}