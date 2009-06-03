#include "stdafx.h"
#include "bittree.h"

/*#define BOOST_TEST_MODULE BitTreeTest
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE( test_test )
{
}*/

int main()
{
  RayTracer tracer(NULL, NULL, 0);
  tracer.trace(point_3f(0.0, 0.5, 0.5), point_3f(1.0, 0.0, 0.0));
  
  return 0;
}
