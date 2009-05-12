#include "stdafx.h"
#include "offset_geom.h"

using namespace offset_geom;

BOOST_AUTO_TEST_CASE( offset_geom_test )
{
  const float eps = 1e-7f;
  Sphere sph(2.0f);
  BOOST_CHECK_CLOSE( sph(point_3f(0, 0, 0)),  2.0f, eps );
  BOOST_CHECK_CLOSE( sph(point_3f(0, -2, 0)), 0.0f, eps );
  BOOST_CHECK_CLOSE( sph(point_3f(2, 2, 2)), 2.0f - sqrt(12.0f), eps );

  Translate<Sphere> tsph(sph, point_3f(0, 0, 2));
  BOOST_CHECK_CLOSE( tsph(point_3f(0, 0, 2)), 2.0f, eps );
  BOOST_CHECK_CLOSE( tsph(point_3f(0, -2, 2)), 0.0f, eps );
}