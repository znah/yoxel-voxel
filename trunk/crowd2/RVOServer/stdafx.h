#pragma once

#include <iostream>
#include <limits>
#include <queue>
#include <cassert>
#include <vector>
#include <set>

//#define  PY_ARRAY_UNIQUE_SYMBOL 1

#pragma warning( push )
#pragma warning (disable:4996; disable:4290)
#include <boost/multi_array.hpp>
#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <boost/noncopyable.hpp>
#include <boost/mpl/vector.hpp>
#include "numpy_boost.hpp"

#include <boost/scoped_ptr.hpp>
#pragma warning( pop )

#include <glm/glm.hpp>
#include <glm/gtx/noise.hpp>

/*#pragma warning (disable:4018)

#include "Box2D/Collision/b2DynamicTree.h"
#include "Box2D/Collision/Shapes/b2EdgeShape.h"
#include "Box2D/Collision/Shapes/b2PolygonShape.h"

#include "common.h"
*/

/*#pragma push_macro("_DEBUG")
#undef _DEBUG
#include <Python.h>
#pragma pop_macro("_DEBUG")*/
