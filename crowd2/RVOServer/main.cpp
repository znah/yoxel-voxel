#include "stdafx.h"
#include "common.h"
#include "RVOServer.h"

//#include "path_field.h"
//#include "RVO.h"
//#include "KdTree.h"


/*py::object py_calc_distmap(py::object py_obst) 
{
    numpy_boost<float, 2> obst_array(py_obst.ptr());
    numpy_boost<float, 2> dist_array(obst_array.shape());
    int h = obst_array.shape()[0], w = obst_array.shape()[1];
    int path_shape[3] = {h, w, 2};
    numpy_boost<float, 3> path_array(path_shape);
    
    calc_distmap(obst_array, dist_array, path_array);
    return py::make_tuple(to_object(dist_array), to_object(path_array));
}*/


//extern void export_sim();
//extern void export_graph();

void test123(float *p, int dim[2])
{
  boost::multi_array_ref<float, 2> arr(p, boost::extents[dim[0]][dim[0]]);

  for (int i = 0; i < dim[0]; ++i)
  {
  for (int j = 0; j < dim[1]; ++j)
  {
    printf("%f ", arr[i][j]);
  }
    printf("\n");
  }
}
