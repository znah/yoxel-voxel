#pragma once

namespace cg
{

struct matrix_4f
{
  float data[4][4];
};

/*matrix_4f MakeLookAtMatrix(const point_3f & eye, const point_3f & center, const point_3f & up)
{
  point_3f f = normlaize(center - eye);
  point_3f upn = normlaize(up);
  point_3f s = f ^ up;
  point_3f u = s ^ f;
  



};*/

}
