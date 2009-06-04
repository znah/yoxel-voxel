#include "stdafx.h"
#include "bittree.h"
#include <conio.h>

/*#define BOOST_TEST_MODULE BitTreeTest
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_CASE( test_test )
{
}*/

template <class T>
void LoadBuf(const char * fn, std::vector<T> & dst)
{
  std::ifstream file(fn, std::ios::binary);
  file.seekg(0, std::ios::end);
  int sz = file.tellg();
  file.seekg(0, std::ios::beg);
  Assert(sz % sizeof(T) == 0);
  dst.resize(sz / sizeof(T));
  file.read((char*)&dst[0], sz);
}


int main()
{
  std::vector<Grid> grids;
  std::vector<Brick> bricks;
  LoadBuf("grids.dat", grids);
  LoadBuf("bit_bricks.dat", bricks);

  RayTracer tracer(&bricks[0], &grids[0], (int)grids.size() - 1);
  point_3f target(0.5f, 0.5f, 0.5f);
  point_3f eye(3, 2, 1);
  tracer.trace(eye, target - eye);
  getch();
  return 0;
}
