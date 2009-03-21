#include "stdafx.h"
#include "scene.h"

#include <conio.h>
#include <Magick++.h>


using std::cout;
using std::endl;


template <class T>
void LoadBuf(const char * fn, std::vector<T> & res)
{
  std::ifstream file(fn, std::ios::binary);
  file.read((char*)&res[0], (int)(res.size() * sizeof(T)));
}

void RenderImage(Scene & scene, const char * file, point_2i size)
{

  point_3f target(0.5f, 0.5f, 0.5f);
  point_3f eye(2.0f, 1.5f, 1.5f);
  matrix_4f view2wld = MakeViewToWld(eye, target - eye, point_3f(0, 0, 1));
  
  const float fov = 45.0f;
  float fovCoef = tan(cg::grad2rad(fov/2));
  float dx = 2*fovCoef / size.x;
  
  std::vector<uchar4> buf(size.x * size.y);
  for (int y = 0; y != size.y; ++y)
    for (int x = 0; x != size.x; ++x)
    {
      point_3f dir(dx * (x - size.x / 2), dx * (y - size.y / 2), -1.0f);
      normalize(dir);
      dir = view2wld * dir - eye;
      buf[y * size.x + x] = scene.TraceRay(eye, dir);
    }
  Magick::Image img(size.x, size.y, "rgba", Magick::CharPixel, &buf[0]);
  img.write(file);
}

int main()
{
  Scene scene;
  scene.SetTreeDepth(3);

  const int n = 32;
  std::vector<uchar> raw(n*n*n);
  LoadBuf("..\\data\\bucky.raw", raw);
  //LoadBuf("..\\data\\bonsai32.raw", raw);

  std::vector<uchar4> data(raw.size());
  for (size_t i = 0; i < raw.size(); ++i)
  {
    int s = raw[i];
    /*s = (s - 32) * 16;  // [32; 48)
    s = cg::bound(s, 0, 255);*/
    uchar4 d = {192, 128, 128, s};
    data[i] = d;
  }
  
  cout << "adding tree" << endl;
  scene.AddVolume( point_3i(0, 0, 0), point_3i(n, n, n), &data[0] );
  cout << scene.GetStats() << endl;
  RenderImage(scene, "test.jpg", point_2i(400, 300));
  cout << "ready" << endl;

  _getch();
  return 0;
}