#include "stdafx.h"
#include "scene.h"
#include <conio.h>


template <class T>
void LoadBuf(const char * fn, std::vector<T> & res)
{
  std::ifstream file(fn, std::ios::binary);
  file.read((char*)&res[0], (int)(res.size() * sizeof(T)));
}

int main()
{
  Scene scene;

  const int n = 256;
  std::vector<uchar> raw(n*n*n);
  LoadBuf("..\\data\\bonsai.raw", raw);

  std::vector<uchar4> data(raw.size());
  for (size_t i = 0; i < raw.size(); ++i)
  {
    int s = raw[i];
    s = (s - 32) * 16;  // [32; 48)
    s = cg::bound(s, 0, 255);
    uchar4 d = {s, s, s, s};
    data[i] = d;
  }
  
  scene.AddVolume( point_3i(100, 100, 100), point_3i(n, n, n), &data[0] );
  std::cout << scene.GetStats() << std::endl;

  getch();

  return 0;
}