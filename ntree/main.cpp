#include "stdafx.h"
#include "scene.h"
#include "ntree/ntree.h"

#include <conio.h>


using std::cout;
using std::endl;


template <class T>
void LoadBuf(const char * fn, std::vector<T> & dst)
{
  std::ifstream file(fn, std::ios::binary);
  file.read((char*)&dst[0], (int)(dst.size() * sizeof(T)));
}

template <class T>
void SaveBuf(const char * fn, const std::vector<T> & src)
{
  std::ofstream file(fn, std::ios::binary);
  file.write((char*)&src[0], (int)(src.size() * sizeof(T)));
}


int main()
{
  Scene scene;
  
  scene.SetTreeDepth(8);
  const int n = 256;
  const char * sceneName = "..\\data\\bonsai32.raw";
  //scene.SetTreeDepth(3);
  //const int n = 32;
  //const char * sceneName = "..\\data\\bucky.raw";

  std::vector<uchar> raw(n*n*n);
  LoadBuf(sceneName, raw);

  std::vector<uchar4> data(raw.size());
  for (size_t i = 0; i < raw.size(); ++i)
  {
    int s = raw[i];
    uchar4 d = {192, 128, 128, s};
    data[i] = d;
  }

  
  cout << "adding tree" << endl;
  scene.AddVolume( point_3i(0, 0, 0), point_3i(n, n, n), &data[0] );
  cout << scene.GetStats() << endl;

  cout << "ready" << endl;

  _getch();
  return 0;
}