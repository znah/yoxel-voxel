#pragma once

template <class Stream, class T>
inline void write(Stream & s, const T & data)
{
  s.write( reinterpret_cast<const char *>(&data), sizeof(data) );
}

template <class Stream, class T>
inline void read(Stream & s, T & data)
{
  s.read( reinterpret_cast<char *>(&data), sizeof(data) );
}

inline int iDivUp(int a, int b)
{
  int r = a / b;
  return (r*b < a) ? r+1 : r;
}
