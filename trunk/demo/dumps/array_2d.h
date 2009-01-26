#pragma once

template <class T> T min(const T & a, const T & b) { return (a < b) ? a : b; }
template <class T> T max(const T & a, const T & b) { return (a > b) ? a : b; }


template <class T>
class array_2d
{
public:
  array_2d(int sx, int sy, T * data) : m_sx(sx), m_sy(sy), m_data(data) {}
  T & operator()(int x, int y) { return m_data[y * m_sx + x]; }

  int width() const { return m_sx; }
  int height() const { return m_sy; }
                                                             
private:
  int m_sx, m_sy;
  T * m_data;
};

