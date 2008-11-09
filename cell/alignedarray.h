#pragma once

#ifdef TARGET_PPU
  #include <libmisc.h>
#endif

template <class T, int Log2Align>
class AlignedArray
{
private:
  T * m_ptr;
  int m_size;

public:
  AlignedArray() : m_ptr(0), m_size(0) {}
  explicit AlignedArray(int size) : m_ptr(0), m_size(0) { resize(size); }
  ~AlignedArray() { resize(0); }

  void resize(int size)
  {
    if (m_size == size)
      return;
    if (m_size > 0)
    {
      #ifdef TARGET_PPU
        free_align(m_ptr);
      #else
        free(m_ptr);
      #endif

      m_ptr = 0;
      m_size = 0;
    }
    if (size > 0)
    {
      #ifdef TARGET_PPU
        m_ptr = (VoxNode*)malloc_align(sizeof(T)*size, Log2Align);
      #else
        m_ptr = (VoxNode*)malloc(sizeof(T)*size);
      #endif
      m_size = size;
    }
  }

  T & operator[](int i) { return m_ptr[i]; }
  const T & operator[](int i) const { return m_ptr[i]; }
};

