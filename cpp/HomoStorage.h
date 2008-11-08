#pragma once

// #define STORAGE_VALID_CHECK

#include "assert.h"
#include "utils.h"

template <class T>
class HomoStorage
{
private:
  std::vector<T> m_data;
  int m_head;
  int m_count;

  int m_pageCapacity;
  std::vector<int> m_pages;

#ifdef STORAGE_VALID_CHECK
  std::vector<int> m_mark;
#endif


public:
  HomoStorage(int pageCapacity = 256) : m_head(-1), m_count(0), m_pageCapacity(pageCapacity)
  { 
    STATIC_ASSERT(sizeof(T) >= sizeof(int), too_small_datatype);
  }

  int getPageSize() const { return m_pageCapacity * sizeof(T); }
  int getPageCapacity() const { return m_pageCapacity; }
  int getPageNum() const { return m_pages.size(); }
  int getPageVer(int page) const { return m_pages[page]; }
  void setItemVer(int ptr, int ver)
  {
    Assert(ptr >= 0 && ptr < (int)m_data.size());
#ifdef STORAGE_VALID_CHECK
    Assert(m_mark[ptr] == 1);
#endif

    int pagesNeeded = iDivUp(m_data.size(), m_pageCapacity);
    m_pages.resize(pagesNeeded, 0);
    m_pages[ptr / m_pageCapacity] = ver;
  }

  int countPages(int ver) const { return std::count(m_pages.begin(), m_pages.end(), ver); }

  int insert()
  {
    ++m_count;
    if (m_head < 0)
    {
      T t;
      m_data.push_back(t);
#ifdef STORAGE_VALID_CHECK
      m_mark.push_back(1);
#endif
      setItemVer(m_data.size()-1, 0);
      return m_data.size()-1;
    }
    int ptr = m_head;
    m_head = *reinterpret_cast<int*>(&(m_data[m_head]));
#ifdef STORAGE_VALID_CHECK
    m_mark[ptr] = 1;
#endif
    setItemVer(ptr, 0);
    return ptr;
  }

  void erase(int ptr) 
  {
    Assert(ptr >= 0 && ptr < (int)m_data.size());
#ifdef STORAGE_VALID_CHECK
    Assert(m_mark[ptr] == 1);
#endif
    --m_count;
    *reinterpret_cast<int*>(&(m_data[ptr])) = m_head;
    m_head = ptr;
#ifdef STORAGE_VALID_CHECK
    m_mark[ptr] = 0;
#endif
  }

  int size() const {return m_data.size(); }
  int count() const { return m_count; }

  T & operator[](int i) 
  { 
    Assert(i >= 0 && i < (int)m_data.size());
#ifdef STORAGE_VALID_CHECK
    Assert(m_mark[i] == 1);
#endif
    return m_data[i]; 
  }

  const T & operator[](int i) const 
  {
    Assert(i >= 0 && i < (int)m_data.size());
#ifdef STORAGE_VALID_CHECK
    Assert(m_mark[i] == 1);
#endif
    return m_data[i]; 
  }

  const std::vector<T> & GetBuf() const { return m_data; }

  template <class Stream>
  void save(Stream & s)
  {
    write(s, m_head);
    write(s, m_count);
    write(s, m_data.size());
    s.write(reinterpret_cast<char *>(&m_data[0]), sizeof(T)*m_data.size());
  }

  template <class Stream>
  void load(Stream & s)
  {
    read(s, m_head);
    read(s, m_count);
    size_t sz(0);
    read(s, sz);
    m_data.resize(sz);
    s.read(reinterpret_cast<char *>(&m_data[0]), sizeof(T)*m_data.size());

    m_pages.clear();
    if (m_data.size() > 0)
      setItemVer(0, 0);
  }
};