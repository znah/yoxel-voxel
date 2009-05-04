#pragma once

template <class T>
class array_3d_ref
{
private:
  point_3i m_extent;
  T * m_data;

protected:

public:
  array_3d_ref() : m_data(NULL) {}
  array_3d_ref(const point_3i & extent, T * data) 
  { 
    m_extent = extent;
    m_data = data;
  }
  virtual ~array_3d_ref() {}

  virtual void assign(const point_3i & extent, T * data)
  {
    m_extent = extent;
    m_data = data;
  }

  point_3i extent() const { return m_extent; }
  int size() const {return m_extent.x * m_extent.y * m_extent.z; }

  T * data() { return m_data; }
  int ofs(const point_3i & p) const { return (p.z * m_extent.y + p.y) * m_extent.x + p.x; }

  T & operator[](const point_3i & p) { return m_data[ofs(p)]; }
  const T & operator[](const point_3i & p) const { return m_data[ofs(p)]; }
};


template <class T>
class array_3d
{
private:
  std::vector<T> m_storage;

public:
  array_3d() {}
  array_3d(const point_3i & extent, T * data)
  {
    assign(extent, data);
  }

  void resize(const point_3i & extent)
  {
    m_storage.resize(extent.x * extent.y * extent.z);
    array_3d_base::assign(extent, &m_storage[0]);
  }

  virtual void assign(const point_3i & extent, T * data)
  {
    int sz = extent.x * extent.y * extent.z;
    m_storage.assign(data, data + sz);
    array_3d_base::assign(extent, &m_storage[0]);
  }
};
