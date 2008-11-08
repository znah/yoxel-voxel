#pragma once

#include "vox_node.h"
#include "alignedarray.h"


inline void reverseBytes(uint & v)
{
  char * p = (char*)&v;
  std::swap(p[0], p[3]);
  std::swap(p[1], p[2]);
}


class SVOData
{
private:
  VoxNodeId m_root;
  AlignedArray<VoxNode, 4> m_data;

public:
  SVOData() : m_root(EmptyNode) {}
 
  void Load(const char * fn)
  {
    std::ifstream input(fn, std::ios::binary);
    input >> m_root; reverseBytes(m_root);
    uint t, size;
    input >> t >> t >> size; reverseBytes(size);
    m_data.resize(size);
    input.read((char*) &m_data[0], size*sizeof(VoxNode));
    
    int intNum = size*sizeof(VoxNode) / sizeof(uint);
    uint * p = (uint*) &m_data[0];
    for (int i = 0; i < intNum; ++i)
      reverseBytes(*(p+i));
  }

  const VoxNode & operator[](VoxNodeId id) const { assert(!IsNull(id)); return m_data[id]; }
};
