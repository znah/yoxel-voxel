#pragma once

#include "vox_node.h"
#include "alignedarray.h"
#include "utils.h"


inline void reverseBytes(uint & v)
{
  char * p = (char*)&v;
  std::swap(p[0], p[3]);
  std::swap(p[1], p[2]);
}

#ifdef TARGET_PPU
  #define REV_BYTES(v) ( reverseBytes(v) )
#else
  #define REV_BYTES(v)
#endif


class SVOData
{
private:
  VoxNodeId m_root;
  AlignedArray<VoxNode, 4> m_data;

public:
  SVOData() : m_root(EmptyNode) {}
 
  void Load(const char * fn)
  {
    std::cout << "Loading " << fn << " ... ";

    std::ifstream input(fn, std::ios::binary);
    read(input, m_root); REV_BYTES(m_root);
    uint t, size;
    read(input, t);
    read(input, t);
    read(input, size); REV_BYTES(size);
    m_data.resize(size);
    input.read((char*) &m_data[0], size*sizeof(VoxNode));
    
    int intNum = size*sizeof(VoxNode) / sizeof(uint);
    uint * p = (uint*) &m_data[0];
    for (int i = 0; i < intNum; ++i)
      REV_BYTES(*(p+i));

    std::cout << "DONE" << std::endl;
  }

  VoxNodeId GetRoot() const { return m_root; }

  const VoxNode & operator[](VoxNodeId id) const { assert(!IsNull(id)); return m_data[id]; }
};
