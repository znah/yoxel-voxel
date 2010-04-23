#pragma once

#include "packed_normal.h"
#include "packed_color.h"

typedef uint VoxData; // color - 16bit, normal - 16bit
typedef uint VoxChild; // VoxData or VoxNodeId
typedef uint VoxNodeInfo; //  0 .. 7   - leafFlags, 
                          //  8 .. 15  - nullFlags, 
                          // 16 .. 18 - selfChildId,  
                          // 19     - emptyFlag
typedef uint VoxNodeId;

#pragma pack(push, 4)  
struct VoxNode
{
  VoxNodeInfo flags;
  VoxNodeId   parent;
  VoxData     data;
  VoxChild    child[8];

  uint        _pad; // pad to 16 byte multiple
};
#pragma pack(pop)


inline GLOBAL_FUNC bool IsNull(VoxNodeId node) { return (node & 0x80000000) != 0; }
const VoxNodeId EmptyNode = 0x80000000;
const VoxNodeId FullNode  = 0x80000001;


inline GLOBAL_FUNC bool GetLeafFlag(VoxNodeInfo ni, int i) { return (ni & (1<<i)) != 0; }
inline GLOBAL_FUNC uchar GetLeafFlags(VoxNodeInfo ni) { return ni & 0xff; }
inline GLOBAL_FUNC void SetLeafFlag(VoxNodeInfo & ni, int i, bool v) 
{ 
  uint mask = 1<<i;
  ni = v ? (ni | mask) : (ni & ~mask);
}
inline GLOBAL_FUNC bool GetNullFlag(VoxNodeInfo ni, int i) { return (ni & (1<<(i+8))) != 0; }
inline GLOBAL_FUNC uchar GetNullFlags(VoxNodeInfo ni) { return (ni>>8) & 0xff; }
inline GLOBAL_FUNC void SetNullFlag(VoxNodeInfo & ni, int i, bool v) 
{ 
  uint mask = 1<<(i+8);
  ni = v ? (ni | mask) : (ni & ~mask);
}
inline GLOBAL_FUNC int  GetSelfChildId(VoxNodeInfo ni) { return (ni>>16) & 7; }
inline GLOBAL_FUNC void SetSelfChildId(VoxNodeInfo & ni, int v) 
{ 
  ni &= ~(7<<16);
  ni |= v<<16;
}
inline GLOBAL_FUNC bool GetEmptyFlag(VoxNodeInfo ni) { return (ni & (1<<19)) != 0; }
inline GLOBAL_FUNC void SetEmptyFlag(VoxNodeInfo & ni, bool v) 
{ 
  uint mask = 1<<19;
  ni = v ? (ni | mask) : (ni & ~mask);
}


inline GLOBAL_FUNC VoxData PackVoxData(Color16 c, Normal16 n)
{
  VoxData vd = 0;
  vd |= c;
  vd |= n << 16;
  return vd;
}

inline GLOBAL_FUNC void UnpackVoxData(VoxData vd, Color16 & c, Normal16 & n)
{
  c = vd & 0xffff;
  n = vd >> 16;
}
