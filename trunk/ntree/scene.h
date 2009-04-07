#pragma once

#include "ntree/nodes.h"
//#include "BrickManager.h"
#include "cu_cpp.h"

class Scene
{
public:
  Scene();
  ~Scene();

  void SetTreeDepth(int depth) { m_treeDepth = depth; }
  int GetTreeDepth() const { return m_treeDepth; }

  void Load(const char * filename);
  void Save(const char * filename);

  ntree::Node * GetRoot() { return &m_root; }

  void AddVolume(cg::point_3i pos, cg::point_3i size, const ntree::ValueType * data);

  std::string GetStats();

  //ntree::ValueType TraceRay(const point_3f & p, point_3f dir);

  //void UpdateGPU();

  void SetViewSize(point_2i size);
  point_2i GetViewSize() const { return m_viewSize; }

  void Render(uchar4 * img, float * debug);
  
private:
  point_2i m_viewSize;
  CuVector<uchar4> m_imgBuf;
  CuVector<float> m_debugBuf;

  ntree::Node m_root;
  int m_treeDepth;

//  BrickPool<uchar4> m_dataPool;
//  BrickPool<uint4> m_gridPool;
//  GPURef m_rootGrid;
//  GPURef m_rootData;

//  struct NHood
//  {
//    ntree::NodePtr p[8];
//    ntree::ValueType data[8];
//  };

//  void UpdateGPURec(const NHood & nhood, GPURef & dataRef, GPURef & childRef);
//  void UploadData(const NHood & nhood, GPURef & dataRef);
//  void GetNHood(const NHood & nhood, const point_3i & p, NHood & res);
};
