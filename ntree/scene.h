#pragma once

#include "ntree/nodes.h"
#include "BrickManager.h"
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

  ntree::NodePtr GetRoot() { return m_root; }

  void AddVolume(cg::point_3i pos, cg::point_3i size, const ntree::ValueType * data);

  std::string GetStats();

  ntree::ValueType TraceRay(const point_3f & p, point_3f dir);

  void UpdateGPU();

  void SetViewSize(point_2i size) { m_viewSize = size; }
  point_2i GetViewSize() const { return m_viewSize; }

  void Render(uchar4 * img);
  
private:
  point_2i m_viewSize;
  CuVector<uchar4> m_imgBuf;

  ntree::NodePtr m_root;
  int m_treeDepth;

  BrickPool<uchar4> m_dataPool;
  BrickPool<uint4> m_gridPool;

  struct NHood
  {
    ntree::NodePtr p[8];
    ntree::ValueType data[8];
  };

  void UpdateGPURec(const NHood & nhood, GPURef & dataRef, GPURef & childRef);
  void UploadData(const NHood & nhood, GPURef & dataRef);
  void GetNHood(const NHood & nhood, const point_3i & p, NHood & res);
};
