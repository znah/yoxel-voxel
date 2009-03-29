#pragma once

#include "ntree/nodes.h"
#include "BrickManager.h"

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
  
private:
  ntree::NodePtr m_root;
  int m_treeDepth;

  BrickPool<uchar4> m_dataPool;
  BrickPool<uint4> m_gridPool;

  struct NHood
  {
    ntree::NodePtr p[8];
  };

  void UpdateGPURec(const NHood & nhood, GPURef & dataRef, GPURef & childRef);
  void UploadData(const NHood & nhood, GPURef gpuRef);
  void GetNHood(NHood nhood, const point_3i & p);
};
