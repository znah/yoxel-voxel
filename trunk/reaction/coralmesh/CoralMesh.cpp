#include "stdafx.h"
#include "CoralMesh.h"

int CoralMesh::AddVert(const point_3f & p)
{
  m_pos.push_back(p);
  m_normal.push_back(point_3f(0, 0, 0));
  return m_pos.size() - 1;
}

int CoralMesh::AddFace(int a, int b, int c)
{
  int fid = m_faces.size();
  m_faces.push_back(face_t());
  setFace(fid, a, b, c);
  return fid;
}

void CoralMesh::setFace(int fid, int a, int b, int c)
{
  m_faces[fid] = face_t(a, b, c);
  edge_t e1(a, b), e2(b, c), e3(c, a);
  m_edgeNext[e1] = e2;
  m_edgeNext[e2] = e3;
  m_edgeNext[e3] = e1;
  m_edgeFace[e1] = fid;
  m_edgeFace[e2] = fid;
  m_edgeFace[e3] = fid;
}

void CoralMesh::UpdateNormals()
{
  m_normal.assign(m_pos.size(), point_3f(0, 0, 0));
  for (int i = 0; i < m_faces.size(); ++i)
  {
    const face_t & f = m_faces[i];
    point_3f v0 = m_pos[f.a];
    point_3f v1 = m_pos[f.b];
    point_3f v2 = m_pos[f.c];
    point_3f n = cross(v1 - v0, v2 - v0);
    m_normal[f.a] += n;
    m_normal[f.b] += n;
    m_normal[f.c] += n;
  }
  for (int i = 0; i < m_normal.size(); ++i)
    m_normal[i] = normalize(m_normal[i]);
}

float CoralMesh::edgeLen(const edge_t & e)
{
  return length(m_pos[e.b] - m_pos[e.a]);
}

void CoralMesh::Grow(float mergeDist, float splitDist, const float * amounts)
{
  for (int i = 0; i < m_pos.size(); ++i)
    m_pos[i] += m_normal[i] * amounts[i];
  
  int shrinkCount = 0;
  while (true)
  {
    std::vector<edge_t> toShrink;
    for (EDGE_ITER iter = m_edgeNext.begin(); iter != m_edgeNext.end(); ++iter)
    {
      edge_t e = iter->first;
      if (e.a < e.b && edgeLen(iter->first) < mergeDist)
        toShrink.push_back(e);
    }
    if (toShrink.empty())
      break;
    for (int i = 0; i < toShrink.size(); ++i )
      if (m_edgeNext.find(toShrink[i]) != m_edgeNext.end())
      {
        shrinkEdge(toShrink[i]);
        ++shrinkCount;
      }
  }

  int splitCount = 0;
  while (true)
  {
    std::vector<edge_t> toSplit;
    for (EDGE_ITER iter = m_edgeNext.begin(); iter != m_edgeNext.end(); ++iter)
    {
      edge_t e = iter->first;
      if (e.a < e.b && edgeLen(iter->first) > splitDist)
        toSplit.push_back(e);
    }
    if (toSplit.empty())
      break;
    for (int i = 0; i < toSplit.size(); ++i )
    {
      splitEdge(toSplit[i]);
      ++splitCount;
    }
  }

  UpdateNormals();
  //printf("splits: %d,  shrinks: %d\n", splitCount, shrinkCount);
}

void CoralMesh::splitEdge(const edge_t & e)
{
  point_3f pos = interpolateVertex(e.a, e.b);
  int vid = AddVert(pos);
  splitEdgeFace(e, vid);
  splitEdgeFace(e.flip(), vid);
}

point_3f CoralMesh::interpolateVertex(int a, int b)
{
  return 0.5f * (m_pos[a] + m_pos[b]);
}

void CoralMesh::removeEdge(const edge_t & e)
{
  m_edgeNext.erase(e);
  m_edgeFace.erase(e);
}

void CoralMesh::splitEdgeFace(const edge_t & e, int vid)
{
  int a = e.a, b = e.b, c = m_edgeNext[e].b;
  setFace(m_edgeFace[e], vid, b, c);
  AddFace(vid, c, a);
  removeEdge(e);
}

void CoralMesh::shrinkEdge(const edge_t & edge)
{
  m_pos[edge.b] = interpolateVertex(edge.a, edge.b);
  std::vector<int> holeBorder;
  holeBorder.push_back(edge.b);
  for (edge_t e = m_edgeNext[edge.flip()]; e != edge; e = m_edgeNext[e.flip()] )
  {
    assert(e.valid());
    holeBorder.push_back(e.b);
  }
  for (int i = 0; i < holeBorder.size(); ++i)
    removeFace( m_edgeFace[ edge_t(edge.a, holeBorder[i]) ] );
  for (int i = 1; i < holeBorder.size() - 1; ++i)
    AddFace(edge.b, holeBorder[i+1], holeBorder[i]);
}

void CoralMesh::removeFace(int fid)
{
  face_t f = m_faces[fid];
  removeEdge(edge_t(f.a, f.b));
  removeEdge(edge_t(f.b, f.c));
  removeEdge(edge_t(f.c, f.a));
  if (fid != m_faces.size() - 1)
  {
    face_t last = m_faces.back();
    setFace(fid, last.a, last.b, last.c);
  }
  m_faces.pop_back();
}