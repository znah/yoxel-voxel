#include "stdafx.h"
#include "CoralMesh.h"

CoralMesh::CoralMesh() 
: m_totalMergeCount(0)
{
}

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
  m_edges[e1] = EdgeData(fid, e2);
  m_edges[e2] = EdgeData(fid, e3);
  m_edges[e3] = EdgeData(fid, e1);
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

float CoralMesh::edgeLen2(const edge_t & e)
{
  return square_norm(m_pos[e.b] - m_pos[e.a]);
}

void CoralMesh::Grow(float mergeDist, float splitDist, const float * amounts)
{
  float mergeDist2 = mergeDist * mergeDist;
  float splitDist2 = splitDist * splitDist;
  for (int i = 0; i < m_pos.size(); ++i)
    m_pos[i] += m_normal[i] * amounts[i];
 
  int shrinkCount = 0;
  while (true)
  {
    std::vector<std::pair<float, edge_t> > toShrink;
    for (EDGE_ITER iter = m_edges.begin(); iter != m_edges.end(); ++iter)
    {
      edge_t e = iter->first;
      float len = edgeLen2(iter->first);
      if (e.a < e.b && len < mergeDist2)
        toShrink.push_back(std::make_pair(len, e));
    }
    if (toShrink.empty())
      break;
    std::sort(toShrink.begin(), toShrink.end());
    for (int i = 0; i < toShrink.size(); ++i )
    {
      if (m_edges.find(toShrink[i].second) != m_edges.end())
      {
        shrinkEdge(toShrink[i].second);
        ++shrinkCount;
        ++m_totalMergeCount;
      }
    }
    //printf(".");
    if (shrinkCount > 100)
        break;
  }

  int splitCount = 0;
  while (true)
  {
    std::vector<edge_t> toSplit;
    for (EDGE_ITER iter = m_edges.begin(); iter != m_edges.end(); ++iter)
    {
      edge_t e = iter->first;
      if (e.a < e.b && edgeLen2(iter->first) > splitDist2)
        toSplit.push_back(e);
    }
    if (toSplit.empty())
      break;
    for (int i = 0; i < toSplit.size(); ++i )
    {
      splitEdge(toSplit[i]);
      ++splitCount;
    }
    //printf(".");
    //if (splitCount > 100)
    //  break;
  }
  
  UpdateNormals();
}

void CoralMesh::splitEdge(const edge_t & e)
{
  point_3f n;
  point_3f pos = interpolateVertex(e.a, e.b, n);
  int vid = AddVert(pos);
  m_normal[vid] = n;
  splitEdgeFace(e, vid);
  splitEdgeFace(e.flip(), vid);
}

point_3f CoralMesh::interpolateVertex(int a, int b, point_3f & n)
{
  point_3f u = m_pos[b] - m_pos[a];
  float len = length(u);
  point_3f avgNorm = normalize(0.5f * (m_normal[a] + m_normal[b]));
  point_3f w = cross(u, avgNorm);
  point_3f v = normalize(cross(w, u))*len;

  point_2f n1(dot(m_normal[a], u)/len, dot(m_normal[a], v)/len);
  point_2f n2(dot(m_normal[b], u)/len, dot(m_normal[b], v)/len);

  float k1 = -n1.x / n1.y;
  float k2 = -n2.x / n2.y;

  const float x = 0.5f, x2 = x*x, x3 = x*x2;
  float y = (k1+k2)*x3 + (-2*k1-k2)*x2 + k1*x;

  point_3f interpPos = m_pos[a] + u*x + v*y;;
  point_3f avgPos = 0.5f*(m_pos[a] + m_pos[b]);

  n = avgNorm;
  // UGLY fix
  if (length(avgPos - interpPos) < len)
    return interpPos;
  else
    return avgPos;
}

void CoralMesh::splitEdgeFace(const edge_t & e, int vid)
{
  const EdgeData & edata = m_edges[e];
  int a = e.a, b = e.b, c = edata.next.b;
  setFace(edata.face, vid, b, c);
  AddFace(vid, c, a);
  m_edges.erase(e);
}

void CoralMesh::shrinkEdge(const edge_t & edge)
{
  std::vector<int> neibA, neibB;
  getAdjacentVerts(edge, neibA);
  getAdjacentVerts(edge.flip(), neibB);

  std::vector<int> holeBorder;
  edge_t shrinked;
  if (neibA.size() < neibB.size())
  {
    holeBorder = neibA;
    shrinked = edge;
  }
  else
  {
    holeBorder = neibB;
    shrinked = edge.flip();
  }

  std::sort(neibA.begin(), neibA.end());
  std::sort(neibB.begin(), neibB.end());
  std::vector<int> shared;
  std::set_intersection(neibA.begin(), neibA.end(), neibB.begin(), neibB.end(), std::back_inserter(shared));
  if (shared.size() > 2)
    return;
  
  m_pos[shrinked.b] = interpolateVertex(shrinked.a, shrinked.b, m_normal[shrinked.b]);
  for (int i = 0; i < holeBorder.size(); ++i)
    removeFace( m_edges[ edge_t(shrinked.a, holeBorder[i]) ].face );
  for (int i = 1; i < holeBorder.size() - 1; ++i)
    AddFace(shrinked.b, holeBorder[i+1], holeBorder[i]);
}

void CoralMesh::removeFace(int fid)
{
  face_t f = m_faces[fid];
  m_edges.erase(edge_t(f.a, f.b));
  m_edges.erase(edge_t(f.b, f.c));
  m_edges.erase(edge_t(f.c, f.a));
  if (fid != m_faces.size() - 1)
  {
    face_t last = m_faces.back();
    setFace(fid, last.a, last.b, last.c);
  }
  m_faces.pop_back();
}

void CoralMesh::getAdjacentVerts(const edge_t & edge, std::vector<int> & res)
{
  res.push_back(edge.b);
  for (edge_t e = m_edges[edge.flip()].next; e != edge; e = m_edges[e.flip()].next )
  {
    assert(e.valid());
    res.push_back(e.b);
    if (res.size() > 100)
    {
      printf("too large nhood!\n");
      break;
    }
  }
}