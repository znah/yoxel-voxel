#include "stdafx.h"
#include "VoxelScene.h"

void save(array_3d_ref<uint8> & a, const char * fn)
{
  std::ofstream file(fn, std::ios::binary);
  write(file, a.extent());
  file.write((char *)a.data(), a.size() * sizeof(uint8));
}

BOOST_AUTO_TEST_CASE( VoxelScene_test )
{
  VoxelScene scene(512, 1.0f);
  offset_geom::Sphere sph(50.0f);
  offset_geom::Translate tsph(sph);
  tsph.ofs = point_3f(100, 100, 100);
  scene.AddSurface(tsph);
  tsph.ofs = point_3f(160, 100, 100);
  scene.SubSurface(tsph);

  BOOST_TEST_MESSAGE("SceneStat: " << scene.GatherStat());

  array_3d<uint8> buf;
  range_3f fetchRng(point_3i(0, 0, 100), point_3i(200, 200, 100));
  scene.FetchAlphaRange(fetchRng, buf);
  BOOST_CHECK_EQUAL(buf[point_3i(100, 100, 100)], 255);
  BOOST_CHECK_EQUAL(buf[point_3i(120, 100, 100)], 0);

  save(buf, "VoxelScene_test_slice.dat");
}