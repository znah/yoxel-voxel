#pragma once

namespace ntree
{

typedef uchar4 ValueType;
const uchar4 DefValue = {0, 0, 0, 0};

const int GridSizePow = 3;
const int GridSize = 1 << GridSizePow;
const int GridSize3 = GridSize*GridSize*GridSize;

const int BrickBoundary = 0;
const int BrickSizePow = 2;
const int BrickSize = (1 << BrickSizePow) + BrickBoundary;
const int BrickSize3 = BrickSize*BrickSize*BrickSize;


}