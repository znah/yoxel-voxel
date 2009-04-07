#pragma once

namespace ntree
{

typedef uchar4 ValueType;
const uchar4 DefValue = {0, 0, 0, 0};

const int GridSizePow = 2;
const int GridSize = 1 << GridSizePow;
const int GridSize3 = GridSize*GridSize*GridSize;

const int BrickSizePow = 3;
const int BrickSize = (1 << BrickSizePow) + 1;
const int BrickSize3 = BrickSize*BrickSize*BrickSize;


}