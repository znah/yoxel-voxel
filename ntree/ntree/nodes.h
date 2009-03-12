#pragma once

namespace ntree
{

typedef uchar4 ValueType;


class Node
{
  int size;
  ValueType * data;
  Node      * children;
};


}