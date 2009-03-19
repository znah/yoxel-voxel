#pragma once

template<class ValueType>
class BrickPool
{
public:
  BrickManager(const textureReference * tex, int brickSize) 
  {
  }

  uchar4 CreateBrick(const ValueType * data = NULL)
  {
    uchar4 id;
    if (data != NULL)
      SetBrick(id, data);
    return id;
  }


  void SetBrick(uchar4 id, const ValueType * data)
  {

  }


private:
  std::vector<int> 


};