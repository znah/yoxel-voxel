#pragma once

namespace cg 
{

inline float determinant ( float m00, float m01, float m02, float m03,
                           float m10, float m11, float m12, float m13,
                           float m20, float m21, float m22, float m23, 
                           float m30, float m31, float m32, float m33 ) 
{
  return  m00*m11*m22*m33 + m00*m12*m23*m31 + m00*m13*m21*m32
          +m01*m10*m23*m32 + m01*m12*m20*m33 + m01*m13*m22*m30
          +m02*m10*m21*m33 + m02*m11*m23*m30 + m02*m13*m20*m31
          +m03*m10*m22*m31 + m03*m11*m20*m32 + m03*m12*m21*m30
          -m00*m11*m23*m32 - m00*m12*m21*m33 - m00*m13*m22*m31
          -m01*m10*m22*m33 - m01*m12*m23*m30 - m01*m13*m20*m32
          -m02*m10*m23*m31 - m02*m11*m20*m33 - m02*m13*m21*m30
          -m03*m10*m21*m32 - m03*m11*m22*m30 - m03*m12*m20*m31 ;
} 

inline float determinant ( const matrix_4f & m ) 
{
  return determinant ( m(0,0), m(0,1), m(0,2), m(0,3),
                       m(1,0), m(1,1), m(1,2), m(1,3),
                       m(2,0), m(2,1), m(2,2), m(2,3),
                       m(3,0), m(3,1), m(3,2), m(3,3) ) ;  
}

inline bool inverse ( const matrix_4f & m, matrix_4f & res ) 
{
  float det = determinant(m);
  if (cg::eq_zero(det))
      return false;

  float detInv = 1.0f / det;

  float m00 = m(0,0), m01 = m(0,1), m02 = m(0,2), m03 = m(0,3),
        m10 = m(1,0), m11 = m(1,1), m12 = m(1,2), m13 = m(1,3),
        m20 = m(2,0), m21 = m(2,1), m22 = m(2,2), m23 = m(2,3),
        m30 = m(3,0), m31 = m(3,1), m32 = m(3,2), m33 = m(3,3);

  res(0,0) = (m11*m22*m33 + m12*m23*m31 + m13*m21*m32
              -m11*m23*m32 - m12*m21*m33 - m13*m22*m31) * detInv;

  res(0,1) = (m01*m23*m32 + m02*m21*m33 + m03*m22*m31
              -m01*m22*m33 - m02*m23*m31 - m03*m21*m32) * detInv;

  res(0,2) = (m01*m12*m33 + m02*m13*m31 + m03*m11*m32
              -m01*m13*m32 - m02*m11*m33 - m03*m12*m31) * detInv;

  res(0,3) = (m01*m13*m22 + m02*m11*m23 + m03*m12*m21
              -m01*m12*m23 - m02*m13*m21 - m03*m11*m22) * detInv;

  res(1,0) = (m10*m23*m32 + m12*m20*m33 + m13*m22*m30
              -m10*m22*m33 - m12*m23*m30 - m13*m20*m32) * detInv;

  res(1,1) = (m00*m22*m33 + m02*m23*m30 + m03*m20*m32
              -m00*m23*m32 - m02*m20*m33 - m03*m22*m30) * detInv;

  res(1,2) = (m00*m13*m32 + m02*m10*m33 + m03*m12*m30
              -m00*m12*m33 - m02*m13*m30 - m03*m10*m32) * detInv;

  res(1,3) = (m00*m12*m23 + m02*m13*m20 + m03*m10*m22
              -m00*m13*m22 - m02*m10*m23 - m03*m12*m20) * detInv;

  res(2,0) = (m10*m21*m33 + m11*m23*m30 + m13*m20*m31
              -m10*m23*m31 - m11*m20*m33 - m13*m21*m30) * detInv;

  res(2,1) = (m00*m23*m31 + m01*m20*m33 + m03*m21*m30
              -m00*m21*m33 - m01*m23*m30 - m03*m20*m31) * detInv;

  res(2,2) = (m00*m11*m33 + m01*m13*m30 + m03*m10*m31
              -m00*m13*m31 - m01*m10*m33 - m03*m11*m30) * detInv;

  res(2,3) = (m00*m13*m21 + m01*m10*m23 + m03*m11*m20
              -m00*m11*m23 - m01*m13*m20 - m03*m10*m21) * detInv;

  res(3,0) = (m10*m22*m31 + m11*m20*m32 + m12*m21*m30
              -m10*m21*m32 - m11*m22*m30 - m12*m20*m31) * detInv;

  res(3,1) = (m00*m21*m32 + m01*m22*m30 + m02*m20*m31
              -m00*m22*m31 - m01*m20*m32 - m02*m21*m30) * detInv;

  res(3,2) = (m00*m12*m31 + m01*m10*m32 + m02*m11*m30
              -m00*m11*m32 - m01*m12*m30 - m02*m10*m31) * detInv;

  res(3,3) = (m00*m11*m22 + m01*m12*m20 + m02*m10*m21
              -m00*m12*m21 - m01*m10*m22 - m02*m11*m20) * detInv;


  return true;
}


inline matrix_4f MakeViewToWld(const point_3f & eye, const point_3f & dir, const point_3f & up)
{
  point_3f fwd = normalized(dir);
  point_3f right = normalized(fwd ^ up);
  point_3f vup = right ^ fwd;

  matrix_4f res;
  res.setcol(0, right, 0);
  res.setcol(1, vup,   0);
  res.setcol(2, -fwd,  0);
  res.setcol(3, eye,   1);
  return res;
};

}