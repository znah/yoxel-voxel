#pragma once

#include <math.h>
#include <stdlib.h>
#include <limits>

//#include <common/meta.h>

#include "scalar_traits.h"

#undef min
#undef max

namespace cg
{

#pragma pack ( push, 1 )

  template<class T>  struct scalar_traits;
  template <> struct scalar_traits<float> 
  {
    static float epsilon() {  return 1e-8f; }
  };

  template <> struct scalar_traits<double> 
  {
    static double epsilon() {  return 1e-10; }
  };

  template <> struct scalar_traits<int> 
  {
    static int epsilon() {  return 0; }
  };

  // глобальная (пока что) используемая при сравнении вещественных чисел точность
  //const double epsilon = scalar_traits<double>::epsilon();
  template < class Scalar > Scalar epsilon() { return scalar_traits< Scalar >::epsilon( ); }

  inline void compiler_error_workaround()
  {
     epsilon<float> () ; 
     epsilon<int>   () ; 
     epsilon<double>() ; 
  }

  // число pi
  const double pi = 3.14159265358979323846; 


  template <typename Scalar> Scalar mod ( Scalar x, Scalar y ) { return x % y ; }
  inline                     double mod ( double x, double y ) { return fmod(x,y) ; }
  inline                     float  mod ( float  x, float  y ) { return fmodf(x,y) ; }

  
  // перевод градусы в радианы
  inline double grad2rad(double grad) { return grad * pi / 180.0; }
  inline float  grad2rad(float  grad) { return grad * float( pi / 180.0f ); }
  inline double grad2rad(int    grad) { return grad * pi / 180.0; }
  inline double grad2rad()            { return pi / 180.0; }
  
  // перевод из радиан в градусы
  inline double rad2grad(double rad)  { return rad * 180.0 / pi; }
  inline float  rad2grad(float  rad)  { return rad * float( 180.0 / pi ); }
  inline double rad2grad(int    rad)  { return rad * 180.0 / pi; }
  inline double rad2grad()            { return 180.0 / pi; }
  
  
  // возводит x в квадрат
  template <class T> inline T sqr(T x) { return x * x; }
  
  // равномерное распределение на отрезке [0, max)
  inline double rand(double max) { return max * ::rand() / (double)RAND_MAX; }
  inline float  rand(float max)  { return max * ::rand() / (float) RAND_MAX; }
  inline int    rand(int max)    { return ::rand() % max; }
  
  // равномерное распределение на отрезке [-max, max)
  inline double symmetric_rand(double max) { return rand(max * 2.0)  - max; }
  inline float  symmetric_rand(float max)  { return rand(max * 2.0f) - max; }
  inline int    symmetric_rand(int max)    { return rand(max * 2)    - max; }

  inline bool is_inf(double x) { return !( -1e30 < x && x < 1e30 ); }
  
  // определяет принадлежность x отрезку [0,1]
  inline bool between01(double x) { return 0 <= x && x <= 1; }
  
  // определяет принадлежность полуоткрытому отрезку [0;1)
  inline bool between01so(double x) { return 0 <= x && x < 1; }
  
  inline bool between01eps(double x, double eps=epsilon< double >( )) { return -eps <= x && x - 1 <= eps; }

  // Линейная комбинация: ta + (1-t)b
  template <class T> T blend(T const & a, T const & b, double t)
  {
      return t*a + (1-t)*b;
  }

  template <class T> T blend(T const & a, T const & b, float t)
  {
      return t*a + (1-t)*b;
  }

  // определяет знак числа
  inline int sign (int    x)  { return x < 0 ? -1 : x > 0 ? 1 : 0; }
  inline int sign (float  x)  { return x < 0 ? -1 : x > 0 ? 1 : 0; }
  inline int sign (double x)  { return x < 0 ? -1 : x > 0 ? 1 : 0; }
  
  // определяет, имеют ли x и y строго разные знаки
  inline bool different_signs(double x, double y) 
  {   return x * y < 0; }
  
  inline int round(double x) { return x > 0 ? int(x + .5) : int(x - .5);  }
  inline int round(float  x) { return x > 0 ? int(x + .5) : int(x - .5);  }

  inline double round(double x, double step)
  {
      return step * round(x / step);
  }

  // сравнение двух вещественных чисел с точностью epsilon

  // fuzzy greater equal
  inline bool ge (float a,  float b,  float  eps = epsilon< float >( ) ) { return a - b >= - eps; }
  inline bool ge (double a, double b, double eps = epsilon< double >( )) { return a - b >= - eps; }
  
  // fuzzy less equal
  inline bool le (float a,  float b,  float  eps = epsilon< float >( ) ) { return a - b <= + eps; }
  inline bool le (double a, double b, double eps = epsilon< double >( )) { return a - b <= + eps; }

  // fuzzy equal
  inline bool eq (float a,  float b,  float  eps = epsilon< float >( )) { return ge(a,b,eps) && le(a,b,eps); }
  inline bool eq (float a,  int   b,  float  eps = epsilon< float >( )) { return ge(a,float(b),eps) && le(a,float(b),eps); }
  inline bool eq (double a,  double b,  double  eps = epsilon< double >( )) { return ge(a,b,eps) && le(a,b,eps); }
  inline bool eq (double a,  int    b,  double  eps = epsilon< double >( )) { return ge(a,double(b),eps) && le(a,double(b),eps); }
  inline bool eq (float  a,  int    b,  double  eps = epsilon< double >( )) { return ge(double(a),double(b),eps) && le(double(a),double(b),eps); }

  // fuzzy equality to 0
  inline bool eq_zero (float  a, float  eps = epsilon< float >( ) ) { return eq(a, 0.f, eps); }
  inline bool eq_zero (double a, double eps = epsilon< double >( )) { return eq(a, 0. , eps); }
  inline bool eq_zero (int a, double eps = epsilon< int >( )) { return eq(a, 0. , eps); }
  
  // если x "близко" к 0, делаем его 0
  template < class Scalar > inline Scalar adjust(Scalar x, Scalar e = epsilon< Scalar >( )) { return eq(x, (Scalar) 0,e) ? 0 : x; }
  
  inline bool is_closer(double x, double step, double dist = epsilon< double >( ))
  {
      return eq(x, round(x, step), dist);
  }

  // альтернатива обычному floor(double)
  inline int nfloor(double f) {
    return f > 0 ? int(f) : int(f - 0.9999999999);
  }
  
  inline int64 nfloor64(double f) {
    return f > 0 ? int64(f) : int64(f - 0.9999999999);
  }
  
  inline int floor(double f) { return nfloor(f); }
  
  inline int ceil(double f) {
    return f > 0 ? int(f + 1) : int(f + 0.000000001);
  }
  
  //  using std::max;
  //  using std::min;
  
  template <class T> void make_min(T &to_be_min, T x) { if (to_be_min > x) to_be_min = x; }
  template <class T> void make_max(T &to_be_max, T x) { if (to_be_max < x) to_be_max = x; }
  
  template <typename T, typename D>
      void make_min ( T & to_be_min, T x, D & assign_if, D const & y )
  {
      if ( to_be_min > x )
      {
          to_be_min = x;
          assign_if = y;
      }
  }

  template <typename T, typename D>
      void make_max ( T & to_be_max, T x, D & assign_if, D const & y )
  {
      if ( to_be_max < x )
      {
          to_be_max = x;
          assign_if = y;
      }
  }

  template <class T> bool make_min_ret(T &to_be_min, T x) { if (to_be_min > x) { to_be_min = x; return true; } return false; }
  template <class T> bool make_max_ret(T &to_be_max, T x) { if (to_be_max < x) { to_be_max = x; return true; } return false; }
  
  inline int min(int a, int b) { return a < b ? a : b; }
  inline int max(int a, int b) { return a > b ? a : b; }
  
  inline unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }
  inline unsigned max(unsigned a, unsigned b) { return a > b ? a : b; }
  
  inline unsigned long min(unsigned long a, unsigned long b) { return a < b ? a : b; }
  inline unsigned long max(unsigned long a, unsigned long b) { return a > b ? a : b; }

  inline int64 min(int64 a, int64 b) { return a < b ? a : b; }
  inline int64 max(int64 a, int64 b) { return a > b ? a : b; }
  
  inline uint64 min(uint64 a, uint64 b) { return a < b ? a : b; }
  inline uint64 max(uint64 a, uint64 b) { return a > b ? a : b; }

  inline double min(double a, double b) { return a < b ? a : b; }
  inline double max(double a, double b) { return a > b ? a : b; }

  inline float min(float a, float b) { return a < b ? a : b; }
  inline float max(float a, float b) { return a > b ? a : b; }

  template <class T> T max(T a, T b, T c) { return max(a, max(b,c)); }
  template <class T> T min(T a, T b, T c) { return min(a, min(b,c)); }
  
  template <class T> T max(T a, T b, T c, T d) { return max(max(a,b),max(c,d)); }
  template <class T> T min(T a, T b, T c, T d) { return min(min(a,b),min(c,d)); }
  
  const int    INT_ETERNITY         = std::numeric_limits<int   >::max();
  const double FLOAT_ETERNITY       = std::numeric_limits<double>::max();
  const double FLOAT_FLOAT_ETERNITY = std::numeric_limits<float >::max();
  const double SENSOR_ETERNITY      = 1e10;
  
  inline bool is_pow2( long x )
  {
    if ( x <= 0 )
      return false;
    return ( (x & (x - 1)) == 0 );
  }
  
  
  inline int log2i( long x )
  {
      if ( x <= 0 )
      return -1;
    int res = 0;
    while ( x >>= 1 )
      res++;
    return res;
  }
  
/*  template <class T, class D = T>
    struct Lerp
  {
    typedef 
       typename meta::_if< meta::_is_integral< D >, T, D >::type 
       tform_type;

    // линейное преобразование [x0,x1] --> [y0,y1]
    Lerp(T x0, T x1, D y0, D y1)
      : K_(cg::eq_zero(x1 - x0) ? tform_type() * T(0.) : ( y1 - y0 ) * ( T(1) / (x1 - x0) ))
      , D_( y0 - K_ * x0 )
    {}
    
    D operator() (T x) const {
      return ( D ) ( K_*x + D_ );
    }
    
  private:
    tform_type K_;
    tform_type D_;
  };
  
  template <class T> inline Lerp<T> lerp(T x0, T x1, T y0, T y1) 
  {
    return Lerp<T>(x0,x1,y0,y1);
  }

  template <class T, class D> inline Lerp<T,D> lerp_d(T x0, T x1, D y0, D y1) 
  {
    return Lerp<T,D>(x0,x1,y0,y1);
  }*/

  template <class V, class S> inline V slerp(S x0, S x1, V y0, V y1, S x) 
  {
    S t = (x - x0) / (x1 - x0);
    S s = (3 - 2 * t) * t * t;

    return y0 * (1 - s) + y1 * s;
  }

  // three-point lerp
  template <class T, class D = T>
    struct Lerp3
  {
    // линейное преобразование [x0,x1] --> [y0,y1]
    Lerp3(T x0, T x1, T x2, D y0, D y1, D y2)
      : K1_(cg::eq_zero(x1 - x0) ? D() * T(0.) : (y1 - y0) * ( T(1) / (x1 - x0) ) )
      , D1_(y0 - K1_*x0)
      , K2_(cg::eq_zero(x2 - x1) ? D() * T(0.) : (y2 - y1) * ( T(1) / (x2 - x1) ) )
      , D2_(y1 - K2_*x1)
      , xx_(x1)
    {}
    
    D operator() (T x) const {
      return x < xx_ 
          ?  x*K1_ + D1_
          :  x*K2_ + D2_;
    }
    
  private:
    T    xx_;
    D    K1_,D1_;
    D    K2_,D2_;
  };
  
  template <class T> inline Lerp3<T> lerp3(T x0, T x1, T x2,  T y0, T y1, T y2) 
  {
    return Lerp3<T>(x0,x1,x2, y0,y1,y2);
  }

  // делает out1 = min(in1,in2); out2 = max(in1,in2);
  template <class T>
    inline void sort2(T in1, T in2, T &out1, T &out2)
  {
    if (in1 < in2) { out1 = in1; out2 = in2; }
    else           { out1 = in2; out2 = in1; }
  }

  template <class T>
      void sort2(T & v1, T & v2)
  {
      if (v1 > v2)
          std::swap(v1, v2);
  }
  
  /*template <class T, class D = T>
    struct Clamp
  {
    Clamp(T x0, T x1, D y0, D y1) 
      :   x0(x0), x1(x1), y0(y0), y1(y1)
    {}
    
    D operator () (T x) const {
      return 
        x <= x0 ? y0 :
      x >= x1 ? y1 :
      Lerp<T,D>(x0, x1, y0, y1)(x);
    }
    
  private:
    T   x0, x1;
    D   y0, y1;
  };
  
  template <class T> inline Clamp<T> clamp(T x0, T x1, T y0, T y1)
  {
    return Clamp<T>(x0, x1, y0, y1);
  }

  template <class T, class D> inline Clamp<T,D> clamp_d(T x0, T x1, D y0, D y1) 
  {
    return Clamp<T,D>(x0,x1,y0,y1);
  }*/

  template <class V, class S> inline V sclamp(S x0, S x1, V y0, V y1, S x) 
  {
      if (x <= x0) return y0;
      if (x >= x1) return y1;
      return slerp<V, S>(x0, x1, y0, y1, x);
  }

  // non-linear clamp (three-point clamp)
  template <class T, class D = T>
    struct Clamp3
  {
    Clamp3(T x0, T x1, T x2,  D y0, D y1, D y2) 
      :   x0(x0), x1(x1), x2(x2),  y0(y0), y1(y1), y2(y2)
    {}
    
    D operator () (T x) const {
      return 
        x <= x0 ? y0 :
        x >= x2 ? y2 :
        Lerp3<T,D>(x0, x1, x2, y0, y1, y2)(x);
    }
    
  private:
    T   x0, x1, x2;
    D   y0, y1, y2;
  };
  
  template <class T> inline Clamp3<T> clamp3(T x0, T x1, T x2,  T y0, T y1, T y2)
  {
    return Clamp3<T>(x0, x1, x2,  y0, y1, y2);
  }

  template < class point_type >
    struct BoundPoint
  {

     BoundPoint( const point_type & xy, const point_type & XY )
        : xy_( xy )
        , XY_( XY )
     {}

     point_type operator ( ) ( point_type result ) const
     {
        // проверим что не выходим за нижние значения
        cg::make_max( result.x, xy_.x );
        cg::make_max( result.y, xy_.y );

        // проверим что не выходим за верхние значения
        cg::make_min( result.x, XY_.x );
        cg::make_min( result.y, XY_.y );

        return result;
     }

  private:
     point_type const xy_;
     point_type const XY_;
  };

  template< class T> inline 
      BoundPoint<T> bound_point( const T & vmin, const T & vmax)
     {
          return BoundPoint<T>( vmin, vmax );
     }

  template< class T >
    struct Bound
  {
      Bound(T vmin, T vmax) 
          : vmin(vmin)
          , vmax(vmax)
      {}

      T operator()(T val)
      {
          if ( val < vmin ) return vmin;
          if ( val > vmax ) return vmax;
          return val;
      }

  private:
      T vmin, vmax;
  };

  template< class T> inline 
      T bound(T x, T vmin, T vmax)
  {
      return x < vmin ? vmin :
        x > vmax ? vmax : x;
  }
  
  template< class T> inline 
      Bound<T> bound(T vmin, T vmax)
     {
          return Bound<T>(vmin, vmax);
     }

  /* Next & previous indexes for closed (cyclic) arrays */  
  int inline prev(int index, int size) 
  {
    return (index == 0) ? size - 1 : index - 1;
  }
  
  int inline next(int index, int size)
  {
    return (index == size - 1) ? 0 : index + 1;
  }
  
  // приведение произвольной величины к диапазону [0, 360)
  template < class T >
     inline T norm360 ( T x ) 
  {
     if ( x >= 360 )
          x = T( fmod( x, 360 ) ); 
     else if ( x < 0 ) 
               x = T( fmod( x, 360 ) ) + T(360) ; 
     return x ; 
  }

  // приведение произвольной величины к диапазону [-180, 180)
  template < class T >
      inline T norm180 ( T x ) 
  {
     x = norm360(x) ; 
     if ( x >= 180 ) 
          x -= 360 ; 
     return x ; 
  }

  // приведение произвольной величины к диапазону [0, 2*Pi)
  inline double norm_2pi ( double x ) 
  {
     return x - 2*pi * floor(x / (2*pi));
  }

  // приведение произвольной величины к диапазону [-Pi, Pi)
  inline double norm_pi ( double x ) 
  {
     x = norm_2pi(x) ; 
     if ( x >= pi ) 
          x -= 2*pi ; 
     return x ; 
  }

#pragma pack ( pop )

  inline double limit ( double x, double min, double max ) 
  {
    return x > max ? max : x < min ? min : x ; 
  }    

  template <class T>
      struct Limit
  {
      Limit(T min, T max) : min_(min), max_(max) {}

      T operator () (T x) const
      {
          return x > max_ ? max_ : x < min_ ? min_ : x;
      }

  private:
      // cg::range_2
      T  min_, max_;
  };

  template <class T>
      Limit<T> limit(T min, T max)
  {
      return Limit<T>(min, max);
  }

  inline double distance_sqr(double A, double B)
  {   return (A - B)*(A - B); }

  inline double distance(double A, double B)
  {   return fabs(A - B); }

  // greatest common divisor
  /*template<class T>
      inline T gcd( T a, T b )
  {
      STATIC_ASSERT(meta::_is_integral<T>::value, T_must_be_integral)
      if ( b == 0 ) return a;
      return gcd( b, a % b );
  }*/
}
