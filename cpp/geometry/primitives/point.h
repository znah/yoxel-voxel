#pragma once

#include "point_fwd.h"

#ifndef SIMPLE_STRUCTS

#include <common/no_deduce.h>
#include <common/assert.h>
#include <geometry/xmath.h>

namespace cg
{
   template < class Scalar, size_t Dim >     struct point_t;
   template < class Scalar, size_t Dim >     struct point_decomposition_t;

#define MAX_POINT(scalar_a, scalar_b, dim)   point_t< MAX_TYPE(scalar_a, scalar_b), dim >

   // --------------------------------------------------------------------------------------------------------- addition
   template < class ScalarT, class ScalarU, size_t Dim >     
      MAX_POINT(ScalarT, ScalarU, Dim) operator + ( point_t< ScalarT, Dim > const & a, point_t< ScalarU, Dim > const & b );

   template < class ScalarT, class ScalarU, size_t Dim >     
      MAX_POINT(ScalarT, ScalarU, Dim) operator - ( point_t< ScalarT, Dim > const & a, point_t< ScalarU, Dim > const & b );

   // --------------------------------------------------------------------------------------------------------- component-wise multiplication
   template < class ScalarT, class ScalarU, size_t Dim >     
      MAX_POINT(ScalarT, ScalarU, Dim) operator & ( point_t< ScalarT, Dim > const & a, point_t< ScalarU, Dim > const & b );

   template < class ScalarT, class ScalarU, size_t Dim >     
      MAX_POINT(ScalarT, ScalarU, Dim) operator / ( point_t< ScalarT, Dim > const & a, point_t< ScalarU, Dim > const & b );

   template < class ScalarT, class ScalarU, size_t Dim >     
      MAX_POINT(ScalarT, ScalarU, Dim) operator % ( point_t< ScalarT, Dim > const & a, point_t< ScalarU, Dim > const & b );

   // --------------------------------------------------------------------------------------------------------- constant multiplication
   template < class ScalarT, class ScalarU, size_t Dim >     
      MAX_POINT(ScalarT, ScalarU, Dim) operator * ( point_t< ScalarT, Dim > const & a, ScalarU alpha );

   template < class ScalarT, class ScalarU, size_t Dim >     
      MAX_POINT(ScalarT, ScalarU, Dim) operator * ( ScalarT alpha, point_t< ScalarU, Dim > const & a );

   template < class ScalarT, class ScalarU, size_t Dim >     
      MAX_POINT(ScalarT, ScalarU, Dim) operator / ( point_t< ScalarT, Dim > const & a, ScalarU alpha );

   template < class ScalarT, class ScalarU, size_t Dim >     
      MAX_POINT(ScalarT, ScalarU, Dim) operator % ( point_t< ScalarT, Dim > const & a, ScalarU alpha );

   // --------------------------------------------------------------------------------------------------------- scalar product
   template < class ScalarT, class ScalarU, size_t Dim >     
      MAX_TYPE(ScalarT, ScalarU)     operator * ( point_t< ScalarT, Dim > const & a, point_t< ScalarU, Dim > const & b );

   // --------------------------------------------------------------------------------------------------------- vector product
   template < class ScalarT, class ScalarU >     
      MAX_TYPE(ScalarT, ScalarU)     operator ^ ( point_t< ScalarT, 2 > const & a, point_t< ScalarU, 2 > const & b );
   template < class ScalarT, class ScalarU >     
      MAX_POINT(ScalarT, ScalarU, 3) operator ^ ( point_t< ScalarT, 3 > const & a, point_t< ScalarU, 3 > const & b );

   // --------------------------------------------------------------------------------------------------------- unary operations
   template < class Scalar, size_t Dim >     point_t< Scalar, Dim > operator - ( point_t< Scalar, Dim > a );

   // --------------------------------------------------------------------------------------------------------- norm and distance operations
   // returns squared norm of point
   template < class Scalar, size_t Dim >     Scalar norm_sqr( point_t< Scalar, Dim > const & a );

   // returns norm of point
   template < class Scalar, size_t Dim >     Scalar norm( point_t< Scalar, Dim > const & a );

   // returns distance between two points
   template < class ScalarT, class ScalarU, size_t Dim >
      MAX_TYPE(ScalarT, ScalarU) distance_sqr( point_t< ScalarT, Dim > const & a, point_t< ScalarU, Dim > const & b );

   // returns distance between two points
   template < class ScalarT, class ScalarU, size_t Dim >
      MAX_TYPE(ScalarT, ScalarU) distance( point_t< ScalarT, Dim > const & a, point_t< ScalarU, Dim > const & b );

   // --------------------------------------------------------------------------------------------------------- normaization operations
   // returns norm of point
   template < class Scalar, size_t Dim >     Scalar normalize( point_t< Scalar, Dim > & point );

   // returns normalized point
   template < class Scalar, size_t Dim >     point_t< Scalar, Dim > normalized( point_t< Scalar, Dim > point );

   // returns direction and length: direction * length == point
   template < class Scalar, size_t Dim >     point_decomposition_t< Scalar, Dim > decompose( point_t< Scalar, Dim > const & point, NO_DEDUCE(Scalar) eps = epsilon< Scalar >( ) );

   // --------------------------------------------------------------------------------------------------------- comparison
   // fuzzy
   template < class Scalar, size_t Dim >     bool eq( point_t< Scalar, Dim > const & a, point_t< Scalar, Dim > const & b, NO_DEDUCE(Scalar) eps = epsilon< Scalar >( ) );
   template < class Scalar, size_t Dim >     bool eq_zero( point_t< Scalar, Dim > const & a, NO_DEDUCE(Scalar) eps = epsilon< Scalar >( ) );

   // strong
   template < class Scalar, size_t Dim >     bool operator == ( point_t< Scalar, Dim > const & a, point_t< Scalar, Dim > const & b );
   template < class Scalar, size_t Dim >     bool operator != ( point_t< Scalar, Dim > const & a, point_t< Scalar, Dim > const & b );

   // ---------------------------------------------------------------------------------------------------------- 
   template < class Scalar, size_t Dim >     point_t< int,    Dim > round( point_t< Scalar, Dim > const & );
   template < class Scalar, size_t Dim >     point_t< Scalar, Dim > floor( point_t< Scalar, Dim > const & );
   template < class Scalar, size_t Dim >     point_t< Scalar, Dim > ceil ( point_t< Scalar, Dim > const & );

   // ---------------------------------------------------------------------------------------------------------- operator less
   template < class Scalar, size_t Dim >     bool operator < ( point_t< Scalar, Dim > const & a, point_t< Scalar, Dim > const & b ) ; 
}

namespace cg
{
#pragma pack( push, 1 )

   // ----------------------------------------------------------------------------------------------------- point_2_t
   template < class Scalar >
      struct point_t< Scalar, 2 >
   {
      typedef  Scalar   scalar_type;
      enum  { dimension = 2 };

      // -------------------------------------------------------------- ctor
      point_t( );

      point_t( Scalar x, Scalar y );

      template < class _Scalar >
      point_t( point_t< _Scalar, 2 > const & point );

      // -------------------------------------------------------------- arithmetic
      // vector operations
      point_t & operator += ( point_t const & );
      point_t & operator -= ( point_t const & );

      point_t & operator *= ( Scalar );
      point_t & operator /= ( Scalar );
      point_t & operator %= ( Scalar );

      // component-wise multiplication
      point_t & operator &= ( point_t const & );

      // component-wise division
      point_t & operator /= ( point_t const & );

      // component-wise remainder
      point_t & operator %= ( point_t const & );

      // -------------------------------------------------------------- data
      Scalar   x;
      Scalar   y;

      // -------------------------------------------------------------- data accessors
      Scalar         & operator [] ( size_t i )       { Assert( i < 2 ); return (&x)[i]; }
      Scalar const   & operator [] ( size_t i ) const { Assert( i < 2 ); return (&x)[i]; }
   };

   // ----------------------------------------------------------------------------------------------------- point_3_t
   template < class Scalar >
      struct point_t< Scalar, 3 >
         : point_t< Scalar, 2 >
   {
      typedef  Scalar   scalar_type;
      enum  { dimension = 3 };

      // -------------------------------------------------------------- ctor
      point_t( );

      point_t( Scalar x, Scalar y, Scalar z );

      point_t( point_t< Scalar, 2 > const & point, Scalar h );

      explicit point_t( point_t< Scalar, 2 > const & point );

      template < class _Scalar >
      point_t( point_t< _Scalar, 3 > const & point );

      // -------------------------------------------------------------- arithmetic
      // vector operations
      point_t & operator += ( point_t const & );
      point_t & operator -= ( point_t const & );

      point_t & operator *= ( Scalar );
      point_t & operator /= ( Scalar );
      point_t & operator %= ( Scalar );

      // component-wise multiplication
      point_t & operator &= ( point_t const & );

      // component-wise division
      point_t & operator /= ( point_t const & );

      // component-wise remainder
      point_t & operator %= ( point_t const & );

      // -------------------------------------------------------------- data
      Scalar   z;

      // -------------------------------------------------------------- data accessors
      Scalar         & operator [] ( size_t i )       { Assert( i < 3 ); return (&this->x)[i]; }
      Scalar const   & operator [] ( size_t i ) const { Assert( i < 3 ); return (&this->x)[i]; }
   };


   // ----------------------------------------------------------------------------------------------------- point_4_t
   template < class Scalar >
      struct point_t< Scalar, 4 >
         : point_t< Scalar, 3 >
   {
      typedef  Scalar   scalar_type;
      enum  { dimension = 4 };

      // -------------------------------------------------------------- ctor
      point_t( );

      point_t( Scalar x, Scalar y, Scalar z, Scalar w );

      point_t( point_t< Scalar, 3 > const & point, Scalar w );

      explicit point_t( point_t< Scalar, 3 > const & point );

      template < class _Scalar >
      point_t( point_t< _Scalar, 4 > const & point );

      // -------------------------------------------------------------- arithmetic
      // vector operations
      point_t & operator += ( point_t const & );
      point_t & operator -= ( point_t const & );

      point_t & operator *= ( Scalar );
      point_t & operator /= ( Scalar );
      point_t & operator %= ( Scalar );

      // component-wise multiplication
      point_t & operator &= ( point_t const & );

      // component-wise division
      point_t & operator /= ( point_t const & );

      // component-wise remainder
      point_t & operator %= ( point_t const & );

      // -------------------------------------------------------------- data
      Scalar   w;

      // -------------------------------------------------------------- data accessors
      Scalar         & operator [] ( size_t i )       { Assert( i < 4 ); return (&this->x)[i]; }
      Scalar const   & operator [] ( size_t i ) const { Assert( i < 4 ); return (&this->x)[i]; }
   };

   template < class Scalar, size_t Dim >
      struct point_decomposition_t
   {
      // -------------------------------------------------------------- ctor
      point_decomposition_t( );
      point_decomposition_t( point_t< Scalar, Dim > const & direction, Scalar length );

      // -------------------------------------------------------------- data
      point_t< Scalar, Dim >     direction;
      Scalar                     length;
   };

#pragma pack( pop )
}

// implementation
namespace cg 
{
   // ----------------------------------------------------------------------------------------------------- point_2_t
   // -------------------------------------------------------------- ctor
   template < class Scalar >
   inline point_t< Scalar, 2 > :: point_t( )
      : x( 0 )
      , y( 0 )
   {}

   template < class Scalar >
   inline point_t< Scalar, 2 > :: point_t( Scalar x, Scalar y )
      : x( x )
      , y( y )
   {}

   template < class Scalar >
   template < class _Scalar >
   inline point_t< Scalar, 2 > :: point_t( point_t< _Scalar, 2 > const & point )
      : x( ( Scalar ) point.x )
      , y( ( Scalar ) point.y )
   {}

   // ----------------------------------------------------------------------------------------------------- point_3_t
   // -------------------------------------------------------------- ctor
   template < class Scalar >
   point_t< Scalar, 3 > :: point_t( )
      : z( 0 )
   {}

   template < class Scalar >
   point_t< Scalar, 3 > :: point_t( Scalar x, Scalar y, Scalar z )
      : point_t< Scalar, 2 >( x, y )
      , z( z )
   {}

   template < class Scalar >
   point_t< Scalar, 3 > :: point_t( point_t< Scalar, 2 > const & point, Scalar h )
      : point_t< Scalar, 2 >( point )
      , z( h )
   {}

   template < class Scalar >
   point_t< Scalar, 3 > :: point_t( point_t< Scalar, 2 > const & point )
      : point_t< Scalar, 2 >( point )
      , z( 0 )
   {}

   template < class Scalar >
   template < class _Scalar >
   point_t< Scalar, 3 > :: point_t( point_t< _Scalar, 3 > const & point )
      : point_t< Scalar, 2 >( point )
      , z( ( Scalar ) point.z )
   {}

   // ----------------------------------------------------------------------------------------------------- point_4_t
   // -------------------------------------------------------------- ctor
   template < class Scalar >
      point_t< Scalar, 4 > :: point_t( )
      : w( 0 )
   {}

   template < class Scalar >
      point_t< Scalar, 4 > :: point_t( Scalar x, Scalar y, Scalar z, Scalar w )
      : point_t< Scalar, 3 >( x, y, z )
      , w( w )
   {}

   template < class Scalar >
      point_t< Scalar, 4 > :: point_t( point_t< Scalar, 3 > const & point, Scalar w )
      : point_t< Scalar, 3 >( point )
      , w( w )
   {}

   template < class Scalar >
      point_t< Scalar, 4 > :: point_t( point_t< Scalar, 3 > const & point )
      : point_t< Scalar, 3 >( point )
      , w( 0 )
   {}

   template < class Scalar >
      template < class _Scalar >
      point_t< Scalar, 4 > :: point_t( point_t< _Scalar, 4 > const & point )
      : point_t< Scalar, 3 >( point )
      , w( ( Scalar ) point.w )
   {}


   // ------------------------------------------------------------------------------------------------- point_decomposition_t
   template < class Scalar, size_t Dim >
   point_decomposition_t< Scalar, Dim > :: point_decomposition_t( )
      : length( 0 )
   {}

   template < class Scalar, size_t Dim >
   point_decomposition_t< Scalar, Dim > :: point_decomposition_t( point_t< Scalar, Dim > const & direction, Scalar length )
      : direction ( direction )
      , length    ( length    )
   {}

   // --------------------------------------------------------------------------------------------------------- addition
   template < class ScalarT, class ScalarU, size_t Dim >     
      MAX_POINT( ScalarT, ScalarU, Dim ) operator + ( point_t< ScalarT, Dim > const & a, point_t< ScalarU, Dim > const & b )
   {
      return MAX_POINT( ScalarT, ScalarU, Dim )( a ) += b;
   }

   template < class ScalarT, class ScalarU, size_t Dim >     
      MAX_POINT( ScalarT, ScalarU, Dim ) operator - ( point_t< ScalarT, Dim > const & a, point_t< ScalarU, Dim > const & b )
   {
      return MAX_POINT( ScalarT, ScalarU, Dim )( a ) -= b;
   }

   // --------------------------------------------------------------------------------------------------------- component-wise multiplication
   template < class ScalarT, class ScalarU, size_t Dim >     
      MAX_POINT( ScalarT, ScalarU, Dim ) operator & ( point_t< ScalarT, Dim > const & a, point_t< ScalarU, Dim > const & b )
   {
      return MAX_POINT( ScalarT, ScalarU, Dim )( a ) &= b;
   }

   template < class ScalarT, class ScalarU, size_t Dim >     
      MAX_POINT(ScalarT, ScalarU, Dim) operator % ( point_t< ScalarT, Dim > const & a, point_t< ScalarU, Dim > const & b )
   {
      return MAX_POINT(ScalarT, ScalarU, Dim) (a) %= b ; 
   }

   template < class ScalarT, class ScalarU, size_t Dim >     
      MAX_POINT( ScalarT, ScalarU, Dim ) operator / ( point_t< ScalarT, Dim > const & a, point_t< ScalarU, Dim > const & b )
   {
      return MAX_POINT( ScalarT, ScalarU, Dim )( a ) /= b;
   }

   // --------------------------------------------------------------------------------------------------------- constant multiplication
   template < class ScalarT, class ScalarU, size_t Dim >     
      MAX_POINT(ScalarT, ScalarU, Dim) operator * ( point_t< ScalarT, Dim > const & a, ScalarU alpha )
   {
      return MAX_POINT(ScalarT, ScalarU, Dim)( a ) *= static_cast< MAX_TYPE( ScalarT, ScalarU ) >( alpha );
   }

   template < class ScalarT, class ScalarU, size_t Dim >     
      MAX_POINT(ScalarT, ScalarU, Dim) operator * ( ScalarT alpha, point_t< ScalarU, Dim > const & a )
   {
      return MAX_POINT(ScalarT, ScalarU, Dim)( a ) *= static_cast< MAX_TYPE( ScalarT, ScalarU ) >( alpha );
   }

   template < class ScalarT, class ScalarU, size_t Dim >     
      MAX_POINT(ScalarT, ScalarU, Dim) operator / ( point_t< ScalarT, Dim > const & a, ScalarU alpha )
   {
      return MAX_POINT(ScalarT, ScalarU, Dim)( a ) /= static_cast< MAX_TYPE( ScalarT, ScalarU ) >( alpha );
   }

   template < class ScalarT, class ScalarU, size_t Dim >     
      MAX_POINT(ScalarT, ScalarU, Dim) operator % ( point_t< ScalarT, Dim > const & a, ScalarU alpha )
   {
      return MAX_POINT(ScalarT, ScalarU, Dim)( a ) %= static_cast< MAX_TYPE( ScalarT, ScalarU ) >( alpha );
   }

   // --------------------------------------------------------------------------------------------------------- scalar product
   template < class ScalarT, class ScalarU, size_t Dim >     
      MAX_TYPE(ScalarT, ScalarU)     operator * ( point_t< ScalarT, Dim > const & a, point_t< ScalarU, Dim > const & b )
   {
      MAX_TYPE(ScalarT, ScalarU) res = 0;
      for ( size_t n = 0; n != Dim; ++n )
         res += a[n] * b[n];

      return res;
   }

   // --------------------------------------------------------------------------------------------------------- vector product
   template < class ScalarT, class ScalarU >     
      MAX_TYPE(ScalarT, ScalarU)     operator ^ ( point_t< ScalarT, 2 > const & a, point_t< ScalarU, 2 > const & b )
   {
      return a.x * b.y - a.y * b.x;  
   }

   template < class ScalarT, class ScalarU >     
      MAX_POINT(ScalarT, ScalarU, 3) operator ^ ( point_t< ScalarT, 3 > const & a, point_t< ScalarU, 3 > const & b )
   {
      return MAX_POINT(ScalarT, ScalarU, 3) ( a.y * b.z - a.z * b.y, 
                                              a.z * b.x - a.x * b.z, 
                                              a.x * b.y - a.y * b.x );
   }

   // --------------------------------------------------------------------------------------------------------- unary operations
   template < class Scalar, size_t Dim >
      point_t< Scalar, Dim > operator - ( point_t< Scalar, Dim > a )
   {
      for ( size_t n = 0; n != Dim; ++n )
         a[n] = -a[n];

      return a;
   }

   // --------------------------------------------------------------------------------------------------------- norm and distance operations
   // returns squared norm of point
   template < class Scalar, size_t Dim >
      Scalar norm_sqr( point_t< Scalar, Dim > const & a )
   {
      return a * a;
   }

   // returns norm of point
   template < class Scalar, size_t Dim >
      Scalar norm( point_t< Scalar, Dim > const & a )
   {
      STATIC_ASSERT( is_floating_point_f<Scalar>::value, norm_for_integral_points_undefined );
      return sqrt( a * a );
   }

   // returns distance between two points
   template < class ScalarT, class ScalarU, size_t Dim >
      MAX_TYPE(ScalarT, ScalarU) distance_sqr( point_t< ScalarT, Dim > const & a, point_t< ScalarU, Dim > const & b )
   {
      return norm_sqr( b - a );
   }

   // returns distance between two points
   template < class ScalarT, class ScalarU, size_t Dim >
      MAX_TYPE(ScalarT, ScalarU) distance( point_t< ScalarT, Dim > const & a, point_t< ScalarU, Dim > const & b )
   {
      return norm( b - a );
   }

   // --------------------------------------------------------------------------------------------------------- normaization operations
   // returns norm of point
   template < class Scalar, size_t Dim >
      Scalar normalize( point_t< Scalar, Dim > & point )
   {
      Scalar point_norm = norm( point );
      point /= point_norm;

      return point_norm;
   }

   // returns normalized point
   template < class Scalar, size_t Dim >
      point_t< Scalar, Dim > normalized( point_t< Scalar, Dim > point )
   {
      normalize( point );
      return point;
   }

   // returns normalized point
   template < class Scalar, size_t Dim >
      point_t< Scalar, Dim > normalized_safe( point_t< Scalar, Dim > const & point )
   {
      point_decomposition_t<Scalar, Dim> dec = decompose( point );

      if ( dec.length != 0 )
         return dec.direction;

      point_t< Scalar, Dim > res;
      res[Dim - 1] = 1.;

      return res;
   }

   // returns direction and length: direction * length == point
   // if eq_zero( length, eps ) returns point( 0, 0, ... , 0 ), length == 0
   template < class Scalar, size_t Dim >
      point_decomposition_t< Scalar, Dim > decompose( point_t< Scalar, Dim > const & point, NO_DEDUCE(Scalar) eps )
   {
      Scalar point_norm = norm( point );
      if ( eq_zero( point_norm, eps ) )
         return point_decomposition_t< Scalar, Dim >( );

      return point_decomposition_t< Scalar, Dim >( point / point_norm, point_norm );
   }

   // --------------------------------------------------------------------------------------------------------- comparison
   // fuzzy
   template < class Scalar, size_t Dim >
      bool eq( point_t< Scalar, Dim > const & a, point_t< Scalar, Dim > const & b, NO_DEDUCE(Scalar) eps )
   {
      for ( size_t l = 0; l != Dim; ++l )
         if ( !eq( a[l], b[l], eps ) )
            return false;

      return true;
   }

   template < class Scalar, size_t Dim >
      bool eq_zero( point_t< Scalar, Dim > const & a, NO_DEDUCE(Scalar) eps )
   {
      for ( size_t l = 0; l != Dim; ++l )
         if ( !eq_zero( a[l], eps ) )
            return false;

      return true;
   }

   // strong
   template < class Scalar, size_t Dim >
      bool operator == ( point_t< Scalar, Dim > const & a, point_t< Scalar, Dim > const & b )
   {
      for ( size_t l = 0; l != Dim; ++l )
         if ( a[l] != b[l] )
            return false;

      return true;
   }

   template < class Scalar, size_t Dim >
      bool operator != ( point_t< Scalar, Dim > const & a, point_t< Scalar, Dim > const & b )
   {
      return !( a == b );
   }

   // ----------------------------------------------------------------------------------------------------- 
   template < class Scalar, size_t Dim >     
      point_t< int, Dim > round( point_t< Scalar, Dim > const & point )
   {
      STATIC_ASSERT( is_floating_point_f<Scalar>::value, norm_for_integral_points_undefined );

      point_t< int, Dim > res;
      for ( size_t l = 0; l != Dim; ++l )
         res[l] = round( point[l] );

      return res;
   }

   template < class Scalar, size_t Dim >     
      point_t< Scalar, Dim > floor( point_t< Scalar, Dim > const & point )
   {
      STATIC_ASSERT( is_floating_point_f<Scalar>::value, norm_for_integral_points_undefined );

      point_t< Scalar, Dim > res;
      for ( size_t l = 0; l != Dim; ++l )
         res[l] = floor( point[l] );

      return res;
   }

   template < class Scalar, size_t Dim >     
      point_t< Scalar, Dim > ceil ( point_t< Scalar, Dim > const & point )
   {
      STATIC_ASSERT( is_floating_point_f<Scalar>::value, norm_for_integral_points_undefined );

      point_t< Scalar, Dim > res;
      for ( size_t l = 0; l != Dim; ++l )
         res[l] = ceil( point[l] );

      return res;
   }

   // ---------------------------------------------------------------------------------------------------------- operator less
   template < class Scalar, size_t Dim >
      bool operator < ( point_t< Scalar, Dim > const & a, point_t< Scalar, Dim > const & b )
   {
      for ( size_t i = 0 ; i != Dim ; i ++ ) 
      {
         if ( a[i] < b[i] ) 
            return true ; 
         if ( a[i] > b[i] ) 
            return false ;
      }

      return false ;
   }

   // ----------------------------------------------------------------------------------------------------- point_2_t

   template < class Scalar > 
      point_t< Scalar, 2 > & point_t< Scalar, 2 > :: operator += ( point_t< Scalar, 2 > const & point )
   {
      x += point.x;
      y += point.y;

      return *this;
   }

   template < class Scalar > 
      point_t< Scalar, 2 > & point_t< Scalar, 2 > :: operator -= ( point_t< Scalar, 2 > const & point )
   {
      x -= point.x;
      y -= point.y;

      return *this;
   }

   template < class Scalar > 
      point_t< Scalar, 2 > & point_t< Scalar, 2 > :: operator *= ( Scalar alpha )
   {
      x *= alpha;
      y *= alpha;

      return *this;
   }

   template < class Scalar > 
      point_t< Scalar, 2 > & point_t< Scalar, 2 > :: operator /= ( Scalar alpha )
   {
      x /= alpha;
      y /= alpha;

      return *this;
   }

   template < class Scalar > 
      point_t< Scalar, 2 > & point_t< Scalar, 2 > :: operator %= ( Scalar alpha )
   {
      x = cg::mod(x,alpha);
      y = cg::mod(y,alpha);

      return *this;
   }

   template < class Scalar > 
      point_t< Scalar, 2 > & point_t< Scalar, 2 > :: operator &= ( point_t< Scalar, 2 > const & point )
   {
      x *= point.x;
      y *= point.y;

      return *this;
   }

   template < class Scalar > 
      point_t< Scalar, 2 > & point_t< Scalar, 2 > :: operator /= ( point_t< Scalar, 2 > const & point )
   {
      x /= point.x;
      y /= point.y;

      return *this;
   }

   template < class Scalar > 
      point_t< Scalar, 2 > & point_t< Scalar, 2 > :: operator %= ( point_t< Scalar, 2 > const & point )
   {
      x = cg::mod(x, point.x);
      y = cg::mod(y, point.y);

      return *this;
   }

   // ----------------------------------------------------------------------------------------------------- point_3_t

   template < class Scalar > 
      point_t< Scalar, 3 > & point_t< Scalar, 3 > :: operator += ( point_t< Scalar, 3 > const & point )
   {
      this->x += point.x;
      this->y += point.y;
      this->z += point.z;

      return *this;
   }

   template < class Scalar > 
      point_t< Scalar, 3 > & point_t< Scalar, 3 > :: operator -= ( point_t< Scalar, 3 > const & point )
   {
      this->x -= point.x;
      this->y -= point.y;
      this->z -= point.z;

      return *this;
   }

   template < class Scalar > 
      point_t< Scalar, 3 > & point_t< Scalar, 3 > :: operator *= ( Scalar alpha )
   {
      this->x *= alpha;
      this->y *= alpha;
      this->z *= alpha;

      return *this;
   }

   template < class Scalar > 
      point_t< Scalar, 3 > & point_t< Scalar, 3 > :: operator /= ( Scalar alpha )
   {
      this->x /= alpha;
      this->y /= alpha;
      this->z /= alpha;

      return *this;
   }

   template < class Scalar > 
      point_t< Scalar, 3 > & point_t< Scalar, 3 > :: operator %= ( Scalar alpha )
   {
      this->x = cg::mod(x,alpha);
      this->y = cg::mod(y,alpha);
      this->z = cg::mod(z,alpha);

      return *this;
   }

   template < class Scalar > 
      point_t< Scalar, 3 > & point_t< Scalar, 3 > :: operator &= ( point_t< Scalar, 3 > const & point )
   {
      this->x *= point.x;
      this->y *= point.y;
      this->z *= point.z;

      return *this;
   }

   template < class Scalar > 
      point_t< Scalar, 3 > & point_t< Scalar, 3 > :: operator /= ( point_t< Scalar, 3 > const & point )
   {
      x /= point.x;
      y /= point.y;
      z /= point.z;

      return *this;
   }

   template < class Scalar > 
      point_t< Scalar, 3 > & point_t< Scalar, 3 > :: operator %= ( point_t< Scalar, 3 > const & point )
   {
      x = cg::mod(x, point.x);
      y = cg::mod(y, point.y);
      z = cg::mod(z, point.z);

      return *this;
   }

   // ----------------------------------------------------------------------------------------------------- point_4_t

   template < class Scalar > 
      point_t< Scalar, 4 > & point_t< Scalar, 4 > :: operator += ( point_t< Scalar, 4 > const & point )
   {
      x += point.x;
      y += point.y;
      z += point.z;
      w += point.w;

      return *this;
   }

   template < class Scalar > 
      point_t< Scalar, 4 > & point_t< Scalar, 4 > :: operator -= ( point_t< Scalar, 4 > const & point )
   {
      x -= point.x;
      y -= point.y;
      z -= point.z;
      w -= point.w;

      return *this;
   }

   template < class Scalar > 
      point_t< Scalar, 4 > & point_t< Scalar, 4 > :: operator *= ( Scalar alpha )
   {
      x *= alpha;
      y *= alpha;
      z *= alpha;
      w *= alpha;

      return *this;
   }

   template < class Scalar > 
      point_t< Scalar, 4 > & point_t< Scalar, 4 > :: operator /= ( Scalar alpha )
   {
      x /= alpha;
      y /= alpha;
      z /= alpha;
      w /= alpha;

      return *this;
   }

   template < class Scalar > 
      point_t< Scalar, 4 > & point_t< Scalar, 4 > :: operator %= ( Scalar alpha )
   {
      x = cg::mod(x,alpha);
      y = cg::mod(y,alpha);
      z = cg::mod(z,alpha);
      w = cg::mod(w,alpha);

      return *this;
   }

   template < class Scalar > 
      point_t< Scalar, 4 > & point_t< Scalar, 4 > :: operator &= ( point_t< Scalar, 4 > const & point )
   {
      x *= point.x;
      y *= point.y;
      z *= point.z;
      w *= point.w;

      return *this;
   }

   template < class Scalar > 
      point_t< Scalar, 4 > & point_t< Scalar, 4 > :: operator /= ( point_t< Scalar, 4 > const & point )
   {
      x /= point.x;
      y /= point.y;
      z /= point.z;
      w /= point.w;

      return *this;
   }

   template < class Scalar > 
      point_t< Scalar, 4 > & point_t< Scalar, 4 > :: operator %= ( point_t< Scalar, 4 > const & point )
   {
      x = cg::mod(x, point.x);
      y = cg::mod(y, point.y);
      z = cg::mod(z, point.z);
      w = cg::mod(w, point.w);

      return *this;
   }


#undef MAX_POINT

}

#else

namespace cg 
{
#pragma pack( push, 1 )
   template<typename scalar>
      struct point_2_t
      {
         scalar x, y ;

         point_2_t ( scalar x = 0, scalar y = 0 ) : x ( x ), y ( y ) {} 

         template< typename scalar1>
            explicit point_2_t ( point_2_t<scalar1> const& p ) : x ( (scalar)p.x ), y ( (scalar)p.y ) {} 
      } ; 

   template<typename scalar>
      struct point_3_t
      {
         scalar x, y, z ;

         point_3_t ( scalar x = 0, scalar y = 0, scalar z = 0 ) : x ( x ), y ( y ), z ( z ) {} 

         template< typename scalar1>
            explicit point_3_t ( point_3_t<scalar1> const& p ) : x ( (scalar)p.x ), y ( (scalar)p.y ), z ( (scalar)p.z ) {} 
      } ; 

   template<typename scalar>
      struct point_4_t
      {
         scalar x, y, z, w ;

         point_4_t ( scalar x = 0, scalar y = 0, scalar z = 0, scalar w = 0 ) : x ( x ), y ( y ), z ( z ), w ( w ) {} 

         template< typename scalar1>
            explicit point_4_t ( point_4_t<scalar1> const& p ) : x ( (scalar)p.x ), y ( (scalar)p.y ), z ( (scalar)p.z ), w ( (scalar)p.w ) {} 
      } ; 
#pragma pack( pop )
} ; 

#endif // SIMPLE_STRUCTS
