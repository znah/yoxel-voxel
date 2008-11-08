#pragma once

namespace cg 
{
   template < class T > struct is_floating_point_f ;
   template < class T > struct is_integer_f ;
   template < class T, class U > struct max_type_f ; 
}

#define MAX_TYPE(a,b)   typename max_type_f<a,b> :: type

//////////////////////////////////////////////////////////////////////////////////////////////
// implementation
namespace cg 
{
   template < class T > struct is_floating_point_f { static const bool value = false; }; 
   template < class T > struct is_integer_f        { static const bool value = false; };

   template < class T, class U > struct max_type_f ; 

   #pragma push_macro ( "DECLARE_EQUAL" )
   #pragma push_macro ( "DECLARE_LESS" )
   #pragma push_macro ( "DECLARE_INTEGRAL" )
   #pragma push_macro ( "DECLARE_INTEGER" )
   #pragma push_macro ( "DECLARE_FLOATING" )

   #define DECLARE_INTEGER(S)  template <> struct is_integer_f<S>         { static const bool value = true; };
   #define DECLARE_FLOATING(S) template <> struct is_floating_point_f<S>  { static const bool value = true; };
   #define DECLARE_EQUAL(S)    template <> struct max_type_f <S, S> { typedef S type ; } ;   
   #define DECLARE_LESS(L,G)   template <> struct max_type_f <L, G> { typedef G type ; } ; template <> struct max_type_f <G, L> { typedef G type ; } ; 

   #define DECLARE_INTEGRAL(scalar)                     \
      DECLARE_INTEGER( scalar )                         \
      DECLARE_EQUAL  ( scalar )                         \
      DECLARE_LESS   ( scalar, double )                 \
      DECLARE_LESS   ( scalar, float )                  \
      DECLARE_INTEGER( unsigned scalar )                \
      DECLARE_EQUAL  ( unsigned scalar )                \
      DECLARE_LESS   ( unsigned scalar, double )        \
      DECLARE_LESS   ( unsigned scalar, float )  

   DECLARE_EQUAL    (double)                                               
   DECLARE_FLOATING (double)                                               

   DECLARE_EQUAL    (float)   
   DECLARE_FLOATING (float)                                               

   DECLARE_LESS    (float,double)

   DECLARE_INTEGRAL(long)                                            
   DECLARE_INTEGRAL(int)                                            
   DECLARE_INTEGRAL(short)                                            
   DECLARE_INTEGRAL(char)                                            

   #pragma pop_macro ( "DECLARE_FLOATING" )
   #pragma pop_macro ( "DECLARE_INTEGER" )
   #pragma pop_macro ( "DECLARE_INTEGRAL" )
   #pragma pop_macro ( "DECLARE_LESS" )
   #pragma pop_macro ( "DECLARE_EQUAL" )
}

