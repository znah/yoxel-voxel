#pragma once

#ifndef SIMPLE_STRUCTS

namespace cg
{
   // point definition
   template < class, size_t >             struct point_t;
   template < class Struct, size_t Dim >  struct point_decomposition_t;

   // point types
   typedef     point_t< double, 2 >       point_2;
   typedef     point_t< float,  2 >       point_2f;
   typedef     point_t< int,    2 >       point_2i;

   typedef     point_t< double, 3 >       point_3;
   typedef     point_t< float,  3 >       point_3f;
   typedef     point_t< int,    3 >       point_3i;

   typedef     point_t< double, 4 >       point_4;
   typedef     point_t< float,  4 >       point_4f;
   typedef     point_t< int,    4 >       point_4i;

   typedef     point_t< unsigned char ,   2 >       point_2b;
   typedef     point_t< unsigned short,   2 >       point_2us;
   typedef     point_t< short,            2 >       point_2s;

   typedef     point_t< unsigned char,    3 >       point_3b;
   typedef     point_t< short,            3 >       point_3s;
   typedef     point_t< unsigned short,   3 >       point_3us;

   typedef     point_t< unsigned char,    4 >       point_4b;
   typedef     point_t< char,             4 >       point_4sb;
   typedef     point_t< short,            4 >       point_4s;
   typedef     point_t< unsigned short,   4 >       point_4us;


   typedef     point_decomposition_t< double, 2 >       point_decomposition_2;
   typedef     point_decomposition_t< float,  2 >       point_decomposition_2f;

   typedef     point_decomposition_t< double, 3 >       point_decomposition_3;
   typedef     point_decomposition_t< float,  3 >       point_decomposition_3f;

   typedef     point_decomposition_t< double, 4 >       point_decomposition_4;
   typedef     point_decomposition_t< float,  4 >       point_decomposition_4f;
}

#else

namespace cg
{ 
   template<typename scalar> struct point_2_t ; 
   template<typename scalar> struct point_3_t ; 

   typedef point_2_t<double> point_2 ; 
   typedef point_2_t<float>  point_2f ; 
   typedef point_2_t<int>    point_2i ; 

   typedef point_3_t<double> point_3 ; 
   typedef point_3_t<float>  point_3f ; 
   typedef point_3_t<int>    point_3i ; 

   typedef point_4_t<double> point_4 ; 
   typedef point_4_t<float>  point_4f ; 
   typedef point_4_t<int>    point_4i ; 
}

#endif // SIMPLE_STRUCTS
