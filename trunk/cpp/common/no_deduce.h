#pragma once

// ������ ��������� ������������� ��������� ����������� �������� �������� ������� �������.
// ��� ���������, ��������, � ����� ������:
//    
// template < class Scalar >
//   vector_t< Scalar > operator * ( Scalar alpha, vector_t< Scalar > vector );
// 
// 5. * vector_t< float >( ) -- ������� ������ ������ ������ ������ ���������� -- ���������� �� ������ ������, 
// ����� ����� ������ ���� Scalar. � �����-�� ������� ������ ��������� ��������� ��������� -- ��������, � ������ ��������� 
// vector_t<float> ( ) * vector_t<double> ������������� ���������, ��� ����� ������������ -- ��� ����� ���� ������������� ���� �� ��������.
// ������ � ������ ������ ����� �������, ��� ������ ������ ������������ �������� vector, � �� ���������� alpha. 
// ���� ��������� ����������� �������� �������� Scalar �� ��������� alpha, �� ��� ���������, ������������� �������������, ����� ����������� 
// � ���� ������� �������. 
// ������� ������������ ������ ��������� ����� �������� �������������� ������� NO_DEDUCE:
// 
// template < class Scalar >
//   vector_t< Scalar > operator * ( NO_DEDUCE(Scalar) alpha, vector_t< Scalar > vector );
// 
// ������ � ������� 5. * vector_t< float >( ) ������� �� ����� -- 5. ���������� � float

namespace meta
{
   template < class T >
      struct no_deduce_f
   {
      typedef     T     type;
   };
}

#ifndef NO_DEDUCE
#  define   NO_DEDUCE(t)   typename meta::no_deduce_f< t >::type
#endif

#ifndef NO_DEDUCE2
#  define   NO_DEDUCE2(p1,p2)   typename meta::no_deduce_f< p1,p2 >::type
#endif