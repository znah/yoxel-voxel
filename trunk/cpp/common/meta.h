#pragma once


//  Compile time assert ( from A.Alexandrescu, "Modern C++ Design", p.25)
//  ---------------------------------------------------------------------
template< bool pred >
struct CompileAssert
{
    CompileAssert(...);
};

template<> struct CompileAssert<false> {};

#define STATIC_CHECK(expr,msg) \
{   \
    class ERR_##msg{} x;   \
    (void)sizeof(CompileAssert<(expr) >( x ) ); \
}

namespace meta
{
   // ��-�� ����, ��� ������ ��� ����� ���������������  VC7.0, 
   // ��������� ��� ��������� �������������
   namespace details
   {
      template < bool condition >
         struct _if
      {
         template < typename true_result, typename false_result >
            struct _if_impl
         {
            typedef 
               true_result
               type;
         };
      };

      template < >
         struct _if< false >
      {
         template < typename true_result, typename false_result >
            struct _if_impl
         {
            typedef 
               false_result
               type;
         };
      };

   }

   template < typename condition, typename true_result, typename false_result >
      struct _if
   {
      typedef 
         typename details::_if< condition::value >::_if_impl< true_result, false_result >::type
         type;
   };

   // ---------------------------------------------------------- integral type function

   template < typename T >
      struct _is_integral
   {
      static const bool value = false;
   };

   template < > struct _is_integral< short >           { static const bool value = true; };
   template < > struct _is_integral< unsigned short >  { static const bool value = true; };

   template < > struct _is_integral< char >           { static const bool value = true; };
   template < > struct _is_integral< unsigned char >  { static const bool value = true; };

   template < > struct _is_integral< int >           { static const bool value = true; };
   template < > struct _is_integral< unsigned int >  { static const bool value = true; };

   template < > struct _is_integral< long >           { static const bool value = true; };
   template < > struct _is_integral< unsigned long >  { static const bool value = true; };
}