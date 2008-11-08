#pragma once 

#ifndef Assert // ����� ���������� ������
        #ifdef _DEBUG 
    #define Assert(x) do { if (!(x)) __asm {int 3} } while( 0 )
        #else
                #define Assert(x)
        #endif
#endif

#ifndef TracedAssert  // � release ����������� � TRACE
        #ifdef _DEBUG 
                #define TracedAssert(x) do { if (!(x)) __asm {int 3} } while ( 0 )
        #elif defined (__TRACE_H)
                #define TracedAssert(x) do { if (!(x)) __TRACE(#x); } while ( 0 )
        #else
                #define TracedAssert(x)
        #endif
#endif

#ifndef Verify // ��� � ������� ����������� � ����� ������������
        #ifdef _DEBUG 
                #define Verify(x) do { if (!(x)) __asm {int 3} } while ( 0 )
        #elif defined (__TRACE_H)
                #define Verify(x) do { if (!(x)) __TRACE(#x); } while ( 0 )
        #else
                #define Verify(x) do { (x); } while ( 0 )
        #endif
#endif

#ifndef AssertRelease // ����� ������� ����� - ���� � release ��������� ���������� ��������� ��� ������������ �������
    #define AssertRelease(x) do { if (!(x)) __asm {int 3} } while ( 0 )
#endif

#ifndef STATIC_ASSERT

namespace meta_details
{
   template< bool pred >
      struct StaticAssert
   {
      StaticAssert(...);
   };

   template<> struct StaticAssert<false> {};
}

#define STATIC_ASSERT(expr,msg) \
{   \
    class ERR_##msg{} x;   \
    (void)sizeof(meta_details::StaticAssert<(expr)>( x ) ); \
}

#endif
