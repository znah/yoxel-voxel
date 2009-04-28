#pragma once 

#ifndef Assert // чисто отладочный ассерт
        #ifdef _DEBUG 
    #define Assert(x) do { if (!(x)) __asm {int 3} } while( 0 )
        #else
                #define Assert(x)
        #endif
#endif

#ifndef TracedAssert  // в release вырождается в TRACE
        #ifdef _DEBUG 
                #define TracedAssert(x) do { if (!(x)) __asm {int 3} } while ( 0 )
        #elif defined (__TRACE_H)
                #define TracedAssert(x) do { if (!(x)) __TRACE(#x); } while ( 0 )
        #else
                #define TracedAssert(x)
        #endif
#endif

#ifndef Verify // код в скобках выполняется в ЛЮБОЙ конфигурации
        #ifdef _DEBUG 
                #define Verify(x) do { if (!(x)) __asm {int 3} } while ( 0 )
        #elif defined (__TRACE_H)
                #define Verify(x) do { if (!(x)) __TRACE(#x); } while ( 0 )
        #else
                #define Verify(x) do { (x); } while ( 0 )
        #endif
#endif

#ifndef AssertRelease // самая жесткая штука - даже в release прерывает выполнение программы при невыполнении условия
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
