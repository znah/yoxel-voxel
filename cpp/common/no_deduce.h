#pragma once

// »ногда возникает необходимость запретить компил€тору выводить параметр шаблона функции.
// Ёто возникает, например, в таком случае:
//    
// template < class Scalar >
//   vector_t< Scalar > operator * ( Scalar alpha, vector_t< Scalar > vector );
// 
// 5. * vector_t< float >( ) -- попытка такого вызова выдаст ошибку компил€ции -- компил€тор не сможет пон€ть, 
// каким типом должен быть Scalar. ¬ каких-то случа€х данное сообщение полностью оправдано -- например, в случае умножени€ 
// vector_t<float> ( ) * vector_t<double> действительно непон€тно, что хочет пользователь -- ему нужно €вно преобразовать один из векторов.
// ќднако в первом случае можно считать, что скал€р должен определ€тьс€ вектором vector, а не множителем alpha. 
// ≈сли запретить компил€тору выводить параметр Scalar по множителю alpha, то тип множител€, передаваемого пользователем, будет приводитьс€ 
// к типу скал€ра вектора. 
// «апрета компил€тором вывода параметра можно добитьс€ использованием макроса NO_DEDUCE:
// 
// template < class Scalar >
//   vector_t< Scalar > operator * ( NO_DEDUCE(Scalar) alpha, vector_t< Scalar > vector );
// 
// теперь с вызовом 5. * vector_t< float >( ) проблем не будет -- 5. приведетс€ к float

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