project(RVOServer)
cmake_minimum_required(VERSION 2.8)

# get boost
#find_package(Boost COMPONENTS
#                python
#             REQUIRED)
find_package(Boost)
SET(Boost_USE_STATIC_LIBS OFF)
SET(Boost_USE_MULTITHREADED ON)
SET(Boost_USE_STATIC_RUNTIME OFF)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

include_directories("ext/include")

add_library(RVOServer
    RVOServer.h
    main.cpp 
    stdafx.h 
    stdafx.cpp
    common.h
    path_field.h
    path_field.cpp
    )

install (TARGETS RVOServer DESTINATION "${CMAKE_SOURCE_DIR}/lib")


#target_link_libraries(RVOServer
#        ${Boost_LIBRARIES}
#        ${PYTHON_LIBRARIES} )