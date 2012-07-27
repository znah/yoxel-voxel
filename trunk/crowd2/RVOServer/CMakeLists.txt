project(RVOServer)
cmake_minimum_required(VERSION 2.8)

# get boost
find_package(Boost COMPONENTS
                python
             REQUIRED)
SET(Boost_USE_STATIC_LIBS OFF)
SET(Boost_USE_MULTITHREADED ON)
SET(Boost_USE_STATIC_RUNTIME OFF)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# get python
find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
link_directories(${PYTHON_LIBRARIES})

add_library(RVOServer SHARED  
    main.cpp 
    stdafx.h 
    stdafx.cpp)
set_target_properties(RVOServer 
    PROPERTIES SUFFIX ".pyd")

install (TARGETS RVOServer DESTINATION .)

#target_link_libraries(RVOServer
#        ${Boost_LIBRARIES}
#        ${PYTHON_LIBRARIES} )