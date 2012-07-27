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
find_package(PythonInterp REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})
link_directories(${PYTHON_LIBRARIES})


execute_process(COMMAND python -c "import numpy;print numpy.get_include()" 
                OUTPUT_VARIABLE  NUMPY_PATH)
message("numpy: ${NUMPY_PATH}")
include_directories(${NUMPY_PATH})

include_directories("ext/include")

add_library(RVOServer SHARED  
    main.cpp 
    stdafx.h 
    stdafx.cpp
    numpy_boost.hpp
    common.h
    #path_field.h
    #path_field.cpp
    )
set_target_properties(RVOServer 
    PROPERTIES SUFFIX ".pyd")

install (TARGETS RVOServer DESTINATION "${CMAKE_SOURCE_DIR}/..")


target_link_libraries(RVOServer
        ${Boost_LIBRARIES}
        ${PYTHON_LIBRARIES} )