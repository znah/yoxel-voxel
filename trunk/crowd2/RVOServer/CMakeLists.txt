project(RVOServer)
cmake_minimum_required(VERSION 2.8)

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

set(CMAKE_BUILD_TYPE Release)

if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "-m32")
endif()

install (TARGETS RVOServer DESTINATION "${CMAKE_SOURCE_DIR}/lib")
