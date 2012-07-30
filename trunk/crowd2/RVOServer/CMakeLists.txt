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

install (TARGETS RVOServer DESTINATION "${CMAKE_SOURCE_DIR}/lib")
