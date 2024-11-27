cmake_minimum_required(VERSION 3.20)

# set (NANOVDB_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/openvdb/nanovdb PARENT_SCOPE)

find_package (ZLIB)
if (NOT ZLIB_FOUND)
  # Build zlib
  set (ZLIB_BUILD_STATIC_LIBS ON CACHE BOOL " " FORCE)
  set (ZLIB_BUILD_SHARED_LIBS OFF CACHE BOOL " " FORCE)
  add_subdirectory (zlib)

  set (ZLIB_LIBRARIES zlibstatic)
  set (ZLIB_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/zlib ${CMAKE_CURRENT_BINARY_DIR}/zlib)

  set (ZLIB_LIBRARY zlibstatic)
  set (ZLIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/zlib ${CMAKE_CURRENT_BINARY_DIR}/zlib)
  set (ZLIB_FOUND TRUE)

  # set_property (TARGET zlibstatic PROPERTY FOLDER "third_party")

  add_library (ZLIB::ZLIB ALIAS zlibstatic)
  include_directories(${ZLIB_INCLUDE_DIRS})  # yuck, but so openexr/ptex can find zlib.h...
endif ()

set (ZLIB_INCLUDE_DIRS ${ZLIB_INCLUDE_DIRS} PARENT_SCOPE)
set (ZLIB_LIBRARIES ${ZLIB_LIBARIES} PARENT_SCOPE)

set (PTEX_BUILD_SHARED_LIBS OFF CACHE BOOL " " FORCE)
if (WIN32)
    add_definitions(-DPTEX_STATIC)
endif()

add_subdirectory(ptex)
set (PTEX_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/ptex/src/ptex PARENT_SCOPE)
