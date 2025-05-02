cmake_minimum_required(VERSION 3.20)

find_package (ZLIB)
if (NOT ZLIB_FOUND)
  # Build zlib
  set (ZLIB_BUILD_STATIC_LIBS ON CACHE BOOL " " FORCE)
  set (ZLIB_BUILD_SHARED_LIBS OFF CACHE BOOL " " FORCE)
  add_subdirectory (zlib)

  set (ZLIB_LIBRARY zlibstatic)
  set (ZLIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/zlib ${CMAKE_CURRENT_BINARY_DIR}/zlib)
  include_directories(${ZLIB_INCLUDE_DIR})

  set_property (TARGET zlibstatic PROPERTY FOLDER "third_party")

  add_library (ZLIB::ZLIB ALIAS zlibstatic)

  install(TARGETS zlibstatic EXPORT ZLIBTargets DESTINATION lib)
  install(EXPORT ZLIBTargets
          DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ZLIB)
endif ()

set (ZLIB_INCLUDE_DIR ${ZLIB_INCLUDE_DIR} PARENT_SCOPE)
set (ZLIB_LIBRARY ${ZLIB_LIBRARY} PARENT_SCOPE)

set (PTEX_BUILD_SHARED_LIBS OFF CACHE BOOL " " FORCE)
set (PTEX_BUILD_STATIC_LIBS ON CACHE BOOL " " FORCE)
if (WIN32)
    add_definitions(-DPTEX_STATIC)
endif()

add_subdirectory(ptex)
set (PTEX_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/ptex/src/ptex PARENT_SCOPE)

# For opensubdiv visibility
set (NO_TESTS ON CACHE BOOL " " FORCE)
set (NO_EXAMPLES ON CACHE BOOL " " FORCE)
set (NO_TUTORIALS ON CACHE BOOL " " FORCE)
set (NO_REGRESSION ON CACHE BOOL " " FORCE)
set (BUILD_SHARED_LIBS OFF)

set (PTEX_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ptex/src/ptex)
set (PTEX_LIBRARY Ptex_static PARENT_SCOPE)

add_subdirectory(OpenSubdiv)
set (OPENSUBDIV_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/OpenSubdiv
    ${CMAKE_CURRENT_BINARY_DIR}/OpenSubdiv PARENT_SCOPE)

set(OSD_LIBRARY osd_static_cpu PARENT_SCOPE)

set (NANOVDB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/openvdb/nanovdb 
    ${CMAKE_CURRENT_BINARY_DIR}openvdb/nanovdb PARENT_SCOPE)

# nvapi
set (NVAPI_LIBRARY ${CMAKE_CURRENT_SOURCE_DIR}/nvapi/amd64/nvapi64.lib PARENT_SCOPE)
