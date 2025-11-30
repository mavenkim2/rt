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

# TBB for openvdb
set(TBB_TEST OFF)
add_subdirectory("tbb")
set(Tbb_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/tbb/include)
get_target_property(TBB_LIBRARY_OUTPUT_DIR tbb LIBRARY_OUTPUT_DIRECTORY)
set(Tbb_tbb_LIBRARY_RELEASE ${TBB_LIBRARY_OUTPUT_DIR}/tbb12.lib)

# Blosc for openvdb
add_subdirectory("blosc")
set(Blosc_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/blosc/blosc)
set(Blosc_LIBRARY_DEBUG ${CMAKE_CURRENT_BINARY_DIR}/blosc/blosc/Debug/blosc.lib)
set(Blosc_LIBRARY_RELEASE ${CMAKE_CURRENT_BINARY_DIR}/blosc/blosc/Release/blosc.lib)

# openvdb
set(USE_NANOVDB ON)
set(USE_BLOSC ON)
set(NANOVDB_USE_OPENVDB ON CACHE BOOL "" FORCE)
set(OPENVDB_CORE_SHARED ON CACHE BOOL "" FORCE)
set(OPENVDB_USE_DELAYED_LOADING OFF)
set(OPENVDB_BUILD_BINARIES OFF)
add_subdirectory("openvdb")
set(OPENVDB_LIBRARY openvdb_shared PARENT_SCOPE)
set(NANOVDB_LIBRARY nanovdb PARENT_SCOPE)

set(OPENVDB_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/openvdb PARENT_SCOPE)
set(NANOVDB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/openvdb/nanovdb PARENT_SCOPE)

set (NANOVDB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/openvdb/nanovdb 
    ${CMAKE_CURRENT_BINARY_DIR}openvdb/nanovdb PARENT_SCOPE)

# nvapi
set (NVAPI_LIBRARY ${CMAKE_CURRENT_SOURCE_DIR}/nvapi/amd64/nvapi64.lib PARENT_SCOPE)

if (USE_GPU)
# gklib
    add_subdirectory("GKlib")
    set (GKLIB_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/GKlib)
    include_directories(${GKLIB_INCLUDE_DIR})
    set(GK_LIBRARY GKlib PARENT_SCOPE)

# metis
    set (METIS_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/METIS/include)
    add_subdirectory("METIS")
    target_include_directories(metis PRIVATE ${METIS_INCLUDE_DIR})
    set (METIS_LIBRARY metis PARENT_SCOPE)
endif()

# streamline
#set(STREAMLINE_LIBRARY streamline PARENT_SCOPE)
