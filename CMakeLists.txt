cmake_minimum_required(VERSION 3.20)
project(RT LANGUAGES CXX C)
    
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    add_definitions(/D _CRT_SECURE_NO_WARNINGS /Dstrdup=_strdup
        /wd4018 /wd4100 /wd4101 /wd4127 /wd4146 /wd4232 /wd4242 /wd4244 /wd4245 /wd4267 /wd4305 /wd4309 
        /wd4310 /wd4334 /wd4456 /wd4464 /wd4668 /wd4701 /wd4703 /wd4711 /wd4756 /wd4820 /wd5045 /wd5250)
endif ()
add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/src/third_party)

set (FlagsSSE2 "-D __SSE__ -D __SSE2__")
set (FlagsSSE42 "${FlagsSSE2} -D __SSE3__ -D __SSSE3__ -D __SSE4_1__ -D __SSE4_2__")

include (CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Using Clang Compiler")
    list (APPEND RT_CXX_FLAGS "-Wno-unused-variable")
    if (CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL 14)
        message (STATUS "Disabling -ffp-contract")
        list (APPEND RT_CXX_FLAGS "-ffp-contract=off")
        endif()
    if (COMPILER_SUPPORTS_MARCH_NATIVE)
        list (APPEND RT_CXX_FLAGS "-march=native")
    endif()
elseif (MSVC)
    message(STATUS "Using MSVC Compiler")
    list (APPEND RT_CXX_FLAGS "-FC" "-Zi" "-EHsc" "-nologo" "-Oi" "-WX" "-W4" "-wd4305" "-wd4324" "-wd4127" 
        "-wd4700" "-wd4701" "-wd4505" "-wd4189" "-wd4201" "-wd4100" "/arch:AVX2")
endif ()

# add_executable(rt "src/rt/rt.cpp")

if (CMAKE_BUILD_TYPE MATCHES "Debug")
    list (APPEND RT_DEFINITIONS "TRACK_MEMORY" "DEBUG")
endif()

if (CMAKE_SYSTEM_NAME STREQUAL Windows)
    list (APPEND RT_DEFINITIONS "NOMINMAX")
else()
    message (SEND_ERROR "Windows only supported")
endif()

set(RT_SOURCE_FILES
    src/rt/base.cpp
    src/rt/win32.cpp
    src/rt/jobsystem.cpp
    src/rt/string.cpp
    src/rt/debug.cpp
    src/rt/base_types.cpp
    src/rt/thread_context.cpp
    src/rt/spectrum.cpp
    src/rt/memory.cpp
    src/rt/scene.cpp
    src/rt/bvh.cpp
    src/rt/lights.cpp
)

set(RT_HEADER_FILES
    src/rt/bvh/bvh_aos.h
    src/rt/bvh/bvh_build.h
    src/rt/bvh/bvh_types.h
    src/rt/bvh/parallel.h
    src/rt/bvh/partial_rebraiding.h
    src/rt/geometry/bspline.h
    src/rt/math/basemath.h
    src/rt/math/bounds.h
    src/rt/math/lane4f32.h
    src/rt/math/lane4u32.h
    src/rt/math/lane8f32.h
    src/rt/math/lane8u32.h
    src/rt/math/math.h
    src/rt/math/matx.h
    src/rt/math/simd_base.h
    src/rt/math/simd_include.h
    src/rt/math/vec2.h
    src/rt/math/vec3.h
    src/rt/math/vec4.h
    src/rt/tables/primes.h
    src/rt/tables/sobolmatrices.h
    src/rt/tests/sampling_test.cpp
    src/rt/tests/test.cpp
    src/rt/algo.h
    src/rt/base.h
    src/rt/base_types.h
    src/rt/bsdf.h
    src/rt/bvh.h
    src/rt/containers.h
    src/rt/debug.h
    src/rt/hash.h
    src/rt/integrate.h
    src/rt/jobsystem.h
    src/rt/lights.h
    src/rt/low_discrepancy.h
    src/rt/memory.h
    src/rt/random.h
    src/rt/rt.h
    src/rt/sampler.h
    src/rt/sampling.h
    src/rt/scene.h
    src/rt/spectrum.h
    src/rt/string.h
    src/rt/template.h
    src/rt/thread_context.h
    src/rt/win32.h
)

list (APPEND RT_DEFINITIONS "__SSE3__" "__SSE4_1__")

function (CHECK_EXT NAME DIR HASH)
  if (NOT IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/${DIR}")
    message (FATAL_ERROR "The ${NAME} submodule directory is missing! "
      "Either that submodule was recently added to pbrt or you did not clone the project with --recursive. "
      "In order to update the submodules, run:\n"
      "  \"git submodule update --init --recursive\"")
  endif ()

  find_package(Git)
  if (GIT_FOUND)
    execute_process (
      COMMAND ${GIT_EXECUTABLE} rev-parse HEAD
      WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/${DIR}"
      RESULT_VARIABLE "git_return"
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
      OUTPUT_VARIABLE "git_hash")
    if (NOT ${git_hash} MATCHES "^${HASH}")
      message (FATAL_ERROR "The ${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/${DIR} "
        "submodule isn't up to date (${git_hash} vs ${HASH}). Please run:\n"
        "  \"git submodule update --recursive\"")
    else ()
      message (STATUS "${NAME} at commit: ${git_hash}")
    endif()
  else (GIT_FOUND)
    message (STATUS "git not found: unable to verify revisions in submodules")
  endif (GIT_FOUND)
endfunction ()

# check_ext ("OpenVDB" "openvdb/nanovdb" 3ce7f9ca2533e9186f984c4d08455d5abb4f0b45)

# target_link_libraries(Ptex_static PRIVATE ${ZLIB_LIBRARIES})
set(deps STATIC 
    ${ZLIB_LIBRARIES}
    Ptex_static)

# target_compile_definitions(Ptex_static PRIVATE "PTEX_STATIC")
add_library(rt_lib INTERFACE)# STATIC 
    # src/rt/rt.cpp)

# target_link_libraries(deps PRIVATE Ptex_static)

# add_library(rt_lib STATIC 
#     src/rt/rt.cpp) 
#     # ${RT_HEADER_FILES}
#     # ${RT_SOURCE_FILES})

list (APPEND RT_DEFINITIONS "PTEX_STATIC")
target_compile_definitions(rt_lib INTERFACE ${RT_DEFINITIONS})

message(STATUS "${ZLIB_INCLUDE_DIRS}")
message(STATUS "${PTEX_INCLUDE}")

target_include_directories(rt_lib INTERFACE 
    ${ZLIB_INCLUDE_DIRS}
    ${PTEX_INCLUDE})
target_compile_options(rt_lib INTERFACE ${RT_CXX_FLAGS})
target_link_libraries(rt_lib INTERFACE ${deps})
# target_link_libraries(rt PRIVATE rt_lib)
#     
# target_compile_definitions(rt PRIVATE ${RT_DEFINITIONS})
# target_compile_options(rt PRIVATE ${RT_CXX_FLAGS})
install (TARGETS 
    rt_lib 
    DESTINATION 
    lib)
