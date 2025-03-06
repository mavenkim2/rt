cmake_minimum_required(VERSION 3.20)
project(RT LANGUAGES CXX C)
    
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

if (MSVC)
    add_definitions(/D _CRT_SECURE_NO_WARNINGS /Dstrdup=_strdup
        /wd4018 /wd4100 /wd4101 /wd4127 /wd4146 /wd4232 /wd4242 /wd4244 /wd4245 /wd4267 /wd4305 /wd4309 
        /wd4310 /wd4334 /wd4456 /wd4464 /wd4668 /wd4701 /wd4703 /wd4711 /wd4756 /wd4820 /wd5045 /wd5250)
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options("-Wno-unsafe-buffer-usage")
endif()

# Windows specific
if (CMAKE_SYSTEM_NAME STREQUAL Windows)
    # if (CMAKE_BUILD_TYPE MATCHES "Debug")
    #     set (CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLLDebug)
    # else()
    #     set (CMAKE_MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
    # endif()
    list (APPEND RT_DEFINITIONS "WIN32_LEAN_AND_MEAN")
    set (deps "ws2_32")
else()
    message (FATAL_ERROR "Sorry, only windows supported for now.")
endif()

set (MEM_INCLUDE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt)

string(TOUPPER "${CMAKE_BUILD_TYPE}" uppercase_CMAKE_BUILD_TYPE)

# Visual Studio uses run time configs, so handle that
if (CMAKE_CONFIGURATION_TYPES)
    foreach(config ${CMAKE_CONFIGURATION_TYPES})
        string(TOUPPER "${config}" config_upper)
        # allow asserts in relwithdebinfo mode
        if (config_upper STREQUAL "DEBUG" OR config_upper STREQUAL "RELWITHDEBINFO")
            list (APPEND RT_DEFINITIONS "TRACK_MEMORY" "DEBUG")
        endif()
        if(config_upper STREQUAL "RELWITHDEBINFO")
            if (NOT MSVC)
                add_definitions(-D_DEBUG)
            endif()
            add_compile_options($<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-UNDEBUG>)
            foreach (flags_var_to_scrub
                    # CMAKE_CXX_FLAGS_RELEASE
                    CMAKE_CXX_FLAGS_RELWITHDEBINFO
                    CMAKE_CXX_FLAGS_MINSIZEREL)
                string (REGEX REPLACE "(^| )[/-]D*NDEBUG($| )" " "
                    "${flags_var_to_scrub}" "${${flags_var_to_scrub}}")
            endforeach()
        endif()
    endforeach()
else()
    if (uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG" OR uppercase_CMAKE_BUILD_TYPE STREQUAL "RELWITHDEBINFO")
        list (APPEND RT_DEFINITIONS "TRACK_MEMORY" "DEBUG")
    endif()
    if (uppercase_CMAKE_BUILD_TYPE STREQUAL "RELWITHDEBINFO")
        if (NOT MSVC)
            add_definitions(-D_DEBUG)
        endif()
        add_compile_options($<$<OR:$<COMPILE_LANGUAGE:C>,$<COMPILE_LANGUAGE:CXX>>:-UNDEBUG>)
        foreach (flags_var_to_scrub
                # CMAKE_CXX_FLAGS_RELEASE
                CMAKE_CXX_FLAGS_RELWITHDEBINFO
                CMAKE_CXX_FLAGS_MINSIZEREL)
            string (REGEX REPLACE "(^| )[/-]D*NDEBUG($| )" " "
                "${flags_var_to_scrub}" "${${flags_var_to_scrub}}")
        endforeach()
        # string(REPLACE "-DNDEBUG" "" CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")
    endif()
endif()


add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/src/third_party)

# set (FlagsSSE2 "-D __SSE__ -D __SSE2__")
# set (FlagsSSE42 "${FlagsSSE2} -D __SSE3__ -D __SSSE3__ -D __SSE4_1__ -D __SSE4_2__")

include (CheckCXXCompilerFlag)
check_cxx_compiler_flag("-march=native" COMPILER_SUPPORTS_MARCH_NATIVE)
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(STATUS "Using Clang Compiler")
    list (APPEND RT_CXX_FLAGS 
        "-Wno-unused-variable" 
        "-Wno-unused-but-set-variable" 
        "-Wno-reorder-ctor"
        "-Wno-nullability-completeness"
    )
    if (CMAKE_CXX_COMPILER_VERSION GREATER_EQUAL 14)
        message (STATUS "Disabling -ffp-contract")
        if (CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
            list (APPEND RT_CXX_FLAGS "-fp:precise")
        else ()
            list (APPEND RT_CXX_FLAGS "-ffp-contract=off")
        endif()
    endif()
    if (COMPILER_SUPPORTS_MARCH_NATIVE)
        list (APPEND RT_CXX_FLAGS "-march=native")
    endif()
elseif (MSVC)
    message(STATUS "Using MSVC Compiler")
    list (APPEND RT_CXX_FLAGS "-FC" "-Zi" "-EHsc" "-nologo" "-Oi" "-WX" "-W4" "-wd4305" "-wd4324" "-wd4127" 
        "-wd4700" "-wd4701" "-wd4505" "-wd4189" "-wd4201" "-wd4100" "/arch:SSE2" "/arch:AVX2")
endif ()

if (CMAKE_SYSTEM_NAME STREQUAL Windows)
    list (APPEND RT_DEFINITIONS "NOMINMAX")
else()
    message (SEND_ERROR "Windows only supported")
endif()

list (APPEND RT_DEFINITIONS "__SSE3__" "__SSE4_1__")

# Create thread context static lib (shared w/ ptex)
set (MEM_SOURCE_FILES 
    src/rt/thread_context.cpp
    src/rt/memory.cpp
    src/rt/string.cpp
    src/rt/base.cpp
)

if (CMAKE_SYSTEM_NAME STREQUAL Windows)
    list (APPEND PLATFORM_SOURCE_FILES "src/rt/win32.cpp")
endif()

add_library(thread_context OBJECT
    ${MEM_SOURCE_FILES}
    ${PLATFORM_SOURCE_FILES}
)

target_compile_definitions(thread_context PRIVATE ${RT_DEFINITIONS})
target_compile_options(thread_context PRIVATE ${RT_CXX_FLAGS})
target_link_libraries(thread_context PRIVATE ${deps})

set(deps  
    ${deps}
    # thread_context
    ${ZLIB_LIBRARY}
    ${PTEX_LIBRARY}
    ${OSD_LIBRARY}
)

install (TARGETS thread_context EXPORT THREAD_CONTEXT DESTINATION lib)
install (EXPORT THREAD_CONTEXT DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/thread_context")

# convert
add_executable(convert 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/cmd/convert.cpp
    ${PLATFORM_SOURCE_FILES}
)
target_compile_definitions(convert PRIVATE ${RT_DEFINITIONS})
target_compile_options(convert PRIVATE ${RT_CXX_FLAGS})

# rgb2spec
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/gen)
add_executable(rgb2spec_exe ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/rgb2spec.cpp)
target_compile_definitions(rgb2spec_exe PRIVATE ${RT_DEFINITIONS})
if (NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/gen/rgbspectrum_srgb.cpp")
    add_custom_command(OUTPUT ${CMAKE_CURRENT_SOURCE_DIR}/src/gen/rgbspectrum_srgb.cpp 
        COMMAND rgb2spec_exe ${CMAKE_CURRENT_SOURCE_DIR}/src/gen/rgbspectrum_srgb.cpp
        DEPENDS rgb2spec_exe
    )
endif()

# Main executable
add_executable(rt_exe 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/rt.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gen/rgbspectrum_srgb.cpp
)
target_compile_definitions(rt_exe PRIVATE ${RT_DEFINITIONS})
target_include_directories(rt_exe PRIVATE 
    # ${ZLIB_INCLUDE_DIR}
    ${PTEX_INCLUDE}
    ${OPENSUBDIV_INCLUDE_DIR}
    ${NANOVDB_INCLUDE_DIR}
)

target_compile_options(rt_exe PRIVATE ${RT_CXX_FLAGS})
target_link_libraries(rt_exe PRIVATE ${deps})
set_target_properties(rt_exe PROPERTIES OUTPUT_NAME rt)
