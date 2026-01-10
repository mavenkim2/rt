cmake_minimum_required(VERSION 3.20)
project(RT LANGUAGES CXX C)
    
set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_STANDARD_REQUIRED ON)

option (WITH_CUDA "Build with NVIDIA CUDA. Currently used only for GPU path guiding." ON) 

if (MSVC)
    add_compile_definitions(
        _CRT_SECURE_NO_WARNINGS 
        "strdup=_strdup"
    )
    # add_compile_options($<$<COMPILE_LANGUAGE:CXX>:
    #     /wd4018 /wd4100 /wd4101 /wd4127 /wd4146 /wd4232 /wd4242 /wd4244 /wd4245 /wd4267 /wd4305 /wd4309 
    #     /wd4310 /wd4334 /wd4456 /wd4464 /wd4668 /wd4701 /wd4703 /wd4711 /wd4756 /wd4820 /wd5045 /wd5250
    #     /wd4130 /wd4389 /wd4275>)
    # add_compile_options(
    #     $<$<COMPILE_LANGUAGE:CXX>:
    #         /wd4018;/wd4100;/wd4101;/wd4127;/wd4146;/wd4232;/wd4242;/wd4244;/wd4245;/wd4267;/wd4305;/wd4309;
    #         /wd4310;/wd4334;/wd4456;/wd4464;/wd4668;/wd4701;/wd4703;/wd4711;/wd4756;/wd4820;/wd5045;/wd5250;
    #         /wd4130;/wd4389;/wd4275
    #     >
    # )
endif ()

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options("-Wno-unsafe-buffer-usage")
endif()

# Windows specific
if (CMAKE_SYSTEM_NAME STREQUAL Windows)
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
            set (uppercase_CMAKE_BUILD_TYPE "RELWITHDEBINFO")
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
        elseif(config_upper STREQUAL "DEBUG")
            set (uppercase_CMAKE_BUILD_TYPE "DEBUG")
        elseif(config_upper STREQUAL "RELEASE")
            set (uppercase_CMAKE_BUILD_TYPE "RELEASE")
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

message (STATUS "build type: ${uppercase_CMAKE_BUILD_TYPE}")
add_subdirectory (${CMAKE_CURRENT_SOURCE_DIR}/src/third_party)

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
    if (uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
        list (APPEND RT_CXX_FLAGS "-O0")
    endif()
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
    list (APPEND RT_CXX_FLAGS "-FC" "-Zi" "-EHsc" "-nologo" "-W4" "-wd4305" "-wd4324" "-wd4127" 
        "-wd4700" "-wd4701" "-wd4505" "-wd4189" "-wd4201" "-wd4100" "/arch:SSE2" "/arch:AVX2")
endif ()

if (CMAKE_SYSTEM_NAME STREQUAL Windows)
    list (APPEND RT_DEFINITIONS "NOMINMAX")
    list (APPEND RT_DEFINITIONS "VK_USE_PLATFORM_WIN32_KHR")
else()
    message (SEND_ERROR "Windows only supported")
endif()

if (uppercase_CMAKE_BUILD_TYPE STREQUAL "DEBUG")
    list (APPEND SHADER_FLAGS "-Zi")
endif()

list (APPEND RT_DEFINITIONS "__SSE3__" "__SSE4_1__")

# GPU or CPU path
option(USE_GPU "USE_GPU" OFF)

# Create thread context static lib (shared w/ ptex)
set (MEM_SOURCE_FILES 
    src/rt/thread_context.cpp
    src/rt/thread_statistics.cpp
    src/rt/memory.cpp
    src/rt/string.cpp
    src/rt/base.cpp
)

set (RT_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/debug.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/integrate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/lights.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/scene.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/simd_integrate.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/spectrum.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/parallel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/tests/test.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/bvh/bvh_types.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/low_discrepancy.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/tables/primes.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/tables/sobolmatrices.hpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/sampler.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/graphics/ptex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/bit_packing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/virtual_geometry/virtual_geometry_manager.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/scene/scene.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/image.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/graphics/render_graph.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/volume.cpp
)

if (NOT USE_GPU)
    list (APPEND RT_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/subdivision.cpp)
endif()

set (RT_VULKAN_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/graphics/vulkan.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/vulkan/volk.c
)

set (VULKAN_INCLUDE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/src/third_party/vulkan
)

if (USE_GPU)
    list (APPEND RT_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/gpu_scene.cpp)
    list (APPEND RT_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/photon_mapping/photon_mapping.cpp)

    if (WITH_CUDA)
        list (APPEND RT_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/gpu/path_guiding.cpp)
        list (APPEND RT_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/gpu/cuda/cuda_device.cpp)
    endif()
    list (APPEND RT_DEFINITIONS "USE_GPU")
else()
    list (APPEND RT_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/cpu_scene.cpp)
endif()

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
    ${ZLIB_LIBRARY}
    ${PTEX_LIBRARY}
    ${OSD_LIBRARY}
    ${NVAPI_LIBRARY}
)

install (TARGETS thread_context EXPORT THREAD_CONTEXT DESTINATION lib)
install (EXPORT THREAD_CONTEXT DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/thread_context")

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

# shader compiles
if (USE_GPU)
    set(SHADER_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/shaders)
    set(SHADER_SPV_DIR ${CMAKE_CURRENT_BINARY_DIR}/src/shaders)
    set(SHADER_FILES "${SHADER_SRC_DIR}/render_raytrace_rgen.hlsl" 
        "${SHADER_SRC_DIR}/render_raytrace_miss.hlsl"
        "${SHADER_SRC_DIR}/render_raytrace_hit.hlsl"
        "${SHADER_SRC_DIR}/clas/decode_dgf_clusters.hlsl"
        "${SHADER_SRC_DIR}/clas/decode_voxel_aabbs.hlsl"
        "${SHADER_SRC_DIR}/clas/fill_cluster_triangle_info.hlsl"
        "${SHADER_SRC_DIR}/compress/block_compress.hlsl"
        "${SHADER_SRC_DIR}/tex/update_page_tables.hlsl"
        "${SHADER_SRC_DIR}/tex/generate_mips.hlsl"
        "${SHADER_SRC_DIR}/clas/prepare_indirect.hlsl"
        "${SHADER_SRC_DIR}/clas/test.hlsl"
        "${SHADER_SRC_DIR}/clas/clas_defrag.hlsl"
        "${SHADER_SRC_DIR}/clas/compute_clas_addresses.hlsl"
        "${SHADER_SRC_DIR}/clas/write_clas_defrag_addresses.hlsl"
        "${SHADER_SRC_DIR}/clas/compute_blas_addresses.hlsl"
        "${SHADER_SRC_DIR}/clas/ptlas_write_instances.hlsl"
        "${SHADER_SRC_DIR}/clas/cluster_fixup.hlsl"
        "${SHADER_SRC_DIR}/clas/ptlas_update_unused_instances.hlsl"
        "${SHADER_SRC_DIR}/clas/ptlas_write_command_infos.hlsl"
        "${SHADER_SRC_DIR}/clas/initialize_instance_freelist.hlsl"
        "${SHADER_SRC_DIR}/clas/ptlas_update_partitions.hlsl"
        "${SHADER_SRC_DIR}/clas/decode_merged_instances.hlsl"
        "${SHADER_SRC_DIR}/clas/merged_instances_test.hlsl"
        "${SHADER_SRC_DIR}/clas/allocate_instances.hlsl"
        "${SHADER_SRC_DIR}/clas/free_instances.hlsl"
        "${SHADER_SRC_DIR}/calculate_motion_vectors.hlsl"
        "${SHADER_SRC_DIR}/clas/task.hlsl"
        "${SHADER_SRC_DIR}/clas/reproject_depth.hlsl"
        "${SHADER_SRC_DIR}/tex/generate_mips_naive.hlsl"
        "${SHADER_SRC_DIR}/clas/compact_freelist.hlsl"
        "${SHADER_SRC_DIR}/tex/clear_page_table.hlsl"
        "${SHADER_SRC_DIR}/accumulate_frames.hlsl"
        "${SHADER_SRC_DIR}/copy_denoised_output.hlsl"
        "${SHADER_SRC_DIR}/wavefront/shading_kernel.hlsl"
        # "${SHADER_SRC_DIR}/photon_mapping/photons.hlsl"
        "${SHADER_SRC_DIR}/sort/downsweep.hlsl"
        "${SHADER_SRC_DIR}/sort/upsweep.hlsl"
        "${SHADER_SRC_DIR}/sort/spine.hlsl"
        "${SHADER_SRC_DIR}/kd_tree/update_tags.hlsl"
        "${SHADER_SRC_DIR}/kd_tree/calculate_bounds.hlsl"
        "${SHADER_SRC_DIR}/kd_tree/create_sort_keys.hlsl"
        "${SHADER_SRC_DIR}/kd_tree/kd_tree_prepare_indirect.hlsl"
        "${SHADER_SRC_DIR}/photon_mapping/generate_random_points.hlsl"
        "${SHADER_SRC_DIR}/kd_tree/finalize_calculate_bounds.hlsl"
        "${SHADER_SRC_DIR}/clas/decode_clusters_no_clas.hlsl"
        "${SHADER_SRC_DIR}/kd_tree/initialize_indices.hlsl"
        "${SHADER_SRC_DIR}/volume/volume_rgen.hlsl"
        "${SHADER_SRC_DIR}/path_guiding/wem.hlsl"
        "${SHADER_SRC_DIR}/path_guiding/map.hlsl"
        "${SHADER_SRC_DIR}/path_guiding/initialize_vmms.hlsl"
        "${SHADER_SRC_DIR}/path_guiding/prefix_sum.hlsl"
    )
    set_property(SOURCE 
        "${SHADER_SRC_DIR}/clas/task.hlsl"
        PROPERTY name "task")
    set_property(SOURCE 
        "${SHADER_SRC_DIR}/render_raytrace_rgen.hlsl" 
        "${SHADER_SRC_DIR}/render_raytrace_miss.hlsl"
        "${SHADER_SRC_DIR}/render_raytrace_hit.hlsl"
        "${SHADER_SRC_DIR}/volume/volume_rgen.hlsl"
        # "${SHADER_SRC_DIR}/photon_mapping/photons.hlsl"
        PROPERTY name "rt")
    set_property(SOURCE 
        "${SHADER_SRC_DIR}/compress/block_compress.hlsl"
        "${SHADER_SRC_DIR}/tex/update_page_tables.hlsl"
        "${SHADER_SRC_DIR}/clas/decode_dgf_clusters.hlsl"
        "${SHADER_SRC_DIR}/clas/decode_voxel_aabbs.hlsl"
        "${SHADER_SRC_DIR}/clas/fill_cluster_triangle_info.hlsl"
        "${SHADER_SRC_DIR}/tex/generate_mips.hlsl"
        "${SHADER_SRC_DIR}/clas/prepare_indirect.hlsl"
        "${SHADER_SRC_DIR}/clas/test.hlsl"
        "${SHADER_SRC_DIR}/clas/clas_defrag.hlsl"
        "${SHADER_SRC_DIR}/clas/compute_clas_addresses.hlsl"
        "${SHADER_SRC_DIR}/clas/write_clas_defrag_addresses.hlsl"
        "${SHADER_SRC_DIR}/clas/compute_blas_addresses.hlsl"
        "${SHADER_SRC_DIR}/clas/ptlas_write_instances.hlsl"
        "${SHADER_SRC_DIR}/clas/cluster_fixup.hlsl"
        "${SHADER_SRC_DIR}/clas/ptlas_update_unused_instances.hlsl"
        "${SHADER_SRC_DIR}/clas/ptlas_write_command_infos.hlsl"
        "${SHADER_SRC_DIR}/clas/initialize_instance_freelist.hlsl"
        "${SHADER_SRC_DIR}/clas/ptlas_update_partitions.hlsl"
        "${SHADER_SRC_DIR}/clas/decode_merged_instances.hlsl"
        "${SHADER_SRC_DIR}/clas/merged_instances_test.hlsl"
        "${SHADER_SRC_DIR}/clas/allocate_instances.hlsl"
        "${SHADER_SRC_DIR}/clas/free_instances.hlsl"
        "${SHADER_SRC_DIR}/calculate_motion_vectors.hlsl"
        "${SHADER_SRC_DIR}/clas/reproject_depth.hlsl"
        "${SHADER_SRC_DIR}/tex/generate_mips_naive.hlsl"
        "${SHADER_SRC_DIR}/clas/compact_freelist.hlsl"
        "${SHADER_SRC_DIR}/tex/clear_page_table.hlsl"
        "${SHADER_SRC_DIR}/accumulate_frames.hlsl"
        "${SHADER_SRC_DIR}/copy_denoised_output.hlsl"
        "${SHADER_SRC_DIR}/wavefront/shading_kernel.hlsl"
        "${SHADER_SRC_DIR}/sort/downsweep.hlsl"
        "${SHADER_SRC_DIR}/sort/upsweep.hlsl"
        "${SHADER_SRC_DIR}/sort/spine.hlsl"
        "${SHADER_SRC_DIR}/kd_tree/update_tags.hlsl"
        "${SHADER_SRC_DIR}/kd_tree/calculate_bounds.hlsl"
        "${SHADER_SRC_DIR}/kd_tree/create_sort_keys.hlsl"
        "${SHADER_SRC_DIR}/kd_tree/kd_tree_prepare_indirect.hlsl"
        "${SHADER_SRC_DIR}/photon_mapping/generate_random_points.hlsl"
        "${SHADER_SRC_DIR}/kd_tree/finalize_calculate_bounds.hlsl"
        "${SHADER_SRC_DIR}/clas/decode_clusters_no_clas.hlsl"
        "${SHADER_SRC_DIR}/kd_tree/initialize_indices.hlsl"
        "${SHADER_SRC_DIR}/path_guiding/wem.hlsl"
        "${SHADER_SRC_DIR}/path_guiding/map.hlsl"
        "${SHADER_SRC_DIR}/path_guiding/initialize_vmms.hlsl"
        "${SHADER_SRC_DIR}/path_guiding/prefix_sum.hlsl"
        PROPERTY name "cs")
    
    file(GLOB_RECURSE SHADER_INCLUDES "${SHADER_SRC_DIR}/*.hlsl" "${SHADER_SRC_DIR}/*.hlsli")
    file(GLOB_RECURSE SHADER_INTEROP "${CMAKE_CURRENT_SOURCE_DIR}/src/rt/shader_interop/*.h")
    
    if (NOT DEFINED SHADER_COMPILER)
        set(SHADER_COMPILER "dxc")
    endif()
    
    foreach(SHADER_FILE ${SHADER_FILES})
        get_filename_component(SHADER_NAME ${SHADER_FILE} NAME_WE)
        get_property(SHADER_TYPE SOURCE ${SHADER_FILE} PROPERTY name)
        set(SHADER_SPV ${SHADER_SPV_DIR}/${SHADER_NAME}.spv)
        # set(SHADER_SPV ${SHADER_SPV_DIR}/${SHADER_NAME}.cso)
    
        if (${SHADER_TYPE} STREQUAL "rt")
            set (SHADER_TYPE_FLAG "-T lib_6_6")
        elseif (${SHADER_TYPE} STREQUAL "cs")
            set (SHADER_TYPE_FLAG "-T cs_6_6")
        elseif (${SHADER_TYPE} STREQUAL "task")
            set (SHADER_TYPE_FLAG "-T as_6_6")
        endif()
    
        add_custom_command(
            OUTPUT ${SHADER_SPV}
            COMMAND ${SHADER_COMPILER} 
            "-spirv" "-fspv-target-env=vulkan1.3" "-fvk-use-scalar-layout" 
            "-Wno-parameter-usage"
            ${SHADER_TYPE_FLAG} "$<$<OR:$<CONFIG:Debug>,$<CONFIG:RelWithDebInfo>>:-Zi>" 
            "$<$<CONFIG:Release>:-O3>"
            "-Fo ${SHADER_SPV}" "${SHADER_FILE}" 
            DEPENDS ${SHADER_FILE} ${SHADER_INCLUDES} ${SHADER_INTEROP}
            COMMENT "Compiling HLSL shader: ${SHADER_FILE} -> ${SHADER_SPV}"
            VERBATIM
        )
        list (APPEND SHADER_SPV_FILES ${SHADER_SPV})
    endforeach()
    
    add_custom_target(shaders DEPENDS ${SHADER_SPV_FILES})
endif()

# convert
set (CONVERT_SOURCE_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/cmd/convert.cpp
    ${RT_VULKAN_SOURCE_FILES}
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/graphics/ptex.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/dgfs.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/debug.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/bit_packing.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/parallel.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/scene/scene.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/bvh/bvh_types.cpp
)

if (USE_GPU)
    list (APPEND CONVERT_SOURCE_FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/virtual_geometry/mesh_simplification.cpp)
endif()

add_executable(convert ${CONVERT_SOURCE_FILES})

target_include_directories(convert PRIVATE 
    ${PTEX_INCLUDE}
    ${NANOVDB_INCLUDE_DIR}
    ${ZLIB_INCLUDE_DIR}
    ${VULKAN_INCLUDE_DIR}
    ${OPENVDB_INCLUDE_DIR}
    ${NANOVDB_INCLUDE_DIR}
)

add_custom_command(TARGET convert POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:openvdb_shared>
        $<TARGET_FILE_DIR:convert>

    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:blosc_shared>
        $<TARGET_FILE_DIR:convert>
)

target_link_libraries(convert PRIVATE ${PTEX_LIBRARY} ${GK_LIBRARY} ${METIS_LIBRARY} ${OPENVDB_LIBRARY} ${NANOVDB_LIBRARY})
target_compile_definitions(convert PRIVATE ${RT_DEFINITIONS})
target_compile_options(convert PRIVATE ${RT_CXX_FLAGS})

# CUDA support
include(CheckLanguage)
if (WITH_CUDA)
    check_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    if (CMAKE_CUDA_COMPILER)
        enable_language(CUDA)
        message(STATUS "CUDA Found: ${CMAKE_CUDA_COMPILER} (${CUDA_VERSION})")
        list (APPEND deps CUDA::cuda_driver)
    endif()
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/src/rt/gpu)

# message(STATUS "CUDA Exe: ${CUDA_NVCC_EXECUTABLE}")
#
# if (CMAKE_CUDA_COMPILER)
#     enable_language (CUDA)
#
#     message(STATUS "CUDA Compiler: ${CMAKE_CUDA_COMPILER}")
#     message(STATUS "CUDA Architecture: ${CMAKE_CUDA_ARCHITECTURES}")
#     message(STATUS "CUDA Exe: ${CUDA_NVCC_EXECUTABLE}")
#     set(CMAKE_CUDA_ARCHITECTURES "native")
#     # set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -lineinfo")
#     add_executable(rt_cuda_exe
#         ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/gpu/test.cu
#     )
#
#     target_compile_options(rt_cuda_exe PRIVATE "-lineinfo")
#     set_target_properties(rt_cuda_exe PROPERTIES
#         CUDA_ARCHITECTURES native
#         CMAKE_CUDA_STANDARD 17
#         CMAKE_CUDA_STANDARD_REQUIRED ON
#     )
# else()
#     message (STATUS "could not find cuda compiler")
# endif()

# Main executable
add_executable(rt_exe 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/rt/rt.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/src/gen/rgbspectrum_srgb.cpp
    ${RT_SOURCE_FILES}
    ${RT_VULKAN_SOURCE_FILES}
)

if (USE_GPU)
    add_dependencies(rt_exe shaders)
else()
    list (APPEND RT_DEFINITIONS "USE_SUBDIVISION")
endif()

target_compile_definitions(rt_exe PRIVATE ${RT_DEFINITIONS})
target_include_directories(rt_exe PRIVATE 
    ${PTEX_INCLUDE}
    ${OPENSUBDIV_INCLUDE_DIR}
    ${NANOVDB_INCLUDE_DIR}
    ${VULKAN_INCLUDE_DIR}
    ${ZLIB_INCLUDE_DIR}
)

target_compile_options(rt_exe PRIVATE ${RT_CXX_FLAGS}
    $<$<CONFIG:Debug>:-O0>
    $<$<CONFIG:RelWithDebInfo>:-O2>
    $<$<CONFIG:Release>:-O2>)
target_link_libraries(rt_exe PRIVATE ${deps})
set_target_properties(rt_exe PROPERTIES OUTPUT_NAME rt)
