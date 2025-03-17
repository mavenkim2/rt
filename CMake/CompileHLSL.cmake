# function (DFS FILE_CONTENT OUT)
#     string(REGEX MATCH "include" MATCH)
# endfunction()
#
# function (CompileHLSL HLSL_PATH OUT)
#
#     get_filename_component(FILE_NAME ${HLSL_PATH} NAME_WLE)
#     file(READ ${HLSL_PATH} FILE_CONTENT)
#     string(FIND "${FILE_CONTENT}" "numthreads" NUMTHREADS_POS)
#
#     set (DXC_COMMON_FLAGS "-spirv" "-fspv-target-env=vulkan1.3" "-fvk-use-scalar-layout")
#     set (DXC_COMMON_FLAGS ${DXC_COMMON_FLAGS} "$<$<CONFIG:Debug>:-Zi>" "$<$<CONFIG:Debug>:-Qembed_debug")
#     set (SHADER_SPV ${SHADER_SPV_DIR}/${SHADER_NAME}.spv)
#
#     # Compute shader
#     if (${NUMTHREADS_POS} GREATER 0)
#         add_custom_command(
#             OUTPUT ${SHADER_SPV}
#             COMMAND ${SHADER_COMPILER} ${DXC_COMMON_FLAGS} "-T cs_6_7" "-Fo ${SHADER_SPV}" "${HLSL_PATH}"
#             DEPENDS "${HLSL_PATH}"
#             COMMENT "Compiling HLSL compute shader: ${SHADER_FILE} -> ${SHADER_SPV}"
#             VERBATIM)
#     else() 
#         add_custom_command(
#             OUTPUT ${SHADER_SPV}
#             COMMAND ${SHADER_COMPILER} ${DXC_COMMON_FLAGS} "-T lib_6_6" "-Fo ${SHADER_SPV}" "${HLSL_PATH}"
#             DEPENDS "${HLSL_PATH}"
#             COMMENT "Compiling HLSL rt shader: ${SHADER_FILE} -> ${SHADER_SPV}"
#             VERBATIM)
#     endif()
#     set (OUT ${SHADER_SPV} PARENT_SCOPE)
# endfunction()
